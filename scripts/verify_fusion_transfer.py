"""Synthetic mechanism checks; no data loading, optimizer, or scientific run."""
import gc
import io
import time

import torch
from torch import nn

from scripts.prepare_sr_native import read, write


def preflight(jobs, dest):
    from src import config
    from src.config import ConfigManager, ExperimentConfig
    from src.data import set_random_seed
    from src.models.transmixer_sr import adapter as baseline_adapter
    from src.models.transmixer_fusion_v2 import adapter
    from src.models.transmixer_fusion_v2.model import PairCritic, band_features, frequency_masks
    report = {"scope": "synthetic interfaces only; no real data, optimizer, checkpoint training or metric calculation", "jobs": []}
    device = "cuda:0"
    x = torch.randn(2, 64, 11, 11, device=device, requires_grad=True)
    bands = band_features(x)
    torch.testing.assert_close(sum(bands), x, rtol=1e-5, atol=1e-5)
    masks = frequency_masks(11, 11, device)
    assert torch.all(masks.sum(0) == 1)
    sum(b.square().mean() for b in bands).backward()
    assert torch.isfinite(x.grad).all()
    report["frequency_reconstruction"] = {"passed": True, "band_sizes": masks.flatten(1).sum(1).tolist()}
    critic = PairCritic().to(device)
    a, b = [torch.randn(16, 32, device=device, requires_grad=True) for _ in range(2)]
    for upper in (False, True):
        representation, fit = critic.objectives(a, b, upper=upper)
        critic_grad = torch.autograd.grad(representation, list(critic.parameters()), allow_unused=True, retain_graph=True)
        feature_grad = torch.autograd.grad(fit, (a, b), allow_unused=True, retain_graph=True)
        assert all(g is None for g in critic_grad) and all(g is None for g in feature_grad)
        assert any(g.abs().sum() > 0 for g in torch.autograd.grad(representation, (a, b), retain_graph=True))
        assert any(g.abs().sum() > 0 for g in torch.autograd.grad(fit, list(critic.parameters())))
    report["critic_gradient_isolation"] = {"passed": True, "upper_and_lower": True}
    del x, bands, critic, a, b, representation, fit
    configs = {j["group"]: read(j["config"]) for j in jobs}
    baseline_cfg = configs["A00"]
    config._global_config_manager = ConfigManager(ExperimentConfig(**baseline_cfg))
    set_random_seed(6)
    base = baseline_adapter.build_model(config, 0, device)["net"].eval()
    base_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    h = torch.randn(2, 1, 30, 11, 11, device=device)
    l = torch.randn(2, 1, 11, 11, device=device)
    with torch.no_grad():
        base_logits = base(h, l)
    base_parameters = sum(p.numel() for p in base.parameters())
    del base
    for job in jobs:
        if job["reused"]:
            continue
        cfg = configs[job["group"]]
        config._global_config_manager = ConfigManager(ExperimentConfig(**cfg))
        set_random_seed(6)
        bundle = adapter.build_model(config, 0, device)
        net, stage = bundle["net"], bundle["net"].fusion_conv
        state = net.state_dict()
        for k, v in base_state.items():
            new_key = k.replace("fusion_conv.", "fusion_conv.original.") if k.startswith("fusion_conv.") else k
            torch.testing.assert_close(v, state[new_key].cpu(), rtol=0, atol=0)
        net.eval()
        with torch.no_grad():
            torch.testing.assert_close(net(h, l), base_logits, rtol=0, atol=0)
        for key in ("alpha_h", "alpha_l", "alpha_detail"):
            if hasattr(stage, key):
                assert getattr(stage, key).item() == 0
                getattr(stage, key).data.fill_(0.1)  # Synthetic gradient check ONLY.
        batch = {"hsi_pca": torch.randn(128, 1, 30, 11, 11, device=device),
                 "aux": torch.randn(128, 1, 11, 11, device=device),
                 "label": torch.arange(128, device=device) % 15}
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        net.train()
        output = adapter.forward_train(bundle, batch)
        loss = adapter.compute_loss(bundle, output, batch, nn.CrossEntropyLoss())
        assert output.shape == (128, 15) and torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        grads = {name: float(p.grad.abs().sum()) if p.grad is not None else 0.0 for name, p in stage.named_parameters()
                 if not name.startswith("original.")}
        assert all(v > 0 for v in grads.values()), {k: v for k, v in grads.items() if v <= 0}
        if job["group"].startswith("D"):
            assert stage.interaction.cache is None
        if job["group"] in ("U02", "U03", "U04"):
            assert stage.interaction.kl is None
        torch.cuda.synchronize()
        training_check_seconds = time.monotonic() - start
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        total_parameters = sum(p.numel() for p in net.parameters())
        train_only = sum(p.numel() for name, p in net.named_parameters() if any(
            token in name for token in (".reconstruct.", ".aux_heads.", "_critic.", ".task_critics.", ".label_embedding.")))
        net.eval()
        eval_batch = {"hsi_pca": batch["hsi_pca"], "aux": batch["aux"]}  # Deliberately no labels.
        with torch.no_grad():
            first = adapter.forward_eval(bundle, eval_batch)
            second = adapter.forward_eval(bundle, eval_batch)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            assert torch.isfinite(first).all()
            torch.cuda.synchronize()
            start = time.monotonic()
            for _ in range(3):
                adapter.forward_eval(bundle, eval_batch)
            torch.cuda.synchronize()
            eval_seconds = (time.monotonic() - start) / 3
        # Whole-model serialization must preserve the shallow hooks and clear caches.
        memory = io.BytesIO()
        torch.save(net, memory)
        memory.seek(0)
        restored = torch.load(memory, map_location=device, weights_only=False).eval()
        with torch.no_grad():
            torch.testing.assert_close(adapter.forward_eval({"net": restored}, eval_batch), first, rtol=0, atol=0)
        report["jobs"].append({"id": job["id"], "passed": True, "original_state_and_disabled_logits_bit_exact": True,
            "batch_size": 128, "parameters": total_parameters, "baseline_parameters": base_parameters,
            "new_parameters": total_parameters - base_parameters, "training_only_parameters": train_only,
            "inference_parameters": total_parameters - train_only, "peak_synthetic_allocated_MiB": peak,
            "synthetic_train_forward_backward_seconds": training_check_seconds,
            "synthetic_eval_batch128_seconds_after_two_warmups_mean_of_three": eval_seconds,
            "all_new_parameters_have_finite_nonzero_gradients_with_temporary_residual_0_1": True,
            "whole_checkpoint_roundtrip": True, "last_diagnostics": stage.diagnostics()})
        write(dest / "preflight.json", report)
        print(f"PREFLIGHT_OK {job['id']} parameters={total_parameters} peak_MiB={peak:.1f}", flush=True)
        del restored, net, stage, bundle, state, batch, output, loss, first, second, memory, eval_batch
        gc.collect()
        torch.cuda.empty_cache()
    sizes = {j["id"].split("_")[0]: j["new_parameters"] for j in report["jobs"]}
    assert sizes["F01"] == sizes["F03"] == sizes["F04"]
    assert sizes["U03"] == sizes["U04"]
    report["active_capacity_controls"] = {"F01_F03_F04_equal": True, "U03_U04_equal": True,
        "U02_confidence_head_removed_parameter_difference": sizes["U03"] - sizes["U02"]}
    # Reject an unapproved seed without allocating a model.
    invalid = {**configs["F01"], "random_seed": 7}
    config._global_config_manager = ConfigManager(ExperimentConfig(**invalid))
    try:
        adapter.build_model(config, 0, device)
    except ValueError:
        report["unapproved_seed_rejected"] = True
    else:
        raise AssertionError("Unapproved seed accepted")
    write(dest / "preflight.json", report)
