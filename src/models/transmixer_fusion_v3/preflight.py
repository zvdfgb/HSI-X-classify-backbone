"""Interface/gradient audits only: no optimizer steps, data or metric replacement."""
import gc
import io
import math
import time
import torch
from . import adapter
from .model import GROUPS, intervention_masks, gaussian_projection, sparse_frequency, unbalanced_ot
from ..transmixer_paper.model import NeighborTransMixer


def run_checks(device="cpu", native_batch=None):
    from src import config
    from src.config import ExperimentConfig
    from scripts.prepare_sr_native import ROOT, read
    torch.set_num_threads(4)
    results = []
    rng = torch.Generator().manual_seed(884)
    small = {"hsi_pca": torch.randn(2, 1, 30, 11, 11, generator=rng).to(device),
             "aux": torch.randn(2, 1, 11, 11, generator=rng).to(device),
             "label": torch.tensor([0, 1], device=device)}
    batch = native_batch or small
    for seed in ((6,) if native_batch is not None else (6, 7, 8)):
        cfg = read(ROOT/"configs/runs/sr_native_20261029"/f"M3_seed{seed}.json")
        for group in GROUPS:
            start = time.monotonic()
            torch.manual_seed(seed)
            baseline = NeighborTransMixer("SR2", **cfg["sr"], hsi_channels=30, aux_channels=1,
                num_classes=15, hidden_dim=64, num_blocks=2, window_size=11, mode="decoupled").to(device).eval()
            rng_after_base = torch.get_rng_state().clone()
            cuda_after_base = torch.cuda.get_rng_state().clone() if device != "cpu" else None
            config._init(ExperimentConfig(**{**cfg, "model_name": "TransMixer_FV3_"+group,
                "paper_transfer": {"group": group}}))
            torch.manual_seed(seed)
            bundle = adapter.build_model(config, 0, device)
            net, stage = bundle["net"], bundle["net"].fusion_conv
            assert torch.equal(rng_after_base, torch.get_rng_state()), (seed, group, "CPU RNG")
            if device != "cpu":
                assert torch.equal(cuda_after_base, torch.cuda.get_rng_state()), (seed, group, "CUDA RNG")
            current = net.state_dict()
            for key, tensor in baseline.state_dict().items():
                target = key.replace("fusion_conv.", "fusion_conv.original.", 1) if key.startswith("fusion_conv.") else key
                assert torch.equal(tensor, current[target]), (seed, group, key)
            net.eval(); stage.disabled = True
            with torch.no_grad():
                ref = baseline(small["hsi_pca"], small["aux"])
                disabled = adapter.forward_eval(bundle, small)
            torch.testing.assert_close(ref, disabled, rtol=0, atol=0)
            stage.disabled = False
            counts = {"stem": 0, "fusion": 0, "pool": 0, "head": 0}
            handles = []
            for key, module in (("stem", net.hsi_stem), ("fusion", stage.original), ("pool", net.gap), ("head", net.classifier)):
                def count(module, args, output, name=key):
                    counts[name] += 1
                handles.append(module.register_forward_hook(count))
            net.train()
            if device != "cpu":
                torch.cuda.reset_peak_memory_stats()
            output = adapter.forward_train(bundle, batch)
            loss = adapter.compute_loss(bundle, output, batch, torch.nn.CrossEntropyLoss())
            loss.backward()
            for handle in handles:
                handle.remove()
            assert counts == dict.fromkeys(counts, 1), counts
            assert output.shape == (len(batch["label"]), 15) and torch.isfinite(loss)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
            assert all(p.grad is not None for p in stage.interaction.parameters()), (seed, group, "unused trainable parameters")
            assert any(p.grad.abs().sum() > 0 for p in stage.interaction.parameters()), (seed, group, "no new gradients")
            assert stage.cache is None and stage.step == 1
            assert all(stage.history[-1][k+"_gradient_norm"] > 0 for k in ("h", "l"))
            net.eval()
            with torch.no_grad():
                a = adapter.forward_eval(bundle, small)
                b = adapter.forward_eval(bundle, small)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            stream = io.BytesIO(); torch.save(net, stream); stream.seek(0)
            restored = torch.load(stream, map_location=device, weights_only=False).eval()
            assert restored.fusion_conv._sr_pool is restored.gap
            with torch.no_grad():
                actual = restored(small["hsi_pca"], small["aux"])
            torch.testing.assert_close(a, actual, rtol=0, atol=0)
            count_base = sum(p.numel() for p in baseline.parameters())
            count_new = sum(p.numel() for p in net.parameters())
            if group.startswith("I"):
                assert count_new-count_base == 8320
            if group == "I04":
                torch.testing.assert_close(stage.random_projection.cpu(), gaussian_projection(), rtol=0, atol=0)
            results.append({"seed": seed, "group": group, "device": device, "batch_size": len(batch["label"]),
                "passed": True, "native_shared_initialization_exact": True, "rng_preserved": True,
                "disabled_logits_exact": True, "single_native_forward_per_batch": counts,
                "finite_loss_and_gradients": True, "all_new_parameters_receive_gradients": True,
                "checkpoint_roundtrip_exact": True, "parameters": count_new, "baseline_parameters": count_base,
                "peak_allocated_bytes_including_reference_model": torch.cuda.max_memory_allocated() if device != "cpu" else None,
                "elapsed_seconds": time.monotonic()-start})
            print("PREFLIGHT_OK", device, group, seed, flush=True)
            del baseline, bundle, net, stage, current, restored, stream, output, loss, a, b, actual, ref, disabled
            gc.collect()
            if device != "cpu":
                torch.cuda.empty_cache()
    torch.manual_seed(991)
    state = torch.get_rng_state().clone()
    a, b, shared = intervention_masks(8, 6, 30, "cpu")
    a2, b2, shared2 = intervention_masks(8, 6, 30, "cpu")
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(a, a2) and torch.equal(b, b2) and torch.equal(shared, shared2)
    assert all(torch.all(m[:, :, 5, 5] == 1) and torch.all((1-m).sum((1, 2, 3)) == 9) for m in (a, b, shared))
    assert torch.all((1-a)*(1-b) == 0)
    x = torch.randn(4, 64, requires_grad=True)
    sf = sparse_frequency(x)
    assert sf.shape == x.shape and torch.isfinite(sf).all()
    ot, _ = unbalanced_ot(torch.nn.functional.normalize(x, dim=1), torch.nn.functional.normalize(x.flip(0), dim=1))
    ot.backward(); assert torch.isfinite(ot) and torch.isfinite(x.grad).all()
    for update in ({"random_seed": 9}, {"epochs": 99}, {"paper_transfer": {"group": "C04", "extra": 1}}):
        config._init(ExperimentConfig(**{**cfg, "model_name": "TransMixer_FV3_C04",
            "paper_transfer": {"group": "C04"}, **update}))
        try:
            adapter.build_model(config, 0, "cpu")
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid protocol accepted")
    return {"all_passed": True, "checks": results,
            "paired_center_preserving_masks": True, "mask_rng_isolated": True,
            "fft_ot_finite_gradients": True, "invalid_protocol_rejected": True}


def inference_costs(batch):
    """No training. Estimate clean-path MACs and measure actual batch128 latency."""
    from scripts.prepare_sr_native import ROOT, read
    from src import config
    from src.config import ExperimentConfig
    from .model import CrossAttention, CrossModalRelations
    from ..transmixer_net.model import SelectiveSSM2D
    cfg = read(ROOT/"configs/runs/sr_native_20261029/M3_seed6.json")
    small = {k: v[:1] for k,v in batch.items()}
    rows = []
    for group in ("A0", *GROUPS):
        torch.manual_seed(6)
        if group == "A0":
            net = NeighborTransMixer("SR2", **cfg["sr"], hsi_channels=30, aux_channels=1,
                num_classes=15, hidden_dim=64, num_blocks=2, window_size=11, mode="decoupled").to("cuda:0")
        else:
            config._init(ExperimentConfig(**{**cfg, "model_name": "TransMixer_FV3_"+group,
                "paper_transfer": {"group": group}}))
            net = adapter.build_model(config, 0, "cuda:0")["net"]
        net.eval()
        total, handles = [0], []
        def count(module, args, output):
            if isinstance(module, torch.nn.modules.conv._ConvNd):
                total[0] += output.numel() * module.in_channels//module.groups * math.prod(module.kernel_size)
            elif isinstance(module, torch.nn.Linear):
                total[0] += output.numel()*module.in_features
            elif isinstance(module, torch.nn.MultiheadAttention):
                b, n, d = args[0].shape
                total[0] += 4*b*n*d*d+2*b*n*n*d
            elif isinstance(module, SelectiveSSM2D):
                b,n,d = args[0].shape
                total[0] += 2*b*n*d*module.d_state
            elif isinstance(module, CrossModalRelations):
                b,d,h,w = args[0].shape; n=h*w
                total[0] += 2*b*n*n*d+2*b*d*d*n
            elif isinstance(module, CrossAttention):
                b,d,h,w = args[0].shape
                total[0] += 2*b*h*w*16*d
        for module in net.modules():
            handles.append(module.register_forward_hook(count))
        with torch.no_grad():
            net(small["hsi_pca"], small["aux"])
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            for _ in range(3):
                net(batch["hsi_pca"], batch["aux"])
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(5):
                net(batch["hsi_pca"], batch["aux"])
            torch.cuda.synchronize()
        rows.append({"group":group,"parameters":sum(p.numel() for p in net.parameters()),
            "estimated_clean_macs_per_patch":total[0], "batch_size":len(batch["label"]),
            "warmup_iterations":3,"measured_iterations":5,"batch_latency_ms":(time.perf_counter()-start)*1000/5})
        print("COST_AUDIT_OK",group,flush=True)
        del net; gc.collect(); torch.cuda.empty_cache()
    return {"runs":rows, "mac_estimator":"Conv/Linear/MHA, two explicit cross-attention matrix products, SSM recurrence/readout; excludes nonlinear/normalization/FFT and auxiliary training losses", "not_an_exact_total_flop_count":True}


if __name__ == "__main__":
    from scripts.prepare_sr_native import ROOT, write
    result = run_checks()
    write(ROOT/"docs/experiments/provenance/fusion_transfer_v3_seed678_20261104/cpu_preflight.json", result)
