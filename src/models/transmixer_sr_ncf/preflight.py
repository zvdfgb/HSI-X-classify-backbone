"""Interface, paired-RNG, exact-parameter, operator-equivalence, gradient, and cost preflight for SR-NCF K00-K06."""
import gc
import io
import math
import time
import torch
from . import adapter
from .model import GROUPS, SRNCFFusionStage, fixed_derangement, local_mean_one_tap_weights
from ..transmixer_paper.model import NeighborTransMixer

EXPECTED_PARAMETERS = {
    "K00": 108615,
    "K01": 108111,
    "K02": 108111,
    "K03": 108111,
    "K04": 108111,
    "K05": 108111,
    "K06": 108111,
}


def run_checks(device="cpu", native_batch=None):
    from src import config
    from src.config import ExperimentConfig
    from scripts.prepare_sr_native import ROOT, read

    torch.set_num_threads(4)
    results = []
    rng = torch.Generator().manual_seed(907)
    small = {
        "hsi_pca": torch.randn(2, 1, 30, 11, 11, generator=rng).to(device),
        "aux": torch.randn(2, 1, 11, 11, generator=rng).to(device),
        "label": torch.tensor([0, 1], device=device),
    }
    batch = native_batch or small

    # 1. Verify local_mean_one_tap_weights mathematical properties (valid mean == 1.0, uniform == 1.0, out == 0.0).
    ones = torch.ones((1, 1, 11, 11), device=device, dtype=torch.float32)
    for d in (1, 2):
        mask = torch.nn.functional.unfold(ones, kernel_size=3, dilation=d, padding=d)
        n_val = mask.sum(dim=1)
        w_rand = torch.softmax(torch.randn(3, 1, 121, generator=rng).to(device) * 2.0, dim=-1).view(3, 1, 11, 11)
        alpha = local_mean_one_tap_weights(w_rand, d, mask)
        assert alpha.shape == (3, 1, 9, 121) and torch.isfinite(alpha).all() and (alpha >= 0).all()
        # Out-of-window taps must be strictly 0
        assert torch.all(alpha.squeeze(1)[:, mask[0] == 0] == 0.0)
        # Mean over valid neighbors at every output pixel must equal 1.0
        mean_valid = alpha.squeeze(1).sum(dim=1) / n_val
        torch.testing.assert_close(mean_valid, torch.ones_like(mean_valid), rtol=1e-5, atol=1e-5)
        # Uniform prior must give exact 1.0 on all valid taps
        w_uni = torch.full((3, 1, 11, 11), 1.0 / 121.0, device=device)
        alpha_uni = local_mean_one_tap_weights(w_uni, d, mask).squeeze(1)
        torch.testing.assert_close(alpha_uni, mask.expand_as(alpha_uni), rtol=1e-6, atol=1e-6)
        # Extreme small corner weights (e.g., 1e-32) and full float32 underflow (0.0) must still give valid mean 1.0
        w_tiny = torch.zeros((2, 1, 11, 11), device=device, dtype=torch.float32)
        w_tiny[0, 0, 5, 5] = 1.0
        w_tiny[0, 0, 0, 0] = 2e-32
        w_tiny[0, 0, 0, 1] = 6e-32
        w_tiny[1, 0, 5, 5] = 1.0  # corners are exact 0.0 (full float32 softmax underflow)
        alpha_tiny = local_mean_one_tap_weights(w_tiny, d, mask)
        mean_tiny = alpha_tiny.squeeze(1).sum(dim=1) / n_val
        torch.testing.assert_close(mean_tiny, torch.ones_like(mean_tiny), rtol=1e-5, atol=1e-5)

    for seed in ((6,) if native_batch is not None else (6, 7, 8)):
        cfg = read(ROOT / "configs/runs/sr_native_20261029" / f"M3_seed{seed}.json")
        built_states = {}
        init_eval_logits = {}
        init_readout_weights = {}
        step2_eval_logits = {}
        for group in GROUPS:
            start = time.monotonic()
            torch.manual_seed(seed)
            baseline = NeighborTransMixer(
                "SR2",
                **cfg["sr"],
                hsi_channels=30,
                aux_channels=1,
                num_classes=15,
                hidden_dim=64,
                num_blocks=2,
                window_size=11,
                mode="decoupled",
            ).to(device).eval()
            rng_after_base = torch.get_rng_state().clone()
            cuda_after_base = torch.cuda.get_rng_state().clone() if device != "cpu" else None

            config._init(ExperimentConfig(**{
                **cfg,
                "model_name": "TransMixer_SRNCF_" + group,
                "paper_transfer": {"family": "SRNCF", "group": group},
            }))
            torch.manual_seed(seed)
            bundle = adapter.build_model(config, 0, device)
            net, stage = bundle["net"], bundle["net"].fusion_conv
            assert torch.equal(rng_after_base, torch.get_rng_state()), (seed, group, "CPU RNG")
            if device != "cpu":
                assert torch.equal(cuda_after_base, torch.cuda.get_rng_state()), (seed, group, "CUDA RNG")

            count_base = sum(p.numel() for p in baseline.parameters())
            count_new = sum(p.numel() for p in net.parameters())
            assert count_base == 80887, count_base
            assert count_new == EXPECTED_PARAMETERS[group], (group, count_new, EXPECTED_PARAMETERS[group])

            current = net.state_dict()
            built_states[group] = {k: v.detach().cpu().clone() for k, v in current.items()}
            for key, tensor in baseline.state_dict().items():
                target = key.replace("fusion_conv.", "fusion_conv.original.", 1) if key.startswith("fusion_conv.") else key
                assert torch.equal(tensor, current[target]), (seed, group, key)

            # Verify uniform unfold equivalence to standard dilated Conv2d (in IEEE float64 to isolate TF32 rounding)
            if group == "K01":
                net.eval()
                with torch.no_grad():
                    z_test = torch.randn(2, 42, 11, 11, device=device, dtype=torch.float64)
                    w_uni = torch.full((2, 1, 11, 11), 1.0 / 121.0, device=device, dtype=torch.float64)
                    wt64 = stage.shared_conv3x3.weight.detach().to(torch.float64)
                    for d, m_buf in ((1, stage.mask_d1), (2, stage.mask_d2)):
                        m64 = m_buf.to(torch.float64)
                        alpha64 = local_mean_one_tap_weights(w_uni, d, m64)
                        z_unf = torch.nn.functional.unfold(z_test, kernel_size=3, dilation=d, padding=d).view(2, 42, 9, 121)
                        o_unf = (wt64.view(42, 42 * 9) @ (z_unf * alpha64).view(2, 42 * 9, 121)).view(2, 42, 11, 11)
                        o_std = torch.nn.functional.conv2d(z_test, wt64, bias=None, stride=1, padding=d, dilation=d)
                        torch.testing.assert_close(o_std, o_unf, rtol=1e-12, atol=1e-12)

            net.eval()
            stage.disabled = True
            with torch.no_grad():
                ref = baseline(small["hsi_pca"], small["aux"])
                disabled = adapter.forward_eval(bundle, small)
            torch.testing.assert_close(ref, disabled, rtol=0, atol=0)
            stage.disabled = False
            with torch.no_grad():
                init_eval_logits[group] = adapter.forward_eval(bundle, small).detach().cpu().clone()
                init_readout_weights[group] = net.gap.weights.detach().cpu().clone()

            counts = {"stem": 0, "fusion": 0, "pool": 0, "head": 0}
            handles = []
            for key, module in (("stem", net.hsi_stem), ("fusion", stage.original), ("pool", net.gap), ("head", net.classifier)):
                def count(module, args, output, name=key):
                    counts[name] += 1
                handles.append(module.register_forward_hook(count))

            criterion = torch.nn.CrossEntropyLoss()
            opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
            net.train()
            if device != "cpu":
                torch.cuda.reset_peak_memory_stats()
            opt.zero_grad()
            output = adapter.forward_train(bundle, batch)
            loss = adapter.compute_loss(bundle, output, batch, criterion)
            loss.backward()
            for handle in handles:
                handle.remove()

            assert counts == dict.fromkeys(counts, 1), counts
            assert output.shape == (len(batch["label"]), 15) and torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()), (seed, group, "missing/nonfinite grad step1")
            new_params = [p for n, p in stage.named_parameters() if not n.startswith("original.")]
            assert all(p.grad.abs().sum() > 0 for p in new_params), (seed, group, "zero grad in step1 new module")
            assert stage.step == 1 and stage.history[-1]["auxiliary_loss"] == 0.0
            assert math.isclose(stage.history[-1]["alpha_d1_valid_mean"], 1.0, abs_tol=1e-5)
            assert math.isclose(stage.history[-1]["alpha_d2_valid_mean"], 1.0, abs_tol=1e-5)
            assert all(stage.history[-1][k + "_gradient_norm"] > 0 for k in ("h", "l"))

            opt.step()
            opt.zero_grad()
            output2 = adapter.forward_train(bundle, batch)
            loss2 = adapter.compute_loss(bundle, output2, batch, criterion)
            loss2.backward()
            assert torch.isfinite(loss2)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in new_params), (seed, group, "zero/nonfinite grad step2")

            net.eval()
            with torch.no_grad():
                a = adapter.forward_eval(bundle, small)
                b = adapter.forward_eval(bundle, small)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            step2_eval_logits[group] = a.detach().cpu().clone()

            stream = io.BytesIO()
            torch.save(net, stream)
            stream.seek(0)
            restored = torch.load(stream, map_location=device, weights_only=False).eval()
            assert restored.fusion_conv._sr_pool is restored.gap
            with torch.no_grad():
                actual_direct = restored(small["hsi_pca"], small["aux"])
                actual_adapter = adapter.forward_eval({"net": restored}, small)
            torch.testing.assert_close(a, actual_direct, rtol=0, atol=0)
            torch.testing.assert_close(a, actual_adapter, rtol=0, atol=0)

            results.append({
                "seed": seed,
                "group": group,
                "device": device,
                "batch_size": len(batch["label"]),
                "passed": True,
                "native_shared_initialization_exact": True,
                "rng_preserved": True,
                "disabled_logits_exact": True,
                "single_native_forward_per_batch": counts,
                "finite_loss_and_gradients": True,
                "all_new_parameters_nonzero_gradients_step1_and_step2": True,
                "checkpoint_roundtrip_exact": True,
                "parameters": count_new,
                "baseline_parameters": count_base,
                "added_parameters": count_new - count_base,
                "peak_allocated_bytes_including_reference_model": torch.cuda.max_memory_allocated() if device != "cpu" else None,
                "elapsed_seconds": time.monotonic() - start,
            })
            print("SRNCF_PREFLIGHT_OK", device, group, seed, count_new, flush=True)
            del baseline, bundle, net, stage, current, restored, stream, output, output2, loss, loss2, opt, a, b, actual_direct, actual_adapter, ref, disabled
            gc.collect()
            if device != "cpu":
                torch.cuda.empty_cache()

        # Verify bit-identical initial weights across K01..K06
        for g in ("K02", "K03", "K04", "K05", "K06"):
            assert set(built_states["K01"].keys()) == set(built_states[g].keys())
            for k in built_states["K01"]:
                assert torch.equal(built_states["K01"][k], built_states[g][k]), ("K01_vs_" + g, k)

        # Verify 2x2 readout/fusion SR invariants & K03 fusion-only misalignment
        for g in ("K02", "K03", "K05"):
            assert torch.equal(init_readout_weights["K01"], init_readout_weights[g]), ("readout_SR2_intact", g)
        uni_ro = torch.full_like(init_readout_weights["K01"], 1.0 / 121.0)
        for g in ("K04", "K06"):
            torch.testing.assert_close(init_readout_weights[g], uni_ro, rtol=1e-6, atol=1e-6)
        assert not torch.equal(init_eval_logits["K01"], init_eval_logits["K02"]), ("K01_vs_K02_init", seed)
        assert not torch.equal(init_eval_logits["K02"], init_eval_logits["K03"]), ("K02_vs_K03_init", seed)
        assert not torch.equal(init_eval_logits["K04"], init_eval_logits["K06"]), ("K04_vs_K06_init", seed)
        assert not torch.equal(init_eval_logits["K02"], init_eval_logits["K04"]), ("K02_vs_K04_init", seed)
        assert not torch.equal(init_eval_logits["K02"], init_eval_logits["K05"]), ("K02_vs_K05_init", seed)

    perm = fixed_derangement(121)
    assert perm.shape == (121,) and torch.all(perm != torch.arange(121))
    assert torch.equal(perm.sort().values, torch.arange(121))

    cfg6 = read(ROOT / "configs/runs/sr_native_20261029/M3_seed6.json")
    for update in (
        {"random_seed": 9},
        {"epochs": 99},
        {"paper_transfer": {"family": "SRNCF", "group": "K02", "extra": 1}},
    ):
        config._init(ExperimentConfig(**{
            **cfg6,
            "model_name": "TransMixer_SRNCF_K02",
            "paper_transfer": {"family": "SRNCF", "group": "K02"},
            **update,
        }))
        try:
            adapter.build_model(config, 0, "cpu")
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid protocol accepted")

    return {
        "all_passed": True,
        "checks": results,
        "expected_parameters": EXPECTED_PARAMETERS,
        "k01_to_k06_identical_init_and_params_verified": True,
        "local_mean_one_normalization_and_uniform_degeneration_verified": True,
        "k03_fusion_only_derangement_zero_fixed_points_verified": True,
        "k04_k06_uniform_gap_readout_verified": True,
        "derangement_121": perm.tolist(),
        "invalid_protocol_rejected": True,
    }


def inference_costs(batch):
    """Estimate clean-path MACs and measure actual batch128 latency on GPU."""
    from scripts.prepare_sr_native import ROOT, read
    from src import config
    from src.config import ExperimentConfig
    from ..transmixer_net.model import SelectiveSSM2D

    cfg = read(ROOT / "configs/runs/sr_native_20261029/M3_seed6.json")
    small = {k: v[:1] for k, v in batch.items()}
    rows = []
    for group in ("A0", *GROUPS):
        torch.manual_seed(6)
        if group == "A0":
            net = NeighborTransMixer(
                "SR2",
                **cfg["sr"],
                hsi_channels=30,
                aux_channels=1,
                num_classes=15,
                hidden_dim=64,
                num_blocks=2,
                window_size=11,
                mode="decoupled",
            ).to("cuda:0")
        else:
            config._init(ExperimentConfig(**{
                **cfg,
                "model_name": "TransMixer_SRNCF_" + group,
                "paper_transfer": {"family": "SRNCF", "group": group},
            }))
            net = adapter.build_model(config, 0, "cuda:0")["net"]
        net.eval()
        total, handles = [0], []

        def count(module, args, output):
            if isinstance(module, torch.nn.modules.conv._ConvNd):
                total[0] += output.numel() * module.in_channels // module.groups * math.prod(module.kernel_size)
            elif isinstance(module, torch.nn.Linear):
                total[0] += output.numel() * module.in_features
            elif isinstance(module, torch.nn.MultiheadAttention):
                b, n, d = args[0].shape
                total[0] += 4 * b * n * d * d + 2 * b * n * n * d
            elif isinstance(module, SelectiveSSM2D):
                b, n, d = args[0].shape
                total[0] += 2 * b * n * d * module.d_state
            elif isinstance(module, SRNCFFusionStage) and module.group != "K00":
                b, _, h, w = args[0].shape
                c_s = module.shared_conv3x3.out_channels
                # Two evaluations (dilation=1 and dilation=2) of shared_conv3x3 [42, 42, 3, 3] + tap modulation
                total[0] += 2 * b * h * w * c_s * c_s * 9 + 2 * b * h * w * c_s * 9

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
        rows.append({
            "group": group,
            "parameters": sum(p.numel() for p in net.parameters()),
            "estimated_clean_macs_per_patch": total[0],
            "batch_size": len(batch["label"]),
            "warmup_iterations": 3,
            "measured_iterations": 5,
            "batch_latency_ms": (time.perf_counter() - start) * 1000 / 5,
        })
        print("SRNCF_COST_AUDIT_OK", group, rows[-1]["parameters"], rows[-1]["estimated_clean_macs_per_patch"], flush=True)
        del net
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "runs": rows,
        "mac_estimator": "Conv/Linear/MHA, shared atrous 3x3 evaluations (and neighbor modulation products where active), SSM recurrence/readout; excludes nonlinear/normalization/pooling",
        "not_an_exact_total_flop_count": True,
    }


if __name__ == "__main__":
    from scripts.prepare_sr_native import ROOT, write
    out_path = ROOT / "docs/experiments/provenance/fusion_sr_ncf_seed678_20261107/cpu_preflight.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_checks()
    write(out_path, result)
