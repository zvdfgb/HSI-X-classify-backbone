"""Interface, paired-RNG, exact-parameter, gradient, and latency/MACs preflight for BMAF N00-N05."""
import gc
import io
import math
import time
import torch
from . import adapter
from .model import GROUPS, CrossModalRelations, fixed_derangement, compute_boundary_map
from ..transmixer_paper.model import NeighborTransMixer

EXPECTED_PARAMETERS = {
    "N00": 157799,
    "N01": 171919,
    "N02": 182631,
    "N03": 184729,
    "N04": 185041,
    "N05": 185041,
}


def run_checks(device="cpu", native_batch=None):
    from src import config
    from src.config import ExperimentConfig
    from scripts.prepare_sr_native import ROOT, read

    torch.set_num_threads(4)
    results = []
    rng = torch.Generator().manual_seed(906)
    small = {
        "hsi_pca": torch.randn(2, 1, 30, 11, 11, generator=rng).to(device),
        "aux": torch.randn(2, 1, 11, 11, generator=rng).to(device),
        "label": torch.tensor([0, 1], device=device),
    }
    batch = native_batch or small

    # Verify boundary map invariance to global offset and positive scaling.
    b_ref = compute_boundary_map(small["hsi_pca"], small["aux"])
    b_shifted = compute_boundary_map(small["hsi_pca"] * 3.5 - 12.0, small["aux"] * 4.0 + 100.0)
    assert b_ref.shape == (2, 4, 11, 11) and torch.isfinite(b_ref).all() and (b_ref >= 0).all()
    torch.testing.assert_close(b_ref, b_shifted, rtol=1e-4, atol=1e-4)

    for seed in ((6,) if native_batch is not None else (6, 7, 8)):
        cfg = read(ROOT / "configs/runs/sr_native_20261029" / f"M3_seed{seed}.json")
        built_states = {}
        init_eval_logits = {}
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
                "model_name": "TransMixer_BMAF_" + group,
                "paper_transfer": {"family": "BMAF", "group": group},
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

            net.eval()
            stage.disabled = True
            with torch.no_grad():
                ref = baseline(small["hsi_pca"], small["aux"])
                disabled = adapter.forward_eval(bundle, small)
            assert stage._raw_inputs is None
            torch.testing.assert_close(ref, disabled, rtol=0, atol=0)
            stage.disabled = False
            with torch.no_grad():
                init_eval_logits[group] = adapter.forward_eval(bundle, small).detach().cpu().clone()
            assert stage._raw_inputs is None

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
            assert stage._raw_inputs is None
            loss = adapter.compute_loss(bundle, output, batch, criterion)
            loss.backward()
            for handle in handles:
                handle.remove()

            assert counts == dict.fromkeys(counts, 1), counts
            assert output.shape == (len(batch["label"]), 15) and torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()), (seed, group, "missing/nonfinite grad step1")
            first_layer_gated = (
                {id(stage.modulator.feat_gate[0].weight), id(stage.modulator.feat_gate[0].bias)}
                if stage.modulator is not None
                else set()
            )
            if stage.modulator is not None and stage.modulator.bnd_gate is not None:
                first_layer_gated.update({id(stage.modulator.bnd_gate[0].weight), id(stage.modulator.bnd_gate[0].bias)})
            step1_active = [p for n, p in stage.named_parameters() if not n.startswith("original.") and id(p) not in first_layer_gated]
            assert all(p.grad.abs().sum() > 0 for p in step1_active), (seed, group, "zero grad in step1 active module")
            assert stage.step == 1 and stage.history[-1]["auxiliary_loss"] == 0.0
            assert (
                stage.history[-1]["g3_min"] == 1.0
                and stage.history[-1]["g3_max"] == 1.0
                and stage.history[-1]["g5_min"] == 1.0
                and stage.history[-1]["g5_max"] == 1.0
            )
            assert math.isclose(stage.history[-1]["g3_mean"], 1.0, abs_tol=1e-6)
            assert math.isclose(stage.history[-1]["g5_mean"], 1.0, abs_tol=1e-6)
            assert all(stage.history[-1][k + "_gradient_norm"] > 0 for k in ("h", "l"))

            opt.step()
            opt.zero_grad()
            output2 = adapter.forward_train(bundle, batch)
            assert stage._raw_inputs is None
            loss2 = adapter.compute_loss(bundle, output2, batch, criterion)
            loss2.backward()
            assert torch.isfinite(loss2)
            new_params = [p for n, p in stage.named_parameters() if not n.startswith("original.")]
            assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in new_params), (seed, group, "zero/nonfinite grad step2")

            net.eval()
            with torch.no_grad():
                a = adapter.forward_eval(bundle, small)
                b = adapter.forward_eval(bundle, small)
            assert stage._raw_inputs is None
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            step2_eval_logits[group] = a.detach().cpu().clone()

            stream = io.BytesIO()
            torch.save(net, stream)
            stream.seek(0)
            restored = torch.load(stream, map_location=device, weights_only=False).eval()
            assert restored.fusion_conv._sr_pool is restored.gap
            assert restored.fusion_conv._raw_inputs is None
            with torch.no_grad():
                actual_direct = restored(small["hsi_pca"], small["aux"])
                actual_adapter = adapter.forward_eval({"net": restored}, small)
            assert restored.fusion_conv._raw_inputs is None
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
                "all_new_parameters_nonzero_gradients_by_step2": True,
                "checkpoint_roundtrip_exact": True,
                "no_cached_input_tensors_retained": True,
                "parameters": count_new,
                "baseline_parameters": count_base,
                "added_parameters": count_new - count_base,
                "peak_allocated_bytes_including_reference_model": torch.cuda.max_memory_allocated() if device != "cpu" else None,
                "elapsed_seconds": time.monotonic() - start,
            })
            print("BMAF_PREFLIGHT_OK", device, group, seed, count_new, flush=True)
            del baseline, bundle, net, stage, current, restored, stream, output, output2, loss, loss2, opt, a, b, actual_direct, actual_adapter, ref, disabled
            gc.collect()
            if device != "cpu":
                torch.cuda.empty_cache()

        # Verify exact parameter & initial weight pairing between N04 and N05, shared relations/branches,
        # exact step-0 unit-gate equivalence across N02..N05, and step-2 divergence of N03/N04/N05.
        assert set(built_states["N04"].keys()) == set(built_states["N05"].keys())
        for k in built_states["N04"]:
            assert torch.equal(built_states["N04"][k], built_states["N05"][k]), ("N04_vs_N05_init", k)
        for g in ("N03", "N04", "N05"):
            torch.testing.assert_close(init_eval_logits["N02"], init_eval_logits[g], rtol=0, atol=0)
        assert not torch.equal(step2_eval_logits["N03"], step2_eval_logits["N04"]), ("N03_vs_N04_step2", seed)
        assert not torch.equal(step2_eval_logits["N04"], step2_eval_logits["N05"]), ("N04_vs_N05_step2", seed)
        for g in ("N00", "N02", "N03", "N04", "N05"):
            for k in built_states["N00"]:
                if ".branches." in k:
                    assert torch.equal(built_states["N00"][k], built_states[g][k]), ("shared_branches", g, k)
        for g in ("N01", "N02", "N03", "N04", "N05"):
            for k in built_states["N01"]:
                if ".relations." in k or ".branches.b3." in k:
                    assert torch.equal(built_states["N01"][k], built_states[g][k]), ("shared_relations_b3", g, k)

    perm = fixed_derangement(121)
    assert perm.shape == (121,) and torch.all(perm != torch.arange(121))
    assert torch.equal(perm.sort().values, torch.arange(121))

    cfg6 = read(ROOT / "configs/runs/sr_native_20261029/M3_seed6.json")
    for update in (
        {"random_seed": 9},
        {"epochs": 99},
        {"paper_transfer": {"family": "BMAF", "group": "N04", "extra": 1}},
    ):
        config._init(ExperimentConfig(**{
            **cfg6,
            "model_name": "TransMixer_BMAF_N04",
            "paper_transfer": {"family": "BMAF", "group": "N04"},
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
        "n04_n05_identical_init_and_params_verified": True,
        "step0_unit_modulation_bit_exact": True,
        "boundary_scale_shift_invariance_verified": True,
        "derangement_verified": True,
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
                "model_name": "TransMixer_BMAF_" + group,
                "paper_transfer": {"family": "BMAF", "group": group},
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
            elif isinstance(module, CrossModalRelations):
                b, d, h, w = args[0].shape
                n = h * w
                total[0] += 2 * b * n * n * d + 2 * b * d * d * n

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
        print("BMAF_COST_AUDIT_OK", group, rows[-1]["parameters"], flush=True)
        del net
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "runs": rows,
        "mac_estimator": "Conv/Linear/MHA, explicit cross-modal matrix products, SSM recurrence/readout; excludes nonlinear/normalization/pooling/local-variance",
        "not_an_exact_total_flop_count": True,
    }


if __name__ == "__main__":
    from scripts.prepare_sr_native import ROOT, write
    out_path = ROOT / "docs/experiments/provenance/fusion_bmaf_seed678_20261106/cpu_preflight.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_checks()
    write(out_path, result)
