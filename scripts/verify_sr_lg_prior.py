"""Numerical/interface preflight only; no optimizer, training or new metrics."""
import gc
import io
from pathlib import Path

import torch
from torch import nn

from src import config
from src.config import ConfigManager, ExperimentConfig
from src.data import getMyData, set_random_seed
from src.model_registry import get_model_adapter
from src.models.transmixer_paper.model import NeighborTransMixer
from scripts.prepare_sr_native import read, write


def preflight(jobs, destination):
    torch.set_num_threads(4)
    device = "cuda:0"
    set_random_seed(6)
    loaders = getMyData(0, 30, 11, 128, 0)
    assert len(loaders[0].dataset) == 2832 and len(loaders[1].dataset) == 12197
    assert loaders[0].drop_last and len(loaders[0]) == 22
    hsi, raw, aux, labels = next(iter(loaders[0]))
    batch = {"hsi_pca": hsi.to(device), "aux": aux.to(device), "label": labels.to(device)}
    del loaders, raw
    results = []
    for seed in (6, 7, 8):
        pair = [j for j in jobs if not j["reused"] and j["seed"] == seed]
        bundles, rng = [], []
        for job in pair:
            set_random_seed(seed)
            exp = ExperimentConfig(**read(job["config"]))
            config._global_config_manager = ConfigManager(exp)
            adapter = get_model_adapter(job["model"])
            assert not hasattr(adapter, "compute_loss")
            bundles.append(adapter.build_model(config, 0, device))
            rng.append((torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()))
        left, right = [b["net"] for b in bundles]
        assert all(torch.equal(rng[0][i], rng[1][i]) for i in (0, 1))
        assert left.state_dict().keys() == right.state_dict().keys()
        assert all(torch.equal(v, right.state_dict()[k]) for k, v in left.state_dict().items())
        set_random_seed(seed)
        baseline = NeighborTransMixer("SR2", spectral_scale=0.2775005102157593,
            spatial_sigma=3.0, hsi_channels=30, aux_channels=1, num_classes=15,
            hidden_dim=64, num_blocks=2, window_size=11, mode="decoupled").to(device)
        assert torch.equal(torch.get_rng_state(), rng[0][0])
        assert torch.equal(torch.cuda.get_rng_state(), rng[0][1])
        native_state = baseline.state_dict()
        for key, value in native_state.items():
            wrapped = key.replace(".global_attn.", ".global_attn.original.")
            assert torch.equal(value, left.state_dict()[wrapped]), key
        baseline.eval(); left.eval(); right.eval()
        with torch.no_grad():
            reference = baseline(batch["hsi_pca"], batch["aux"])
            aligned = left(batch["hsi_pca"], batch["aux"])
            misaligned = right(batch["hsi_pca"], batch["aux"])
        assert reference.shape == aligned.shape == misaligned.shape == (128, 15)
        torch.testing.assert_close(aligned, reference, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(aligned, misaligned, atol=0, rtol=0)
        assert torch.equal(left.gap.weights, right.gap.weights)
        prior_a = left.blocks[0].global_attn.log_prior
        prior_b = right.blocks[0].global_attn.log_prior
        permutation = left.prior_permutation
        assert torch.all(permutation != torch.arange(121, device=device))
        torch.testing.assert_close(prior_b, prior_a[:, permutation], atol=0, rtol=0)
        torch.testing.assert_close(prior_a.sort(1).values, prior_b.sort(1).values, atol=0, rtol=0)
        # Test the actual native MHA with a nonzero additive key-position bias.
        attention = left.blocks[0].global_attn
        query = torch.randn(3, 121, 32, device=device)
        attention.log_prior = prior_a[:3]
        attention.beta.data.fill_(0.2)
        actual = attention(query, query, query)[0]
        explicit_mask = (0.2 * prior_a[:3])[:, None, None, :].expand(3, 4, 121, 121).reshape(12, 121, 121)
        expected = attention.original(query, query, query, attn_mask=explicit_mask)[0]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        attention.beta.data.zero_()
        gradients = []
        for bundle in bundles:
            net = bundle["net"]
            net.train(); net.zero_grad(set_to_none=True)
            out = net(batch["hsi_pca"], batch["aux"])
            loss = nn.CrossEntropyLoss()(out, batch["label"])
            assert torch.isfinite(loss) and torch.isfinite(out).all()
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
            beta_grad = [float(b.global_attn.beta.grad) for b in net.blocks]
            assert all(abs(g) > 0 for g in beta_grad)
            gradients.append(beta_grad)
            net.eval()
        with torch.no_grad():
            output = left(batch["hsi_pca"], batch["aux"])
            stream = io.BytesIO(); torch.save(left, stream); stream.seek(0)
            restored = torch.load(stream, map_location=device, weights_only=False)
            torch.testing.assert_close(restored(batch["hsi_pca"], batch["aux"]), output, atol=0, rtol=0)
        results.append({"seed": seed, "all_passed": True,
            "baseline_parameters": sum(p.numel() for p in baseline.parameters()),
            "new_parameters_each_group": sum(p.numel() for p in left.parameters()),
            "added_parameters": 2, "zero_beta_max_abs_error": float((aligned-reference).abs().max()),
            "beta_gradients_aligned_misaligned": gradients,
            "permutation": permutation.cpu().tolist(),
            "native_batch128_forward_backward": True, "serialization": True,
            "paired_initialization_and_rng": True, "nonzero_explicit_mask_exact": True,
            "spatial_permutation_and_multiset_preservation": True})
        print(f"Preflight seed{seed} passed; beta gradients={gradients}", flush=True)
        del baseline, left, right, restored, bundles, output, aligned, reference, misaligned, loss, out
        gc.collect(); torch.cuda.empty_cache()
    write(Path(destination)/"preflight.json", {"all_passed": True, "native_batch": 128,
        "training_samples": 2832, "test_samples": 12197, "native_train_batches": 22, "checks": results})
