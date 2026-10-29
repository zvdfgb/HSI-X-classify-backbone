"""Scientific controls, native adapter compatibility, and checkpoint regressions."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from src.config import ConfigManager, ExperimentConfig
from src import fusion_experiment as lab
from src.model_registry import get_model_adapter
from src.models.transmixer_net.model import TransMixerNet
from src.models.transmixer_fusion.model import FusionTransMixerNet
from src.models.transmixer_paper import adapter
from src.models.transmixer_paper.model import (
    NeighborTransMixer, ConceptTransMixer, FrozenCorrector, geometry_targets, state_digest,
)
from src.paper_transfer_experiment import resolve_protocol, training_statistics


class PaperTests(unittest.TestCase):
    def setUp(self):
        lab.seed_everything(6, 1)
        self.args = dict(hsi_channels=4, aux_channels=1, num_classes=3,
                         hidden_dim=8, num_blocks=2, window_size=5)
        self.h = torch.randn(3, 1, 4, 5, 5)
        self.a = torch.randn(3, 1, 5, 5)

    def test_backbone_unchanged_and_rng_paired(self):
        torch.manual_seed(6)
        base = TransMixerNet(**self.args)
        after_base = torch.get_rng_state()
        for variant in ("SR1", "SR2", "SR3", "OC1", "OC2", "OC3"):
            torch.manual_seed(6)
            net = (NeighborTransMixer(variant, **self.args) if variant.startswith("SR")
                   else ConceptTransMixer(variant, **self.args))
            self.assertTrue(torch.equal(after_base, torch.get_rng_state()))
            for name, value in base.state_dict().items():
                if not name.startswith("classifier."):
                    self.assertTrue(torch.equal(value, net.state_dict()[name]), name)
        torch.manual_seed(6)
        oc1 = ConceptTransMixer("OC1", **self.args).eval()
        base.eval()
        self.assertTrue(torch.equal(base(self.h, self.a), oc1(self.h, self.a)))

    def test_neighbor_weights_geometry_and_no_added_parameters(self):
        base = TransMixerNet(**self.args)
        counts = sum(p.numel() for p in base.parameters())
        weights = []
        for variant in ("SR1", "SR2", "SR3"):
            net = NeighborTransMixer(variant, **self.args)
            w = net.neighbor_weights(self.h, self.a)
            torch.testing.assert_close(w.sum((1, 2, 3)), torch.ones(3))
            self.assertTrue((w > 0).all())
            self.assertEqual(sum(p.numel() for p in net.parameters()), counts)
            torch.testing.assert_close(net.neighbor_weights(self.h.flip(-1), self.a.flip(-1)), w.flip(-1))
            net(self.h, self.a).sum().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))
            weights.append(w)
        self.assertFalse(torch.equal(weights[0], weights[1]))
        self.assertFalse(torch.equal(weights[1], weights[2]))

    def test_concept_pair_identical_except_loss_and_targets_from_input(self):
        torch.manual_seed(6); c2 = ConceptTransMixer("OC2", **self.args)
        torch.manual_seed(6); c3 = ConceptTransMixer("OC3", **self.args)
        self.assertEqual(c2.state_dict().keys(), c3.state_dict().keys())
        self.assertTrue(all(torch.equal(x, c3.state_dict()[k]) for k, x in c2.state_dict().items()))
        torch.testing.assert_close(geometry_targets(self.a.flip(-1)), geometry_targets(self.a))
        batch = dict(hsi_pca=self.h, aux=self.a, label=torch.tensor([0, 1, 2]))
        logits = c3(self.h, self.a)
        loss = adapter.compute_loss(dict(net=c3, variant="OC3", settings={}), logits, batch, nn.CrossEntropyLoss())
        self.assertGreaterEqual(loss.item(), nn.functional.cross_entropy(logits, batch['label']).item())
        loss.backward()
        self.assertTrue(torch.isfinite(c3.classifier.concept_head.weight.grad).all())

    def test_corrector_identity_matching_capacity_and_frozen_state(self):
        base = TransMixerNet(**self.args).eval()
        networks = []
        for variant in ("PN1", "PN2"):
            torch.manual_seed(6)
            net = FrozenCorrector(variant, copy.deepcopy(base))
            networks.append(net)
            before = state_digest(net.base)
            original = base(self.h, self.a)
            self.assertTrue(torch.equal(net(self.h, self.a), original))
            net.train()
            self.assertFalse(net.base.training)
            self.assertFalse(net.base.classifier[2].training)
            optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
            for _ in range(2):
                optimizer.zero_grad()
                nn.functional.cross_entropy(net(self.h, self.a), torch.tensor([0, 1, 2])).backward()
                optimizer.step()
            adapter.on_training_complete(dict(net=net, variant=variant, baseline_initial_sha256=before))
            self.assertTrue(all(p.grad is None for p in net.base.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.corrector.parameters()))
            buffer = io.BytesIO()
            torch.save(net, buffer); buffer.seek(0)
            restored = torch.load(buffer, weights_only=False)
            torch.testing.assert_close(restored.eval()(self.h, self.a), net.eval()(self.h, self.a), rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in networks[0].parameters()), sum(p.numel() for p in networks[1].parameters()))

    def fixture(self):
        cfg = json.loads((lab.ROOT / "configs/fusion_pilot.json").read_text(encoding="utf-8"))
        cfg.update(epochs=1, batch_size=2, channels=4, window_size=5, device="cpu", cpu_threads=1)
        rng = np.random.default_rng(3)
        gt = np.ones((7, 7), dtype=np.int64)
        gt[1, 2] = gt[2, 2] = gt[3, 4] = 2
        data = dict(features=rng.normal(size=(11, 11, 4)).astype(np.float32),
                    aux=rng.normal(size=(11, 11, 1)).astype(np.float32), gt=gt,
                    train=np.array([[1, 1], [1, 2], [2, 1], [2, 2]]),
                    validation=np.array([[3, 3], [3, 4]]),
                    info=dict(aux_channels=1, classes=15, dataset="synthetic", split_sha256="fixture"))
        return cfg, data

    def test_existing_training_loop_adapter_parity_and_no_test_access(self):
        cfg, data = self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = lab.run_one(cfg, "F0", 6, data, root / "old")
            manager = ConfigManager(ExperimentConfig(model_name="TransMixer_B0_Decoupled", channels=4, window_size=5))
            new = lab.run_one(cfg, "G0", 6, data, root / "new",
                              model_adapter=get_model_adapter(manager.config.model_name), model_config=manager)
            a = torch.load(old["checkpoint"], weights_only=True)["state_dict"]
            b = torch.load(new["checkpoint"], weights_only=True)["state_dict"]
            self.assertTrue(all(torch.equal(v, b[k]) for k, v in a.items()))
            self.assertEqual(old["validation"], new["validation"])
            self.assertIsNone(new["test"])

    def test_registry_real_training_save_reload_and_native_eval(self):
        cfg, data = self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for variant in ("SR3", "OC3", "PN2"):
                settings = {}
                if variant.startswith("PN"):
                    base = TransMixerNet(hsi_channels=4, num_classes=15, window_size=5)
                    p = root / "base.pth"
                    torch.save(dict(state_dict=base.state_dict(), selection_split="validation"), p)
                    settings = dict(baseline_checkpoint=str(p), baseline_sha256=lab.sha256(p))
                manager = ConfigManager(ExperimentConfig(model_name=adapter.MODEL_NAMES[variant],
                                                         channels=4, window_size=5, paper_transfer=settings))
                selected = get_model_adapter(manager.config.model_name)
                result = lab.run_one(cfg, variant, 6, data, root / variant,
                                     model_adapter=selected, model_config=manager)
                self.assertIsNone(result["test"])
                restored = selected.build_model(manager, 0, torch.device("cpu"))
                restored["net"].load_state_dict(torch.load(result["checkpoint"], weights_only=True)["state_dict"])
                logits = selected.forward_eval({"net": restored["net"].eval()},
                                               dict(hsi_pca=torch.randn(2, 1, 4, 5, 5), aux=torch.randn(2, 1, 5, 5)))
                self.assertEqual(logits.shape, (2, 15))
                self.assertTrue(torch.isfinite(logits).all())

    def test_plan_cannot_silently_change_shared_protocol(self):
        plan = json.loads((lab.ROOT / "configs/paper_transfer.json").read_text(encoding="utf-8"))
        reference, cfg = resolve_protocol(plan)
        old = json.loads((reference / "batch_config.json").read_text(encoding="utf-8"))
        self.assertEqual({k:v for k,v in cfg.items() if k!='evaluate_test'},
                         {k:v for k,v in old.items() if k!='evaluate_test'})
        self.assertFalse(cfg["evaluate_test"])
        with self.assertRaises(ValueError):
            resolve_protocol({**plan, "learning_rate": .01})
        with self.assertRaises(ValueError):
            resolve_protocol({**plan, "correction_epochs": 10})

    def test_auxiliary_statistics_only_need_training_inputs(self):
        cfg, data = self.fixture()
        del data["validation"]
        stats = training_statistics(data, cfg)
        self.assertEqual(stats["samples"], len(data["train"]))
        self.assertGreater(stats["height_scale"], 0)
        self.assertTrue(np.isfinite(stats["concept_std"]).all())


if __name__ == "__main__":
    unittest.main()
