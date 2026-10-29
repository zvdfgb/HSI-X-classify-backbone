import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.fusion_experiment import (ROOT, check_positions, stratified_positions,
                                   validate_settings, run_one, FusionPatchDataset, seed_everything)
from src.models.transmixer_fusion.model import FusionTransMixerNet
from src.models.transmixer_fusion.adapter import build_model
from src.config import ConfigManager, ExperimentConfig


class ProtocolTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA regression requires a GPU")
    def test_strict_cuda_attention_forward_backward_repeatability(self):
        outputs = []
        for _ in range(2):
            seed_everything(6, 1)
            net = FusionTransMixerNet(variant="F5", hsi_channels=4, aux_channels=1,
                                      num_classes=3, hidden_dim=8, num_blocks=2, window_size=5).cuda()
            h = torch.randn(2, 1, 4, 5, 5, device="cuda")
            a = torch.randn(2, 1, 5, 5, device="cuda")
            for strength in net.interaction.strength.values():
                strength.data.fill_(.2)
            logits = net(h, a)
            logits.square().mean().backward()
            outputs.append((logits.detach().cpu(), [p.grad.detach().cpu() for p in net.parameters()]))
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
        self.assertFalse(torch.backends.cuda.mem_efficient_sdp_enabled())
        self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
        self.assertTrue(torch.equal(outputs[0][0], outputs[1][0]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(outputs[0][1], outputs[1][1])))

    def config(self):
        return json.loads((ROOT / "configs/fusion_pilot.json").read_text(encoding="utf-8"))

    def test_split_repeatability_class_coverage_and_disjointness(self):
        gt = np.tile(np.arange(1, 4), (6, 1))
        pos = np.array([[r, c] for r in range(5) for c in range(3)])
        test = np.array([[5, c] for c in range(3)])
        check_positions(pos, test, gt, 3)
        train, val = stratified_positions(pos, gt, .2, 25)
        again = stratified_positions(pos, gt, .2, 25)
        np.testing.assert_array_equal(train, again[0]); np.testing.assert_array_equal(val, again[1])
        self.assertFalse(set(map(tuple, train)) & set(map(tuple, val)))
        self.assertEqual(len(train)+len(val), len(pos))
        self.assertEqual(set(gt[val[:, 0], val[:, 1]]), {1, 2, 3})
        with self.assertRaises(ValueError):
            check_positions(pos, pos[:1], gt, 3)
        with self.assertRaises(ValueError):
            check_positions(np.concatenate((pos, pos[:1])), test, gt, 3)

    def test_configuration_and_pilot_test_guard(self):
        cfg = self.config(); validate_settings(cfg)
        for key, bad in (("window_size", 4), ("evaluate_test", True), ("epochs", 0),
                         ("seeds", [6, 6]), ("learning_rate", float("nan")), ("variants", ["F9"])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_settings({**cfg, key: bad})

    def test_registry_builds_named_model(self):
        manager = ConfigManager(ExperimentConfig(model_name="TransMixer_Fusion_F5", channels=4))
        net = build_model(manager, 0, torch.device("cpu"))["net"]
        self.assertEqual(net.variant, "F5")

    def test_paired_spatial_augmentation(self):
        feature = np.arange(81, dtype=np.float32).reshape(9, 9, 1)
        data = dict(features=feature, aux=feature.copy(), gt=np.ones((5, 5)),
                    train=np.array([[1, 1], [2, 2]]))
        dataset = FusionPatchDataset(data, "train", 5, augment=True)
        for _ in range(10):
            h, a, _ = dataset[0]
            torch.testing.assert_close(h[0], a)

    def test_real_train_select_save_reload_without_test_access(self):
        cfg = self.config()
        cfg.update(epochs=1, batch_size=2, channels=4, window_size=5, device="cpu", cpu_threads=1)
        rng = np.random.default_rng(3)
        gt = np.ones((7, 7), dtype=np.int64)
        gt[1, 2] = gt[2, 2] = gt[3, 4] = 2
        data = dict(features=rng.normal(size=(11, 11, 4)).astype(np.float32),
                    aux=rng.normal(size=(11, 11, 1)).astype(np.float32), gt=gt,
                    train=np.array([[1, 1], [1, 2], [2, 1], [2, 2]]),
                    validation=np.array([[3, 3], [3, 4]]),
                    info=dict(aux_channels=1, classes=2, dataset="synthetic", split_sha256="fixture"))
        # Deliberately no data['test']: any attempted test evaluation fails.
        def small_model(**kwargs):
            return FusionTransMixerNet(**{**kwargs, "hidden_dim": 8})
        with tempfile.TemporaryDirectory(prefix="transmixer-fusion-test-") as temp:
            out = Path(temp) / "job"
            with patch("src.fusion_experiment.FusionTransMixerNet", small_model):
                result = run_one(cfg, "F4", 6, data, out)
            self.assertIsNone(result["test"])
            self.assertEqual(result["selection_split"], "validation")
            self.assertTrue((out / "best.pth").is_file())
            record = json.loads((out / "history.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["train_samples"], 4)
            self.assertEqual(result["best_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
