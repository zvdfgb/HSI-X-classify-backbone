"""Synthetic interface checks only; no research data or experiment training."""
import io
import unittest

import torch

from src import config
from src.config import ConfigManager, ExperimentConfig
from src.model_registry import MODEL_REGISTRY, LEGACY_MODEL_REGISTRY, get_model_adapter
from src.models.transmixer_net.model import TransMixerNet
from src.models.transmixer_paper.model import NeighborTransMixer
from . import adapter


class SRIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.previous = config._global_config_manager
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self):
        config._global_config_manager = self.previous
        torch.set_num_threads(self.threads)

    def build(self, mode="decoupled", variant="SR2", settings=None):
        exp = ExperimentConfig(model_name=f"TransMixer_{variant}", dataset_type=0,
                               ablation_mode=mode, sr=settings)
        config._global_config_manager = ConfigManager(exp)
        return adapter.build_model(config, 0, "cpu")["net"]

    def test_native_adapter_shapes_gradients_and_checkpoint(self):
        for mode in ("decoupled", "pure_ssm"):
            with self.subTest(mode=mode):
                net = self.build(mode, settings={"spectral_scale": .3})
                batch = {"hsi_pca": torch.randn(2, 1, 30, 5, 5),
                         "aux": torch.randn(2, 1, 5, 5)}
                logits = adapter.forward_train({"net": net}, batch)
                self.assertEqual(logits.shape, (2, 15))
                torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
                self.assertTrue(torch.isfinite(net.hsi_stem[0].weight.grad).all())
                net.eval()
                expected = adapter.forward_eval({"net": net}, batch)
                stream = io.BytesIO()
                torch.save(net, stream)  # Native trainer saves whole model objects.
                stream.seek(0)
                loaded = torch.load(stream, map_location="cpu", weights_only=False)
                torch.testing.assert_close(loaded(**{"hsi_pca": batch["hsi_pca"],
                                                      "aux": batch["aux"]}), expected)

    def test_sr_has_identical_initial_trainable_parameters(self):
        for mode in ("decoupled", "pure_ssm"):
            torch.manual_seed(6)
            original = TransMixerNet(mode=mode)
            torch.manual_seed(6)
            sr = self.build(mode, settings={"spectral_scale": .3})
            left, right = dict(original.named_parameters()), dict(sr.named_parameters())
            self.assertEqual(left.keys(), right.keys())
            for name in left:
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)

    def test_pooling_weights_and_spectral_effect(self):
        net = self.build(settings={"spectral_scale": 1.})
        hsi = torch.zeros(1, 1, 30, 5, 5)
        hsi[:, :, :, 2, 1] = 2.
        weights = net.neighbor_weights(hsi, torch.zeros(1, 1, 5, 5))
        torch.testing.assert_close(weights.sum(), torch.tensor(1.))
        self.assertTrue((weights >= 0).all())
        self.assertLess(weights[0, 0, 2, 1], weights[0, 0, 2, 3])
        torch.testing.assert_close(net.neighbor_weights(hsi[:, 0], torch.zeros(1, 1, 5, 5)), weights)

    def test_sr1_needs_no_spectral_calibration(self):
        net = self.build(variant="SR1")
        a, b = torch.randn(2, 1, 30, 5, 5), torch.randn(2, 1, 30, 5, 5)
        aux = torch.randn(2, 1, 5, 5)
        torch.testing.assert_close(net.neighbor_weights(a, aux), net.neighbor_weights(b, aux))

    def test_invalid_or_missing_model_settings_fail(self):
        for settings in (None, {}, {"spectral_scale": 0.}, {"spectral_scale": float("nan")},
                         {"spectral_scale": 1., "spatial_sigma": -1.},
                         {"spectral_scale": 1., "epochs": 10}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.build(settings=settings)

    def test_retired_models_stay_out_of_default_queue(self):
        self.assertFalse(set(MODEL_REGISTRY) & set(LEGACY_MODEL_REGISTRY))
        self.assertIs(get_model_adapter("TransMixer_SR2"), adapter)
        self.assertIn("TransMixer_Paper_PN1", LEGACY_MODEL_REGISTRY)
        self.assertNotIn("TransMixer_Paper_PN1", MODEL_REGISTRY)
        self.assertIsNotNone(get_model_adapter("TransMixerNet"))
        self.assertIsNotNone(get_model_adapter("TransMixer_Paper_SR2"))
        self.assertEqual(NeighborTransMixer.__module__, "src.models.transmixer_paper.model")


if __name__ == "__main__":
    unittest.main()
