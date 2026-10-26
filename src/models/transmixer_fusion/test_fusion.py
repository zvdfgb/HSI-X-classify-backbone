import unittest

import torch

from src.models.transmixer_net.model import TransMixerNet
from src.models.transmixer_fusion.model import CrossModalInteraction, FusionTransMixerNet, VARIANTS


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def build(self, variant, aux_channels=1):
        torch.manual_seed(123)
        return FusionTransMixerNet(variant, hsi_channels=4, aux_channels=aux_channels,
                                  num_classes=3, hidden_dim=8, num_blocks=2, window_size=5)

    def test_paired_initialization_and_zero_identity(self):
        torch.manual_seed(123)
        baseline = TransMixerNet(hsi_channels=4, aux_channels=1, num_classes=3,
                                 hidden_dim=8, num_blocks=2, window_size=5).eval()
        expected_rng = torch.rand(4)
        x, a = torch.randn(2, 1, 4, 5, 5), torch.randn(2, 1, 5, 5)
        expected = baseline(x, a)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = self.build(variant).eval()
                torch.testing.assert_close(torch.rand(4), expected_rng, rtol=0, atol=0)
                for key, value in baseline.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
                torch.testing.assert_close(model(x, a), expected, rtol=1e-6, atol=1e-6)

    def test_lidar_sar_shapes_and_trainable_new_paths(self):
        for variant in VARIANTS:
            for aux_channels in (1, 4):
                with self.subTest(variant=variant, aux=aux_channels):
                    model = self.build(variant, aux_channels).train()
                    for p in model.interaction.strength.values():
                        p.data.fill_(0.2)
                    out = model(torch.randn(2, 4, 5, 5), torch.randn(2, aux_channels, 5, 5))
                    self.assertEqual(tuple(out.shape), (2, 3))
                    torch.nn.functional.cross_entropy(out, torch.tensor([0, 2])).backward()
                    for name, p in model.named_parameters():
                        if p.grad is not None:
                            self.assertTrue(torch.isfinite(p.grad).all(), name)
                    for name, p in model.interaction.named_parameters():
                        self.assertIsNotNone(p.grad, name)
                        self.assertGreater(p.grad.abs().sum().item(), 0, name)

    def test_directionality_and_simultaneous_updates(self):
        torch.manual_seed(44)
        h, a = torch.randn(2, 9, 8), torch.randn(2, 9, 8)
        messages = {}
        for mode in ("aux_to_hsi", "hsi_to_aux", "bidirectional"):
            torch.manual_seed(10)
            module = CrossModalInteraction(8, mode).eval()
            for p in module.strength.values():
                p.data.fill_(0.3)
            messages[mode] = module(h, a, 3, 3)
        torch.testing.assert_close(messages["aux_to_hsi"][1], a)
        torch.testing.assert_close(messages["hsi_to_aux"][0], h)
        torch.testing.assert_close(messages["bidirectional"][0], messages["aux_to_hsi"][0])
        torch.testing.assert_close(messages["bidirectional"][1], messages["hsi_to_aux"][1])
        self.assertFalse(torch.allclose(messages["aux_to_hsi"][0], h))
        self.assertFalse(torch.allclose(messages["hsi_to_aux"][1], a))

    def test_split_control_matches_parameters_but_changes_assignment(self):
        torch.manual_seed(2)
        matched = CrossModalInteraction(8, "split_matched")
        torch.manual_seed(2)
        swapped = CrossModalInteraction(8, "split_swapped")
        for key, value in matched.state_dict().items():
            torch.testing.assert_close(value, swapped.state_dict()[key], rtol=0, atol=0)
        for m in (matched, swapped):
            for p in m.strength.values():
                p.data.fill_(0.2)
        h, a = torch.randn(2, 9, 8), torch.randn(2, 9, 8)
        self.assertFalse(torch.allclose(matched(h, a, 3, 3)[0], swapped(h, a, 3, 3)[0]))

    def test_invalid_mode_and_unaligned_features(self):
        with self.assertRaises(ValueError):
            CrossModalInteraction(8, "unknown")
        with self.assertRaises(ValueError):
            CrossModalInteraction(8, "concat")(torch.zeros(1, 9, 8), torch.zeros(1, 4, 8), 3, 3)


if __name__ == "__main__":
    unittest.main()
