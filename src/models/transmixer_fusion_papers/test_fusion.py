"""Mechanism/interface checks, never scientific training or dataset evaluation."""
import io
import unittest

import torch
from torch import nn

from src.models.transmixer_net.model import SelectiveSSM2D, TransMixerNet
from src.models.transmixer_paper.model import NeighborTransMixer
from .model import ConditionedSSM, FusionStage, GROUPS, ScanInteraction


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_self_condition_reuses_original_recurrence_and_gradients(self):
        torch.manual_seed(6)
        original = SelectiveSSM2D(8)
        torch.manual_seed(6)
        wrapped = ConditionedSSM(8, "self")
        x = torch.randn(2, 7, 8)
        control = torch.randn_like(x)
        a, da = original(x)
        b, db = wrapped(x, control)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(da, db, rtol=0, atol=0)
        a.sum().backward()
        b.sum().backward()
        torch.testing.assert_close(original.x_proj.weight.grad,
                                   wrapped.x_proj.projection.weight.grad, rtol=0, atol=0)
        self.assertIsNone(wrapped.x_proj.control)

    def test_condition_sources_and_reverse_scan(self):
        x = torch.randn(2, 7, 8)
        c1 = torch.randn_like(x, requires_grad=True)
        c2 = c1.detach() + 2.0
        for mode in ("cross_bc", "cross_bc_delta"):
            ssm = ConditionedSSM(8, mode)
            y1, d1 = ssm(x, c1)
            y2, d2 = ssm(x, c2)
            self.assertFalse(torch.allclose(y1, y2))
            if mode == "cross_bc":
                torch.testing.assert_close(d1, d2, rtol=0, atol=0)
            else:
                self.assertFalse(torch.allclose(d1, d2))
            self.assertTrue((d1 > 0).all())
            y1.sum().backward()
            self.assertTrue(torch.isfinite(c1.grad).all())
            self.assertGreater(float(c1.grad.abs().sum()), 0)
            self.assertIsNone(ssm.x_proj.control)
        scan = ScanInteraction(8, "interleave")
        sequence = torch.randn(2, 18, 8)
        expected = (scan.ssm(sequence)[0] + scan.ssm(sequence.flip(1))[0].flip(1)) / 2
        torch.testing.assert_close(scan.bidirectional(sequence), expected, rtol=0, atol=0)
        ht, lt = torch.randn(2, 9, 8), torch.randn(2, 9, 8)
        packed = torch.stack((ht, lt), 2).flatten(1, 2).reshape(2, 9, 2, 8)
        torch.testing.assert_close(packed[:, :, 0], ht)
        torch.testing.assert_close(packed[:, :, 1], lt)

    def test_parameter_matching_and_loss_switches(self):
        counts = {}
        x = torch.randn(2, 128, 3, 3)
        for group in GROUPS:
            stage = FusionStage(nn.Identity(), group)
            counts[group] = sum(p.numel() for p in stage.parameters())
            if group >= "G10":
                stage.train()
                stage(x)
                ce = torch.tensor(2.0, requires_grad=True)
                expected = ce
                if group in ("G12", "G13", "G14"):
                    expected = expected + 0.01 * stage.interaction.balance
                if group in ("G13", "G14"):
                    expected = expected + 0.001 * stage.interaction.consensus
                torch.testing.assert_close(stage.additional_loss(ce), expected, rtol=0, atol=0)
                stage.eval()
                stage(x)
                info = stage.diagnostics()
                self.assertAlmostEqual(sum(info["mean_weights"]), 1.0, places=6)
                if group == "G10":
                    self.assertEqual(info["mean_weights"], [0.25] * 4)
        for members in (("G02", "G03", "G04", "G05"),
                        ("G06", "G07", "G08", "G09"),
                        ("G10", "G11", "G12", "G13", "G14")):
            self.assertEqual(len({counts[g] for g in members}), 1)

    def test_original_forward_zero_residual_and_serialization(self):
        for group in ("G04", "G05", "G08", "G13", "G14"):
            use_a = GROUPS[group][2]
            kwargs = dict(mode="decoupled" if use_a else "pure_ssm")
            def build():
                return (NeighborTransMixer("SR2", spectral_scale=0.2775005102157593, **kwargs)
                        if use_a else TransMixerNet(**kwargs))
            torch.manual_seed(6)
            baseline = build().eval()
            torch.manual_seed(6)
            net = build()
            net.fusion_conv = FusionStage(net.fusion_conv, group)
            net.eval()
            hsi, aux = torch.randn(2, 1, 30, 11, 11), torch.randn(2, 1, 11, 11)
            with torch.no_grad():
                torch.testing.assert_close(net(hsi, aux), baseline(hsi, aux), rtol=0, atol=0)
                net.fusion_conv.alpha_h.fill_(0.1)
                net.fusion_conv.alpha_l.fill_(0.1)
                expected = net(hsi, aux)
                buffer = io.BytesIO()
                torch.save(net, buffer)
                buffer.seek(0)
                restored = torch.load(buffer, weights_only=False)
                torch.testing.assert_close(restored(hsi, aux), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
