import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.config import ConfigManager, ExperimentConfig
from src.model_registry import get_model_adapter
from src.models.transmixer_net.model import TransMixerNet
from src.models.transmixer_ablation.model import FeatureAblationNet
from scripts import train_transmixer_ablation as batch_runner


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shared_initialization_and_zero_scaled_baseline_equivalence(self):
        for seed in (6, 7):
            torch.manual_seed(seed)
            baseline = TransMixerNet().eval()
            rng_after_baseline = torch.get_rng_state()
            hsi, aux = torch.randn(2, 30, 11, 11), torch.randn(2, 1, 11, 11)
            with torch.no_grad():
                expected = baseline(hsi, aux)
            for variant in batch_runner.VARIANTS:
                with self.subTest(seed=seed, variant=variant):
                    torch.manual_seed(seed)
                    model = FeatureAblationNet(variant).eval()
                    self.assertTrue(torch.equal(torch.get_rng_state(), rng_after_baseline))
                    for key, value in model.state_dict().items():
                        base_key = key.replace("local_dw.0.dw3.", "local_dw.0.")
                        if base_key not in baseline.state_dict():
                            continue
                        if variant == "B3" and key.endswith("ssm.dt_proj.bias"):
                            continue
                        torch.testing.assert_close(value, baseline.state_dict()[base_key], rtol=0, atol=0)
                    if variant != "B3":
                        with torch.no_grad():
                            torch.testing.assert_close(model(hsi, aux), expected, rtol=0, atol=0)

    def test_registry_shapes_gradients_and_checkpoint_roundtrip(self):
        for dataset in (0, 3):
            for variant in batch_runner.VARIANTS:
                with self.subTest(dataset=dataset, variant=variant):
                    torch.manual_seed(6)
                    exp = ExperimentConfig(model_name=f"TransMixer_{variant}", dataset_type=dataset)
                    adapter = get_model_adapter(exp.model_name)
                    bundle = adapter.build_model(ConfigManager(exp), dataset, torch.device("cpu"))
                    model = bundle["net"]
                    batch = {"hsi_pca": torch.randn(2, 1, 30, 11, 11),
                             "aux": torch.randn(2, exp.lidar_or_sar_channels[dataset], 11, 11)}
                    out = adapter.forward_train(bundle, batch)
                    self.assertEqual(tuple(out.shape), (2, exp.out_features[dataset]))
                    loss = torch.nn.functional.cross_entropy(out, torch.tensor([0, 1]))
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    for parameter in model.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    if variant in ("B1", "B2", "B4"):
                        scales = [b.local_dw[0].beta if variant == "B4" else b.alpha for b in model.blocks]
                        self.assertTrue(any(p.grad.abs().item() > 0 for p in scales))
                    # Use the same whole-model serialization as the existing trainer.
                    model.eval()
                    stream = io.BytesIO()
                    torch.save(model, stream)
                    stream.seek(0)
                    restored = torch.load(stream, weights_only=False)
                    with torch.no_grad():
                        torch.testing.assert_close(restored(batch["hsi_pca"], batch["aux"]),
                                                   adapter.forward_eval(bundle, batch))

    def test_b1_and_b2_bypass_sources(self):
        torch.manual_seed(6)
        b1 = FeatureAblationNet("B1").blocks[0].eval()
        torch.manual_seed(6)
        b2 = FeatureAblationNet("B2").blocks[0].eval()
        u_input = torch.randn(2, 9, 64)
        with torch.no_grad():
            baseline = b1(u_input, 3, 3)
            b1.alpha.fill_(1)
            b2.alpha.fill_(1)
            y1, y2 = b1(u_input, 3, 3), b2(u_input, 3, 3)
        self.assertFalse(torch.allclose(y1, baseline))
        self.assertFalse(torch.allclose(y2, baseline))
        self.assertFalse(torch.allclose(y1, y2))

    def test_b3_dt_initialization_range(self):
        model = FeatureAblationNet("B3")
        for block in model.blocks:
            dt = torch.nn.functional.softplus(block.ssm.dt_proj.bias)
            self.assertTrue(((dt >= .001) & (dt <= .1)).all())

    def test_b4_new_kernel_learns_after_scale_opens(self):
        torch.manual_seed(6)
        module = FeatureAblationNet("B4").blocks[0].local_dw[0]
        x = torch.randn(2, 32, 5, 5)
        optimizer = torch.optim.SGD(module.parameters(), lr=.01)
        module(x).square().mean().backward()
        self.assertGreater(module.beta.grad.abs().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        module(x).square().mean().backward()
        self.assertGreater(module.dw5.weight.grad.abs().sum().item(), 0)


class LauncherTests(unittest.TestCase):
    def test_default_plan_and_unique_artifacts(self):
        settings = json.loads(batch_runner.DEFAULT_CONFIG.read_text())
        configs = batch_runner.make_plan(settings, "test")
        self.assertEqual([c.model_name for c in configs],
                         [batch_runner.CANONICAL_NAMES[v] for v in batch_runner.VARIANTS])
        self.assertTrue(all(c.random_seed == 6 and c.epochs == 100 and c.enable_testing for c in configs))
        settings["seeds"] = [6, 7]
        configs = batch_runner.make_plan(settings, "test")
        self.assertEqual(len({c.model_save_path for c in configs}), 8)
        for field in ("model_save_path", "report_path", "log_path"):
            self.assertTrue(all(getattr(c, field) != getattr(batch_runner.make_plan(settings, "other")[i], field)
                                for i, c in enumerate(configs)))

    def test_bad_settings_fail(self):
        for settings in ({"models": ["B0"]}, {"seeds": [6, 6]}, {"datasets": [9]},
                         {"training": {"epoch": 1}}, {"training": {"window_size": 10}}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                batch_runner.make_plan(settings, "test")

    def test_dry_run_never_starts_training(self):
        with patch.object(batch_runner, "preflight") as preflight, contextlib.redirect_stdout(io.StringIO()):
            code = batch_runner.main(["--dry-run"])
        self.assertEqual(code, 0)
        preflight.assert_not_called()

    def test_metrics_and_batch_failure_continuation(self):
        configs = batch_runner.make_plan({"models": ["B1", "B2", "B3"]}, "test")
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for i, cfg in enumerate(configs):
                cfg.model_save_path = str(folder / f"{i}.pth")
                cfg.report_path = str(folder / f"{i}.txt")
                cfg.log_path = str(folder / f"{i}.log")

            def fake_child(config_path, console_path):
                cfg = json.loads(config_path.read_text())
                if cfg["model_name"] == batch_runner.CANONICAL_NAMES["B2"]:
                    return 7
                Path(cfg["model_save_path"]).touch()
                Path(cfg["report_path"]).write_text("88.5 Overall accuracy (%)\n89 Average accuracy (%)\n87.2 Kappa accuracy (%)\n")
                return 0

            with patch.object(batch_runner, "run_child", side_effect=fake_child), contextlib.redirect_stdout(io.StringIO()):
                code = batch_runner.execute_plan(configs, folder)
            self.assertEqual(code, 1)
            rows = json.loads((folder / "summary.json").read_text())
            self.assertEqual([r["status"] for r in rows], ["success", "failed", "success"])
            self.assertEqual(rows[0]["oa"], 88.5)
            self.assertTrue((folder / "summary.csv").exists())
            bad_report = folder / "bad.txt"
            bad_report.write_text("incomplete report")
            with self.assertRaises(ValueError):
                batch_runner.read_metrics(bad_report)


if __name__ == "__main__":
    unittest.main()
