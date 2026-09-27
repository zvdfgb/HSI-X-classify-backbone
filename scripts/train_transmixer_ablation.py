"""Run the B1-B4 TransMixer ablation batch through the existing backbone."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATASET_LABELS, ExperimentConfig


DEFAULT_CONFIG = ROOT / "configs" / "feature_ablation.json"
VARIANTS = ("B1", "B2", "B3", "B4")
CANONICAL_NAMES = {
    "B1": "TransMixer_B1_LocalResidual",
    "B2": "TransMixer_B2_PreSSMResidual",
    "B3": "TransMixer_B3_DeltaInit",
    "B4": "TransMixer_B4_MultiScaleDW",
}
TRAINING_KEYS = {
    "cuda_device", "learning_rate", "epochs", "batch_size", "num_workers",
    "channels", "window_size", "enable_visualization",
}


def make_plan(settings, run_tag):
    """Validate settings before any filesystem writes or expensive imports."""
    unknown = set(settings) - {"models", "datasets", "seeds", "training"}
    if unknown:
        raise ValueError(f"Unknown batch settings: {sorted(unknown)}")
    training = settings.get("training", {})
    if set(training) - TRAINING_KEYS:
        raise ValueError(f"Unknown training settings: {sorted(set(training) - TRAINING_KEYS)}")
    models = settings.get("models", list(VARIANTS))
    datasets = settings.get("datasets", [0])
    seeds = settings.get("seeds", [6])
    for name, items in (("models", models), ("datasets", datasets), ("seeds", seeds)):
        if not isinstance(items, list) or not items or len(set(items)) != len(items):
            raise ValueError(f"{name} must be a nonempty list without duplicates")
    if any(m not in VARIANTS for m in models):
        raise ValueError("models must contain only B1, B2, B3, B4")
    if any(type(d) is not int or d not in DATASET_LABELS for d in datasets):
        raise ValueError("datasets must contain integer IDs 0..7")
    if any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("seeds must contain integers in [0, 2**32)")
    configs = []
    for dataset in datasets:
        for seed in seeds:
            for variant in models:
                cfg = ExperimentConfig(
                    experiment_name=f"{run_tag}_{variant}_seed{seed}",
                    model_name=CANONICAL_NAMES[variant], dataset_type=dataset,
                    random_seed=seed, enable_training=True, enable_testing=True,
                    **training,
                )
                for key in ("epochs", "batch_size", "channels", "window_size"):
                    value = getattr(cfg, key)
                    if type(value) is not int or value <= 0:
                        raise ValueError(f"{key} must be a positive integer")
                if cfg.window_size % 2 != 1:
                    raise ValueError("window_size must be odd for center-pixel patches")
                if cfg.channels > cfg.data_channels[dataset]:
                    raise ValueError("PCA channels exceed dataset spectral bands")
                if not math.isfinite(cfg.learning_rate) or cfg.learning_rate <= 0:
                    raise ValueError("learning_rate must be finite and positive")
                if type(cfg.num_workers) is not int or cfg.num_workers < 0:
                    raise ValueError("num_workers must be a nonnegative integer")
                configs.append(cfg)
    return configs


def preflight(configs):
    import torch
    from src.data import dataset_path
    from scripts.train_all import DATASET_REQUIRED_FILES

    missing = set()
    for cfg in configs:
        device = torch.device(cfg.cuda_device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable. Select your PyTorch GPU interpreter in VS Code, "
                                   "or explicitly set training.cuda_device to cpu.")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise ValueError(f"GPU does not exist: {device}")
        elif device.type != "cpu":
            raise ValueError("This runner supports cpu or cuda devices")
        for parts in DATASET_REQUIRED_FILES[cfg.dataset_type]:
            path = Path(dataset_path(*parts))
            if not path.is_file():
                missing.add(str(path))
    if missing:
        raise FileNotFoundError("Missing dataset files:\n" + "\n".join(sorted(missing)))
    return {"python": sys.version, "executable": sys.executable,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda}


def read_metrics(report):
    content = Path(report).read_text(encoding="utf-8")
    result = {}
    for key, label in (("oa", "Overall accuracy"), ("aa", "Average accuracy"),
                       ("kappa", "Kappa accuracy")):
        matches = re.findall(r"([-+0-9.eE]+)\s+" + label, content)
        if not matches or not math.isfinite(float(matches[-1])):
            raise ValueError(f"Missing/invalid {label} in {report}")
        result[key] = float(matches[-1])
    return result


def run_child(config_path, console_path):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    command = [sys.executable, "-u", str(ROOT / "scripts" / "train.py"),
               "--config", str(config_path)]
    with console_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()


def save_summary(folder, rows):
    (folder / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["model", "dataset", "seed", "status", "oa", "aa", "kappa",
              "config", "checkpoint", "report", "console", "error"]
    with (folder / "summary.csv").open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def execute_plan(configs, folder):
    rows = []
    for i, cfg in enumerate(configs, 1):
        stem = f"{DATASET_LABELS[cfg.dataset_type]}_{cfg.model_name}_seed{cfg.random_seed}"
        config_path = folder / f"{stem}.json"
        console_path = folder / f"{stem}.console.log"
        config_path.write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
        row = dict(model=cfg.model_name, dataset=DATASET_LABELS[cfg.dataset_type],
                   seed=cfg.random_seed, status="running", config=str(config_path),
                   checkpoint=cfg.model_save_path, report=cfg.report_path,
                   console=str(console_path), error="")
        rows.append(row)
        save_summary(folder, rows)
        print(f"\n[{i}/{len(configs)}] {stem}", flush=True)
        try:
            # Never let a stale artifact masquerade as a successful new experiment.
            if any(Path(p).exists() for p in (cfg.model_save_path, cfg.log_path, cfg.report_path)):
                raise FileExistsError("Output already exists; choose a new run tag")
            code = run_child(config_path, console_path)
            if code:
                raise RuntimeError(f"Training/testing exited with code {code}; see console log")
            if not Path(cfg.model_save_path).is_file():
                raise FileNotFoundError("Training did not produce a checkpoint")
            row.update(read_metrics(cfg.report_path))
            row["status"] = "success"
        except KeyboardInterrupt:
            row.update(status="interrupted", error="Interrupted by user")
            save_summary(folder, rows)
            return 130
        except Exception as exc:
            row.update(status="failed", error=str(exc))
            print(f"FAILED: {exc}", flush=True)
        save_summary(folder, rows)
    return 1 if any(row["status"] != "success" for row in rows) else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sequential B1-B4 training + testing (B0 is not run)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="+", choices=VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--datasets", nargs="+", type=int, choices=range(8))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without training or writing files")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args(argv)
    try:
        settings = json.loads(args.config.read_text(encoding="utf-8-sig"))
        for key in ("models", "seeds", "datasets"):
            if getattr(args, key) is not None:
                settings[key] = getattr(args, key)
        training = settings.setdefault("training", {})
        if args.epochs is not None:
            training["epochs"] = args.epochs
        if args.device is not None:
            training["cuda_device"] = args.device
        tag = args.run_tag or datetime.now().strftime("b1b4_%Y%m%d_%H%M%S_%f")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", tag):
            raise ValueError("run-tag must be 1-64 letters/digits/underscores/hyphens")
        folder = ROOT / "report" / "feature_ablation" / tag
        configs = make_plan(settings, tag)
        for cfg in configs:
            print(f"{cfg.model_name} | {DATASET_LABELS[cfg.dataset_type]} | seed={cfg.random_seed} "
                  f"| epochs={cfg.epochs} | lr={cfg.learning_rate} | {cfg.cuda_device}")
        print(f"Batch output: {folder}")
        print("Protocol: existing trainer selects checkpoints by test accuracy (same as legacy B0).")
        if args.dry_run:
            return 0
        runtime = preflight(configs)
        folder.mkdir(parents=True, exist_ok=False)
        source_files = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__), ROOT / "scripts" / "train.py"]
        manifest = {"settings": settings, "runtime": runtime,
                    "checkpoint_selection": "legacy_test_accuracy",
                    "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in source_files}}
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        result = execute_plan(configs, folder)
        print(f"\nResults: {folder / 'summary.csv'}")
        return result
    except (ValueError, TypeError, OSError, RuntimeError, ImportError) as exc:
        print(f"Cannot start batch: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
