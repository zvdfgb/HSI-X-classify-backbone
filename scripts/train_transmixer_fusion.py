"""Run from VS Code/F5 for the pilot; --config selects the confirmation batch."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paired TransMixer fusion experiments with validation-only selection")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fusion_pilot.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-tag")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    for key in ("variants", "datasets", "seeds", "epochs", "device"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    from src.fusion_experiment import execute, validate_settings, LABELS
    validate_settings(cfg)
    tag = args.run_tag or datetime.now().strftime("fusion_%Y%m%d_%H%M%S_%f")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", tag):
        raise ValueError("run-tag must be 1-64 letters, digits, underscores or hyphens")
    folder = ROOT / "report/fusion_ablation" / tag
    for dataset in cfg["datasets"]:
        for seed in cfg["seeds"]:
            for variant in cfg["variants"]:
                print(f"{dataset} {variant} {LABELS[variant]} seed{seed} epochs={cfg['epochs']}", flush=True)
    print(f"Output: {folder}\nSelection: validation only; evaluate_test={cfg['evaluate_test']}", flush=True)
    if args.dry_run:
        return 0
    return execute(cfg, folder)


if __name__ == "__main__":
    raise SystemExit(main())
