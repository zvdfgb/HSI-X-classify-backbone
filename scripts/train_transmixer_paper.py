"""Run the approved paper-transfer grid using existing lab components."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper_transfer.json")
    parser.add_argument("--run-tag", default="paper_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_tag):
        raise ValueError("Invalid run tag")
    plan = json.loads(args.config.read_text(encoding="utf-8-sig"))
    from src.paper_transfer_experiment import resolve_protocol, audit_protocol, execute
    reference, cfg = resolve_protocol(plan)
    audit = audit_protocol(reference, cfg)
    print(json.dumps({"plan": plan, "inherited_protocol": cfg, "runtime": audit}, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    return execute(plan, ROOT / "report/paper_transfer" / args.run_tag, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
