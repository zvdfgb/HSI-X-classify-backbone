"""Full-size real-data CUDA smoke checks, kept separate from formal results."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from src import fusion_experiment as lab
    from src.paper_transfer_experiment import (
        read, write, now, resolve_protocol, audit_protocol, training_statistics,
        manager_for, settings_for, reuse_baseline, get_model_adapter,
    )
    plan = read(ROOT / "configs/paper_transfer.json")
    reference, cfg = resolve_protocol(plan)
    audit = audit_protocol(reference, cfg)
    import torch
    folder = ROOT / "report/paper_transfer_preflight" / ("check_" + now()[:19].replace(":", "").replace("-", "").replace("T", "_"))
    folder.mkdir(parents=True, exist_ok=False)
    lab.seed_everything(6, cfg["cpu_threads"])
    data = lab.prepare_data("Houston2013", cfg, folder / "Houston2013")
    stats = training_statistics(data, cfg)
    checks = []
    for seed in plan["seeds"]:
        out = folder / "Houston2013" / f"G0_seed{seed}"
        row = reuse_baseline(reference, seed, cfg, data, out, {})
        checks.append(dict(check="archived_baseline_reproduction", seed=seed, validation=row["validation"], passed=True))
        print(f"BASELINE seed{seed}: every archived validation prediction reproduced", flush=True)
    for variant in ("G0", "SR3", "PN2", "OC3"):
        settings = settings_for(variant, 6, plan, stats, folder)
        job_cfg = {**cfg, "epochs": 1}
        manager = manager_for(variant, 6, job_cfg, settings)
        selected = get_model_adapter(manager.config.model_name)
        row = lab.run_one(job_cfg, variant, 6, data, folder / "smoke" / variant,
                          model_adapter=selected, model_config=manager)
        check = dict(check="real_data_one_epoch_cuda", variant=variant, passed=True,
                     validation=row["validation"], test_accessed=row["test"] is not None)
        if variant == "G0":
            original = json.loads((reference / "Houston2013/F0_seed6/history.jsonl").read_text(encoding="utf-8").splitlines()[0])
            current = json.loads((folder / "smoke/G0/history.jsonl").read_text(encoding="utf-8").splitlines()[0])
            check["training_loss_absolute_difference"] = abs(original["train_loss"] - current["train_loss"])
            check["validation_ce_absolute_difference"] = abs(original["validation"]["loss"] - current["validation"]["loss"])
            if check["training_loss_absolute_difference"] > 1e-6 or check["validation_ce_absolute_difference"] > 1e-6 or original["validation"]["confusion_matrix"] != current["validation"]["confusion_matrix"]:
                write(folder / "baseline_parity_failure.json", check)
                raise RuntimeError("Real-data first epoch no longer reproduces the previous baseline protocol")
        checks.append(check)
        print(f"SMOKE {variant}: complete, validation selected, no test scoring", flush=True)
        torch.cuda.empty_cache()
    write(folder / "preflight.json", dict(passed=True, created_at=now(), audit=audit,
                                          checks=checks, note="One-epoch smoke only; not formal 100-epoch results"))
    print(json.dumps(dict(passed=True, folder=str(folder), checks=len(checks)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
