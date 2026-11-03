"""Approved 13-result queue. Scientific execution is ONLY scripts/train.py.

Reuses original batch JSON/hash/report readers, pin audit, state writer and
Windows power guard. Does not implement training, loading or evaluation.
"""
import argparse
import csv
import gc
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_fusion_papers as core
from scripts.prepare_sr_native import read, write, sha, report_metrics

TAG = "fusion_transfer_seed6_20261102"
DEST = ROOT / "report/fusion_papers" / TAG
LOGS = ROOT / "log/fusion_papers" / TAG
ARCHIVE = ROOT / "docs/experiments/provenance" / TAG
PLAN = ROOT / "docs/experiments/20261102_三篇融合机制_实验矩阵_已批准.json"
OLD = ROOT / "report/fusion_papers/fusion_B1_seed78_20261101"
core.DEST, core.LOGS = DEST, LOGS


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def check_no_training():
    query = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' "
             "-and $_.CommandLine -match '(scripts[/\\\\]train\\.py|train_all\\.py|run_batch\\.py)' } | "
             "Select-Object -ExpandProperty ProcessId")
    result = subprocess.run(["powershell", "-NoProfile", "-Command", query],
        capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode or result.stdout.strip():
        raise RuntimeError(f"Existing training or process inspection failed: {result.stdout} {result.stderr}")


def prepare():
    import numpy as np
    import torch
    from src.config import ExperimentConfig
    from src.models.transmixer_fusion_v2.model import GROUPS
    check_no_training()
    plan, old = read(PLAN), read(OLD / "manifest.json")
    assert plan["training_authorized_for_this_matrix"] and plan["status"] == "approved"
    assert [g["id"] for g in plan["groups"]] == ["A00", *GROUPS]
    assert all(g["seed"] == 6 and g["epochs"] == 100 for g in plan["groups"])
    if DEST.exists():
        raise FileExistsError(f"Refusing overwrite of existing batch: {DEST}")
    assert read(OLD / "运行状态.json")["state"] == "completed"
    assert Path(sys.executable).resolve() == Path(old["python"]).resolve()
    assert platform.python_version() == old["python_version"]
    assert str(torch.__version__) == old["torch"] and np.__version__ == old["numpy"]
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == old["gpu"]
    migration = read(ARCHIVE / "pre_edit_audit.json")
    registry = "src/model_registry.py"
    assert sha(ARCHIVE / "model_registry_before.py.txt") == old["source_and_config_sha256"][registry]
    for category in ("source_and_config_sha256", "data_sha256"):
        for rel, expected in old[category].items():
            if rel != registry and sha(ROOT / rel) != expected:
                raise RuntimeError(f"Original source/config/data changed: {rel}")
    migration.update(registry_after_sha256=sha(ROOT / registry), verified_at=now())
    write(ARCHIVE / "source_migration.json", migration)
    baseline = next(j for j in old["jobs"] if j["id"] == "G01_seed6")
    for kind in ("checkpoint", "report"):
        assert sha(baseline[kind]) == baseline[f"{kind}_sha256"]
    defaults = read(ROOT / "configs/train.json")
    baseline_cfg = read(baseline["config"])
    for key, value in defaults.items():
        if key not in ("experiment_name", "model_name"):
            assert baseline_cfg.get(key) == value, key
    assert baseline_cfg["ablation_mode"] == "decoupled"
    assert baseline_cfg["sr"] == {"spatial_sigma": 3.0, "spectral_scale": plan["fixed_protocol"]["sr_spectral_scale"]}
    native = ROOT / "report/sr_native/sr_native_20261029"
    assert read(native / "training_input_statistics.json")["6"]["spectral_scale"] == baseline_cfg["sr"]["spectral_scale"]
    jobs = [{**baseline, "id": "A00_seed6", "group": "A00", "seed": 6, "reused": True,
             "reuse_source": "G01_seed6 / M3_seed6"}]
    for gid in GROUPS:
        cfg = {**defaults, "experiment_name": f"{TAG}_{gid}_seed6", "model_name": f"TransMixer_FV2_{gid}",
               "random_seed": 6, "ablation_mode": "decoupled", "sr": baseline_cfg["sr"],
               "paper_transfer": {"group": gid}}
        exp = ExperimentConfig(**cfg)
        path = ROOT / "configs/runs" / TAG / f"{gid}_seed6.json"
        for candidate in (path, Path(exp.model_save_path), Path(exp.report_path), Path(exp.log_path)):
            if candidate.exists():
                raise FileExistsError(candidate)
        write(path, cfg)
        jobs.append({"id": f"{gid}_seed6", "group": gid, "seed": 6, "model": exp.model_name,
                     "config": str(path), "reused": False, "checkpoint": exp.model_save_path,
                     "report": exp.report_path, "native_log": exp.log_path,
                     "stdout": str(LOGS / f"{gid}_seed6.stdout.log"),
                     "stderr": str(LOGS / f"{gid}_seed6.stderr.log")})
    DEST.mkdir(parents=True)
    write(DEST / "reuse_audit.json", {"baseline": jobs[0], "metrics": report_metrics(baseline["report"]),
        "original_source_files_verified": len(old["source_and_config_sha256"]),
        "data_files_verified": len(old["data_sha256"]), "source_migration": str(ARCHIVE / "source_migration.json")})
    from scripts.verify_fusion_transfer import preflight
    preflight(jobs, DEST)
    sources = list((ROOT / "src").rglob("*.py"))
    sources += [ROOT / rel for rel in ("scripts/train.py", "scripts/run_fusion_papers.py", "scripts/prepare_sr_native.py",
        "scripts/keep_awake_fusion.py", "scripts/run_fusion_transfer.py", "scripts/verify_fusion_transfer.py",
        "configs/train.json", "AGENTS.md")]
    sources += [PLAN, PLAN.with_name("20261102_三篇融合机制_实验审核稿.md"),
                ARCHIVE / "pre_edit_audit.json", ARCHIVE / "source_migration.json", ARCHIVE / "model_registry_before.py.txt",
                native / "training_input_statistics.json", DEST / "preflight.json", DEST / "reuse_audit.json"]
    sources += [Path(j["config"]) for j in jobs]
    manifest = {"run_tag": TAG, "created_at": now(), "python": sys.executable,
        "python_version": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0), "jobs": jobs, "total": 13, "reused": 1, "new_training": 12,
        "source_and_config_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        "data_sha256": old["data_sha256"], "training_entry": "scripts/train.py",
        "checkpoint_selection": "native_test_OA_best", "additional_loss_design": str(PLAN),
        "FINE_optimization": "approved synchronous gradient isolation; not official alternating optimization"}
    write(DEST / "manifest.json", manifest)
    core.verify_pins(manifest)
    core.state_write({"state": "prepared", "total": 13, "reused": 1, "completed": 1, "new_completed": 0,
                      "new_total": 12, "current": None, "training_pid": None})
    summarize()


def summarize():
    manifest, plan = read(DEST / "manifest.json"), read(PLAN)
    rows = []
    for job in manifest["jobs"]:
        if job["reused"] or (DEST / "completed" / f"{job['id']}.json").exists():
            rows.append({k: job[k] for k in ("id", "group", "seed", "reused", "report")} | report_metrics(job["report"]))
    table = {r["group"]: r for r in rows}
    comparisons = []
    for pair in plan["comparisons"]["primary"] + plan["comparisons"]["mechanism"]:
        a, b = pair.split("-")
        if a in table and b in table:
            comparisons.append({"comparison": pair, **{f"{k}_delta_pp": table[a][k] - table[b][k] for k in ("oa", "aa", "kappa")}})
    write(DEST / "summary.json", {"total": 13, "completed": len(rows), "reused": 1,
        "new_completed": len(rows) - 1, "new_total": 12, "seeds": [6], "runs": rows, "comparisons": comparisons,
        "limitations": plan["interpretation"]["limitations"]})
    with (DEST / "逐组结果.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 三篇论文融合机制迁移实验", "", f"已完成{len(rows)}/13：复用A基线1项，新训{len(rows)-1}/12项；仅seed6，每项100轮。", "",
             "A=局部/全局固定1:1通道分工+SR2。保留原测试OA选模；单种子探索，不是独立测试或统计显著证据。",
             "UMF的KL及FTRE附加目标为已审核变量；FINE采用单AdamW梯度隔离适配，不等价于官方交替优化。", "",
             "| 组别 | 模块 | OA % | AA % | Kappa % | 来源 |", "|---|---|---:|---:|---:|---|"]
    for group in plan["groups"]:
        row = table.get(group["id"])
        result = f"{row['oa']:.4f} | {row['aa']:.4f} | {row['kappa']:.4f}" if row else "待完成 | — | —"
        lines.append(f"| {group['id']} | {group['module']} | {result} | {'复用' if group['id']=='A00' else '新训'} |")
    lines += ["", "## 预先固定的比较", "", "差值单位：百分点。", "", "| 比较 | OA差值 | AA差值 | Kappa差值 |", "|---|---:|---:|---:|"]
    lines += [f"| {c['comparison']} | {c['oa_delta_pp']:+.4f} | {c['aa_delta_pp']:+.4f} | {c['kappa_delta_pp']:+.4f} |" for c in comparisons]
    lines += ["", "完整原生分类报告包含逐类指标。未完成组的中间OA不作最终结果。容量匹配与机制梯度见preflight.json，完成证据和诊断见completed。", ""]
    (DEST / "实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def final_audit():
    manifest = read(DEST / "manifest.json")
    core.verify_pins(manifest)
    outputs = []
    for job in manifest["jobs"]:
        if job["reused"]:
            outputs.append({"id": job["id"], "reused": True, "checkpoint_sha256": sha(job["checkpoint"]), "report_sha256": sha(job["report"])})
            continue
        seal = read(DEST / "completed" / f"{job['id']}.json")
        assert seal["epochs"] == 100
        for key in ("checkpoint", "report", "stdout", "stderr", "native_log"):
            assert sha(job[key]) == seal[f"{key}_sha256"]
        outputs.append(seal)
    write(DEST / "final_integrity_audit.json", {"verified_at": now(), "all_passed": True,
        "new_training_runs": 12, "reused_runs": 1, "source_and_data_pins_verified": True, "outputs": outputs})


def run():
    import msvcrt
    from scripts.keep_awake_fusion import kernel
    manifest = read(DEST / "manifest.json")
    if read(DEST / "运行状态.json")["state"] != "prepared":
        raise RuntimeError("Only fresh prepared queue may start; no implicit resume/retraining")
    lock = (DEST / "queue.lock").open("xb+")
    lock.write(b"0"); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    state = {"state": "starting", "total": 13, "reused": 1, "completed": 1, "new_completed": 0,
             "new_total": 12, "queue_pid": os.getpid(), "training_pid": None, "current": None, "started_at": now()}
    child = None
    (DEST / "completed").mkdir(exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    try:
        if not kernel.SetThreadExecutionState(0x80000001):
            raise RuntimeError("Temporary idle sleep protection request failed")
        core.state_write(state)
        for job in manifest["jobs"]:
            if job["reused"]:
                continue
            core.verify_pins(manifest)
            check_no_training()
            for key in ("stdout", "stderr", "checkpoint", "report", "native_log"):
                if Path(job[key]).exists():
                    raise FileExistsError(f"Refusing overwrite: {job[key]}")
            started, clock = now(), time.monotonic()
            state.update(state="launching", current=job, training_pid=None, current_epoch=None)
            core.state_write(state)
            env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
            env.pop("CUBLAS_WORKSPACE_CONFIG", None)
            with Path(job["stdout"]).open("xb") as out, Path(job["stderr"]).open("xb") as err:
                child = subprocess.Popen([manifest["python"], "-X", "utf8", "-u", "scripts/train.py", "--config", job["config"]],
                    cwd=ROOT, env=env, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
                state.update(state="training", training_pid=child.pid, current_started_at=started)
                core.state_write(state)
                print(f"Started {job['id']} pid={child.pid}", flush=True)
                while child.poll() is None:
                    time.sleep(10)
                    with Path(job["stderr"]).open("rb") as progress:
                        progress.seek(max(0, Path(job["stderr"]).stat().st_size - 16000))
                        epochs = re.findall(r"Epoch (\d+)/100", progress.read().decode("utf-8", errors="replace"))
                    state.update(current_epoch=int(epochs[-1]) if epochs else None,
                                 current_elapsed_seconds=round(time.monotonic() - clock, 1))
                    core.state_write(state)
            if child.returncode != 0:
                raise RuntimeError(f"Native training failed: {job['id']} exit={child.returncode}")
            stdout = Path(job["stdout"]).read_text(encoding="utf-8", errors="replace")
            stderr = Path(job["stderr"]).read_text(encoding="utf-8", errors="replace")
            if "Report Success!" not in stdout or not re.search(r"Epoch 100/100:\s+100%[^\r\n]*22/22", stderr):
                raise RuntimeError(f"Native 100-epoch completion evidence missing: {job['id']}")
            metrics = report_metrics(job["report"])
            core.verify_pins(manifest)
            import torch
            net = torch.load(job["checkpoint"], map_location="cpu", weights_only=False)
            diagnostic = net.fusion_conv.diagnostics()
            del net
            done = {"id": job["id"], "seed": 6, "started_at": started, "finished_at": now(),
                "elapsed_seconds": time.monotonic() - clock, "epochs": 100, "metrics": metrics,
                "best_checkpoint_fusion_diagnostics": diagnostic,
                **{f"{key}_sha256": sha(job[key]) for key in ("checkpoint", "report", "stdout", "stderr", "native_log")}}
            write(DEST / "completed" / f"{job['id']}.json", done)
            summarize()
            state.update(completed=state["completed"] + 1, new_completed=state["new_completed"] + 1, training_pid=None)
            core.state_write(state)
            print(f"Completed {job['id']}: OA={metrics['oa']:.4f}", flush=True)
        final_audit()
        state.update(state="completed", current=None, training_pid=None, finished_at=now())
        core.state_write(state)
        print("All 13 results complete: 1 reused + 12 new.", flush=True)
    except BaseException as exc:
        state.update(state="failed", error=repr(exc), training_pid=child.pid if child and child.poll() is None else None)
        core.state_write(state)
        raise
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


def guard():
    from scripts import keep_awake_fusion as g
    g.BATCH, g.STATUS, g.EVIDENCE = DEST, DEST / "运行状态.json", DEST / "power_protection.json"
    info = {"helper_pid": os.getpid(), "scope": "temporary idle sleep protection during approved queue",
            "power_plan_changed": False, "display_required": False, "started_at": now()}
    try:
        while True:
            state = read(g.STATUS)
            if not (g.is_running(state.get("queue_pid")) or g.is_running(state.get("training_pid"))):
                break
            if not g.kernel.SetThreadExecutionState(0x80000001):
                raise RuntimeError("Temporary sleep protection failed")
            supply = g.PowerStatus()
            valid = g.kernel.GetSystemPowerStatus(g.ctypes.byref(supply))
            info.update(state="active", native_request_succeeded=True,
                        ac_connected=supply.ac == 1 if valid else None,
                        battery_percent=int(supply.battery_percent) if valid else None, checked_at=now())
            g.write(info)
            time.sleep(30)
    except BaseException as exc:
        info.update(state="failed", error=repr(exc))
        raise
    finally:
        g.kernel.SetThreadExecutionState(0x80000000)
        if info.get("state") != "failed":
            info["state"] = "released"
        info["updated_at"] = now()
        g.write(info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "guard", "summarize", "audit"))
    args = parser.parse_args()
    os.chdir(ROOT)
    {"prepare": prepare, "run": run, "guard": guard, "summarize": summarize, "audit": final_audit}[args.action]()
