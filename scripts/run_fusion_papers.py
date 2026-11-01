"""Audit/queue the approved 15-group seed6 batch using scripts/train.py only.

This is process orchestration and saved-report arithmetic, not a training,
data-loading, checkpoint-selection, prediction or classification-metric loop.
Reuses the existing SR batch's hash/JSON/native report readers.
"""
import argparse
import csv
import ctypes
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_sr_native import read, write, sha, report_metrics

TAG = "fusion_papers_seed6_20261031"
DEST = ROOT / "report/fusion_papers" / TAG
LOGS = ROOT / "log/fusion_papers" / TAG
PLAN = ROOT / "docs/experiments/20261031_三篇Mamba论文_15组_seed6_已批准.json"
PROTECTED = ["configs/train.json", "scripts/train.py", "src/trainer.py", "src/runner.py",
             "src/data.py", "src/evaluator.py", "src/metrics.py", "src/config.py",
             "src/models/transmixer_net/model.py", "src/models/transmixer_paper/model.py",
             "src/models/transmixer_sr/adapter.py"]


def now():
    return datetime.now().astimezone().isoformat()


def state_write(value):
    path = DEST / "运行状态.json"
    value["updated_at"] = now()
    tmp = path.with_suffix(".tmp")
    write(tmp, value)
    os.replace(tmp, path)


def verify_pins(manifest):
    for collection in (manifest["source_and_config_sha256"], manifest["data_sha256"]):
        for rel, expected in collection.items():
            if sha(ROOT / rel) != expected:
                raise RuntimeError(f"Pinned file changed: {rel}")
    for job in manifest["jobs"]:
        if job["reused"]:
            for kind in ("checkpoint", "report"):
                if sha(job[kind]) != job[f"{kind}_sha256"]:
                    raise RuntimeError(f"Reused output changed: {job['id']} / {kind}")


def prepare():
    import numpy as np
    import torch
    from src import config
    from src.config import ExperimentConfig, ConfigManager
    from src.data import set_random_seed
    from src.model_registry import get_model_adapter

    plan = read(PLAN)
    if not plan["training_authorized"] or plan["protocol"]["seeds"] != [6] or len(plan["groups"]) != 15:
        raise RuntimeError("User-approved original 15 groups with seed6 required")
    if DEST.exists():
        raise FileExistsError(f"Batch exists; refusing overwrite: {DEST}")
    historic = ROOT / "report/sr_native/sr_native_20261029"
    original = read(historic / "manifest.json")
    audit = read(historic / "final_integrity_audit.json")
    if str(torch.__version__) != original["torch"] or np.__version__ != original["numpy"]:
        raise RuntimeError("Original Torch/NumPy environment required")
    if Path(sys.executable).resolve() != Path(original["python"]).resolve() or not torch.cuda.is_available():
        raise RuntimeError("Original CUDA/Python runtime required")
    if torch.cuda.get_device_name(0) != original["gpu"]:
        raise RuntimeError("Original training GPU required")
    protected = {}
    for rel in PROTECTED:
        digest = sha(ROOT / rel)
        if digest != original["source_and_config_sha256"][rel]:
            raise RuntimeError(f"Original protected lab source changed: {rel}")
        protected[rel] = digest
    data_hashes = read(historic / "data_sha256.json")
    for rel, expected in data_hashes.items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f"Original research data changed: {rel}")
    stats_path = historic / "training_input_statistics.json"
    stats_rel = stats_path.relative_to(ROOT).as_posix()
    if sha(stats_path) != original["source_and_config_sha256"][stats_rel]:
        raise RuntimeError("Original training-only SR statistics changed")
    scale = read(stats_path)["6"]["spectral_scale"]
    if scale != plan["protocol"]["recorded_spectral_scale"]:
        raise RuntimeError("Approved SR2 scale does not match native input statistics")
    defaults = read(ROOT / "configs/train.json")
    jobs, reuse_audit = [], {}
    for group in plan["groups"]:
        gid = group["id"]
        if group["seeds"] != [6]:
            raise RuntimeError("Unapproved seed")
        if group["planned_action"] == "reuse":
            name = group["reuse_from"]["group"]
            old = next(j for j in original["jobs"] if j["id"] == f"{name}_seed6")
            sealed = next(j for j in audit["outputs"] if j["id"] == old["id"])
            previous_config = read(old["config"])
            for key, value in defaults.items():
                if key not in ("experiment_name", "model_name") and previous_config.get(key) != value:
                    raise RuntimeError(f"Reuse protocol differs: {gid}/{key}")
            for kind in ("checkpoint", "report"):
                if sha(old[kind]) != sealed[f"{kind}_sha256"]:
                    raise RuntimeError(f"Sealed historical {kind} changed: {gid}")
            jobs.append({**old, "id": f"{gid}_seed6", "group": gid, "reused": True,
                         "checkpoint_sha256": sealed["checkpoint_sha256"],
                         "report_sha256": sealed["report_sha256"], "reuse_source": old["id"]})
            reuse_audit[gid] = {"accepted": True, "source": old["id"],
                               "metrics": report_metrics(old["report"])}
            continue
        cfg = {**defaults, "experiment_name": f"{TAG}_{gid}_seed6",
               "model_name": f"TransMixer_FP_{gid}", "random_seed": 6,
               "ablation_mode": group["backbone_mode"], "paper_transfer": {"group": gid}}
        if group["pooling"] == "SR2":
            cfg["sr"] = {"spatial_sigma": 3.0, "spectral_scale": scale}
        exp = ExperimentConfig(**cfg)
        cfg_path = ROOT / "configs/runs" / TAG / f"{gid}_seed6.json"
        if cfg_path.exists():
            raise FileExistsError(cfg_path)
        for path in (exp.model_save_path, exp.report_path, exp.log_path):
            if Path(path).exists():
                raise FileExistsError(path)
        write(cfg_path, cfg)
        jobs.append({"id": f"{gid}_seed6", "group": gid, "seed": 6, "model": cfg["model_name"],
                     "config": str(cfg_path), "reused": False, "checkpoint": exp.model_save_path,
                     "report": exp.report_path, "native_log": exp.log_path,
                     "stdout": str(LOGS / f"{gid}_seed6.stdout.log"),
                     "stderr": str(LOGS / f"{gid}_seed6.stderr.log")})
    DEST.mkdir(parents=True)
    write(DEST / "reuse_audit.json", {"protected_original_files": protected,
                                     "data_files_verified": len(data_hashes), "reused": reuse_audit,
                                     "sr_statistics_source": str(stats_path), "spectral_scale": scale})
    preflight = []
    for job in jobs:
        if job["reused"]:
            continue
        exp = ExperimentConfig(**read(job["config"]))
        set_random_seed(6)
        config._global_config_manager = ConfigManager(exp)
        adapter = get_model_adapter(exp.model_name)
        bundle = adapter.build_model(config, 0, "cuda:0")
        net = bundle["net"]
        net.fusion_conv.alpha_h.data.fill_(0.1)
        net.fusion_conv.alpha_l.data.fill_(0.1)
        batch = {"hsi_pca": torch.randn(128, 1, 30, 11, 11, device="cuda:0"),
                 "aux": torch.randn(128, 1, 11, 11, device="cuda:0"),
                 "label": torch.arange(128, device="cuda:0") % 15}
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        net.train()
        logits = adapter.forward_train(bundle, batch)
        loss = adapter.compute_loss(bundle, logits, batch, torch.nn.CrossEntropyLoss())
        loss.backward()
        if logits.shape != (128, 15) or not torch.isfinite(loss):
            raise RuntimeError(f"Synthetic output/loss failure: {job['id']}")
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
            raise RuntimeError(f"Synthetic gradient failure: {job['id']}")
        interaction = net.fusion_conv.interaction
        if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in interaction.parameters() if p.requires_grad):
            raise RuntimeError(f"Fusion gradients do not propagate: {job['id']}")
        net.eval()
        with torch.no_grad():
            prediction = adapter.forward_eval(bundle, batch)
        if not torch.isfinite(prediction).all():
            raise RuntimeError(f"Synthetic inference failure: {job['id']}")
        torch.cuda.synchronize()
        info = {"group": job["group"], "passed": True, "batch_size": 128,
                "parameters": sum(p.numel() for p in net.parameters()),
                "trainable_parameters": sum(p.numel() for p in net.parameters() if p.requires_grad),
                "peak_synthetic_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
                "synthetic_check_seconds": time.monotonic() - start}
        preflight.append(info)
        write(DEST / "preflight.json", preflight)
        print(f"CUDA batch128 preflight passed: {job['id']}, {info['peak_synthetic_allocated_MiB']:.0f} MiB", flush=True)
        del net, bundle, batch, logits, loss, prediction, interaction
        gc.collect()
        torch.cuda.empty_cache()
    sources = list((ROOT / "src").rglob("*.py"))
    sources += [ROOT / "scripts/train.py", ROOT / "scripts/prepare_sr_native.py", Path(__file__),
                ROOT / "configs/train.json", ROOT / "AGENTS.md", PLAN,
                DEST / "preflight.json", DEST / "reuse_audit.json", stats_path]
    sources += [Path(j["config"]) for j in jobs]
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    manifest = {"run_tag": TAG, "created_at": now(), "python": sys.executable,
                "python_version": platform.python_version(), "torch": str(torch.__version__),
                "numpy": np.__version__, "gpu": torch.cuda.get_device_name(0), "jobs": jobs,
                "total": 15, "reused": 2, "new_training": 13,
                "source_and_config_sha256": hashes, "data_sha256": data_hashes,
                "training_entry": "scripts/train.py", "checkpoint_selection": "native_test_OA_best"}
    write(DEST / "manifest.json", manifest)
    state_write({"state": "prepared", "total": 15, "completed": 2, "new_completed": 0,
                 "reused": 2, "training_pid": None, "current": None})
    summarize()


def summarize():
    manifest = read(DEST / "manifest.json")
    plan = read(PLAN)
    values = []
    for job in manifest["jobs"]:
        marker = DEST / "completed" / f"{job['id']}.json"
        if job["reused"] or marker.exists():
            values.append({"id": job["id"], "group": job["group"], "seed": 6,
                           "reused": job["reused"], "report": job["report"], **report_metrics(job["report"])})
    table = {v["group"]: v for v in values}
    comparisons = []
    for candidate, control in plan["primary_comparisons"] + plan["mechanism_comparisons"] + plan["without_A_comparisons"]:
        if candidate in table and control in table:
            comparisons.append({"candidate": candidate, "control": control,
                                "oa_delta_pp": table[candidate]["oa"] - table[control]["oa"]})
    interactions = {}
    for name, candidate, alone in (("B1", "G04", "G05"), ("B2", "G08", "G09"), ("B3", "G13", "G14")):
        if all(k in table for k in (candidate, alone, "G00", "G01")):
            interactions[name] = (table[candidate]["oa"] - table["G01"]["oa"]) - (table[alone]["oa"] - table["G00"]["oa"])
    write(DEST / "summary.json", {"total": 15, "completed": len(values), "seeds": [6], "runs": values,
                                 "paired_comparisons": comparisons, "single_seed_interactions_pp": interactions,
                                 "evidence_scope": "single seed; original test-OA selection; exploratory only"})
    with (DEST / "逐组结果.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "group", "seed", "reused", "report", "oa", "aa", "kappa"])
        writer.writeheader()
        writer.writerows(values)
    lines = ["# 三篇论文融合实验：seed6原流程进度与结果", "",
             f"已完成 {len(values)}/15；计划复用2项、新训13项。", "",
             "仅seed6，不计算跨种子标准差，不宣称稳定性或统计显著性。原测试OA参与选模，不是独立测试证据。", "",
             "| 组别 | 方法 | OA % | AA % | Kappa % | 来源 |", "|---|---|---:|---:|---:|---|"]
    for group in plan["groups"]:
        gid = group["id"]
        if gid in table:
            v = table[gid]
            lines.append(f"| {gid} | {group['label']} | {v['oa']:.4f} | {v['aa']:.4f} | {v['kappa']:.4f} | {'复用' if v['reused'] else '新训'} |")
        else:
            lines.append(f"| {gid} | {group['label']} | 待完成 | — | — | 新训 |")
    lines += ["", "## 已具备结果的比较", "", "| 比较 | OA差值／百分点 |", "|---|---:|"]
    lines += [f"| {v['candidate']}−{v['control']} | {v['oa_delta_pp']:+.4f} |" for v in comparisons]
    lines += ["", "逐组原始报告由原evaluator/metrics生成。本文仅读取报告并相减，不执行新的评估。",
              "各已完成任务的耗时、检查点哈希和融合诊断见completed目录；原始训练stdout/stderr见log/fusion_papers批次目录。", ""]
    (DEST / "实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def run(resume=False):
    import msvcrt
    lock_path = DEST / "queue.lock"
    lock = lock_path.open("a+b")
    if lock_path.stat().st_size == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    manifest = read(DEST / "manifest.json")
    state = {"state": "starting", "total": 15, "reused": 2, "completed": 2, "new_completed": 0,
             "queue_pid": os.getpid(), "training_pid": None, "current": None, "started_at": now()}
    (DEST / "completed").mkdir(exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    child = None
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        state_write(state)
        for job in manifest["jobs"]:
            if job["reused"]:
                continue
            marker = DEST / "completed" / f"{job['id']}.json"
            if marker.exists():
                if not resume:
                    raise RuntimeError("Completed jobs exist; explicit --resume required")
                done = read(marker)
                for kind in ("checkpoint", "report"):
                    if sha(job[kind]) != done[f"{kind}_sha256"]:
                        raise RuntimeError("Completed artifact hash changed")
                state["completed"] += 1
                state["new_completed"] += 1
                continue
            verify_pins(manifest)
            check = subprocess.run(["powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' -and $_.CommandLine -match '(scripts[/\\\\]train\\.py|train_all\\.py|run_batch\\.py)' } | Select-Object -ExpandProperty ProcessId"],
                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            if check.returncode != 0 or check.stdout.strip():
                raise RuntimeError(f"Another native training may be active: {check.stdout.strip()}")
            for key in ("stdout", "stderr", "checkpoint", "report", "native_log"):
                if Path(job[key]).exists():
                    raise FileExistsError(f"Refusing to overwrite unfinished output: {job[key]}")
            state.update(state="launching", current=job, training_pid=None, current_epoch=None)
            state_write(state)
            started = now()
            clock = time.monotonic()
            env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
            env.pop("CUBLAS_WORKSPACE_CONFIG", None)
            with Path(job["stdout"]).open("wb") as out, Path(job["stderr"]).open("wb") as err:
                child = subprocess.Popen([manifest["python"], "-X", "utf8", "-u", "scripts/train.py", "--config", job["config"]],
                                         cwd=ROOT, env=env, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
                state.update(state="training", training_pid=child.pid, current_started_at=started)
                state_write(state)
                print(f"Started {job['id']} pid={child.pid}", flush=True)
                while child.poll() is None:
                    time.sleep(10)
                    with Path(job["stderr"]).open("rb") as progress:
                        progress.seek(max(0, Path(job["stderr"]).stat().st_size - 12000))
                        tail = progress.read().decode("utf-8", errors="replace")
                    epochs = re.findall(r"Epoch (\d+)/100", tail)
                    state["current_epoch"] = int(epochs[-1]) if epochs else None
                    state["current_elapsed_seconds"] = round(time.monotonic() - clock, 1)
                    state_write(state)
            if child.returncode != 0:
                raise RuntimeError(f"Native training failed: {job['id']} exit={child.returncode}")
            stdout = Path(job["stdout"]).read_text(encoding="utf-8", errors="replace")
            stderr = Path(job["stderr"]).read_text(encoding="utf-8", errors="replace")
            if "Report Success!" not in stdout or "Epoch 100/100" not in stderr:
                raise RuntimeError(f"Native epoch100/report completion evidence missing: {job['id']}")
            metrics = report_metrics(job["report"])
            verify_pins(manifest)
            import torch
            net = torch.load(job["checkpoint"], map_location="cpu", weights_only=False)
            diagnostic = net.fusion_conv.diagnostics()
            del net
            done = {"id": job["id"], "started_at": started, "finished_at": now(),
                    "elapsed_seconds": time.monotonic() - clock, "epochs": 100,
                    "checkpoint_sha256": sha(job["checkpoint"]), "report_sha256": sha(job["report"]),
                    "metrics": metrics, "best_checkpoint_fusion_diagnostics": diagnostic}
            write(marker, done)
            summarize()
            state["completed"] += 1
            state["new_completed"] += 1
            state["training_pid"] = None
            state_write(state)
            print(f"Completed {job['id']}: OA={metrics['oa']:.4f}", flush=True)
        verify_pins(manifest)
        state.update(state="completed", current=None, training_pid=None, finished_at=now())
        state_write(state)
        print("All 15 results complete: 2 reused + 13 new.", flush=True)
    except BaseException as exc:
        state.update(state="failed", error=repr(exc))
        state_write(state)
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        run(args.resume)
    else:
        summarize()
