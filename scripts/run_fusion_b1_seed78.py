"""Approved B1 seed7/8 extension: orchestration and saved-report reading only.

Every scientific run uses the unchanged scripts/train.py -> runner -> trainer.
No training/data/selection/metric implementation is introduced here.
"""
import argparse
import csv
import gc
import importlib.machinery
import importlib.util
import json
import os
import platform
import re
import statistics
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

TAG = "fusion_B1_seed78_20261101"
DEST = ROOT / "report/fusion_papers" / TAG
LOGS = ROOT / "log/fusion_papers" / TAG
PLAN = ROOT / "docs/experiments/20261101_B1补充种子7_8_实验矩阵_已批准.json"
ARCHIVE = ROOT / "docs/experiments/provenance" / TAG
OLD = ROOT / "report/fusion_papers/fusion_papers_seed6_20261031"
NATIVE = ROOT / "report/sr_native/sr_native_20261029"
GROUPS = ("G02", "G03", "G04", "G05")
core.DEST, core.LOGS = DEST, LOGS


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def prepare():
    import numpy as np
    import torch
    from src import config
    from src.config import ExperimentConfig, ConfigManager
    from src.data import set_random_seed
    from src.models.transmixer_fusion_papers import adapter

    plan = read(PLAN)
    expected = [(g, s) for s in (7, 8) for g in GROUPS]
    assert plan["training_authorized"] and plan["status"] == "approved"
    assert [(j["group"], j["seed"]) for j in plan["new_jobs"]] == expected
    assert all(j["epochs"] == 100 for j in plan["new_jobs"])
    if DEST.exists():
        raise FileExistsError(f"Refusing to overwrite batch: {DEST}")
    old, native = read(OLD / "manifest.json"), read(NATIVE / "manifest.json")
    assert read(OLD / "运行状态.json")["state"] == "completed"
    assert Path(sys.executable).resolve() == Path(old["python"]).resolve()
    assert str(torch.__version__) == old["torch"] and np.__version__ == old["numpy"]
    assert platform.python_version() == old["python_version"]
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == old["gpu"]
    adapter_rel = "src/models/transmixer_fusion_papers/adapter.py"
    migration = read(ARCHIVE / "source_migration.json")
    assert sha(ARCHIVE / "adapter_seed6_before_extension.py.txt") == old["source_and_config_sha256"][adapter_rel]
    for category in ("source_and_config_sha256", "data_sha256"):
        for rel, expected_hash in old[category].items():
            if rel != adapter_rel and sha(ROOT / rel) != expected_hash:
                raise RuntimeError(f"Previous batch source/config/data drift: {rel}")
    protected = {rel: sha(ROOT / rel) for rel in core.PROTECTED}
    assert all(digest == native["source_and_config_sha256"][rel] for rel, digest in protected.items())
    migration.update(new_adapter_sha256=sha(ROOT / adapter_rel), verified_at=now())
    write(ARCHIVE / "source_migration.json", migration)
    defaults = read(ROOT / "configs/train.json")
    jobs, reuse = [], []
    stats = NATIVE / "training_input_statistics.json"
    assert sha(stats) == native["source_and_config_sha256"][stats.relative_to(ROOT).as_posix()]
    scale = read(stats)["6"]["spectral_scale"]
    assert scale == plan["protocol"]["recorded_spectral_scale"]

    def reuse_job(source, source_id, gid, seed, folder):
        original = next(j for j in source["jobs"] if j["id"] == source_id)
        seal = original if original["reused"] else read(folder / "completed" / f"{source_id}.json")
        cfg = read(original["config"])
        for key, value in defaults.items():
            if key not in ("experiment_name", "model_name", "random_seed") and cfg.get(key) != value:
                raise RuntimeError(f"Reused config mismatch: {source_id}/{key}")
        assert cfg["random_seed"] == seed
        assert cfg["ablation_mode"] == ("pure_ssm" if gid in ("G00", "G05") else "decoupled")
        if gid not in ("G00", "G05"):
            assert cfg["sr"] == {"spatial_sigma": 3.0, "spectral_scale": scale}
        for kind in ("checkpoint", "report"):
            assert sha(original[kind]) == seal[f"{kind}_sha256"]
        metrics = report_metrics(original["report"])
        jobs.append({**original, "id": f"{gid}_seed{seed}", "group": gid, "seed": seed,
                     "reused": True, "reuse_source": source_id,
                     "checkpoint_sha256": seal["checkpoint_sha256"], "report_sha256": seal["report_sha256"]})
        reuse.append({"id": jobs[-1]["id"], "source": source_id, "accepted": True, "metrics": metrics})

    for gid in ("G00", "G01", *GROUPS):
        reuse_job(old, f"{gid}_seed6", gid, 6, OLD)
    for seed in (7, 8):
        for gid, source_id in (("G00", "M0"), ("G01", "M3")):
            reuse_job(native, f"{source_id}_seed{seed}", gid, seed, NATIVE)
    for gid, seed in expected:
        cfg = read(ROOT / "configs/runs/fusion_papers_seed6_20261031" / f"{gid}_seed6.json")
        cfg.update(experiment_name=f"{TAG}_{gid}_seed{seed}", random_seed=seed)
        exp = ExperimentConfig(**cfg)
        path = ROOT / "configs/runs" / TAG / f"{gid}_seed{seed}.json"
        for candidate in (path, Path(exp.model_save_path), Path(exp.report_path), Path(exp.log_path)):
            if candidate.exists():
                raise FileExistsError(candidate)
        write(path, cfg)
        jobs.append({"id": f"{gid}_seed{seed}", "group": gid, "seed": seed, "model": exp.model_name,
                     "config": str(path), "reused": False, "checkpoint": exp.model_save_path,
                     "report": exp.report_path, "native_log": exp.log_path,
                     "stdout": str(LOGS / f"{gid}_seed{seed}.stdout.log"),
                     "stderr": str(LOGS / f"{gid}_seed{seed}.stderr.log")})
    DEST.mkdir(parents=True)
    write(DEST / "reuse_audit.json", {"protected_original_files": protected, "reused": reuse,
                                      "data_files_verified": len(old["data_sha256"]),
                                      "spectral_scale": scale, "sr_statistics_source": str(stats),
                                      "source_migration": migration})
    # A synthetic interface check only: no real data, optimizer or scientific epochs.
    loader = importlib.machinery.SourceFileLoader(
        "src.models.transmixer_fusion_papers._adapter_before_extension",
        str(ARCHIVE / "adapter_seed6_before_extension.py.txt"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    previous = importlib.util.module_from_spec(spec)
    loader.exec_module(previous)
    preflight = {"seed6_regression": [], "seed_restrictions": [], "new_jobs": []}
    for gid in GROUPS:
        cfg = read(ROOT / "configs/runs/fusion_papers_seed6_20261031" / f"{gid}_seed6.json")
        config._global_config_manager = ConfigManager(ExperimentConfig(**cfg))
        set_random_seed(6)
        before = previous.build_model(config, 0, "cuda:0")["net"].eval()
        set_random_seed(6)
        after = adapter.build_model(config, 0, "cuda:0")["net"].eval()
        assert before.state_dict().keys() == after.state_dict().keys()
        for key, value in before.state_dict().items():
            torch.testing.assert_close(value, after.state_dict()[key], rtol=0, atol=0)
        h = torch.randn(2, 1, 30, 11, 11, device="cuda:0")
        l = torch.randn(2, 1, 11, 11, device="cuda:0")
        with torch.no_grad():
            torch.testing.assert_close(before(h, l), after(h, l), rtol=0, atol=0)
        preflight["seed6_regression"].append({"group": gid, "state_and_logits_bit_exact": True})
        del before, after, h, l
    for gid, seed in (("G02", 9), ("G06", 7), ("G06", 8), ("G13", 7), ("G14", 8)):
        cfg = read(ROOT / "configs/runs/fusion_papers_seed6_20261031" / f"{gid}_seed6.json")
        cfg["random_seed"] = seed
        config._global_config_manager = ConfigManager(ExperimentConfig(**cfg))
        try:
            adapter.build_model(config, 0, "cuda:0")
        except ValueError as exc:
            assert "Approved fusion protocol" in str(exc)
        else:
            raise AssertionError(f"Unapproved group/seed accepted: {gid}/{seed}")
        preflight["seed_restrictions"].append({"group": gid, "seed": seed, "rejected": True})
    for job in jobs:
        if job["reused"]:
            continue
        exp = ExperimentConfig(**read(job["config"]))
        set_random_seed(exp.random_seed)
        config._global_config_manager = ConfigManager(exp)
        bundle = adapter.build_model(config, 0, "cuda:0")
        net = bundle["net"]
        assert net.fusion_conv.alpha_h.item() == net.fusion_conv.alpha_l.item() == 0.0
        net.fusion_conv.alpha_h.data.fill_(0.1)
        net.fusion_conv.alpha_l.data.fill_(0.1)
        batch = {"hsi_pca": torch.randn(128, 1, 30, 11, 11, device="cuda:0"),
                 "aux": torch.randn(128, 1, 11, 11, device="cuda:0"),
                 "label": torch.arange(128, device="cuda:0") % 15}
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        net.train()
        logits = adapter.forward_train(bundle, batch)
        loss = adapter.compute_loss(bundle, logits, batch, torch.nn.CrossEntropyLoss())
        assert logits.shape == (128, 15) and torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in net.fusion_conv.interaction.parameters())
        net.eval()
        with torch.no_grad():
            prediction = adapter.forward_eval(bundle, batch)
        assert torch.isfinite(prediction).all()
        torch.cuda.synchronize()
        preflight["new_jobs"].append({"id": job["id"], "passed": True, "batch_size": 128,
            "parameters": sum(p.numel() for p in net.parameters()),
            "peak_synthetic_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
            "seconds": time.monotonic() - started})
        write(DEST / "preflight.json", preflight)
        print(f"CUDA preflight passed: {job['id']}", flush=True)
        del net, bundle, batch, logits, loss, prediction
        gc.collect()
        torch.cuda.empty_cache()
    sources = list((ROOT / "src").rglob("*.py"))
    sources += [ROOT / "scripts/train.py", ROOT / "scripts/prepare_sr_native.py",
                ROOT / "scripts/run_fusion_papers.py", ROOT / "scripts/keep_awake_fusion.py", Path(__file__),
                ROOT / "configs/train.json", ROOT / "AGENTS.md", PLAN, stats,
                DEST / "preflight.json", DEST / "reuse_audit.json",
                ARCHIVE / "source_migration.json", ARCHIVE / "adapter_seed6_before_extension.py.txt"]
    sources += [Path(j["config"]) for j in jobs]
    manifest = {"run_tag": TAG, "created_at": now(), "python": sys.executable,
                "python_version": platform.python_version(), "torch": str(torch.__version__),
                "numpy": np.__version__, "gpu": torch.cuda.get_device_name(0), "jobs": jobs,
                "total": 18, "reused": 10, "new_training": 8,
                "source_and_config_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
                "data_sha256": old["data_sha256"], "training_entry": "scripts/train.py",
                "checkpoint_selection": "native_test_OA_best"}
    write(DEST / "manifest.json", manifest)
    core.verify_pins(manifest)
    core.state_write({"state": "prepared", "total": 18, "reused": 10, "completed": 10,
                      "new_completed": 0, "new_total": 8, "current": None, "training_pid": None})
    summarize()


def summarize():
    manifest, plan = read(DEST / "manifest.json"), read(PLAN)
    rows = []
    for j in manifest["jobs"]:
        if j["reused"] or (DEST / "completed" / f"{j['id']}.json").exists():
            rows.append({k: j[k] for k in ("id", "group", "seed", "reused", "report")} |
                        report_metrics(j["report"]))
    table = {(x["group"], x["seed"]): x for x in rows}
    aggregates = []
    for g in plan["groups"]:
        members = [table[g["id"], s] for s in (6, 7, 8) if (g["id"], s) in table]
        item = {"group": g["id"], "n": len(members), "complete": len(members) == 3}
        if item["complete"]:
            for metric in ("oa", "aa", "kappa"):
                values = [x[metric] for x in members]
                item[metric] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}
        aggregates.append(item)
    pairs = []
    for a, b in plan["primary_comparisons"] + plan["secondary_comparisons"]:
        differences = [{"seed": s, "oa_delta_pp": table[a, s]["oa"] - table[b, s]["oa"]}
                       for s in (6, 7, 8) if (a, s) in table and (b, s) in table]
        pairs.append({"candidate": a, "control": b, "n": len(differences), "differences": differences,
                      "mean_oa_delta_pp": statistics.mean(x["oa_delta_pp"] for x in differences),
                      "positive_seeds": sum(x["oa_delta_pp"] > 0 for x in differences)})
    interactions = [{"seed": s, "oa_interaction_pp":
        (table["G04", s]["oa"] - table["G01", s]["oa"]) -
        (table["G05", s]["oa"] - table["G00", s]["oa"])}
        for s in (6, 7, 8) if all((g, s) in table for g in ("G00", "G01", "G04", "G05"))]
    write(DEST / "summary.json", {"total": 18, "completed": len(rows), "reused": 10,
        "new_completed": sum(not x["reused"] for x in rows), "new_total": 8,
        "seeds": [6, 7, 8], "runs": rows, "aggregates": aggregates,
        "paired_comparisons": pairs, "A_B1_interactions": interactions,
        "evidence_scope": "Original test-OA model selection; three seeds only; not independent testing."})
    with (DEST / "逐组结果.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# B1补充seed7、8：原流程实验结果", "",
        f"已完成{len(rows)}/18项；复用10项，新训{sum(not x['reused'] for x in rows)}/8项，每项100轮。", "",
        "A为局部/全局固定1:1通道分工+SR2；R为纯SSM+原融合+平均池化。原测试OA参与选模，不构成独立测试证据。", "",
        "| 组别 | 方法 | seed | OA % | AA % | Kappa % | 来源 |",
        "|---|---|---:|---:|---:|---:|---|"]
    for group in plan["groups"]:
        for s in (6, 7, 8):
            v = table.get((group["id"], s))
            if v:
                lines.append(f"| {group['id']} | {group['label']} | {s} | {v['oa']:.4f} | {v['aa']:.4f} | {v['kappa']:.4f} | {'复用' if v['reused'] else '新训'} |")
            else:
                lines.append(f"| {group['id']} | {group['label']} | {s} | 待完成 | — | — | 新训 |")
    lines += ["", "## 三种子汇总", "", "仅在同组6/7/8三个种子全部完成后计算均值±样本标准差（ddof=1）。", "",
              "| 组别 | OA % | AA % | Kappa % |", "|---|---:|---:|---:|"]
    for a in aggregates:
        cells = [f"{a[k]['mean']:.4f} ± {a[k]['sample_std']:.4f}" for k in ("oa", "aa", "kappa")] if a["complete"] else [f"待完成（{a['n']}/3）", "—", "—"]
        lines.append(f"| {a['group']} | " + " | ".join(cells) + " |")
    lines += ["", "## 同种子配对比较", "", "G04−G01是主比较；所有差值单位为OA百分点。", "",
              "| 比较 | 已具备种子差值 | 平均差值 | 正增益种子数 |", "|---|---|---:|---:|"]
    for p in pairs:
        diff = "; ".join(f"seed{x['seed']}: {x['oa_delta_pp']:+.4f}" for x in p["differences"])
        lines.append(f"| {p['candidate']}−{p['control']} | {diff} | {p['mean_oa_delta_pp']:+.4f} | {p['positive_seeds']}/{p['n']} |")
    lines += ["", "## A整体与B1的交互", "", "逐种子计算 `(G04−G01)−(G05−G00)`："]
    lines += [f"- seed{x['seed']}: {x['oa_interaction_pp']:+.4f}个百分点。" for x in interactions]
    lines += ["", "全部逐组成绩均读取原evaluator/metrics报告（12197个样本）。不挑选最好种子，不从三个种子直接宣称统计显著或独立测试收益。", "",
              "运行状态见运行状态.json；新训完成证据、检查点/报告哈希见completed；复用与最小源码迁移见reuse_audit.json；源文件与配置锁定见manifest.json。", ""]
    (DEST / "实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def run():
    import msvcrt
    from scripts.keep_awake_fusion import kernel
    manifest = read(DEST / "manifest.json")
    if read(DEST / "运行状态.json")["state"] != "prepared":
        raise RuntimeError("Only a fresh prepared batch may start; no implicit restart/resume")
    lock = (DEST / "queue.lock").open("xb+")
    lock.write(b"0")
    lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    state = {"state": "starting", "total": 18, "reused": 10, "completed": 10,
             "new_completed": 0, "new_total": 8, "queue_pid": os.getpid(),
             "training_pid": None, "current": None, "started_at": now()}
    child = None
    (DEST / "completed").mkdir(exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    try:
        if not kernel.SetThreadExecutionState(0x80000001):
            raise RuntimeError("Temporary sleep protection request failed")
        core.state_write(state)
        for job in manifest["jobs"]:
            if job["reused"]:
                continue
            core.verify_pins(manifest)
            check = subprocess.run(["powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' -and $_.CommandLine -match '(scripts[/\\\\]train\\.py|train_all\\.py|run_batch\\.py)' } | Select-Object -ExpandProperty ProcessId"],
                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            if check.returncode or check.stdout.strip():
                raise RuntimeError(f"Another training may be active: {check.stdout.strip()}")
            for key in ("stdout", "stderr", "checkpoint", "report", "native_log"):
                if Path(job[key]).exists():
                    raise FileExistsError(f"Refusing overwrite: {job[key]}")
            state.update(state="launching", current=job, training_pid=None, current_epoch=None)
            core.state_write(state)
            started, clock = now(), time.monotonic()
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
                        progress.seek(max(0, Path(job["stderr"]).stat().st_size - 12000))
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
            write(DEST / "completed" / f"{job['id']}.json", {"id": job["id"], "seed": job["seed"],
                "started_at": started, "finished_at": now(), "elapsed_seconds": time.monotonic() - clock,
                "epochs": 100, "checkpoint_sha256": sha(job["checkpoint"]), "report_sha256": sha(job["report"]),
                "metrics": metrics, "best_checkpoint_fusion_diagnostics": diagnostic})
            summarize()
            state.update(completed=state["completed"] + 1, new_completed=state["new_completed"] + 1, training_pid=None)
            core.state_write(state)
            print(f"Completed {job['id']}: OA={metrics['oa']:.4f}", flush=True)
        core.verify_pins(manifest)
        state.update(state="completed", current=None, training_pid=None, finished_at=now())
        core.state_write(state)
        print("All 18 results complete: 10 reused + 8 new.", flush=True)
    except BaseException as exc:
        state.update(state="failed", error=repr(exc), training_pid=child.pid if child and child.poll() is None else None)
        core.state_write(state)
        raise
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()


def guard():
    from scripts import keep_awake_fusion as g
    g.BATCH, g.STATUS, g.EVIDENCE = DEST, DEST / "运行状态.json", DEST / "power_protection.json"
    info = {"helper_pid": os.getpid(), "scope": "temporary idle sleep/hibernate prevention during B1 seed7/8 batch",
            "power_plan_changed": False, "display_required": False, "started_at": now()}
    try:
        while True:
            state = read(g.STATUS)
            if not (g.is_running(state.get("queue_pid")) or g.is_running(state.get("training_pid"))):
                break
            if not g.kernel.SetThreadExecutionState(0x80000001):
                raise RuntimeError("Native sleep protection request failed")
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
    parser.add_argument("action", choices=("prepare", "run", "summarize", "guard"))
    args = parser.parse_args()
    os.chdir(ROOT)
    {"prepare": prepare, "run": run, "summarize": summarize, "guard": guard}[args.action]()
