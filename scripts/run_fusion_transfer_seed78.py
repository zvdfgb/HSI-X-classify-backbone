"""Approved follow-on seed7/8 batch, using ONLY the native scientific entry.

Waits for the seed6 batch, audits it read-only, applies the explicitly approved
seed guard/document migration, then queues 24 jobs. No data/metric/train loop.
"""
import argparse
import csv
import gc
import importlib.machinery
import importlib.util
import io
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_fusion_transfer as original
from scripts.prepare_sr_native import read, write, sha, report_metrics
from scripts.keep_awake_fusion import kernel, is_running, PowerStatus, ctypes

TAG = "fusion_transfer_seed78_20261102"
DEST = ROOT / "report/fusion_papers" / TAG
LOGS = ROOT / "log/fusion_papers" / TAG
OLD = ROOT / "report/fusion_papers/fusion_transfer_seed6_20261102"
NATIVE = ROOT / "report/sr_native/sr_native_20261029"
ARCHIVE = ROOT / "docs/experiments/provenance" / TAG
PLAN = ROOT / "docs/experiments/20261102_三篇融合机制_补充种子7_8_已批准.json"
GROUPS = tuple(f"{f}{i:02d}" for f in "FUD" for i in range(1, 5))
MIGRATED = ("src/models/transmixer_fusion_v2/adapter.py", "AGENTS.md")
now = original.now


def state_write(state):
    value = {**state, "updated_at": now()}
    path = DEST / "运行状态.json"
    write(path.with_suffix(".tmp"), value)
    os.replace(path.with_suffix(".tmp"), path)


def power_write(phase):
    if not kernel.SetThreadExecutionState(0x80000001):
        raise RuntimeError("Temporary execution-state protection failed")
    supply = PowerStatus()
    valid = kernel.GetSystemPowerStatus(ctypes.byref(supply))
    write(DEST / "power_protection.json", {"state": "active", "phase": phase,
        "helper_pid": os.getpid(), "queue_pid": os.getpid(), "native_request_succeeded": True,
        "power_plan_changed": False, "display_required": False,
        "ac_connected": supply.ac == 1 if valid else None,
        "battery_percent": int(supply.battery_percent) if valid else None, "checked_at": now()})


def audit_job(job, seal):
    for key in ("checkpoint", "report"):
        if sha(job[key]) != seal[f"{key}_sha256"]:
            raise RuntimeError(f"Sealed output changed: {job['id']}/{key}")
    if not job["reused"]:
        assert seal["epochs"] == 100
        for key in ("stdout", "stderr", "native_log"):
            assert sha(job[key]) == seal[f"{key}_sha256"], (job["id"], key)
        assert "Report Success!" in Path(job["stdout"]).read_text(encoding="utf-8", errors="replace")
        assert re.search(r"Epoch 100/100:\s+100%[^\r\n]*22/22", Path(job["stderr"]).read_text(encoding="utf-8", errors="replace"))
    return report_metrics(job["report"])


def migrate():
    """Run ONLY after source queue exits successfully; preserve old evidence."""
    old = read(OLD / "manifest.json")
    original.core.verify_pins(old)
    source_audit = read(OLD / "final_integrity_audit.json")
    assert source_audit["all_passed"] and source_audit["new_training_runs"] == 12
    assert read(OLD / "summary.json")["completed"] == 13
    verified = []
    for job in old["jobs"]:
        seal = job if job["reused"] else read(OLD / "completed" / f"{job['id']}.json")
        verified.append({"id": job["id"], "metrics": audit_job(job, seal)})
    ARCHIVE.mkdir(parents=True, exist_ok=False)
    archived = {}
    for rel in MIGRATED:
        p = ROOT / rel
        target = ARCHIVE / ("adapter_seed6_before.py.txt" if rel.endswith("adapter.py") else "AGENTS_before.md.txt")
        target.write_bytes(p.read_bytes())
        archived[rel] = {"before_sha256": sha(p), "before_archive": str(target)}
    adapter = ROOT / MIGRATED[0]
    before = adapter.read_text(encoding="utf-8")
    assert before.count("fixed = dict(random_seed=6, channels=30") == 1
    assert before.count("if any(getattr(exp, k) != v for k, v in fixed.items()):") == 1
    after = before.replace("fixed = dict(random_seed=6, channels=30", "fixed = dict(channels=30").replace(
        "if any(getattr(exp, k) != v for k, v in fixed.items()):",
        "if exp.random_seed not in (6, 7, 8) or any(getattr(exp, k) != v for k, v in fixed.items()):")
    adapter.write_text(after, encoding="utf-8")
    agents = ROOT / "AGENTS.md"
    rules = agents.read_text(encoding="utf-8")
    old_rule = "weight_decay=0.01、seed6。多种子实验仅按审核后的矩阵替换种子与模型。"
    new_rule = "weight_decay=0.01；2026-11-02用户更新：后续所有实验至少使用seed6、7、8，不再默认单seed6。组别与其他设置仍按审核矩阵执行，不自动新增模型、组合或更多种子。"
    assert rules.count(old_rule) == 1
    agents.write_text(rules.replace(old_rule, new_rule) +
        "\n- 2026-11-02用户已批准：三篇融合机制seed6批次完成核验后，为F01-F04、U01-U04、D01-D04全部补seed7、8，共24项新训；A三个种子核验复用。保留原计算、损失、训练/数据/评估与测试OA选模规则。\n", encoding="utf-8")
    for rel in MIGRATED:
        archived[rel]["after_sha256"] = sha(ROOT / rel)
    write(ARCHIVE / "source_migration.json", {"authorized_at": read(PLAN)["approved_at"],
        "applied_at": now(), "source_batch": old["run_tag"],
        "source_final_audit_sha256": sha(OLD / "final_integrity_audit.json"),
        "source_completed_evidence": verified, "files": archived,
        "computation_changed": False, "reason": "User-authorized seeds7/8 guard and future minimum three-seed policy"})


def preflight(jobs):
    import torch
    from src import config
    from src.config import ExperimentConfig, ConfigManager
    from src.data import set_random_seed
    from src.models.transmixer_sr import adapter as baseline_adapter
    from src.models.transmixer_fusion_v2 import adapter
    loader = importlib.machinery.SourceFileLoader("src.models.transmixer_fusion_v2._before_seed78",
        str(ARCHIVE / "adapter_seed6_before.py.txt"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    before = importlib.util.module_from_spec(spec)
    loader.exec_module(before)
    report = {"scope": "synthetic checks only; no real data or optimizer", "seed6_regression": [], "jobs": []}
    for gid in GROUPS:
        cfg = read(ROOT / "configs/runs/fusion_transfer_seed6_20261102" / f"{gid}_seed6.json")
        config._global_config_manager = ConfigManager(ExperimentConfig(**cfg))
        set_random_seed(6); a = before.build_model(config, 0, "cpu")["net"]
        set_random_seed(6); b = adapter.build_model(config, 0, "cpu")["net"]
        assert a.state_dict().keys() == b.state_dict().keys()
        for k, v in a.state_dict().items():
            torch.testing.assert_close(v, b.state_dict()[k], rtol=0, atol=0)
        report["seed6_regression"].append({"group": gid, "parameters_bit_exact": True})
        del a, b
    for job in jobs:
        if job["reused"]:
            continue
        cfg = read(job["config"])
        config._global_config_manager = ConfigManager(ExperimentConfig(**{**cfg, "model_name": "TransMixer_SR2"}))
        set_random_seed(job["seed"])
        base = baseline_adapter.build_model(config, 0, "cuda:0")["net"].eval()
        initial = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
        small = {"hsi_pca": torch.randn(2, 1, 30, 11, 11, device="cuda:0"),
                 "aux": torch.randn(2, 1, 11, 11, device="cuda:0")}
        with torch.no_grad():
            base_out = baseline_adapter.forward_eval({"net": base}, small)
        del base
        config._global_config_manager = ConfigManager(ExperimentConfig(**cfg))
        set_random_seed(job["seed"])
        bundle = adapter.build_model(config, 0, "cuda:0")
        net, stage = bundle["net"], bundle["net"].fusion_conv
        for k, v in initial.items():
            nk = k.replace("fusion_conv.", "fusion_conv.original.") if k.startswith("fusion_conv.") else k
            torch.testing.assert_close(v, net.state_dict()[nk].cpu(), rtol=0, atol=0)
        net.eval()
        with torch.no_grad():
            torch.testing.assert_close(adapter.forward_eval(bundle, small), base_out, rtol=0, atol=0)
        for name in ("alpha_h", "alpha_l", "alpha_detail"):
            if hasattr(stage, name):
                assert getattr(stage, name).item() == 0
                getattr(stage, name).data.fill_(0.1)
        batch = {"hsi_pca": torch.randn(128, 1, 30, 11, 11, device="cuda:0"),
                 "aux": torch.randn(128, 1, 11, 11, device="cuda:0"),
                 "label": torch.arange(128, device="cuda:0") % 15}
        net.train()
        out = adapter.forward_train(bundle, batch)
        loss = adapter.compute_loss(bundle, out, batch, torch.nn.CrossEntropyLoss())
        assert out.shape == (128, 15) and torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in stage.interaction.parameters())
        if job["group"].startswith("D"):
            assert stage.interaction.cache is None
            if job["group"] in ("D03", "D04"):
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in stage.interaction.unique_critic.parameters())
        net.eval()
        with torch.no_grad():
            a = adapter.forward_eval(bundle, small)
            b = adapter.forward_eval(bundle, small)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        if job["group"] == "F04":
            checkpoint = io.BytesIO(); torch.save(net, checkpoint); checkpoint.seek(0)
            restored = torch.load(checkpoint, map_location="cuda:0", weights_only=False).eval()
            with torch.no_grad():
                torch.testing.assert_close(adapter.forward_eval({"net": restored}, small), a, rtol=0, atol=0)
            del restored, checkpoint
        report["jobs"].append({"id": job["id"], "passed": True, "batch128": True,
            "original_initialization_and_disabled_logits_exact": True,
            "parameters": sum(p.numel() for p in net.parameters())})
        write(DEST / "preflight.json", report)
        print(f"PREFLIGHT_OK {job['id']}", flush=True)
        del bundle, net, stage, small, batch, initial, out, loss, a, b, base_out
        gc.collect(); torch.cuda.empty_cache()
        power_write("preflight")
    invalid = {**cfg, "random_seed": 9}
    config._global_config_manager = ConfigManager(ExperimentConfig(**invalid))
    try:
        adapter.build_model(config, 0, "cpu")
    except ValueError:
        report["seed9_rejected"] = True
    else:
        raise AssertionError("Unapproved seed9 accepted")
    write(DEST / "preflight.json", report)


def prepare():
    import torch
    import numpy as np
    from src.config import ExperimentConfig
    plan, old = read(PLAN), read(OLD / "manifest.json")
    assert plan["training_authorized"] and len(plan["new_jobs"]) == 24
    assert [(j["group"], j["seed"]) for j in plan["new_jobs"]] == [(g, s) for s in (7, 8) for g in GROUPS]
    assert Path(sys.executable).resolve() == Path(old["python"]).resolve()
    assert platform.python_version() == old["python_version"] and str(torch.__version__) == old["torch"]
    assert np.__version__ == old["numpy"] and torch.cuda.get_device_name(0) == old["gpu"]
    migration = read(ARCHIVE / "source_migration.json")
    for category in ("source_and_config_sha256", "data_sha256"):
        for rel, digest in old[category].items():
            if rel in MIGRATED:
                assert digest == migration["files"][rel]["before_sha256"]
                assert sha(ROOT / rel) == migration["files"][rel]["after_sha256"]
            else:
                assert sha(ROOT / rel) == digest, rel
    jobs = []
    for j in old["jobs"]:
        seal = j if j["reused"] else read(OLD / "completed" / f"{j['id']}.json")
        audit_job(j, seal)
        jobs.append({**j, "reused": True, "reuse_source": f"{old['run_tag']}/{j['id']}",
                     "checkpoint_sha256": seal["checkpoint_sha256"], "report_sha256": seal["report_sha256"]})
    native = read(NATIVE / "manifest.json"); sealed = read(NATIVE / "final_integrity_audit.json")
    defaults = read(ROOT / "configs/train.json")
    stats = read(NATIVE / "training_input_statistics.json")
    for seed in (7, 8):
        j = next(j for j in native["jobs"] if j["id"] == f"M3_seed{seed}")
        seal = next(jj for jj in sealed["outputs"] if jj["id"] == j["id"])
        cfg = read(j["config"])
        assert sha(j["config"]) == native["source_and_config_sha256"][Path(j["config"]).relative_to(ROOT).as_posix()]
        for k, v in defaults.items():
            if k not in ("experiment_name", "model_name", "random_seed"):
                assert cfg[k] == v, k
        assert cfg["random_seed"] == seed and cfg["ablation_mode"] == "decoupled"
        assert cfg["sr"] == {"spatial_sigma": 3.0, "spectral_scale": stats[str(seed)]["spectral_scale"]}
        for key in ("checkpoint", "report"):
            assert sha(j[key]) == seal[f"{key}_sha256"]
        jobs.append({**j, "id": f"A00_seed{seed}", "group": "A00", "seed": seed, "reused": True,
            "reuse_source": j["id"], "checkpoint_sha256": seal["checkpoint_sha256"], "report_sha256": seal["report_sha256"]})
    for requested in plan["new_jobs"]:
        gid, seed = requested["group"], requested["seed"]
        cfg = read(ROOT / "configs/runs/fusion_transfer_seed6_20261102" / f"{gid}_seed6.json")
        cfg.update(random_seed=seed, experiment_name=f"{TAG}_{gid}_seed{seed}")
        assert cfg["sr"]["spectral_scale"] == stats[str(seed)]["spectral_scale"]
        exp = ExperimentConfig(**cfg)
        path = ROOT / "configs/runs" / TAG / f"{gid}_seed{seed}.json"
        for p in (path, Path(exp.model_save_path), Path(exp.report_path), Path(exp.log_path)):
            if p.exists():
                raise FileExistsError(p)
        write(path, cfg)
        jobs.append({"id": f"{gid}_seed{seed}", "group": gid, "seed": seed, "reused": False,
            "model": exp.model_name, "config": str(path), "checkpoint": exp.model_save_path,
            "report": exp.report_path, "native_log": exp.log_path,
            "stdout": str(LOGS / f"{gid}_seed{seed}.stdout.log"), "stderr": str(LOGS / f"{gid}_seed{seed}.stderr.log")})
    write(DEST / "reuse_audit.json", {"verified_at": now(), "reused": 15,
        "source_seed6_results": 13, "original_baseline_seed78_results": 2,
        "results": [{"id": j["id"], "metrics": report_metrics(j["report"])} for j in jobs if j["reused"]]})
    preflight(jobs)
    sources = list((ROOT / "src").rglob("*.py"))
    sources += [ROOT / rel for rel in original.core.PROTECTED]
    sources += [ROOT / rel for rel in ("src/model_registry.py", "scripts/run_fusion_transfer.py", "scripts/run_fusion_papers.py",
        "scripts/prepare_sr_native.py", "scripts/keep_awake_fusion.py", "scripts/run_fusion_transfer_seed78.py", "AGENTS.md")]
    sources += [PLAN, ARCHIVE / "source_migration.json", ARCHIVE / "adapter_seed6_before.py.txt", ARCHIVE / "AGENTS_before.md.txt",
        NATIVE / "training_input_statistics.json", DEST / "preflight.json", DEST / "reuse_audit.json", DEST / "follow_lock.json"]
    sources += [Path(j["config"]) for j in jobs]
    manifest = {"run_tag": TAG, "created_at": now(), "python": sys.executable,
        "python_version": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0), "jobs": jobs, "total": 39, "reused": 15, "new_training": 24,
        "source_and_config_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        "data_sha256": old["data_sha256"], "training_entry": "scripts/train.py",
        "checkpoint_selection": "native_test_OA_best", "source_batch": old["run_tag"]}
    write(DEST / "manifest.json", manifest)
    original.core.verify_pins(manifest)
    state_write({"state": "prepared", "total": 39, "reused": 15, "completed": 15, "new_total": 24,
        "new_completed": 0, "queue_pid": os.getpid(), "training_pid": None, "current": None})
    summarize()


def summarize():
    manifest, plan = read(DEST / "manifest.json"), read(PLAN)
    rows = []
    for j in manifest["jobs"]:
        if j["reused"] or (DEST / "completed" / f"{j['id']}.json").exists():
            rows.append({k: j[k] for k in ("id", "group", "seed", "reused", "report")} | report_metrics(j["report"]))
    table = {(r["group"], r["seed"]): r for r in rows}
    aggregates = []
    for gid in ("A00", *GROUPS):
        members = [table[gid, s] for s in (6, 7, 8) if (gid, s) in table]
        item = {"group": gid, "n": len(members), "complete": len(members) == 3}
        if item["complete"]:
            for metric in ("oa", "aa", "kappa"):
                values = [r[metric] for r in members]
                item[metric] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}
        aggregates.append(item)
    pairs = []
    for pair in plan["comparisons"]["primary"] + plan["comparisons"]["mechanism"]:
        a, b = pair.split("-")
        differences = [{"seed": s, **{k: table[a, s][k] - table[b, s][k] for k in ("oa", "aa", "kappa")}}
            for s in (6, 7, 8) if (a, s) in table and (b, s) in table]
        pairs.append({"comparison": pair, "n": len(differences), "per_seed_delta_pp": differences,
            "mean_delta_pp": {k: statistics.mean(d[k] for d in differences) for k in ("oa", "aa", "kappa")},
            "positive_counts": {k: sum(d[k] > 0 for d in differences) for k in ("oa", "aa", "kappa")}})
    summary = {"total": 39, "completed": len(rows), "reused": 15, "new_completed": sum(not r["reused"] for r in rows),
        "new_total": 24, "seeds": [6, 7, 8], "runs": rows, "aggregates": aggregates, "paired_comparisons": pairs,
        "overall_accounting": "3 A baselines + 36 fusion runs (12 source seed6 runs + 24 supplemental seed7/8 runs)",
        "limitations": "Three seeds; original test OA selects checkpoints; not independent testing or statistical significance."}
    write(DEST / "summary.json", summary)
    with (DEST / "逐种子结果.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = ["# 三篇融合机制：seed6/7/8汇总", "", f"已完成{len(rows)}/39；本补充批次复用15项，补训{summary['new_completed']}/24项。", "",
        "整体为A基线3项、融合方案36项（seed6新训12项和本批次补训24项）。每项100轮；原测试OA选模保留，三种子不直接证明显著性或独立测试收益。", "",
        "| 组别 | seed | OA % | AA % | Kappa % | 来源 |", "|---|---:|---:|---:|---:|---|"]
    for gid in ("A00", *GROUPS):
        for s in (6, 7, 8):
            r = table.get((gid, s))
            v = f"{r['oa']:.4f} | {r['aa']:.4f} | {r['kappa']:.4f}" if r else "待完成 | — | —"
            lines.append(f"| {gid} | {s} | {v} | {'复用' if r and r['reused'] else '新训'} |")
    lines += ["", "## 均值±样本标准差", "", "仅三种子完整后计算；ddof=1。", "", "| 组别 | OA % | AA % | Kappa % |", "|---|---:|---:|---:|"]
    for a in aggregates:
        cells = [f"{a[k]['mean']:.4f} ± {a[k]['sample_std']:.4f}" for k in ("oa", "aa", "kappa")] if a["complete"] else [f"待完成({a['n']}/3)", "—", "—"]
        lines.append(f"| {a['group']} | " + " | ".join(cells) + " |")
    lines += ["", "## 同种子配对比较", "", "全部差值单位为百分点；不挑选最好种子。", "", "| 比较 | 各seed OA差值 | 平均OA差值 | OA正增益次数 |", "|---|---|---:|---:|"]
    for p in pairs:
        values = "; ".join(f"seed{d['seed']}: {d['oa']:+.4f}" for d in p["per_seed_delta_pp"])
        lines.append(f"| {p['comparison']} | {values} | {p['mean_delta_pp']['oa']:+.4f} | {p['positive_counts']['oa']}/{p['n']} |")
    lines += ["", "AA/Kappa逐种子差值及平均差值见summary.json。源码种子校验与未来三种子规范的授权迁移见source_migration.json；原seed6产物只读保留。", ""]
    (DEST / "实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def final_audit():
    m = read(DEST / "manifest.json"); original.core.verify_pins(m)
    results = []
    for j in m["jobs"]:
        seal = j if j["reused"] else read(DEST / "completed" / f"{j['id']}.json")
        results.append({"id": j["id"], "metrics": audit_job(j, seal), "checkpoint_sha256": sha(j["checkpoint"]), "report_sha256": sha(j["report"])})
    write(DEST / "final_integrity_audit.json", {"all_passed": True, "verified_at": now(),
        "reused": 15, "new_training_runs": 24, "results": results, "source_config_data_pins_verified": True})


def run():
    import msvcrt
    manifest = read(DEST / "manifest.json")
    assert read(DEST / "运行状态.json")["state"] == "prepared"
    lock = (DEST / "queue.lock").open("xb+"); lock.write(b"0"); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    state = {"state": "starting", "total": 39, "reused": 15, "completed": 15, "new_completed": 0,
        "new_total": 24, "queue_pid": os.getpid(), "training_pid": None, "current": None, "started_at": now()}
    child = None
    (DEST / "completed").mkdir(exist_ok=True)
    try:
        for j in manifest["jobs"]:
            if j["reused"]: continue
            original.core.verify_pins(manifest); original.check_no_training()
            for k in ("stdout", "stderr", "checkpoint", "report", "native_log"):
                if Path(j[k]).exists(): raise FileExistsError(j[k])
            power_write("training")
            start, clock = now(), time.monotonic()
            state.update(state="launching", current=j, training_pid=None, current_epoch=None)
            state_write(state)
            env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
            env.pop("CUBLAS_WORKSPACE_CONFIG", None)
            with Path(j["stdout"]).open("xb") as out, Path(j["stderr"]).open("xb") as err:
                child = subprocess.Popen([manifest["python"], "-X", "utf8", "-u", "scripts/train.py", "--config", j["config"]],
                    cwd=ROOT, env=env, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
                state.update(state="training", training_pid=child.pid, current_started_at=start); state_write(state)
                print(f"Started {j['id']} pid={child.pid}", flush=True)
                while child.poll() is None:
                    time.sleep(10); power_write("training")
                    with Path(j["stderr"]).open("rb") as f:
                        f.seek(max(0, Path(j["stderr"]).stat().st_size - 16000))
                        epochs = re.findall(r"Epoch (\d+)/100", f.read().decode("utf-8", errors="replace"))
                    state.update(current_epoch=int(epochs[-1]) if epochs else None,
                        current_elapsed_seconds=round(time.monotonic() - clock, 1)); state_write(state)
            if child.returncode != 0: raise RuntimeError(f"Native training failed: {j['id']} exit={child.returncode}")
            metrics = report_metrics(j["report"])
            seal = {"id": j["id"], "seed": j["seed"], "epochs": 100, "started_at": start, "finished_at": now(),
                "elapsed_seconds": time.monotonic() - clock, "metrics": metrics,
                **{f"{k}_sha256": sha(j[k]) for k in ("checkpoint", "report", "stdout", "stderr", "native_log")}}
            audit_job(j, seal); original.core.verify_pins(manifest)
            import torch
            net = torch.load(j["checkpoint"], map_location="cpu", weights_only=False)
            seal["best_checkpoint_fusion_diagnostics"] = net.fusion_conv.diagnostics(); del net
            write(DEST / "completed" / f"{j['id']}.json", seal)
            summarize(); state.update(completed=state["completed"] + 1, new_completed=state["new_completed"] + 1, training_pid=None)
            state_write(state); print(f"Completed {j['id']}: OA={metrics['oa']:.4f}", flush=True)
        final_audit(); state.update(state="completed", current=None, training_pid=None, finished_at=now()); state_write(state)
        print("All 39 results complete: 15 reused + 24 supplemental.", flush=True)
    except BaseException as exc:
        state.update(state="failed", error=repr(exc), training_pid=child.pid if child and child.poll() is None else None)
        state_write(state); raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


def follow():
    if DEST.exists(): raise FileExistsError(f"Refusing duplicate follow-on batch: {DEST}")
    DEST.mkdir(parents=True); LOGS.mkdir(parents=True, exist_ok=True)
    plan = read(PLAN); assert plan["training_authorized"]
    assert len(plan["new_jobs"]) == 24
    original.core.verify_pins(read(OLD / "manifest.json"))
    write(DEST / "follow_lock.json", {"created_at": now(), "plan_sha256": sha(PLAN), "script_sha256": sha(__file__),
        "source_manifest_sha256": sha(OLD / "manifest.json"), "authorization": plan["authorization"]})
    expected = read(DEST / "follow_lock.json")
    try:
        while True:
            if sha(__file__) != expected["script_sha256"] or sha(PLAN) != expected["plan_sha256"]:
                raise RuntimeError("Pending approved plan/controller changed")
            s = read(OLD / "运行状态.json"); power_write("waiting_for_seed6")
            state_write({"state": "waiting_for_seed6", "queue_pid": os.getpid(), "training_pid": None,
                "total": 39, "new_total": 24, "new_completed": 0, "source_completed": s["completed"],
                "source_total": 13, "source_queue_pid": s.get("queue_pid"), "source_training_pid": s.get("training_pid"),
                "source_current": s.get("current"), "source_epoch": s.get("current_epoch"), "current": None})
            if s["state"] == "completed":
                if is_running(s.get("queue_pid")) or is_running(s.get("training_pid")):
                    time.sleep(5); continue
                break
            if s["state"] == "failed": raise RuntimeError(f"Source seed6 batch failed: {s.get('error')}")
            if not is_running(s.get("queue_pid")):
                raise RuntimeError("Source queue exited before verified completion; no silent resume")
            time.sleep(15)
        original.check_no_training()
        state_write({"state": "auditing_and_preparing", "queue_pid": os.getpid(), "training_pid": None,
            "current": None, "total": 39, "new_total": 24, "new_completed": 0})
        migrate(); prepare(); run()
    except BaseException as exc:
        current = read(DEST / "运行状态.json") if (DEST / "运行状态.json").exists() else {}
        if current.get("state") != "failed":
            state_write({**current, "state": "failed", "error": repr(exc), "queue_pid": os.getpid()})
        raise
    finally:
        # If the queue dies while its child still exists, keep protection until
        # the actual child exits; never silently kill/restart scientific work.
        s = read(DEST / "运行状态.json") if (DEST / "运行状态.json").exists() else {}
        while is_running(s.get("training_pid")):
            power_write("failed_queue_waiting_for_active_training_child"); time.sleep(15)
        kernel.SetThreadExecutionState(0x80000000)
        p = read(DEST / "power_protection.json") if (DEST / "power_protection.json").exists() else {}
        write(DEST / "power_protection.json", {**p, "state": "released", "updated_at": now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("follow", "summarize", "audit"))
    args = parser.parse_args(); os.chdir(ROOT)
    {"follow": follow, "summarize": summarize, "audit": final_audit}[args.action]()
