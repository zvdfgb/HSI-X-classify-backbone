"""Authorized 36 native jobs, waiting for the existing queue before ANY GPU work."""
import argparse
import ctypes
import gc
import json
import math
import os
import platform
import re
import runpy
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_sr_native import read, write, sha, report_metrics

TAG = "fusion_transfer_v3_seed678_20261104"
DEST = ROOT/"report/fusion_papers"/TAG
LOG = ROOT/"log/fusion_papers"/TAG
ARCHIVE = ROOT/"docs/experiments/provenance"/TAG
PLAN = ROOT/"docs/experiments/20261104_三篇新论文融合迁移_已批准.json"
REVIEW = ROOT/"docs/experiments/20261104_三篇新论文融合迁移_实验审核稿.md"
PREDECESSOR = ROOT/"report/sr_lg_operator_controls_20261104"
NATIVE = ROOT/"report/sr_native/sr_native_20261029"
GROUPS = tuple(f"{family}{i:02d}" for family in "CSI" for i in range(1, 5))
PROTECTED = ["configs/train.json", "scripts/train.py", "src/runner.py", "src/trainer.py", "src/data.py",
    "src/evaluator.py", "src/metrics.py", "src/config.py", "src/models/transmixer_net/model.py",
    "src/models/transmixer_paper/model.py", "src/models/transmixer_sr/adapter.py"]
ARTIFACTS = ("checkpoint", "report", "native_log", "stdout", "stderr", "diagnostics")
COMPARISONS = [("C04", "A0"), ("S04", "A0"), ("I04", "A0"),
    ("C02", "C01"), ("C03", "C01"), ("C04", "C02"), ("C04", "C03"),
    ("S02", "S01"), ("S03", "S02"), ("S04", "S02"), ("S04", "S03"),
    ("I01", "A0"), ("I02", "I01"), ("I03", "I01"), ("I04", "I01"), ("I04", "I02"), ("I04", "I03")]


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name+".tmp")
    write(temp, value); os.replace(temp, path)


def state(**updates):
    path = DEST/"运行状态.json"
    s = read(path) if path.exists() else {}
    s.update(updates, updated_at=now())
    atomic(path, s)
    return s


def inventory():
    command = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } | "
               "Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command],
        capture_output=True, encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError("Cannot verify training process inventory: "+result.stderr)
    values = json.loads(result.stdout) if result.stdout.strip() else []
    return values if isinstance(values, list) else [values]


def assert_idle():
    active = [p for p in inventory() if p["ProcessId"] != os.getpid() and re.search(
        r"scripts[/\\]train\.py|train_all\.py|run_batch\.py|run_lg_operator_controls\.py|launch_lg_controls_after_analysis\.py", p.get("CommandLine") or "")]
    if active:
        raise RuntimeError("Other training/controller is still active: "+str(active))


def pins(manifest):
    for category in ("source_and_config_sha256", "data_sha256"):
        for rel, digest in manifest[category].items():
            if sha(ROOT/rel) != digest:
                raise RuntimeError("Locked file changed: "+rel)
    for j in manifest["baselines"]:
        for key in ("checkpoint", "report", "config"):
            if sha(j[key]) != j[key+"_sha256"]:
                raise RuntimeError("Reused A0 changed: "+j["id"]+"/"+key)


def reuse_audit():
    old, seal = read(NATIVE/"manifest.json"), read(NATIVE/"final_integrity_audit.json")
    assert seal["mismatches"] == []
    for rel in PROTECTED:
        assert sha(ROOT/rel) == old["source_and_config_sha256"][rel], rel
    for rel, digest in old["data_sha256"].items():
        assert sha(ROOT/rel) == digest, rel
    rows = []
    for seed in (6, 7, 8):
        source = next(j for j in old["jobs"] if j["id"] == f"M3_seed{seed}")
        evidence = next(j for j in seal["outputs"] if j["id"] == source["id"])
        cfg_path = ROOT/"configs/runs/sr_native_20261029"/f"M3_seed{seed}.json"
        assert sha(cfg_path) == old["source_and_config_sha256"][cfg_path.relative_to(ROOT).as_posix()]
        cfg = read(cfg_path)
        assert cfg["model_name"] == "TransMixer_SR2" and cfg["random_seed"] == seed
        assert cfg["sr"] == {"spatial_sigma": 3., "spectral_scale": .2775005102157593}
        for name, value in dict(epochs=100, batch_size=128, num_workers=0, learning_rate=.0001,
                                channels=30, window_size=11, dataset_type=0, ablation_mode="decoupled").items():
            assert cfg[name] == value, name
        assert not cfg.get("distillation") and not cfg.get("paper_transfer")
        for key in ("checkpoint", "report"):
            assert sha(source[key]) == evidence[key+"_sha256"], (seed, key)
        rows.append({**source, "id": f"A0_seed{seed}", "group": "A0", "seed": seed,
            "reused": True, "reuse_source": "sr_native_20261029/"+source["id"],
            "config": str(cfg_path), "config_sha256": sha(cfg_path),
            **{key+"_sha256": evidence[key+"_sha256"] for key in ("checkpoint", "report")},
            "metrics": report_metrics(source["report"])})
    return old, rows


def setup():
    """CPU/file operations only. Does not import CUDA or register new models."""
    from src.config import ExperimentConfig
    if (DEST/"manifest.json").exists():
        raise FileExistsError("Batch exists; refusing overwrite")
    cpu = read(ARCHIVE/"cpu_preflight.json")
    assert cpu["all_passed"] and len(cpu["checks"]) == 36
    old, baselines = reuse_audit()
    assert read(PREDECESSOR/"state.json")["state"] in ("training", "completed", "starting", "launching")
    DEST.mkdir(parents=True, exist_ok=True); LOG.mkdir(parents=True, exist_ok=True)
    registry = ROOT/"src/model_registry.py"
    with (ARCHIVE/"model_registry.before.py").open("xb") as stream:
        stream.write(registry.read_bytes())
    jobs = []
    for seed in (6, 7, 8):
        original = read(ROOT/"configs/runs/sr_native_20261029"/f"M3_seed{seed}.json")
        for group in GROUPS:
            cfg = {**original, "experiment_name": f"{TAG}_{group}_seed{seed}",
                   "model_name": "TransMixer_FV3_"+group, "paper_transfer": {"group": group}}
            path = ROOT/"configs/runs"/TAG/f"{group}_seed{seed}.json"
            if path.exists():
                raise FileExistsError(path)
            write(path, cfg)
            exp = ExperimentConfig(**cfg)
            job = {"id": f"{group}_seed{seed}", "group": group, "seed": seed, "reused": False,
                "config": str(path), "checkpoint": exp.model_save_path, "report": exp.report_path,
                "native_log": exp.log_path, "diagnostics": str(Path(exp.report_path).with_suffix(".diagnostics.json")),
                "stdout": str(LOG/f"{group}_seed{seed}.stdout.log"), "stderr": str(LOG/f"{group}_seed{seed}.stderr.log")}
            assert not any(Path(job[k]).exists() for k in ARTIFACTS)
            jobs.append(job)
    plan = {"training_authorized": True, "approved_at_recorded": now(),
        "user_instruction": "按照上述设计开始实验，严格遵循之前的实验标准并复用代码；等待另一个训练结束后再gpu上训练",
        "review_document": str(REVIEW.relative_to(ROOT)), "review_sha256": sha(REVIEW),
        "total_groups": 13, "total_results": 39, "reused": 3, "new_training_runs": 36,
        "seeds": [6, 7, 8], "epochs": 100, "groups": ["A0", *GROUPS],
        "new_jobs": [{"id": j["id"], "group": j["group"], "seed": j["seed"]} for j in jobs],
        "main_comparisons": ["C04-A0", "S04-A0", "I04-A0"],
        "training_entry": "scripts/train.py -> src/runner.py -> src/trainer.py",
        "wait_for": str(PREDECESSOR.relative_to(ROOT)), "no_gpu_until_predecessor_exits": True,
        "registry_transition_after_predecessor_audit_only": True, "retrain_on_failure": False}
    write(PLAN, plan)
    files = list((ROOT/"src").rglob("*.py"))+[Path(__file__), ROOT/"scripts/prepare_sr_native.py",
        ROOT/"scripts/keep_awake_fusion.py", ROOT/"scripts/train.py", ROOT/"configs/train.json", ROOT/"AGENTS.md",
        PLAN, REVIEW, ARCHIVE/"cpu_preflight.json"]+[Path(j["config"]) for j in jobs]
    manifest = {"tag": TAG, "created_at": now(), "total": 39, "new_total": 36, "reused": 3,
        "python": old["python"], "python_version": old["python_version"], "torch": old["torch"],
        "numpy": old["numpy"], "gpu": old["gpu"], "jobs": jobs, "baselines": baselines,
        "source_and_config_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        "data_sha256": old["data_sha256"], "registry_transition_pending": True,
        "checkpoint_selection": "unchanged native scheduled test OA maximum; not independent test",
        "predecessor_manifest_sha256": sha(PREDECESSOR/"manifest.json"),
        "protocol": dict(dataset="Houston2013 official", channels=30, window=11, epochs=100,
                         batch_size=128, workers=0, learning_rate=.0001, weight_decay=.01, seeds=[6, 7, 8])}
    write(DEST/"manifest.json", manifest); write(ARCHIVE/"reuse_audit.json", {"all_passed": True, "baselines": baselines})
    write(ARCHIVE/"authorization.json", plan); pins(manifest)
    state(state="prepared_waiting", completed=3, total=39, reused=3, new_completed=0, new_total=36,
          current=None, training_pid=None, queue_pid=None, gpu_started=False)
    summarize()


def predecessor_audit():
    from scripts.keep_awake_fusion import is_running
    status, m = read(PREDECESSOR/"state.json"), read(PREDECESSOR/"manifest.json")
    assert status["state"] == "completed" and status["new_completed"] == 6
    assert read(PREDECESSOR/"final_integrity_audit.json")["all_passed"]
    assert not any(is_running(status.get(key)) for key in ("queue_pid", "training_pid", "helper_pid"))
    for category in ("source_and_config_sha256", "data_sha256"):
        for rel, digest in m[category].items():
            assert sha(ROOT/rel) == digest, rel
    for job in m["jobs"]:
        done = read(PREDECESSOR/"completed"/f"{job['id']}.json")
        assert done["epochs"] == 100
        for key in ARTIFACTS[:5]:
            assert sha(job[key]) == done[key+"_sha256"], (job["id"], key)
    assert_idle()
    write(ARCHIVE/"predecessor_final_audit.json", {"all_passed": True, "checked_at": now(),
        "source_config_data_and_six_outputs_verified": True, "no_active_predecessor_or_training": True,
        "final_audit_sha256": sha(PREDECESSOR/"final_integrity_audit.json")})


def migrate():
    assert_idle()
    manifest = read(DEST/"manifest.json"); pins(manifest)
    registry = ROOT/"src/model_registry.py"
    before = registry.read_bytes()
    assert before == (ARCHIVE/"model_registry.before.py").read_bytes()
    previous = runpy.run_path(str(registry))
    anchor = b"MODEL_REGISTRY = {\n"
    if before.count(anchor) != 2 or b"TransMixer_FV3_" in before:
        raise RuntimeError("Unexpected registry bytes; no migration applied")
    addition = (b'    **{f"TransMixer_FV3_{family}{i:02d}": "src.models.transmixer_fusion_v3.adapter"\n'
                b'       for family in "CSI" for i in range(1, 5)},\n')
    registry.write_bytes(before.replace(anchor, anchor+addition, 1))
    after = runpy.run_path(str(registry))
    assert all(after[k] == previous[k] for k in ("LEGACY_MODEL_REGISTRY", "MODEL_ALIASES"))
    assert after["MODEL_REGISTRY"] == {**previous["MODEL_REGISTRY"],
        **{"TransMixer_FV3_"+g: "src.models.transmixer_fusion_v3.adapter" for g in GROUPS}}
    write(ARCHIVE/"registry_transition.json", {"authorized": True, "applied_at": now(),
        "before_sha256": sha(ARCHIVE/"model_registry.before.py"), "after_sha256": sha(registry),
        "only_added_names": ["TransMixer_FV3_"+g for g in GROUPS], "historical_mappings_unchanged": True,
        "predecessor_complete_and_exited_before_change": True})
    manifest["source_and_config_sha256"]["src/model_registry.py"] = sha(registry)
    manifest["registry_transition_pending"] = False
    manifest["source_and_config_sha256"][(ARCHIVE/"registry_transition.json").relative_to(ROOT).as_posix()] = sha(ARCHIVE/"registry_transition.json")
    write(DEST/"manifest.json", manifest); pins(manifest)


def gpu_preflight():
    assert_idle()
    import numpy as np
    import torch
    from src.data import getMyData, set_random_seed
    from src.models.transmixer_fusion_v3.preflight import run_checks, inference_costs
    m = read(DEST/"manifest.json")
    assert Path(sys.executable).resolve() == Path(m["python"]).resolve()
    assert platform.python_version() == m["python_version"] and str(torch.__version__) == m["torch"]
    assert np.__version__ == m["numpy"] and torch.cuda.get_device_name(0) == m["gpu"]
    set_random_seed(6); loaders = getMyData(0, 30, 11, 128, 0)
    assert len(loaders[0].dataset) == 2832 and len(loaders[1].dataset) == 12197
    assert len(loaders[0]) == 22 and loaders[0].drop_last
    h, raw, l, y = next(iter(loaders[0]))
    batch = {"hsi_pca": h.to("cuda:0"), "aux": l.to("cuda:0"), "label": y.to("cuda:0")}
    del loaders, h, raw, l, y; gc.collect()
    result = run_checks("cuda:0", batch)
    costs = inference_costs(batch)
    write(DEST/"cost_audit.json", costs)
    write(DEST/"gpu_preflight.json", result)
    m["source_and_config_sha256"][(DEST/"gpu_preflight.json").relative_to(ROOT).as_posix()] = sha(DEST/"gpu_preflight.json")
    m["source_and_config_sha256"][(DEST/"cost_audit.json").relative_to(ROOT).as_posix()] = sha(DEST/"cost_audit.json")
    write(DEST/"manifest.json", m); pins(m)
    del batch; gc.collect(); torch.cuda.empty_cache()


def summarize():
    m = read(DEST/"manifest.json")
    rows = [{"id": j["id"], "group": "A0", "seed": j["seed"], "reused": True, **j["metrics"]} for j in m["baselines"]]
    for job in m["jobs"]:
        path = DEST/"completed"/f"{job['id']}.json"
        if path.exists():
            done = read(path)
            rows.append({"id": job["id"], "group": job["group"], "seed": job["seed"], "reused": False,
                         "elapsed_seconds": done["elapsed_seconds"], **done["metrics"]})
    stats, comparisons = {}, []
    for group in ("A0", *GROUPS):
        values = [r for r in rows if r["group"] == group]
        if len(values) == 3:
            stats[group] = {key: {"mean": statistics.mean(r[key] for r in values),
                                 "std": statistics.stdev(r[key] for r in values)} for key in ("oa", "aa", "kappa")}
    for a, b in COMPARISONS:
        deltas = []
        for seed in (6, 7, 8):
            ra = next((r for r in rows if r["group"] == a and r["seed"] == seed), None)
            rb = next((r for r in rows if r["group"] == b and r["seed"] == seed), None)
            if ra and rb:
                deltas.append({"seed": seed, **{key: ra[key]-rb[key] for key in ("oa", "aa", "kappa")}})
        comparisons.append({"comparison": a+"-"+b, "paired_deltas_pp": deltas,
            "complete": len(deltas) == 3, "statistics": {key: {"mean_delta_pp": statistics.mean(d[key] for d in deltas),
            "positive_count": sum(d[key] > 0 for d in deltas), "paired_seeds": len(deltas)} for key in ("oa", "aa", "kappa")} if deltas else {}})
    interaction = []
    for seed in (6, 7, 8):
        items = {r["group"]: r for r in rows if r["seed"] == seed and r["group"].startswith("C")}
        if len(items) == 4:
            interaction.append({"seed": seed, **{key: items["C04"][key]-items["C02"][key]-items["C03"][key]+items["C01"][key]
                for key in ("oa", "aa", "kappa")}})
    write(DEST/"summary.json", {"updated_at": now(), "completed": len(rows), "total": 39, "reused": 3,
        "new_completed": len(rows)-3, "new_total": 36, "runs": rows, "statistics": stats,
        "comparisons": comparisons, "cf_interaction_per_seed_pp": interaction,
        "cf_interaction_mean_pp": {key: statistics.mean(r[key] for r in interaction) for key in ("oa", "aa", "kappa")} if interaction else {}})
    lines = ["# 三篇新论文融合子机制迁移实验报告", "", f"更新时间：{now()}。完成{len(rows)}/39，复用3项，新训{len(rows)-3}/36。", "",
        "固定主比较：C04-A0、S04-A0、I04-A0。每项100轮，seed6/7/8；沿用原测试OA选模，不是独立测试。", "",
        "CF-IPT 为显式通道归一化修正后的融合子机制；ISFuse只迁移CFI与干预一致性；IID只迁移DC，不代表完整论文复现。", "",
        "| 组别 | seed | 来源 | OA% | AA% | Kappa% |", "|---|---|---|---:|---:|---:|"]
    lines += [f"| {r['group']} | {r['seed']} | {'复用' if r['reused'] else '新训'} | {r['oa']:.4f} | {r['aa']:.4f} | {r['kappa']:.4f} |" for r in rows]
    lines += ["", "## 三种子统计（样本标准差ddof=1）", "", "|组别|OA%|AA%|Kappa%|", "|---|---:|---:|---:|"]
    lines += ["| "+g+" | "+" | ".join(f"{v[k]['mean']:.4f} ± {v[k]['std']:.4f}" for k in ("oa", "aa", "kappa"))+" |" for g,v in stats.items()]
    lines += ["", "## 同种子预定比较", "", "|比较|已配对种子数|平均OA差/百分点|OA正增益次数|", "|---|---:|---:|---:|"]
    lines += [f"| {c['comparison']} | {len(c['paired_deltas_pp'])} | {c['statistics']['oa']['mean_delta_pp']:.4f} | {c['statistics']['oa']['positive_count']} |" for c in comparisons if c["statistics"]]
    lines += ["", "逐种子差值、AA/Kappa差值及CF交互项见summary.json。部分结构组参数不同；单数据集、三种子不足以直接声称显著性、稳定泛化或整篇论文无效。", ""]
    if (DEST/"cost_audit.json").exists():
        lines += ["## 资源与推理开销", "", "MACs为明确算子范围内的估计，非完整FLOPs；推理计时为原生batch128、3次预热后5次平均。峰值显存见gpu_preflight.json，包含参考模型。", "",
            "|组别|有效参数|每patch估计MACs|batch128延迟/ms|", "|---|---:|---:|---:|"]
        lines += [f"|{r['group']}|{r['parameters']}|{r['estimated_clean_macs_per_patch']}|{r['batch_latency_ms']:.3f}|"
                  for r in read(DEST/"cost_audit.json")["runs"]]
    (DEST/"实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def audit_job(job, done):
    assert done["epochs"] == 100 and done["training_steps"] == 2200
    assert "Report Success!" in Path(job["stdout"]).read_text(encoding="utf-8", errors="replace")
    assert re.search(r"Epoch 100/100:\s+100%[^\r\n]*22/22", Path(job["stderr"]).read_text(encoding="utf-8", errors="replace"))
    for key in ARTIFACTS:
        assert sha(job[key]) == done[key+"_sha256"], (job["id"], key)
    assert report_metrics(job["report"]) == done["metrics"]


def final_audit():
    m = read(DEST/"manifest.json"); pins(m)
    checks = []
    for job in m["jobs"]:
        done = read(DEST/"completed"/f"{job['id']}.json"); audit_job(job, done)
        checks.append({"id": job["id"], "verified": True, "epochs": 100, "steps": 2200,
                       **{key+"_sha256": done[key+"_sha256"] for key in ARTIFACTS}})
    assert len(checks) == 36 and not m["registry_transition_pending"]
    summarize()
    write(DEST/"final_integrity_audit.json", {"all_passed": True, "checked_at": now(),
        "new_completed": 36, "reused_verified": 3, "total": 39, "outputs": checks,
        "source_config_data_locked": True, "native_100_epochs_22_batches_report_success_verified": True,
        "registry_transition_sha256": sha(ARCHIVE/"registry_transition.json")})


def run_queue(helper):
    from scripts.keep_awake_fusion import kernel
    m = read(DEST/"manifest.json"); pins(m)
    assert read(DEST/"gpu_preflight.json")["all_passed"]
    (DEST/"completed").mkdir()
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    child = None
    try:
        for index, job in enumerate(m["jobs"]):
            assert_idle(); pins(m)
            assert helper.poll() is None
            assert not any(Path(job[k]).exists() for k in ARTIFACTS)
            started, clock = now(), time.monotonic()
            state(state="launching", current=job["id"], group=job["group"], seed=job["seed"], current_epoch=0,
                  current_started_at=started, new_completed=index, completed=3+index)
            with Path(job["stdout"]).open("xb") as out, Path(job["stderr"]).open("xb") as err:
                child = subprocess.Popen([m["python"], "-X", "utf8", "-u", "scripts/train.py", "--config", job["config"]],
                    cwd=ROOT, env=env, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
                state(state="training", training_pid=child.pid)
                while child.poll() is None:
                    time.sleep(10)
                    assert kernel.SetThreadExecutionState(0x80000001)
                    assert helper.poll() is None, "Power guard stopped; preserve active scientific child"
                    path = Path(job["stderr"])
                    with path.open("rb") as stream:
                        stream.seek(max(0, path.stat().st_size-16000)); tail = stream.read().decode("utf-8", errors="replace")
                    epochs = re.findall(r"Epoch (\d+)/100", tail)
                    state(current_epoch=int(epochs[-1]) if epochs else 0,
                          elapsed_seconds=round(time.monotonic()-clock, 1), log_updated_at=datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat())
            if child.returncode:
                raise RuntimeError(f"Native training failed {job['id']} exit={child.returncode}; inspect {job['stderr']}")
            diagnostics = read(job["diagnostics"])
            done = {"id": job["id"], "seed": job["seed"], "group": job["group"], "epochs": 100,
                "training_steps": diagnostics["training_steps"], "started_at": started, "finished_at": now(),
                "elapsed_seconds": time.monotonic()-clock, "metrics": report_metrics(job["report"]),
                **{key+"_sha256": sha(job[key]) for key in ARTIFACTS}}
            audit_job(job, done); pins(m)
            write(DEST/"completed"/f"{job['id']}.json", done); summarize()
            state(state="between_jobs", training_pid=None, completed=index+4, new_completed=index+1)
            print("COMPLETED", job["id"], done["metrics"], flush=True)
        final_audit()
        state(state="completed", current=None, group=None, seed=None, training_pid=None, finished_at=now())
    except BaseException as exc:
        state(state="failed", error=repr(exc), failed_at=now(),
              training_pid=child.pid if child and child.poll() is None else None)
        raise
    finally:
        # Never kill or restart a surviving native child when the orchestration fails.
        while child and child.poll() is None:
            state(state="failed_waiting_for_active_child", training_pid=child.pid)
            kernel.SetThreadExecutionState(0x80000001); time.sleep(15)


def guard():
    from scripts import keep_awake_fusion as g
    info = {"helper_pid": os.getpid(), "power_plan_changed": False, "display_required": False, "started_at": now()}
    try:
        while True:
            s = read(DEST/"运行状态.json")
            if not any(g.is_running(s.get(k)) for k in ("queue_pid", "training_pid")):
                break
            if not g.kernel.SetThreadExecutionState(0x80000001):
                raise RuntimeError("Temporary sleep protection request failed")
            supply = g.PowerStatus(); valid = g.kernel.GetSystemPowerStatus(ctypes.byref(supply))
            info.update(state="active", native_request_succeeded=True, checked_at=now(),
                        queue_pid=s.get("queue_pid"), training_pid=s.get("training_pid"),
                        ac_connected=supply.ac == 1 if valid else None,
                        battery_percent=int(supply.battery_percent) if valid else None)
            atomic(DEST/"power_protection.json", info); time.sleep(20)
    except BaseException as exc:
        info.update(state="failed", error=repr(exc)); raise
    finally:
        g.kernel.SetThreadExecutionState(0x80000000)
        if info.get("state") != "failed":
            info["state"] = "released"
        info["updated_at"] = now(); atomic(DEST/"power_protection.json", info)


def follow():
    from scripts.keep_awake_fusion import kernel, is_running
    import msvcrt
    lock = (DEST/"controller.lock").open("a+b")
    lock.seek(0); lock.write(b"0"); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    helper = None
    try:
        assert read(PLAN)["training_authorized"]
        assert read(DEST/"运行状态.json")["state"] == "prepared_waiting"
        state(state="waiting_for_predecessor", queue_pid=os.getpid(), training_pid=None,
              started_at=now(), predecessor=str(PREDECESSOR), gpu_started=False)
        with (LOG/"power.stdout.log").open("xb") as out, (LOG/"power.stderr.log").open("xb") as err:
            helper = subprocess.Popen([sys.executable, "-X", "utf8", "-u", str(Path(__file__)), "guard"],
                cwd=ROOT, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
        state(helper_pid=helper.pid)
        for _ in range(30):
            if (DEST/"power_protection.json").exists():
                break
            if helper.poll() is not None:
                raise RuntimeError("Power guard failed")
            time.sleep(1)
        assert read(DEST/"power_protection.json")["state"] == "active"
        while True:
            assert helper.poll() is None
            assert kernel.SetThreadExecutionState(0x80000001)
            previous = read(PREDECESSOR/"state.json")
            active = any(is_running(previous.get(k)) for k in ("queue_pid", "training_pid", "helper_pid"))
            state(predecessor_state=previous["state"], predecessor_current=previous.get("current"),
                  predecessor_epoch=previous.get("current_epoch"), predecessor_new_completed=previous.get("new_completed"))
            if previous["state"] == "completed" and not active:
                # Parent handoff process may be closing; never race it or any new training.
                try:
                    assert_idle()
                except RuntimeError:
                    time.sleep(15); continue
                break
            if previous["state"] == "failed" and not active:
                raise RuntimeError("Predecessor failed; this batch was NOT launched: "+str(previous.get("error")))
            if previous["state"] != "completed" and not active:
                raise RuntimeError("Predecessor stopped before completion; no automatic retraining")
            time.sleep(20)
        predecessor_audit()
        state(state="migrating_registry")
        migrate()
        state(state="gpu_preflight", gpu_started=True)
        gpu_preflight()
        run_queue(helper)
    except BaseException as exc:
        state(state="failed", error=repr(exc), failed_at=now()); raise
    finally:
        s = read(DEST/"运行状态.json")
        while is_running(s.get("training_pid")):
            kernel.SetThreadExecutionState(0x80000001); time.sleep(15)
            s = read(DEST/"运行状态.json")
        kernel.SetThreadExecutionState(0x80000000)
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == "__main__":
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("setup", "follow", "guard", "summarize", "audit"))
    action = parser.parse_args().action
    {"setup": setup, "follow": follow, "guard": guard, "summarize": summarize, "audit": final_audit}[action]()
