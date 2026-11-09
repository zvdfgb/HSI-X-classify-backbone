"""Orchestration controller for the approved 18-job SR-guided A0 & C02 fusion batch (R00-R05, seeds 6/7/8).

Strictly reuses scripts/train.py -> src/runner.py -> src/trainer.py, src/data.py,
src/evaluator.py, and src/metrics.py without changing training protocol or checkpoint rule.
"""
import argparse
import csv
import ctypes
import gc
import json
import os
import platform
import re
import runpy
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_sr_native import read as read_json, write, sha, report_metrics

TAG = "fusion_sr_a0_c02_seed678_20261108"
DEST = ROOT / "report/fusion_papers" / TAG
LOG = ROOT / "log/fusion_papers" / TAG
ARCHIVE = ROOT / "docs/experiments/provenance" / TAG
PLAN = ROOT / "docs/experiments/20261108_A0与C02接入SR引导融合验证_已批准.json"
REVIEW = ROOT / "docs/experiments/20261108_A0与C02接入SR引导融合验证_实验审核稿.md"
NATIVE = ROOT / "report/sr_native/sr_native_20261029"
FV3 = ROOT / "report/fusion_papers/fusion_transfer_v3_seed678_20261104"
SRNCF = ROOT / "report/fusion_papers/fusion_sr_ncf_seed678_20261107"

GROUPS = tuple(f"R{i:02d}" for i in range(6))
ALL_GROUPS = ("A0", "K01", "K02", "K03", "C02", *GROUPS)
REUSED_COUNT = 15
NEW_COUNT = 18
TOTAL_COUNT = REUSED_COUNT + NEW_COUNT

PROTECTED = [
    "configs/train.json",
    "scripts/train.py",
    "src/runner.py",
    "src/trainer.py",
    "src/data.py",
    "src/evaluator.py",
    "src/metrics.py",
    "src/config.py",
    "src/models/transmixer_net/model.py",
    "src/models/transmixer_paper/model.py",
    "src/models/transmixer_sr/adapter.py",
]
ARTIFACTS = ("checkpoint", "report", "native_log", "stdout", "stderr", "diagnostics")
COMPARISONS = [
    ("R01", "R00"),
    ("R01", "R02"),
    ("R01", "A0"),
    ("R00", "A0"),
    ("R01", "K02"),
    ("R04", "R03"),
    ("R04", "R05"),
    ("R04", "C02"),
    ("R03", "C02"),
    ("R04", "A0"),
]
HOUSTON_CLASSES = (
    "Healthy grass",
    "Stressed grass",
    "Synthetic grass",
    "Trees",
    "Soil",
    "Water",
    "Residential",
    "Commercial",
    "Road",
    "Highway",
    "Railway",
    "Parking lot 1",
    "parking lot 2",
    "Tennis court",
    "Running track",
)
FOCUS_CLASSES = ("Highway", "Road", "Commercial", "Stressed grass")


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path, retry_seconds=8.0):
    deadline = time.monotonic() + retry_seconds
    while True:
        try:
            return read_json(path)
        except OSError as exc:
            if not (isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in (5, 32, 33)):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(0.05, remaining))


def atomic(path, value, retry_seconds=8.0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        deadline, delay = time.monotonic() + retry_seconds, 0.02
        while True:
            try:
                os.replace(temp, path)
                return
            except OSError as exc:
                if not (isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in (5, 32, 33)):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                time.sleep(min(delay, remaining))
                delay = min(delay * 2, 0.2)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def state(**updates):
    path = DEST / "运行状态.json"
    s = read(path) if path.exists() else {}
    s.update(updates, updated_at=now())
    atomic(path, s)
    return s


def inventory():
    command = (
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } | "
        "Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode:
        raise RuntimeError("Cannot verify training process inventory: " + result.stderr)
    values = json.loads(result.stdout) if result.stdout.strip() else []
    return values if isinstance(values, list) else [values]


def assert_idle():
    active = []
    for p in inventory():
        if p["ProcessId"] == os.getpid():
            continue
        cmd = p.get("CommandLine") or ""
        if re.search(
            r"scripts[/\\]train\.py|train_all\.py|run_batch\.py|run_lg_operator_controls\.py|run_fusion_transfer_v3\.py|run_fusion_hamf\.py|run_bmaf_fusion\.py|run_sr_ncf_fusion\.py",
            cmd,
        ):
            active.append(p)
        elif re.search(r"run_sr_a0_c02_fusion\.py", cmd) and not re.search(r"\bguard\b", cmd):
            active.append(p)
    if active:
        raise RuntimeError("Other training/controller is still active: " + str(active))


def pins(manifest):
    for category in ("source_and_config_sha256", "data_sha256"):
        for rel, digest in manifest[category].items():
            if sha(ROOT / rel) != digest:
                raise RuntimeError("Locked file changed: " + rel)
    for j in manifest["baselines"]:
        for key in ("checkpoint", "report", "config"):
            if sha(j[key]) != j[key + "_sha256"]:
                raise RuntimeError("Reused baseline changed: " + j["id"] + "/" + key)


def reuse_audit():
    old, seal = read(NATIVE / "manifest.json"), read(NATIVE / "final_integrity_audit.json")
    assert seal["mismatches"] == []
    for rel in PROTECTED:
        assert sha(ROOT / rel) == old["source_and_config_sha256"][rel], rel
    for rel, digest in old["data_sha256"].items():
        assert sha(ROOT / rel) == digest, rel

    fv3_m, fv3_seal = read(FV3 / "manifest.json"), read(FV3 / "final_integrity_audit.json")
    assert fv3_seal["all_passed"] and fv3_seal["new_completed"] == 36

    srncf_m, srncf_seal = read(SRNCF / "manifest.json"), read(SRNCF / "final_integrity_audit.json")
    assert srncf_seal["all_passed"] and srncf_seal["new_completed"] == 21

    rows = []
    # 1. A0 (= M3 in sr_native_20261029)
    for seed in (6, 7, 8):
        source = next(j for j in old["jobs"] if j["id"] == f"M3_seed{seed}")
        evidence = next(j for j in seal["outputs"] if j["id"] == source["id"])
        cfg_path = ROOT / "configs/runs/sr_native_20261029" / f"M3_seed{seed}.json"
        assert sha(cfg_path) == old["source_and_config_sha256"][cfg_path.relative_to(ROOT).as_posix()]
        cfg = read(cfg_path)
        assert cfg["model_name"] == "TransMixer_SR2" and cfg["random_seed"] == seed
        assert cfg["sr"] == {"spatial_sigma": 3.0, "spectral_scale": 0.2775005102157593}
        for key in ("checkpoint", "report"):
            assert sha(source[key]) == evidence[key + "_sha256"], ("A0", seed, key)
        rows.append({
            "id": f"A0_seed{seed}",
            "group": "A0",
            "seed": seed,
            "reused": True,
            "reuse_source": "sr_native_20261029/" + source["id"],
            "config": str(cfg_path),
            "config_sha256": sha(cfg_path),
            "checkpoint": source["checkpoint"],
            "checkpoint_sha256": evidence["checkpoint_sha256"],
            "report": source["report"],
            "report_sha256": evidence["report_sha256"],
            "metrics": report_metrics(source["report"]),
        })

    # 2. K01, K02, K03 (from fusion_sr_ncf_seed678_20261107)
    for g in ("K01", "K02", "K03"):
        for seed in (6, 7, 8):
            job = next(j for j in srncf_m["jobs"] if j["id"] == f"{g}_seed{seed}")
            done = read(SRNCF / "completed" / f"{g}_seed{seed}.json")
            cfg_path = Path(job["config"])
            assert sha(cfg_path) == srncf_m["source_and_config_sha256"][cfg_path.relative_to(ROOT).as_posix()]
            for key in ("checkpoint", "report"):
                assert sha(job[key]) == done[key + "_sha256"], (g, seed, key)
            assert report_metrics(job["report"]) == done["metrics"]
            rows.append({
                "id": f"{g}_seed{seed}",
                "group": g,
                "seed": seed,
                "reused": True,
                "reuse_source": f"{SRNCF.name}/{g}_seed{seed}",
                "config": str(cfg_path),
                "config_sha256": sha(cfg_path),
                "checkpoint": job["checkpoint"],
                "checkpoint_sha256": done["checkpoint_sha256"],
                "report": job["report"],
                "report_sha256": done["report_sha256"],
                "metrics": done["metrics"],
            })

    # 3. C02 (from fusion_transfer_v3_seed678_20261104)
    for seed in (6, 7, 8):
        job = next(j for j in fv3_m["jobs"] if j["id"] == f"C02_seed{seed}")
        done = read(FV3 / "completed" / f"C02_seed{seed}.json")
        cfg_path = Path(job["config"])
        assert sha(cfg_path) == fv3_m["source_and_config_sha256"][cfg_path.relative_to(ROOT).as_posix()]
        for key in ("checkpoint", "report"):
            assert sha(job[key]) == done[key + "_sha256"], ("C02", seed, key)
        assert report_metrics(job["report"]) == done["metrics"]
        rows.append({
            "id": f"C02_seed{seed}",
            "group": "C02",
            "seed": seed,
            "reused": True,
            "reuse_source": f"{FV3.name}/C02_seed{seed}",
            "config": str(cfg_path),
            "config_sha256": sha(cfg_path),
            "checkpoint": job["checkpoint"],
            "checkpoint_sha256": done["checkpoint_sha256"],
            "report": job["report"],
            "report_sha256": done["report_sha256"],
            "metrics": done["metrics"],
        })

    assert len(rows) == REUSED_COUNT
    return old, rows


def parse_per_class_report(report_path):
    text = Path(report_path).read_bytes().decode("utf-8", errors="replace")
    parsed = []
    for cls in HOUSTON_CLASSES:
        m = re.search(
            rf"(?m)^\s*{re.escape(cls)}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$",
            text,
        )
        if not m:
            raise ValueError(f"Missing class '{cls}' in native report: {report_path}")
        parsed.append({
            "class": cls,
            "native_printed_precision": f"{float(m.group(1)):.4f}",
            "native_printed_recall": f"{float(m.group(2)):.4f}",
            "native_printed_f1": f"{float(m.group(3)):.4f}",
            "support": int(m.group(4)),
        })
    return parsed


def export_per_class_csv(manifest):
    rows = []
    for base in manifest["baselines"]:
        for item in parse_per_class_report(base["report"]):
            rows.append({
                "id": base["id"],
                "group": base["group"],
                "seed": base["seed"],
                **item,
                "source_report": base["report"],
            })
    for job in manifest["jobs"]:
        marker = DEST / "completed" / f"{job['id']}.json"
        if marker.exists():
            for item in parse_per_class_report(job["report"]):
                rows.append({
                    "id": job["id"],
                    "group": job["group"],
                    "seed": job["seed"],
                    **item,
                    "source_report": job["report"],
                })
    csv_path = DEST / "native_per_class_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "group",
                "seed",
                "class",
                "native_printed_precision",
                "native_printed_recall",
                "native_printed_f1",
                "support",
                "source_report",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def apply_authorized_transitions():
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    agents_path = ROOT / "AGENTS.md"
    agents_backup = ARCHIVE / "AGENTS_before.md.txt"
    if not agents_backup.exists():
        with agents_backup.open("xb") as stream:
            stream.write(agents_path.read_bytes())
    agents_bytes = agents_path.read_bytes()
    note = (
        "\r\n- 2026-11-08用户已批准：A0与C02接入原生SR引导邻域融合对照实验R00-R05共6组×seed6/7/8=18项新训，"
        "复用A0、K01-K03与C02共15项参照；保留原训练/数据/评估链路、纯CE损失与测试OA选模规则。\r\n"
    ).encode("utf-8")
    if b"2026-11-08" not in agents_bytes:
        assert agents_bytes == agents_backup.read_bytes()
        updated = agents_bytes.rstrip(b"\r\n") + b"\r\n" + note
        assert b"$" not in updated
        agents_path.write_bytes(updated)
    write(ARCHIVE / "agents_transition.json", {
        "authorized": True,
        "applied_at": now(),
        "before_sha256": sha(agents_backup),
        "after_sha256": sha(agents_path),
    })

    registry = ROOT / "src/model_registry.py"
    reg_backup = ARCHIVE / "model_registry.before.py"
    if not reg_backup.exists():
        with reg_backup.open("xb") as stream:
            stream.write(registry.read_bytes())
    reg_bytes = registry.read_bytes()
    if b"TransMixer_SRA0C02_" not in reg_bytes:
        assert reg_bytes == reg_backup.read_bytes()
        previous = runpy.run_path(str(registry))
        anchor = b"MODEL_REGISTRY = {\n"
        newline = b"\n"
        if anchor not in reg_bytes:
            anchor = b"MODEL_REGISTRY = {\r\n"
            newline = b"\r\n"
        if reg_bytes.count(anchor) != 2:
            raise RuntimeError("Unexpected registry bytes; refusing automatic edit")
        addition = (
            b'    **{f"TransMixer_SRA0C02_R{i:02d}": "src.models.transmixer_sr_a0_c02.adapter"' + newline
            + b"       for i in range(6)}," + newline
        )
        registry.write_bytes(reg_bytes.replace(anchor, anchor + addition, 1))
        after = runpy.run_path(str(registry))
        assert all(after[k] == previous[k] for k in ("LEGACY_MODEL_REGISTRY", "MODEL_ALIASES"))
        assert after["MODEL_REGISTRY"] == {
            **previous["MODEL_REGISTRY"],
            **{"TransMixer_SRA0C02_" + g: "src.models.transmixer_sr_a0_c02.adapter" for g in GROUPS},
        }
    write(ARCHIVE / "registry_transition.json", {
        "authorized": True,
        "applied_at": now(),
        "before_sha256": sha(reg_backup),
        "after_sha256": sha(registry),
        "only_added_names": ["TransMixer_SRA0C02_" + g for g in GROUPS],
        "historical_mappings_unchanged": True,
    })


def setup():
    from src.config import ExperimentConfig

    assert_idle()
    if (DEST / "manifest.json").exists():
        raise FileExistsError("Batch exists; refusing overwrite")
    apply_authorized_transitions()

    from src.models.transmixer_sr_a0_c02.preflight import run_checks as run_cpu_checks
    from src.models.transmixer_sr_ncf.model import fixed_derangement
    cpu = run_cpu_checks("cpu")
    write(ARCHIVE / "cpu_preflight.json", cpu)
    assert cpu["all_passed"] and len(cpu["checks"]) == NEW_COUNT

    old, baselines = reuse_audit()

    DEST.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)

    jobs = []
    for seed in (6, 7, 8):
        original = read(ROOT / "configs/runs/sr_native_20261029" / f"M3_seed{seed}.json")
        for group in GROUPS:
            cfg = {
                **original,
                "experiment_name": f"{TAG}_{group}_seed{seed}",
                "model_name": f"TransMixer_SRA0C02_{group}",
                "paper_transfer": {"family": "SRA0C02", "group": group},
            }
            path = ROOT / "configs/runs" / TAG / f"{group}_seed{seed}.json"
            if path.exists():
                raise FileExistsError(path)
            write(path, cfg)
            exp = ExperimentConfig(**cfg)
            job = {
                "id": f"{group}_seed{seed}",
                "group": group,
                "seed": seed,
                "reused": False,
                "config": str(path),
                "checkpoint": exp.model_save_path,
                "report": exp.report_path,
                "native_log": exp.log_path,
                "diagnostics": str(Path(exp.report_path).with_suffix(".diagnostics.json")),
                "stdout": str(LOG / f"{group}_seed{seed}.stdout.log"),
                "stderr": str(LOG / f"{group}_seed{seed}.stderr.log"),
            }
            assert not any(Path(job[k]).exists() for k in ARTIFACTS), job["id"]
            jobs.append(job)

    param_map = {c["group"]: c["parameters"] for c in cpu["checks"] if c["seed"] == 6}
    plan = {
        "training_authorized": True,
        "approved_at_recorded": now(),
        "user_instruction": "开始",
        "review_document": REVIEW.relative_to(ROOT).as_posix(),
        "review_sha256": sha(REVIEW),
        "total_groups": len(ALL_GROUPS),
        "total_results": TOTAL_COUNT,
        "reused": REUSED_COUNT,
        "new_training_runs": NEW_COUNT,
        "seeds": [6, 7, 8],
        "epochs": 100,
        "groups": list(ALL_GROUPS),
        "verified_parameters": {
            "A0": 80887,
            "K01": 108111,
            "K02": 108111,
            "K03": 108111,
            "C02": 680311,
            **param_map,
        },
        "fusion_derangement_121": fixed_derangement(121).tolist(),
        "new_jobs": [{"id": j["id"], "group": j["group"], "seed": j["seed"]} for j in jobs],
        "main_comparisons": ["R01-R00", "R01-R02", "R01-A0", "R04-R03", "R04-R05", "R04-C02"],
        "predefined_comparisons": [f"{a}-{b}" for a, b in COMPARISONS],
        "training_entry": "scripts/train.py -> src/runner.py -> src/trainer.py",
        "checkpoint_selection": "unchanged native scheduled test OA maximum; not independent test",
        "retrain_on_failure": False,
    }
    write(PLAN, plan)

    files = (
        [p for p in (ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts]
        + [
            Path(__file__),
            ROOT / "scripts/prepare_sr_native.py",
            ROOT / "scripts/keep_awake_fusion.py",
            ROOT / "scripts/train.py",
            ROOT / "configs/train.json",
            ROOT / "AGENTS.md",
            PLAN,
            REVIEW,
            ARCHIVE / "cpu_preflight.json",
            ARCHIVE / "agents_transition.json",
            ARCHIVE / "registry_transition.json",
        ]
        + [Path(j["config"]) for j in jobs]
    )
    manifest = {
        "tag": TAG,
        "created_at": now(),
        "total": TOTAL_COUNT,
        "new_total": NEW_COUNT,
        "reused": REUSED_COUNT,
        "python": old["python"],
        "python_version": old["python_version"],
        "torch": old["torch"],
        "numpy": old["numpy"],
        "gpu": old["gpu"],
        "jobs": jobs,
        "baselines": baselines,
        "source_and_config_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        "data_sha256": old["data_sha256"],
        "registry_transition_pending": False,
        "checkpoint_selection": "unchanged native scheduled test OA maximum; not independent test",
        "fv3_manifest_sha256": sha(FV3 / "manifest.json"),
        "srncf_manifest_sha256": sha(SRNCF / "manifest.json"),
        "protocol": dict(
            dataset="Houston2013 official",
            channels=30,
            window=11,
            epochs=100,
            batch_size=128,
            workers=0,
            learning_rate=0.0001,
            weight_decay=0.01,
            seeds=[6, 7, 8],
        ),
    }
    write(DEST / "manifest.json", manifest)
    write(ARCHIVE / "reuse_audit.json", {"all_passed": True, "baselines": baselines})
    write(ARCHIVE / "authorization.json", plan)
    pins(manifest)
    state(
        state="prepared",
        completed=REUSED_COUNT,
        total=TOTAL_COUNT,
        reused=REUSED_COUNT,
        new_completed=0,
        new_total=NEW_COUNT,
        current=None,
        training_pid=None,
        queue_pid=None,
        gpu_preflight_passed=False,
    )
    summarize()
    print("SRA0C02_SETUP_OK", len(jobs), flush=True)


def gpu_preflight():
    assert_idle()
    import numpy as np
    import torch
    from src.data import getMyData, set_random_seed
    from src.models.transmixer_sr_a0_c02.preflight import run_checks, inference_costs

    m = read(DEST / "manifest.json")
    pins(m)
    assert Path(sys.executable).resolve() == Path(m["python"]).resolve()
    assert platform.python_version() == m["python_version"] and str(torch.__version__) == m["torch"]
    assert np.__version__ == m["numpy"] and torch.cuda.get_device_name(0) == m["gpu"]
    set_random_seed(6)
    loaders = getMyData(0, 30, 11, 128, 0)
    assert len(loaders[0].dataset) == 2832 and len(loaders[1].dataset) == 12197
    assert len(loaders[0]) == 22 and loaders[0].drop_last
    h, raw, l, y = next(iter(loaders[0]))
    batch = {"hsi_pca": h.to("cuda:0"), "aux": l.to("cuda:0"), "label": y.to("cuda:0")}
    del loaders, h, raw, l, y
    gc.collect()

    result = run_checks("cuda:0", batch)
    costs = inference_costs(batch)
    fv3_costs = read(FV3 / "cost_audit.json")["runs"]
    srncf_costs = read(SRNCF / "cost_audit.json")["runs"]
    ref_c02 = next((r for r in fv3_costs if r["group"] == "C02"), None)
    ref_k02 = next((r for r in srncf_costs if r["group"] == "K02"), None)
    if ref_c02 is not None:
        costs["reference_c02_from_fv3"] = ref_c02
    if ref_k02 is not None:
        costs["reference_k02_from_srncf"] = ref_k02

    write(DEST / "cost_audit.json", costs)
    write(DEST / "gpu_preflight.json", result)
    m["source_and_config_sha256"][(DEST / "gpu_preflight.json").relative_to(ROOT).as_posix()] = sha(DEST / "gpu_preflight.json")
    m["source_and_config_sha256"][(DEST / "cost_audit.json").relative_to(ROOT).as_posix()] = sha(DEST / "cost_audit.json")
    write(DEST / "manifest.json", m)
    pins(m)
    state(state="ready_to_run", gpu_preflight_passed=True)
    summarize()
    del batch
    gc.collect()
    torch.cuda.empty_cache()
    print("SRA0C02_GPU_PREFLIGHT_OK", flush=True)


def summarize():
    m = read(DEST / "manifest.json")
    rows = [
        {"id": j["id"], "group": j["group"], "seed": j["seed"], "reused": True, **j["metrics"]}
        for j in m["baselines"]
    ]
    for job in m["jobs"]:
        path = DEST / "completed" / f"{job['id']}.json"
        if path.exists():
            done = read(path)
            rows.append({
                "id": job["id"],
                "group": job["group"],
                "seed": job["seed"],
                "reused": False,
                "elapsed_seconds": done["elapsed_seconds"],
                **done["metrics"],
            })
    stats, comparisons = {}, []
    for group in ALL_GROUPS:
        values = [r for r in rows if r["group"] == group]
        if len(values) == 3:
            stats[group] = {
                key: {
                    "mean": statistics.mean(r[key] for r in values),
                    "std": statistics.stdev(r[key] for r in values),
                }
                for key in ("oa", "aa", "kappa")
            }
    for a, b in COMPARISONS:
        deltas = []
        for seed in (6, 7, 8):
            ra = next((r for r in rows if r["group"] == a and r["seed"] == seed), None)
            rb = next((r for r in rows if r["group"] == b and r["seed"] == seed), None)
            if ra and rb:
                deltas.append({"seed": seed, **{key: ra[key] - rb[key] for key in ("oa", "aa", "kappa")}})
        comparisons.append({
            "comparison": a + "-" + b,
            "paired_deltas_pp": deltas,
            "complete": len(deltas) == 3,
            "statistics": {
                key: {
                    "mean_delta_pp": statistics.mean(d[key] for d in deltas),
                    "positive_count": sum(d[key] > 0 for d in deltas),
                    "paired_seeds": len(deltas),
                }
                for key in ("oa", "aa", "kappa")
            }
            if deltas
            else {},
        })

    per_class_rows = export_per_class_csv(m)
    focus_summary = {}
    for group in ALL_GROUPS:
        g_rows = [r for r in per_class_rows if r["group"] == group]
        if len(g_rows) == 45:
            focus_summary[group] = {}
            for cls_name in FOCUS_CLASSES:
                c_rows = [r for r in g_rows if r["class"] == cls_name]
                focus_summary[group][cls_name] = {
                    "recall_mean": statistics.mean(float(r["native_printed_recall"]) * 100.0 for r in c_rows),
                    "f1_mean": statistics.mean(float(r["native_printed_f1"]) * 100.0 for r in c_rows),
                }

    atomic(
        DEST / "summary.json",
        {
            "updated_at": now(),
            "completed": len(rows),
            "total": TOTAL_COUNT,
            "reused": REUSED_COUNT,
            "new_completed": len(rows) - REUSED_COUNT,
            "new_total": NEW_COUNT,
            "runs": rows,
            "statistics": stats,
            "comparisons": comparisons,
            "focus_classes_mean_percent": focus_summary,
        },
    )

    lines = [
        "# A0 与 C02 接入原生 SR 引导邻域融合对照实验报告（R00-R05）",
        "",
        f"更新时间：{now()}。完成 {len(rows)}/{TOTAL_COUNT}，复用 {REUSED_COUNT} 项（A0、K01-K03 与 C02 各 3 种子），新训 {len(rows) - REUSED_COUNT}/{NEW_COUNT}。",
        "",
        "预设核心比较：`R01 - R00`、`R01 - R02`、`R01 - A0`（A0 基座单尺度 SR 融合净效应）与 `R04 - R03`、`R04 - R05`、`R04 - C02`（C02 基座 +0 参数三尺度 SR 融合净效应）。每项 100 轮，`seed6/7/8`，纯 `CrossEntropyLoss`；沿用原测试 OA 选模规则，不包装为独立测试。",
        "",
        "| 组别 | seed | 来源 | OA% | AA% | Kappa% |",
        "|---|---:|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r['group']} | {r['seed']} | {'复用' if r['reused'] else '新训'} | {r['oa']:.4f} | {r['aa']:.4f} | {r['kappa']:.4f} |"
        for r in rows
    ]
    lines += [
        "",
        "## 三种子统计（样本标准差 ddof=1）",
        "",
        "| 组别 | OA% | AA% | Kappa% |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        "| " + g + " | " + " | ".join(f"{v[k]['mean']:.4f} ± {v[k]['std']:.4f}" for k in ("oa", "aa", "kappa")) + " |"
        for g, v in stats.items()
    ]
    lines += [
        "",
        "## 同种子预定比较",
        "",
        "| 比较 | 已配对种子数 | 平均OA差/百分点 | OA正增益次数 |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {c['comparison']} | {len(c['paired_deltas_pp'])} | {c['statistics']['oa']['mean_delta_pp']:+.4f} | {c['statistics']['oa']['positive_count']} |"
        for c in comparisons
        if c["statistics"]
    ]
    if focus_summary:
        lines += [
            "",
            "## 预设重点地物三种子平均召回率与 F1（%）",
            "",
            "| 组别 | Highway Recall/F1 | Road Recall/F1 | Commercial Recall/F1 | Stressed grass Recall/F1 |",
            "|---|---:|---:|---:|---:|",
        ]
        for g, cmap in focus_summary.items():
            cells = [f"{cmap[c]['recall_mean']:.2f} / {cmap[c]['f1_mean']:.2f}" for c in FOCUS_CLASSES]
            lines.append(f"| {g} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "逐种子差值、AA/Kappa 差值见 `summary.json`；全部 15 类精度/召回/F1 见 `native_per_class_metrics.csv`。单数据集、三种子结果按实报告，不虚构显著性检验。",
        "",
    ]
    if (DEST / "cost_audit.json").exists():
        cost_data = read(DEST / "cost_audit.json")
        lines += [
            "## 资源与推理开销",
            "",
            "MACs 为明确算子范围内的估计，非完整 FLOPs；推理计时为原生 batch128、3 次预热后 5 次平均。",
            "",
            "| 组别 | 有效参数 | 每patch估计MACs | batch128延迟/ms |",
            "|---|---:|---:|---:|",
        ]
        for ref_key, label in (("reference_k02_from_srncf", "K02（SR-NCF参照）"), ("reference_c02_from_fv3", "C02（FV3参照）")):
            if ref_key in cost_data:
                rc = cost_data[ref_key]
                lines.append(
                    f"| {label} | {rc['parameters']} | {rc['estimated_clean_macs_per_patch']} | {rc['batch_latency_ms']:.3f} |"
                )
        lines += [
            f"| {r['group']} | {r['parameters']} | {r['estimated_clean_macs_per_patch']} | {r['batch_latency_ms']:.3f} |"
            for r in cost_data["runs"]
        ]
    report_md = "\n".join(lines) + "\n"
    assert "$" not in report_md
    (DEST / "实验报告.md").write_text(report_md, encoding="utf-8")


def audit_job(job, done):
    assert (done["id"], done["seed"], done["group"]) == (job["id"], job["seed"], job["group"])
    assert done["epochs"] == 100 and done["training_steps"] == 2200
    assert read(job["diagnostics"])["training_steps"] == 2200
    assert "Report Success!" in Path(job["stdout"]).read_text(encoding="utf-8", errors="replace")
    assert re.search(
        r"Epoch 100/100:\s+100%[^\r\n]*22/22",
        Path(job["stderr"]).read_text(encoding="utf-8", errors="replace"),
    )
    for key in ARTIFACTS:
        assert sha(job[key]) == done[key + "_sha256"], (job["id"], key)
    assert report_metrics(job["report"]) == done["metrics"]


def completed_prefix(manifest):
    count, pending = 0, False
    comp_dir = DEST / "completed"
    if not comp_dir.exists():
        for job in manifest["jobs"]:
            existing = [key for key in ARTIFACTS if Path(job[key]).exists()]
            if existing:
                raise RuntimeError("Unsealed artifacts exist: " + job["id"] + str(existing))
        return 0
    known = {j["id"] + ".json" for j in manifest["jobs"]}
    unexpected = {p.name for p in comp_dir.glob("*.json")} - known
    if unexpected:
        raise RuntimeError("Unknown completed markers: " + str(sorted(unexpected)))
    for job in manifest["jobs"]:
        marker = comp_dir / f"{job['id']}.json"
        if marker.exists():
            if pending:
                raise RuntimeError("Non-contiguous completed marker: " + job["id"])
            audit_job(job, read(marker))
            count += 1
        else:
            pending = True
            existing = [key for key in ARTIFACTS if Path(job[key]).exists()]
            if existing:
                raise RuntimeError(
                    "Unsealed artifacts require explicit recovery, not retraining: "
                    + job["id"]
                    + str(existing)
                )
    return count


def final_audit():
    m = read(DEST / "manifest.json")
    pins(m)
    checks = []
    for job in m["jobs"]:
        done = read(DEST / "completed" / f"{job['id']}.json")
        audit_job(job, done)
        checks.append({
            "id": job["id"],
            "verified": True,
            "epochs": 100,
            "steps": 2200,
            **{key + "_sha256": done[key + "_sha256"] for key in ARTIFACTS},
        })
    assert len(checks) == NEW_COUNT and not m["registry_transition_pending"]
    summarize()
    per_class_rows = export_per_class_csv(m)
    assert len(per_class_rows) == TOTAL_COUNT * 15
    write(
        DEST / "final_integrity_audit.json",
        {
            "all_passed": True,
            "checked_at": now(),
            "new_completed": NEW_COUNT,
            "reused_verified": REUSED_COUNT,
            "total": TOTAL_COUNT,
            "per_class_csv_rows": len(per_class_rows),
            "outputs": checks,
            "source_config_data_locked": True,
            "native_100_epochs_22_batches_report_success_verified": True,
            "registry_transition_sha256": sha(ARCHIVE / "registry_transition.json"),
        },
    )


def run_queue(helper, resume=False):
    from scripts.keep_awake_fusion import kernel

    m = read(DEST / "manifest.json")
    pins(m)
    assert read(DEST / "gpu_preflight.json")["all_passed"]
    (DEST / "completed").mkdir(exist_ok=True)
    prefix = completed_prefix(m)
    if not resume and prefix:
        raise RuntimeError("Completed outputs require the authorized resume entry")
    env = dict(
        os.environ,
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTHONIOENCODING="utf-8",
        PYTHONUNBUFFERED="1",
    )
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    child = None
    try:
        for index, job in enumerate(m["jobs"]):
            if index < prefix:
                continue
            assert_idle()
            pins(m)
            assert helper.poll() is None
            assert not any(Path(job[k]).exists() for k in ARTIFACTS)
            started, clock = now(), time.monotonic()
            state(
                state="launching",
                current=job["id"],
                group=job["group"],
                seed=job["seed"],
                current_epoch=0,
                current_started_at=started,
                new_completed=index,
                completed=REUSED_COUNT + index,
            )
            with Path(job["stdout"]).open("xb") as out, Path(job["stderr"]).open("xb") as err:
                child = subprocess.Popen(
                    [m["python"], "-X", "utf8", "-u", "scripts/train.py", "--config", job["config"]],
                    cwd=ROOT,
                    env=env,
                    stdout=out,
                    stderr=err,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                state(state="training", training_pid=child.pid)
                while child.poll() is None:
                    time.sleep(10)
                    assert kernel.SetThreadExecutionState(0x80000001)
                    assert helper.poll() is None, "Power guard stopped; preserve active scientific child"
                    path = Path(job["stderr"])
                    with path.open("rb") as stream:
                        stream.seek(max(0, path.stat().st_size - 16000))
                        tail = stream.read().decode("utf-8", errors="replace")
                    epochs = re.findall(r"Epoch (\d+)/100", tail)
                    state(
                        current_epoch=int(epochs[-1]) if epochs else 0,
                        elapsed_seconds=round(time.monotonic() - clock, 1),
                        log_updated_at=datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(),
                    )
            if child.returncode:
                raise RuntimeError(
                    f"Native training failed {job['id']} exit={child.returncode}; inspect {job['stderr']}"
                )
            diagnostics = read(job["diagnostics"])
            done = {
                "id": job["id"],
                "seed": job["seed"],
                "group": job["group"],
                "epochs": 100,
                "training_steps": diagnostics["training_steps"],
                "started_at": started,
                "finished_at": now(),
                "elapsed_seconds": time.monotonic() - clock,
                "metrics": report_metrics(job["report"]),
                **{key + "_sha256": sha(job[key]) for key in ARTIFACTS},
            }
            audit_job(job, done)
            pins(m)
            with (DEST / "completed" / f"{job['id']}.json").open("x", encoding="utf-8") as stream:
                json.dump(done, stream, ensure_ascii=False, indent=2)
            summarize()
            state(state="between_jobs", training_pid=None, completed=REUSED_COUNT + index + 1, new_completed=index + 1)
            print("COMPLETED", job["id"], done["metrics"], flush=True)
        final_audit()
        state(state="completed", current=None, group=None, seed=None, training_pid=None, finished_at=now())
    except BaseException as exc:
        state(
            state="failed",
            error=repr(exc),
            failed_at=now(),
            training_pid=child.pid if child and child.poll() is None else None,
        )
        raise
    finally:
        while child and child.poll() is None:
            kernel.SetThreadExecutionState(0x80000001)
            try:
                state(state="failed_waiting_for_active_child", training_pid=child.pid)
            except OSError as exc:
                print("State unavailable; protecting surviving child", child.pid, repr(exc), flush=True)
            time.sleep(15)


def guard():
    from scripts import keep_awake_fusion as g

    info = {
        "helper_pid": os.getpid(),
        "power_plan_changed": False,
        "display_required": False,
        "started_at": now(),
    }
    try:
        while True:
            s = read(DEST / "运行状态.json")
            if not any(g.is_running(s.get(k)) for k in ("queue_pid", "training_pid")):
                break
            if not g.kernel.SetThreadExecutionState(0x80000001):
                raise RuntimeError("Temporary sleep protection request failed")
            supply = g.PowerStatus()
            valid = g.kernel.GetSystemPowerStatus(ctypes.byref(supply))
            info.update(
                state="active",
                native_request_succeeded=True,
                checked_at=now(),
                queue_pid=s.get("queue_pid"),
                training_pid=s.get("training_pid"),
                ac_connected=supply.ac == 1 if valid else None,
                battery_percent=int(supply.battery_percent) if valid else None,
            )
            atomic(DEST / "power_protection.json", info)
            time.sleep(20)
    except BaseException as exc:
        info.update(state="failed", error=repr(exc))
        raise
    finally:
        g.kernel.SetThreadExecutionState(0x80000000)
        if info.get("state") != "failed":
            info["state"] = "released"
        info["updated_at"] = now()
        atomic(DEST / "power_protection.json", info)


def run_main():
    from scripts.keep_awake_fusion import kernel, is_running
    import msvcrt

    lock = (DEST / "controller.lock").open("a+b")
    lock.seek(0)
    lock.write(b"0")
    lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    helper = None
    try:
        assert_idle()
        assert read(PLAN)["training_authorized"]
        current_state = read(DEST / "运行状态.json")
        assert current_state["state"] == "ready_to_run" and current_state["gpu_preflight_passed"]
        m = read(DEST / "manifest.json")
        pins(m)
        assert Path(sys.executable).resolve() == Path(m["python"]).resolve()
        assert read(DEST / "gpu_preflight.json")["all_passed"]

        state(state="starting", queue_pid=os.getpid(), training_pid=None, started_at=now())
        assert kernel.SetThreadExecutionState(0x80000001)
        with (LOG / "power.stdout.log").open("xb") as out, (LOG / "power.stderr.log").open("xb") as err:
            helper = subprocess.Popen(
                [sys.executable, "-X", "utf8", "-u", str(Path(__file__)), "guard"],
                cwd=ROOT,
                stdout=out,
                stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        state(helper_pid=helper.pid)
        for _ in range(30):
            if (DEST / "power_protection.json").exists():
                power = read(DEST / "power_protection.json")
                if power.get("helper_pid") == helper.pid and power.get("state") == "active":
                    break
            if helper.poll() is not None:
                raise RuntimeError("Power guard failed")
            time.sleep(1)
        else:
            raise RuntimeError("Power guard did not acknowledge")
        run_queue(helper)
    except BaseException as exc:
        state(state="failed", error=repr(exc), failed_at=now())
        raise
    finally:
        s = read(DEST / "运行状态.json")
        while is_running(s.get("training_pid")):
            kernel.SetThreadExecutionState(0x80000001)
            time.sleep(15)
            s = read(DEST / "运行状态.json")
        kernel.SetThreadExecutionState(0x80000000)
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()


def resume():
    from scripts.keep_awake_fusion import kernel, is_running
    import msvcrt

    lock = (DEST / "controller.lock").open("a+b")
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    entered = False
    try:
        previous = read(DEST / "运行状态.json")
        assert previous["state"] == "ready_for_authorized_resume"
        assert not any(is_running(previous.get(k)) for k in ("queue_pid", "training_pid", "helper_pid"))
        assert_idle()
        m = read(DEST / "manifest.json")
        pins(m)
        assert Path(sys.executable).resolve() == Path(m["python"]).resolve()
        assert completed_prefix(m) >= 1
        assert read(DEST / "gpu_preflight.json")["all_passed"]
        attempt = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        state(
            state="resuming",
            queue_pid=os.getpid(),
            training_pid=None,
            helper_pid=None,
            resumed_at=now(),
            recovery_attempt=attempt,
            error=None,
            failed_at=None,
        )
        entered = True
        assert kernel.SetThreadExecutionState(0x80000001)
        with (LOG / f"power.resume_{attempt}.stdout.log").open("xb") as out, (
            LOG / f"power.resume_{attempt}.stderr.log"
        ).open("xb") as err:
            helper = subprocess.Popen(
                [sys.executable, "-X", "utf8", "-u", str(Path(__file__)), "guard"],
                cwd=ROOT,
                stdout=out,
                stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        state(helper_pid=helper.pid)
        for _ in range(30):
            assert helper.poll() is None, "Recovery sleep guard failed"
            if (DEST / "power_protection.json").exists():
                power = read(DEST / "power_protection.json")
                if power.get("helper_pid") == helper.pid and power.get("state") == "active":
                    break
            time.sleep(1)
        else:
            raise RuntimeError("Recovery sleep guard did not acknowledge")
        run_queue(helper, resume=True)
    except BaseException as exc:
        if entered:
            state(state="failed", error=repr(exc), failed_at=now())
        raise
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()


if __name__ == "__main__":
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("setup", "gpu_preflight", "run", "resume", "guard", "summarize", "audit"),
    )
    action = parser.parse_args().action
    {
        "setup": setup,
        "gpu_preflight": gpu_preflight,
        "run": run_main,
        "resume": resume,
        "guard": guard,
        "summarize": summarize,
        "audit": final_audit,
    }[action]()
