"""Queue/adapt/report only; training, preprocessing and scoring reuse the lab.

The experiment specification inherits all shared values from the sealed previous
three-seed batch. No optimizer, data augmentation or epoch loop is reimplemented.
"""
import csv
from datetime import datetime, timezone, timedelta
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ConfigManager, ExperimentConfig
from .model_registry import get_model_adapter
from . import fusion_experiment as lab
from .models.transmixer_paper.adapter import MODEL_NAMES, VARIANTS
from .models.transmixer_paper.model import geometry_targets, state_digest

ROOT = lab.ROOT
ORDER = ("G0", *VARIANTS)
SOURCE_FILES = (
    "README.md", "src/data.py", "src/trainer.py", "src/evaluator.py", "src/metrics.py", "src/runner.py",
    "src/config.py", "src/model_registry.py", "src/fusion_experiment.py",
    "src/models/transmixer_net/model.py", "src/models/transmixer_net/adapter.py",
    "src/models/transmixer_fusion/model.py", "src/models/transmixer_fusion/adapter.py",
    "src/models/transmixer_paper/model.py", "src/models/transmixer_paper/adapter.py",
    "src/paper_transfer_experiment.py", "scripts/train_transmixer_paper.py",
    "configs/paper_transfer.json",
    "src/models/transmixer_paper/test_paper.py", "scripts/check_transmixer_paper.py",
)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    lab.write_json(temp, value)
    temp.replace(path)


def source_hashes():
    return {p: lab.sha256(ROOT / p) for p in SOURCE_FILES}


def resolve_protocol(plan):
    required = {"protocol_reference", "variants", "seeds", "reuse_verified_baselines",
                "correction_epochs", "spatial_sigma", "correction_hidden", "concept_weight", "final_test"}
    if set(plan) != required:
        raise ValueError("Unexpected plan fields")
    if plan["variants"] != list(ORDER) or plan["seeds"] != [6, 7, 8]:
        raise ValueError("Approved first batch requires the nine groups and paired seeds 6,7,8")
    if plan["correction_epochs"] != 50 or plan["spatial_sigma"] != 3. or plan["correction_hidden"] != 16 or plan["concept_weight"] != .1:
        raise ValueError("Approved method-specific constants changed")
    if type(plan["reuse_verified_baselines"]) is not bool or type(plan["final_test"]) is not bool:
        raise ValueError("Flags must be booleans")
    reference = ROOT / plan["protocol_reference"]
    cfg = read(reference / "batch_config.json")
    lab.validate_settings(cfg)
    if cfg["seeds"] != plan["seeds"] or cfg["datasets"] != ["Houston2013"] or cfg["epochs"] != 100:
        raise ValueError("Reference is not the approved three-seed Houston protocol")
    # Test inference occurs only once after the entire fixed grid has finished.
    return reference, {**cfg, "evaluate_test": False}


def audit_protocol(reference, cfg):
    previous = read(reference / "runtime.json")
    if str(torch.__version__) != previous["torch"] or np.__version__ != previous["numpy"] or torch.version.cuda != previous["cuda"]:
        raise RuntimeError("Torch/NumPy/CUDA differ from the previous experiment; refusing silent drift")
    audit = {"python": sys.executable, "python_version": platform.python_version(),
             "torch": str(torch.__version__), "numpy": np.__version__, "cuda": torch.version.cuda,
             "reference": str(reference), "reference_runtime_sha256": lab.sha256(reference / "runtime.json"),
             "shared_config": cfg, "unchanged_backbone_sources": {}}
    for rel in ("src/models/transmixer_net/model.py", "src/models/transmixer_fusion/model.py",
                "src/models/transmixer_fusion/adapter.py"):
        digest = lab.sha256(ROOT / rel)
        expected = previous["source_sha256"].get(rel.replace("/", "\\"), previous["source_sha256"].get(rel))
        if digest != expected:
            raise RuntimeError(f"Previous backbone source changed: {rel}")
        audit["unchanged_backbone_sources"][rel] = digest
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU fallback")
    audit["gpu"] = torch.cuda.get_device_name(torch.device(cfg["device"]))
    return audit


def training_statistics(data, cfg):
    dataset = lab.FusionPatchDataset(data, "train", cfg["window_size"])
    spectral, height, concepts = [], [], []
    center = cfg["window_size"] // 2
    mask = torch.ones(cfg["window_size"] ** 2, dtype=torch.bool)
    mask[center * cfg["window_size"] + center] = False
    for h, a, _ in DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False):
        h = h[:, 0]
        dh = (h - h[:, :, center:center+1, center:center+1]).square().mean(1).flatten(1)
        da = (a - a[:, :, center:center+1, center:center+1]).square().mean(1).flatten(1)
        spectral.append(dh[:, mask].numpy().reshape(-1))
        height.append(da[:, mask].numpy().reshape(-1))
        concepts.append(geometry_targets(a).numpy())
    concepts = np.concatenate(concepts)
    return {"spectral_scale": max(float(np.median(np.concatenate(spectral))), 1e-8),
            "height_scale": max(float(np.median(np.concatenate(height))), 1e-8),
            "concept_mean": concepts.mean(0).tolist(),
            "concept_std": np.maximum(concepts.std(0), 1e-8).tolist(),
            "samples": len(dataset), "scope": "training input patches only; no validation/test labels"}


def manager_for(variant, seed, cfg, settings):
    return ConfigManager(ExperimentConfig(
        experiment_name=f"paper_{variant}_seed{seed}", dataset_type=0,
        model_name="TransMixer_B0_Decoupled" if variant == "G0" else MODEL_NAMES[variant],
        channels=cfg["channels"], window_size=cfg["window_size"], random_seed=seed,
        cuda_device=cfg["device"], learning_rate=cfg["learning_rate"], epochs=cfg["epochs"],
        batch_size=cfg["batch_size"], num_workers=0, paper_transfer=settings,
        enable_testing=False, enable_visualization=False))


def settings_for(variant, seed, plan, stats, folder):
    settings = {k: stats[k] for k in ("spectral_scale", "height_scale", "concept_mean", "concept_std")}
    settings.update({k: plan[k] for k in ("spatial_sigma", "correction_hidden", "concept_weight")})
    if variant.startswith("PN"):
        checkpoint = folder / "Houston2013" / f"G0_seed{seed}" / "best.pth"
        settings.update(baseline_checkpoint=str(checkpoint), baseline_sha256=lab.sha256(checkpoint))
    return settings


def load_net(row, cfg, device):
    manager = manager_for(row["variant"], row["seed"], cfg, row["settings"])
    adapter = get_model_adapter(manager.config.model_name)
    bundle = adapter.build_model(manager, 0, device)
    checkpoint = torch.load(row["checkpoint"], map_location=device, weights_only=True)
    bundle["net"].load_state_dict(checkpoint["state_dict"], strict=True)
    return bundle["net"].eval()


def reuse_baseline(reference, seed, cfg, data, out, settings):
    source = reference / "Houston2013" / f"F0_seed{seed}"
    old = read(source / "result.json")
    old_cfg = read(source / "config.json")
    for key in cfg:
        if key != "evaluate_test" and old_cfg[key] != cfg[key]:
            raise RuntimeError(f"Baseline config mismatch: {key}")
    history = [json.loads(line) for line in (source / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    if old["status"] != "completed" or len(history) != cfg["epochs"] or old["selection_split"] != "validation":
        raise RuntimeError("Prior baseline is incomplete or selected by a different protocol")
    if [h["epoch"] for h in history] != list(range(1, cfg["epochs"] + 1)):
        raise RuntimeError("Prior baseline epoch coverage mismatch")
    device = torch.device(cfg["device"])
    lab.seed_everything(seed, cfg["cpu_threads"])
    manager = manager_for("G0", seed, cfg, settings)
    net = get_model_adapter(manager.config.model_name).build_model(manager, 0, device)["net"]
    checkpoint = torch.load(source / "best.pth", map_location=device, weights_only=True)
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    val_set = lab.FusionPatchDataset(data, "validation", cfg["window_size"])
    metrics, ys, preds = lab.evaluate(net, DataLoader(val_set, batch_size=cfg["batch_size"]), device, data["info"]["classes"])
    # Reproducing archived validation predictions and loss also checks the current
    # PCA, standardization, checkpoint loading and scoring against the old batch.
    if metrics["confusion_matrix"] != old["validation"]["confusion_matrix"] or abs(metrics["loss"] - old["validation"]["loss"]) > 1e-6:
        raise RuntimeError("Archived baseline predictions/loss did not reproduce; refusing checkpoint reuse")
    with (source / "validation_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        archived = list(csv.DictReader(stream))
    if len(archived) != len(preds) or any(int(x["predicted_class_1based"]) != p + 1 for x, p in zip(archived, preds)):
        raise RuntimeError("Archived per-sample predictions mismatch")
    out.mkdir(parents=True, exist_ok=False)
    for filename in ("best.pth", "history.jsonl", "validation_predictions.csv"):
        shutil.copy2(source / filename, out / filename)
    write(out / "config.json", {**cfg, "variant": "G0", "seed": seed,
                                "model_args": {"model_name": manager.config.model_name, "paper_transfer": settings}})
    row = {**old, "variant": "G0", "label": "固定1:1共同基线", "test": None, "validation": metrics,
           "checkpoint": str(out / "best.pth"), "split_sha256": data["info"]["split_sha256"],
           "reused": True, "source_checkpoint": str(source / "best.pth"),
           "source_checkpoint_sha256": lab.sha256(source / "best.pth"),
           "reproduction": "all validation predictions identical; CE tolerance 1e-6",
           "settings": settings, "benchmark": lab.benchmark(net, val_set, device)}
    write(out / "result.json", row)
    return row


@torch.no_grad()
def diagnostics(net, row, data, cfg, base_predictions):
    loader = DataLoader(lab.FusionPatchDataset(data, "validation", cfg["window_size"]), batch_size=cfg["batch_size"])
    device = torch.device(cfg["device"])
    if row["variant"].startswith("OC"):
        targets, predicted = [], []
        for h, a, y in loader:
            net(h.to(device), a.to(device))
            predicted.append(net.classifier.last_concepts.cpu().numpy())
            targets.append(net.normalized_targets(a.to(device)).cpu().numpy())
        target, pred = np.concatenate(targets), np.concatenate(predicted)
        result = {"normalized_concept_mae": np.abs(target - pred).mean(0).tolist(),
                  "training_mean_predictor_mae": np.abs(target).mean(0).tolist(),
                  "supervised_concepts": row["variant"] in ("OC1", "OC3")}
        if row["variant"] == "OC3":
            net.classifier.replace_concepts_with_mean = True
            metrics, _, predictions = lab.evaluate(net, loader, device, data["info"]["classes"])
            net.classifier.replace_concepts_with_mean = False
            original = []
            with Path(row["checkpoint"]).with_name("validation_predictions.csv").open(encoding="utf-8-sig") as f:
                original = [int(x["predicted_class_1based"]) - 1 for x in csv.DictReader(f)]
            result.update(replace_concepts_with_train_mean_validation_oa=metrics["oa"],
                          changed_predictions=sum(a != b for a, b in zip(original, predictions)))
        return result
    if row["variant"].startswith("PN"):
        _, labels, predictions = lab.evaluate(net, loader, device, data["info"]["classes"])
        old = base_predictions[row["seed"]]
        corrected = sum(a != y and b == y for a, b, y in zip(old, predictions, labels))
        damaged = sum(a == y and b != y for a, b, y in zip(old, predictions, labels))
        return {"baseline_final_sha256": state_digest(net.base), "validation_wrong_to_right": corrected,
                "validation_right_to_wrong": damaged, "validation_net_corrections": corrected - damaged}
    return {}


def paired(rows, candidate, control, split, metric="oa"):
    values = []
    for seed in (6, 7, 8):
        a = next((r for r in rows if r["variant"] == candidate and r["seed"] == seed and r.get("status") == "completed"), None)
        b = next((r for r in rows if r["variant"] == control and r["seed"] == seed and r.get("status") == "completed"), None)
        if a and b and a.get(split) and b.get(split):
            values.append(a[split][metric] - b[split][metric])
    return {"candidate": candidate, "control": control, "split": split, "metric": metric,
            "n": len(values), "differences": values,
            "mean": float(np.mean(values)) if values else None,
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "positive_seeds": sum(v > 0 for v in values)}


def summary(folder, rows, state):
    write(folder / "summary.json", rows)
    write(folder / "运行状态.json", {"updated_at": now(), "pid": os.getpid(), "total": 27,
                                "completed": sum(r.get("status") == "completed" for r in rows), **state})
    with (folder / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["variant", "seed", "status", "reused", "best_epoch", "val_oa", "val_loss", "test_oa", "parameters", "training_seconds"]
        writer = csv.DictWriter(stream, fields); writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row.get(k, "") for k in fields}, "val_oa": row.get("validation", {}).get("oa", ""),
                             "val_loss": row.get("validation", {}).get("loss", ""),
                             "test_oa": (row.get("test") or {}).get("oa", "")})
    contrasts = [paired(rows, a, b, split) for split in ("validation", "test")
                 for a, b in (("SR1", "G0"), ("SR2", "SR1"), ("SR3", "SR2"), ("SR3", "G0"),
                              ("PN1", "G0"), ("PN2", "PN1"), ("PN2", "G0"),
                              ("OC1", "G0"), ("OC3", "OC2"), ("OC3", "G0"))]
    write(folder / "paired_comparisons.json", contrasts)
    lines = ["# 三篇论文迁移实验进度与结果", "", f"更新时间：{now()}；状态：{state['state']}。", "",
             "共同标准继承2026-10-24三种子验证选模实验；保留固定1:1结构。",
             "验证OA可能饱和100%，同时记录分类CE；最终测试在全体固定配置训练完成后统一执行。",
             "旧基线只有在来源、划分、输入文件和逐样本验证预测均核对后才复用。", "",
             "| 组 | 完成种子数 | 验证OA 均值±标准差 | 验证CE均值 | 测试OA 均值±标准差 |", "|---|---:|---:|---:|---:|"]
    for variant in ORDER:
        completed = [r for r in rows if r["variant"] == variant and r.get("status") == "completed"]
        def stats(split, metric):
            values = [r[split][metric] for r in completed if r.get(split)]
            return f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}" if len(values) > 1 else (f"{values[0]:.4f}（单种子）" if values else "待完成")
        losses = [r["validation"]["loss"] for r in completed]
        ce = f"{np.mean(losses):.6f}" if losses else "—"
        lines.append(f"| {variant} | {len(completed)} | {stats('validation','oa')} | {ce} | {stats('test','oa')} |")
    lines += ["", "## 机制对照", "", "差值单位为准确率百分点；所有配对种子完整保留。"]
    for item in contrasts:
        if item["n"]:
            lines.append(f"- {item['split']} {item['candidate']}−{item['control']}：{item['mean']:+.4f}，{item['positive_seeds']}/{item['n']}种子正向。")
    (folder / "实验报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(plan, folder, resume=False):
    reference, cfg = resolve_protocol(plan)
    audit = audit_protocol(reference, cfg)
    folder = Path(folder).resolve()
    if not folder.is_relative_to((ROOT / "report/paper_transfer").resolve()):
        raise ValueError("Run folder must remain under report/paper_transfer")
    if folder.exists() and not resume:
        raise FileExistsError("Run exists; use explicit --resume")
    folder.mkdir(parents=True, exist_ok=True)
    sources = source_hashes()
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        previous = read(manifest_path)
        if previous["plan"] != plan or previous["source_sha256"] != sources:
            raise RuntimeError("Resume source/config drift; existing run will not be modified")
    else:
        write(manifest_path, {"created_at": now(), "plan": plan, "protocol_audit": audit, "source_sha256": sources,
                              "training_loop": "src.fusion_experiment.run_one with the native model adapter hooks",
                              "test_policy": "one evaluation per fixed configuration after all training; no test-based tuning"})
        for rel in SOURCE_FILES:
            dest = folder / "source_snapshot" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dest)
    # Operating-system lock is released on crash; never launch duplicate queues.
    lock = (folder / "queue.lock").open("a+b")
    lock.write(b"0"); lock.flush(); lock.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = []
    try:
        summary(folder, rows, {"state": "preparing_data"})
        lab.seed_everything(6, cfg["cpu_threads"])
        data = lab.prepare_data("Houston2013", cfg, folder / "Houston2013")
        old_info = read(reference / "Houston2013/data_manifest.json")
        for key in ("train", "validation", "test", "class_counts", "pca_solver", "shape", "aux_channels"):
            if data["info"][key] != old_info[key]:
                raise RuntimeError(f"Data protocol drift: {key}")
        for key, item in data["info"]["files"].items():
            if item["sha256"] != old_info["files"][key]["sha256"]:
                raise RuntimeError(f"Input file changed: {key}")
        with np.load(reference / "Houston2013/split.npz") as old_split:
            for key in ("train", "validation", "test"):
                if not np.array_equal(data[key], old_split[key]):
                    raise RuntimeError(f"Center indices differ: {key}")
        # Preserve both byte-level and semantic identity of archived positions.
        shutil.copy2(reference / "Houston2013/split.npz", folder / "Houston2013/split.npz")
        data["info"]["split_sha256"] = lab.sha256(folder / "Houston2013/split.npz")
        write(folder / "Houston2013/data_manifest.json", data["info"])
        statistics = training_statistics(data, cfg)
        write(folder / "training_input_statistics.json", statistics)
        write(folder / "protocol_check.json", {**audit, "data": data["info"], "split_arrays_equal": True,
                                                "split_file_sha256_equal": data["info"]["split_sha256"] == old_info["split_sha256"]})
        for variant in plan["variants"]:
            for seed in plan["seeds"]:
                out = folder / "Houston2013" / f"{variant}_seed{seed}"
                settings = settings_for(variant, seed, plan, statistics, folder)
                if (out / "result.json").exists() and read(out / "result.json").get("status") == "completed":
                    row = read(out / "result.json")
                    if row.get("checkpoint_sha256") != lab.sha256(row["checkpoint"]):
                        raise RuntimeError("Completed checkpoint checksum mismatch")
                else:
                    if out.exists():
                        archive = out.with_name(out.name + ".interrupted_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
                        if not archive.is_relative_to(folder):
                            raise ValueError("Archive target escaped the run folder")
                        out.rename(archive)
                    summary(folder, rows, {"state": "training", "current_variant": variant, "current_seed": seed,
                                           "history": str(out / "history.jsonl")})
                    print(f"START {variant} seed{seed} {now()}", flush=True)
                    if variant == "G0" and plan["reuse_verified_baselines"]:
                        row = reuse_baseline(reference, seed, cfg, data, out, settings)
                    else:
                        job_cfg = {**cfg, "epochs": plan["correction_epochs"] if variant.startswith("PN") else cfg["epochs"]}
                        manager = manager_for(variant, seed, job_cfg, settings)
                        adapter = get_model_adapter(manager.config.model_name)
                        row = lab.run_one(job_cfg, variant, seed, data, out,
                                          model_adapter=adapter, model_config=manager)
                        row.update(settings=settings, reused=False)
                    row["checkpoint_sha256"] = lab.sha256(row["checkpoint"])
                    write(out / "result.json", row)
                rows.append(row)
                summary(folder, rows, {"state": "training", "last_completed": f"{variant}_seed{seed}"})
                gc.collect(); torch.cuda.empty_cache()
        if source_hashes() != sources:
            raise RuntimeError("Sources changed during the run; refusing final evaluation")
        base_predictions = {}
        for seed in plan["seeds"]:
            with (folder / "Houston2013" / f"G0_seed{seed}" / "validation_predictions.csv").open(encoding="utf-8-sig") as stream:
                base_predictions[seed] = [int(x["predicted_class_1based"]) - 1 for x in csv.DictReader(stream)]
        summary(folder, rows, {"state": "final_evaluation", "grid_frozen_at": now()})
        for row in rows:
            net = load_net(row, cfg, torch.device(cfg["device"]))
            row["diagnostics"] = diagnostics(net, row, data, cfg, base_predictions)
            if plan["final_test"] and row.get("test") is None:
                loader = DataLoader(lab.FusionPatchDataset(data, "test", cfg["window_size"]), batch_size=cfg["batch_size"])
                metrics, ys, preds = lab.evaluate(net, loader, torch.device(cfg["device"]), data["info"]["classes"])
                row["test"] = metrics
                path = Path(row["checkpoint"]).with_name("test_predictions.csv")
                with path.open("w", newline="", encoding="utf-8-sig") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["row", "column", "true_class_1based", "predicted_class_1based"])
                    writer.writerows((int(r), int(c), y+1, p+1) for (r,c),y,p in zip(data["test"],ys,preds))
            write(Path(row["checkpoint"]).with_name("result.json"), row)
            del net; gc.collect(); torch.cuda.empty_cache()
            summary(folder, rows, {"state": "final_evaluation", "last_evaluated": f"{row['variant']}_seed{row['seed']}"})
        summary(folder, rows, {"state": "completed", "finished_at": now()})
        return 0
    except BaseException as error:
        (folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        summary(folder, rows, {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": str(error)})
        raise
    finally:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()
