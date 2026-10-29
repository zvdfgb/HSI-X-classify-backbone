"""Prepare the approved SR batch and read native reports; never train/evaluate.

The lab owns data preprocessing, model interfaces, training and metrics.
This helper only audits reuse, measures an input-only SR constant, writes
native configurations, checks synthetic interfaces, and reads saved reports.
"""
import argparse
import gc
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_metrics(path):
    text = Path(path).read_bytes().decode("utf-8", errors="replace")
    result = {}
    for key, label in (("oa", "Overall"), ("aa", "Average"), ("kappa", "Kappa")):
        matches = re.findall(rf"(?m)^([\d.eE+-]+) {label} accuracy \(%\)", text)
        if len(matches) != 1:
            raise ValueError(f"Expected one native {label} result: {path}")
        result[key] = float(matches[0])
    if not re.search(r"(?m)^\s*accuracy\s+[\d.]+\s+12197\s*$", text):
        raise ValueError(f"Native report must cover 12197 samples: {path}")
    return result


def audit_reuse():
    base = ROOT / "docs/results/weekend_20261024_25"
    index = {v["id"]: v for v in read(base / "evidence/index.json")}
    old = read(base / "legacy_fusion/manifest.json")
    kd = read(base / "evidence/kd_manifest.json")
    seal = read(ROOT / "report/legacy_fusion_comparison/lab_fusion_20261025_133847/最终验收封存.json")
    expected_checkpoints = {"F0": seal["runs"][0]["checkpoint_sha256"],
                            "T1": "0f1b9e9b46d4e7340a9bb0c50ef0c682b6f958d96be050e02976372ed42d9797"}
    # These raw-file hashes predate BOTH native runs. No old split is reused.
    data_manifest = read(ROOT / "report/fusion_ablation/fusion_20261024_210925_749584/Houston2013/data_manifest.json")
    data_hashes = {}
    for item in data_manifest["files"].values():
        path = Path(item["path"])
        digest = sha(path)
        if digest != item["sha256"]:
            raise RuntimeError(f"Historical input changed: {path}")
        data_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = digest
    from src import config
    from src.model_registry import get_model_adapter
    audit = {}
    common = ["src/data.py", "src/evaluator.py", "src/metrics.py", "scripts/train.py", "src/runner.py"]
    for key, model, manifest in (("F0", "TransMixer_B0_Decoupled", old), ("T1", "RSCNet", kd)):
        info = index[key]
        checkpoint = Path(info["original_checkpoint"])
        report = base / info["report"]
        cfg = read(base / info["config"])
        if sha(checkpoint) != expected_checkpoints[key] or sha(report) != info["report_sha256"]:
            raise RuntimeError(f"Historical checkpoint/report hash mismatch: {key}")
        defaults = read(ROOT / "configs/train.json")
        for name in ("dataset_type", "cuda_device", "learning_rate", "epochs", "batch_size", "num_workers", "random_seed", "channels", "window_size", "enable_testing", "enable_visualization", "enable_tsne", "enable_erf"):
            if cfg[name] != defaults[name]:
                raise RuntimeError(f"Historical native config mismatch: {key}/{name}")
        if cfg["model_name"] != model or cfg.get("distillation") is not None:
            raise RuntimeError(f"Wrong historical model: {key}")
        hashes = {p.replace("\\", "/"): v for p, v in manifest["source_sha256"].items()}
        folder = "transmixer_net" if key == "F0" else "rscnet"
        for rel in common + [f"src/models/{folder}/model.py", f"src/models/{folder}/adapter.py"]:
            if sha(ROOT / rel) != hashes[rel]:
                raise RuntimeError(f"Historical native source changed: {key}/{rel}")
        trainer = (ROOT / "src/trainer.py").read_text(encoding="utf-8")
        compatibility = "byte-identical"
        if sha(ROOT / "src/trainer.py") != hashes["src/trainer.py"]:
            # Remove only the two optional callbacks; verify all remaining bytes
            # against the sealed original trainer. Default adapters have neither.
            trainer = trainer.replace('            compute_loss = getattr(model_adapter, "compute_loss", None)\n            loss = (compute_loss(model_bundle, outputs, batch, criterion)\n                    if compute_loss is not None else criterion(outputs, batch["label"]))\n', '            loss = criterion(outputs, batch["label"])\n')
            trainer = trainer.replace('    on_training_complete = getattr(model_adapter, "on_training_complete", None)\n    if on_training_complete is not None:\n        on_training_complete(model_bundle)\n\n', '')
            digest = hashlib.sha256(trainer.replace("\n", "\r\n").encode()).hexdigest()
            adapter = get_model_adapter(model)
            if digest != hashes["src/trainer.py"] or any(hasattr(adapter, n) for n in ("compute_loss", "on_training_complete")):
                raise RuntimeError("Historical trainer differs beyond inactive optional callbacks")
            compatibility = "default CE path identical; only inactive optional callbacks were added"
        if Path(manifest["python"]).resolve() != Path(sys.executable).resolve():
            raise RuntimeError("Historical Python interpreter changed")
        audit[key] = {"accepted": True, "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                      "report": str(report), "report_sha256": sha(report), "config": cfg,
                      "trainer_compatibility": compatibility, "metrics": report_metrics(report)}
    return audit, data_hashes


def prepare(tag):
    import numpy as np
    import torch
    from src import config
    from src.config import ConfigManager, ExperimentConfig
    from src.data import getMyData, set_random_seed
    from src.model_registry import get_model_adapter

    plan = read(ROOT / "docs/experiments/20261029_SR原流程_实验矩阵.json")
    if not plan["approved"] or plan["task_count"] != 27 or plan["protocol"]["seeds"] != [6, 7, 8]:
        raise RuntimeError("Approved fixed matrix required")
    if not torch.cuda.is_available() or str(torch.__version__) != "2.6.0+cu124" or np.__version__ != "2.3.5":
        raise RuntimeError("The existing CUDA/Torch/NumPy environment is required")
    destination = ROOT / "report/sr_native" / tag
    if destination.exists():
        raise FileExistsError(f"Batch already exists; refusing overwrite: {destination}")
    audit, data_hashes = audit_reuse()
    destination.mkdir(parents=True)
    write(destination / "reuse_audit.json", audit)
    write(destination / "data_sha256.json", data_hashes)
    defaults = read(ROOT / "configs/train.json")
    scales = {}
    for seed in (6, 7, 8):
        set_random_seed(seed)
        loaders = getMyData(0, 30, 11, 128, 0)  # All preprocessing belongs to the lab.
        dataset = loaders[0].dataset
        if len(dataset) != 2832 or len(loaders[1].dataset) != 12197 or not loaders[0].drop_last:
            raise RuntimeError("Original lab split/loader mismatch")
        previous = dataset.train
        distances = []
        dataset.train = False
        try:
            # Index the ORIGINAL dataset, including the last16 inputs; no new loader.
            for start in range(0, len(dataset), 128):
                h = torch.stack([dataset[i][0][0] for i in range(start, min(start + 128, len(dataset)))])
                d = (h - h[:, :, 5:6, 5:6]).square().mean(1).flatten(1)
                mask = torch.arange(121) != 60
                distances.append(d[:, mask].numpy().reshape(-1))
            scales[str(seed)] = {"spectral_scale": max(float(np.median(np.concatenate(distances))), 1e-8),
                                 "samples": len(dataset), "source": "native HXDataset training inputs; no labels"}
        finally:
            dataset.train = previous
        print(f"SR scale seed{seed}: {scales[str(seed)]}", flush=True)
        del loaders, dataset, distances, h, d
        gc.collect()
    write(destination / "training_input_statistics.json", scales)

    jobs, preflight = [], []
    config_dir = ROOT / "configs/runs" / tag
    for group in plan["groups"]:
        for seed in (6, 7, 8):
            cfg = {**defaults, "experiment_name": f"{tag}_{group['id']}_seed{seed}",
                   "model_name": group["model"], "random_seed": seed,
                   "ablation_mode": group.get("ablation_mode", "decoupled")}
            if group["id"] in ("M2", "M3", "M4"):
                cfg["sr"] = {"spatial_sigma": 3.0}
                if group["id"] != "M4":
                    cfg["sr"]["spectral_scale"] = scales[str(seed)]["spectral_scale"]
            exp = ExperimentConfig(**cfg)
            config_path = config_dir / f"{group['id']}_seed{seed}.json"
            write(config_path, cfg)
            reuse = "F0" if (group["id"], seed) == ("M1", 6) else "T1" if (group["id"], seed) == ("C4", 6) else None
            job = {"id": f"{group['id']}_seed{seed}", "group": group["id"], "seed": seed,
                   "model": group["model"], "config": str(config_path), "reused": bool(reuse),
                   "checkpoint": audit[reuse]["checkpoint"] if reuse else exp.model_save_path,
                   "report": audit[reuse]["report"] if reuse else exp.report_path,
                   "native_log": exp.log_path, "stdout": str(ROOT / "log/sr_native" / tag / f"{group['id']}_seed{seed}.stdout.log"),
                   "stderr": str(ROOT / "log/sr_native" / tag / f"{group['id']}_seed{seed}.stderr.log")}
            if reuse:
                job["checkpoint_sha256"] = audit[reuse]["checkpoint_sha256"]
            else:
                for target in (exp.model_save_path, exp.report_path, exp.log_path):
                    if Path(target).exists():
                        raise FileExistsError(f"Refusing to overwrite output: {target}")
            jobs.append(job)
            if seed == 6:
                set_random_seed(seed)
                config._global_config_manager = ConfigManager(exp)
                adapter = get_model_adapter(exp.model_name)
                bundle = adapter.build_model(config, 0, "cuda:0")
                net = bundle["net"].to("cuda:0")
                net.train()
                batch = {"hsi_pca": torch.randn(128, 1, 30, 11, 11, device="cuda:0"),
                         "hsi": torch.randn(128, 1, 144, 11, 11, device="cuda:0"),
                         "aux": torch.randn(128, 1, 11, 11, device="cuda:0"),
                         "label": torch.arange(128, device="cuda:0") % 15}
                logits = adapter.forward_train(bundle, batch)
                if logits.shape != (128, 15):
                    raise RuntimeError(f"Native model output shape mismatch: {job['id']}")
                loss = torch.nn.functional.cross_entropy(logits, batch["label"])
                loss.backward()  # Synthetic interface test only; no optimizer or research data.
                if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                    raise RuntimeError(f"Nonfinite synthetic gradient: {job['id']}")
                net.eval()
                with torch.no_grad():
                    output = adapter.forward_eval(bundle, batch)
                if not torch.isfinite(output).all():
                    raise RuntimeError(f"Nonfinite synthetic eval: {job['id']}")
                preflight.append({"group": group["id"], "batch_size": 128, "shape": list(output.shape),
                                  "parameters": sum(p.numel() for p in net.parameters()), "passed": True})
                print(f"Synthetic native adapter passed: {group['id']}", flush=True)
                del bundle, net, batch, logits, loss, output
                gc.collect()
                torch.cuda.empty_cache()
    write(destination / "preflight.json", preflight)
    sources = [p for p in (ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts]
    sources += [ROOT / "scripts/train.py", Path(__file__), ROOT / "scripts/run_sr_native.ps1", ROOT / "configs/train.json"]
    sources += [ROOT / "AGENTS.md", ROOT / "docs/experiments/20261029_SR原流程_实验矩阵.json"]
    sources += [destination / "training_input_statistics.json", destination / "reuse_audit.json", destination / "preflight.json"]
    sources += list(config_dir.glob("*.json"))
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sources}
    manifest = {"run_tag": tag, "created_at": datetime.now().astimezone().isoformat(), "python": sys.executable,
                "torch": str(torch.__version__), "numpy": np.__version__, "python_version": platform.python_version(),
                "gpu": torch.cuda.get_device_name(0), "jobs": jobs, "total": 27, "reused": 2, "new_training": 25,
                "source_and_config_sha256": hashes, "data_sha256": data_hashes,
                "training_entry": "scripts/train.py", "checkpoint_selection": "native_test_OA_best"}
    write(destination / "manifest.json", manifest)
    write(destination / "运行状态.json", {"state": "prepared", "total": 27, "reused": 2, "new_completed": 0, "completed": 2})
    summarize(tag)
    print(f"Prepared {tag}: 2 reused + 25 native training jobs; no formal training started.", flush=True)


def summarize(tag):
    import statistics
    folder = ROOT / "report/sr_native" / tag
    manifest = read(folder / "manifest.json")
    values = []
    for job in manifest["jobs"]:
        marker = folder / "completed" / f"{job['id']}.json"
        if job["reused"] or marker.exists():
            values.append({"id": job["id"], "group": job["group"], "seed": job["seed"],
                           "reused": job["reused"], "report": job["report"], **report_metrics(job["report"])})
    paired = []
    table = {(v["group"], v["seed"]): v for v in values}
    plan = read(ROOT / "docs/experiments/20261029_SR原流程_实验矩阵.json")
    for candidate, control in plan["comparisons"]:
        diffs = [{"seed": s, "oa_delta_pp": table[candidate, s]["oa"] - table[control, s]["oa"]}
                 for s in (6, 7, 8) if (candidate, s) in table and (control, s) in table]
        paired.append({"candidate": candidate, "control": control, "n": len(diffs), "differences": diffs})
    write(folder / "summary.json", {"total": 27, "completed": len(values), "runs": values, "paired_comparisons": paired})
    lines = ["# SR原流程实验进度与结果", "", "唯一训练入口：scripts/train.py；保持实验室原数据、训练、选模和指标流程。", "",
             f"已完成 {len(values)}/27（复用 {sum(v['reused'] for v in values)}，新增 {sum(not v['reused'] for v in values)}）。", "",
             "| 组别 | 已有种子 | OA均值 % | 样本标准差 |", "|---|---|---:|---:|"]
    for group in plan["groups"]:
        subset = [v for v in values if v["group"] == group["id"]]
        oa = [v["oa"] for v in subset]
        mean = f"{statistics.mean(oa):.4f}" if oa else "—"
        std = f"{statistics.stdev(oa):.4f}" if len(oa) > 1 else "—"
        lines.append(f"| {group['id']} | {','.join(str(v['seed']) for v in subset) or '待运行'} | {mean} | {std} |")
    lines += ["", "原始报告、逐种子OA/AA/Kappa及配对差值见summary.json。旧验证流程不混入此表。", ""]
    (folder / "实验报告.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "summarize"))
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"sr_native_[0-9]{8}", args.run_tag):
        raise ValueError("Use a dated sr_native_YYYYMMDD batch name")
    (prepare if args.action == "prepare" else summarize)(args.run_tag)
