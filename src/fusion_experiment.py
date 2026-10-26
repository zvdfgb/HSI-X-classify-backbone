"""Validation-selected, paired experiments; legacy train/test behavior is untouched."""
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .models.transmixer_fusion.model import FusionTransMixerNet, VARIANTS

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"Houston2013": ("houston", "lidar", 15), "Berlin": ("berlin", "sar", 8)}
LABELS = {
    "F0": "原始拼接", "F1": "辅助自增强（容量对照）", "F2": "辅助→HSI",
    "F3": "HSI→辅助", "F4": "整体双向交互", "F5": "全局/局部分工交互", "F6": "交换分工对照",
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_settings(cfg):
    required = {"stage", "variants", "datasets", "seeds", "epochs", "batch_size", "learning_rate",
                "weight_decay", "channels", "window_size", "validation_fraction", "split_seed",
                "preprocess_seed", "device", "cpu_threads", "evaluate_test"}
    if set(cfg) != required:
        raise ValueError(f"Config fields: missing={required-set(cfg)}, unknown={set(cfg)-required}")
    if cfg["stage"] not in ("pilot", "confirm"):
        raise ValueError("stage must be pilot or confirm")
    for key, allowed in (("variants", VARIANTS), ("datasets", DATASETS)):
        xs = cfg[key]
        if not isinstance(xs, list) or not xs or len(xs) != len(set(xs)) or any(x not in allowed for x in xs):
            raise ValueError(f"Invalid {key}")
    if not isinstance(cfg["seeds"], list) or not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("seeds must be a nonempty unique list")
    for seed in [*cfg["seeds"], cfg["split_seed"], cfg["preprocess_seed"]]:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Seeds must be integers in [0, 2**32)")
    for key in ("epochs", "batch_size", "channels", "window_size", "cpu_threads"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["window_size"] % 2 != 1:
        raise ValueError("window_size must be odd")
    for key in ("learning_rate", "weight_decay", "validation_fraction"):
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]):
            raise ValueError(f"Invalid {key}")
    if cfg["learning_rate"] <= 0 or cfg["weight_decay"] < 0 or not 0 < cfg["validation_fraction"] < 1:
        raise ValueError("Invalid optimizer or split settings")
    if type(cfg["evaluate_test"]) is not bool or (cfg["stage"] == "pilot" and cfg["evaluate_test"]):
        raise ValueError("Pilot must not evaluate the test set")
    device = torch.device(cfg["device"])
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")


def dataset_files(name):
    prefix, auxiliary, _ = DATASETS[name]
    result = {}
    for kind in ("hsi", auxiliary, "gt", "index"):
        filename = f"{prefix}_{kind}.mat"
        candidates = [ROOT / "data" / name / filename, ROOT / "data" / name / "data" / filename,
                      ROOT.parent / "data" / name / filename, ROOT.parent / "data" / name / "data" / filename]
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            raise FileNotFoundError(f"Missing {name}: {filename}; expected {candidates[0]}")
        result[kind] = found
    return result


def stratified_positions(official_train, gt, fraction, seed):
    positions = np.asarray(official_train, dtype=np.int64)
    labels = gt[positions[:, 0], positions[:, 1]]
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) < 2:
            raise ValueError(f"Class {label} needs at least two training centers")
        count = min(len(indices) - 1, max(1, int(round(len(indices) * fraction))))
        val.extend(indices[:count]); train.extend(indices[count:])
    return positions[np.asarray(train)], positions[np.asarray(val)]


def check_positions(train, test, gt, classes):
    for name, positions in (("official_train", train), ("official_test", test)):
        if positions.ndim != 2 or positions.shape[1] != 2 or not len(positions):
            raise ValueError(f"Invalid {name} shape")
        if not np.issubdtype(positions.dtype, np.integer):
            raise ValueError(f"{name} indices must be integer")
        if (positions < 0).any() or (positions[:, 0] >= gt.shape[0]).any() or (positions[:, 1] >= gt.shape[1]).any():
            raise ValueError(f"{name} indices outside image")
        if len(np.unique(positions, axis=0)) != len(positions):
            raise ValueError(f"Duplicate {name} centers")
        ys = gt[positions[:, 0], positions[:, 1]]
        if (ys < 1).any() or (ys > classes).any():
            raise ValueError(f"Unlabelled/invalid {name} centers")
    if set(map(tuple, train)) & set(map(tuple, test)):
        raise ValueError("Official train/test centers overlap")
    if set(np.unique(gt[train[:, 0], train[:, 1]])) != set(range(1, classes + 1)):
        raise ValueError("Official training set must contain every class")


def standardize(array):
    array = np.asarray(array, dtype=np.float64)
    flat = array.reshape(-1, array.shape[-1])
    mean, std = flat.mean(0), flat.std(0)
    return np.asarray(((flat - mean) / np.where(std > 1e-12, std, 1)).reshape(array.shape), dtype=np.float32)


def prepare_data(name, cfg, folder):
    prefix, aux_name, classes = DATASETS[name]
    files = dataset_files(name)
    gt = np.asarray(loadmat(files["gt"])[f"{prefix}_gt"])
    index = loadmat(files["index"])
    official_train = np.asarray(index[f"{prefix}_train"])
    test = np.asarray(index[f"{prefix}_test"])
    check_positions(official_train, test, gt, classes)
    train, val = stratified_positions(official_train, gt, cfg["validation_fraction"], cfg["split_seed"])
    hsi = loadmat(files["hsi"])[f"{prefix}_hsi"]
    aux = loadmat(files[aux_name])[f"{prefix}_{aux_name}"]
    if aux.ndim == 2:
        aux = aux[..., None]
    if hsi.ndim != 3 or hsi.shape[:2] != gt.shape or aux.shape[:2] != gt.shape:
        raise ValueError("Expected spatially aligned HSI, auxiliary data and GT")
    if cfg["channels"] > hsi.shape[-1]:
        raise ValueError("PCA components exceed available bands")
    if not np.isfinite(hsi).all() or not np.isfinite(aux).all():
        raise ValueError("Non-finite input data")
    print(f"Preparing {name}: PCA on unlabelled full scene, fixed split; no test scoring", flush=True)
    pca = PCA(n_components=cfg["channels"], random_state=cfg["preprocess_seed"])
    features = pca.fit_transform(hsi.reshape(-1, hsi.shape[-1])).reshape(*gt.shape, cfg["channels"])
    features, aux = standardize(features), standardize(aux)
    del hsi
    pad = cfg["window_size"] // 2
    features = np.pad(features, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    aux = np.pad(aux, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "split.npz", train=train, validation=val, test=test)
    def counts(pos):
        return np.bincount(gt[pos[:, 0], pos[:, 1]].astype(int), minlength=classes+1)[1:].tolist()
    info = {
        "dataset": name, "shape": list(gt.shape), "classes": classes, "aux_channels": aux.shape[-1],
        "train": len(train), "validation": len(val), "test": len(test),
        "class_counts": {k: counts(v) for k, v in (("train", train), ("validation", val), ("test", test))},
        "split_sha256": sha256(folder / "split.npz"), "pca_solver": pca._fit_svd_solver,
        "preprocessing": "unlabelled full-scene PCA and standardization (transductive)",
        "spatial_protocol": "disjoint centers; overlapping neighbourhoods possible; same-scene validation",
        "files": {key: {"path": str(p), "sha256": sha256(p)} for key, p in files.items()},
    }
    write_json(folder / "data_manifest.json", info)
    return dict(features=features, aux=aux, gt=gt, train=train, validation=val, test=test, info=info)


class FusionPatchDataset(Dataset):
    def __init__(self, data, split, window, augment=False):
        self.data, self.positions, self.window, self.augment = data, data[split], window, augment

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        row, col = self.positions[i]
        w = self.window
        h = torch.from_numpy(self.data["features"][row:row+w, col:col+w].transpose(2, 0, 1)).unsqueeze(0)
        a = torch.from_numpy(self.data["aux"][row:row+w, col:col+w].transpose(2, 0, 1))
        if self.augment and random.random() < 0.5:
            dimension = random.choice((-1, -2))
            h, a = h.flip(dimension), a.flip(dimension)
        return h, a, int(self.data["gt"][row, col]) - 1


def seed_everything(seed, threads):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # The CUDA memory-efficient attention backward warned about nondeterminism.
    # Use the math backend for every group and fail rather than silently warn.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def classification_metrics(y, pred, classes):
    cm = confusion_matrix(y, pred, labels=np.arange(classes))
    support = cm.sum(1)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(classes, dtype=float), where=support > 0)
    kappa = float(cohen_kappa_score(y, pred, labels=np.arange(classes)))
    if not math.isfinite(kappa):
        raise ValueError("Kappa undefined for this evaluation set")
    return {"oa": float(np.trace(cm) / cm.sum() * 100), "aa": float(recall.mean() * 100),
            "kappa": kappa * 100, "per_class_recall": (recall * 100).tolist(),
            "support": support.tolist(), "confusion_matrix": cm.tolist()}


@torch.no_grad()
def evaluate(net, loader, device, classes):
    net.eval()
    loss_sum, total, ys, preds = 0., 0, [], []
    for h, a, y in loader:
        h, a, y = h.to(device), a.to(device), y.to(device)
        logits = net(h, a)
        loss = nn.functional.cross_entropy(logits, y, reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss")
        loss_sum += loss.item(); total += len(y)
        ys.extend(y.cpu().tolist()); preds.extend(logits.argmax(1).cpu().tolist())
    if not total:
        raise ValueError("Empty evaluation set")
    metrics = classification_metrics(ys, preds, classes)
    return {**metrics, "loss": loss_sum / total}, ys, preds


@torch.no_grad()
def benchmark(net, dataset, device):
    net.eval()
    n = min(32, len(dataset))
    h, a, _ = next(iter(DataLoader(dataset, batch_size=n, shuffle=False)))
    h, a = h.to(device), a.to(device)
    for _ in range(2):
        net(h, a)
    sync(device); start = time.perf_counter()
    for _ in range(5):
        net(h, a)
    sync(device)
    return {"batch_size": n, "warmup": 2, "repeats": 5,
            "forward_ms_per_batch": (time.perf_counter()-start)/5*1000,
            "scope": "model forward only, fixed validation inputs; excludes data transfer"}


def run_one(cfg, variant, seed, data, folder):
    folder.mkdir(parents=True, exist_ok=False)
    seed_everything(seed, cfg["cpu_threads"])
    device = torch.device(cfg["device"])
    info = data["info"]
    model_args = dict(variant=variant, hsi_channels=cfg["channels"], aux_channels=info["aux_channels"],
                      num_classes=info["classes"], hidden_dim=64, num_blocks=2, window_size=cfg["window_size"])
    net = FusionTransMixerNet(**model_args).to(device)
    train_set = FusionPatchDataset(data, "train", cfg["window_size"], augment=True)
    val_set = FusionPatchDataset(data, "validation", cfg["window_size"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, generator=generator,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    parameters = sum(p.numel() for p in net.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best, best_epoch = (-float("inf"), -float("inf")), 0
    start = time.perf_counter()
    history = []
    write_json(folder / "config.json", {**cfg, "variant": variant, "seed": seed, "model_args": model_args})
    for epoch in range(1, cfg["epochs"]+1):
        net.train(); loss_sum, samples = 0., 0
        epoch_start = time.perf_counter()
        for h, a, y in train_loader:
            h, a, y = h.to(device), a.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(net(h, a), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item()*len(y); samples += len(y)
        val, _, _ = evaluate(net, val_loader, device, info["classes"])
        score = (val["oa"], -val["loss"])
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"state_dict": net.state_dict(), "model_args": model_args, "epoch": epoch,
                        "selection_split": "validation", "validation": val}, folder / "best.pth")
        sync(device)
        record = {"epoch": epoch, "train_loss": loss_sum/samples, "train_samples": samples,
                  "validation": val, "seconds": time.perf_counter()-epoch_start,
                  "strengths": net.interaction_strengths(), "best_epoch": best_epoch}
        history.append(record)
        with (folder / "history.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
        print(f"{info['dataset']} {variant} seed{seed} {epoch:03}/{cfg['epochs']} "
              f"loss={record['train_loss']:.4f} valOA={val['oa']:.2f} "
              f"valAA={val['aa']:.2f} best={best_epoch} time={record['seconds']:.1f}s", flush=True)
    training_seconds = time.perf_counter()-start
    peak = torch.cuda.max_memory_allocated(device)/1024**2 if device.type == "cuda" else None
    checkpoint = torch.load(folder / "best.pth", map_location=device, weights_only=True)
    net.load_state_dict(checkpoint["state_dict"])
    val, ys, preds = evaluate(net, val_loader, device, info["classes"])
    with (folder / "validation_predictions.csv").open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.writer(out); writer.writerow(["row", "column", "true_class_1based", "predicted_class_1based"])
        writer.writerows((int(r), int(c), y+1, p+1) for (r, c), y, p in zip(data["validation"], ys, preds))
    result = {"status": "completed", "variant": variant, "label": LABELS[variant], "dataset": info["dataset"],
              "seed": seed, "stage": cfg["stage"], "epochs": cfg["epochs"], "best_epoch": best_epoch,
              "validation": val, "test": None, "selection_split": "validation", "parameters": parameters,
              "training_seconds": training_seconds, "peak_train_memory_mb": peak,
              "strengths": net.interaction_strengths(), "checkpoint": str(folder / "best.pth"),
              "benchmark": benchmark(net, val_set, device), "split_sha256": info["split_sha256"]}
    if cfg["evaluate_test"]:
        test_set = FusionPatchDataset(data, "test", cfg["window_size"])
        test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False)
        result["test"], _, _ = evaluate(net, test_loader, device, info["classes"])
    write_json(folder / "result.json", result)
    return result


def save_summary(folder, cfg, rows):
    write_json(folder / "summary.json", rows)
    fields = ["dataset", "variant", "seed", "status", "best_epoch", "val_oa", "val_aa", "val_kappa",
              "test_oa", "parameters", "training_seconds", "error"]
    with (folder / "summary.csv").open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields); writer.writeheader()
        for row in rows:
            val, test = row.get("validation", {}), row.get("test") or {}
            writer.writerow({**{k: row.get(k, "") for k in fields},
                             "val_oa": val.get("oa", ""), "val_aa": val.get("aa", ""),
                             "val_kappa": val.get("kappa", ""), "test_oa": test.get("oa", "")})
    lines = ["# TransMixer 双模态交互实验结果", "", f"阶段：{cfg['stage']}；每组{cfg['epochs']}轮；种子{cfg['seeds']}。",
             "", "模型均由验证集选取。短程结果只反映早期训练，不能确认最终性能或发表创新。",
             "全图无标签PCA/标准化；同场景分层验证；中心不相交但邻域可能重叠。",
             "旧报告采用不同选模协议，不能直接比较旧OA。", "",
             "| 数据集 | 组 | seed | 状态 | 最佳轮 | 验证OA | 验证AA | 参数量 | 训练秒数 |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        val = row.get("validation", {})
        lines.append(f"| {row['dataset']} | {row['variant']} {LABELS[row['variant']]} | {row['seed']} | "
                     f"{row['status']} | {row.get('best_epoch','')} | {format(val['oa'], '.3f') if val else '—'} | "
                     f"{format(val['aa'], '.3f') if val else '—'} | {row.get('parameters','')} | "
                     f"{format(row['training_seconds'], '.1f') if 'training_seconds' in row else '—'} |")
    lines.extend(["", "## 配对差值与稳定性", "", "以下均为验证OA百分点差；单种子没有标准差证据。"])
    for dataset in cfg["datasets"]:
        for variant in cfg["variants"]:
            diffs = []
            for seed in cfg["seeds"]:
                base = next((r for r in rows if r['dataset']==dataset and r['variant']=='F0' and r['seed']==seed and r['status']=='completed'), None)
                candidate = next((r for r in rows if r['dataset']==dataset and r['variant']==variant and r['seed']==seed and r['status']=='completed'), None)
                if base and candidate:
                    diffs.append(candidate['validation']['oa']-base['validation']['oa'])
            if diffs:
                spread = f"，差值标准差 {np.std(diffs,ddof=1):.3f}" if len(diffs)>1 else "，仅单种子"
                lines.append(f"- {dataset} {variant}：平均差值 {np.mean(diffs):+.3f}{spread}。")
    lines.extend(["", "## 如何解释", "", "- F1对照额外的辅助分支表达能力；F2/F3判断方向，F4判断双向的增量。",
                  "- F5与F6具有相同参数和初始权重，比较两种算子分配；不能仅凭F5对F4的差值证明分工有效。",
                  "- 分工组在真实全局/局部流合并前交互；辅助投影的两组通道不是预先具有物理语义的两类特征。",
                  "- 跨组参数与调用次数存在差异，结果JSON保留实测时延、显存、参数与交互强度。",
                  "- 正式实验固定候选后运行100轮和多个配对种子；Berlin需另备数据。"])
    (folder / "结果解读.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def execute(cfg, folder):
    validate_settings(cfg)
    device = torch.device(cfg["device"])
    if device.type == "cuda" and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
        raise RuntimeError("Requested CUDA device is unavailable; no implicit CPU fallback")
    for dataset in cfg["datasets"]:
        dataset_files(dataset)
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "batch_config.json", cfg)
    sources = [ROOT / "src/models/transmixer_net/model.py", ROOT / "src/models/transmixer_fusion/model.py",
               ROOT / "src/models/transmixer_fusion/adapter.py", ROOT / "src/fusion_experiment.py",
               ROOT / "scripts/train_transmixer_fusion.py", ROOT / "src/model_registry.py"]
    write_json(folder / "runtime.json", {"torch": str(torch.__version__), "numpy": np.__version__,
               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
               "determinism": "strict deterministic algorithms; SDP math only; TF32 disabled; cuDNN benchmark disabled",
               "test_evaluation_enabled": cfg["evaluate_test"], "selection": "validation OA, tie-break lower validation loss"})
    rows = []
    for dataset in cfg["datasets"]:
        data = prepare_data(dataset, cfg, folder / dataset)
        for seed in cfg["seeds"]:
            for variant in cfg["variants"]:
                row = {"variant": variant, "dataset": dataset, "seed": seed, "status": "running"}
                rows.append(row); save_summary(folder, cfg, rows)
                job_folder = folder / dataset / f"{variant}_seed{seed}"
                try:
                    row.update(run_one(cfg, variant, seed, data, job_folder))
                except BaseException as exc:
                    row.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
                    if job_folder.exists():
                        (job_folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    save_summary(folder, cfg, rows)
                    if not isinstance(exc, Exception):
                        raise
                    print(f"FAILED {dataset} {variant} seed{seed}: {exc}", flush=True)
                finally:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                save_summary(folder, cfg, rows)
        del data
    return 0 if all(r["status"] == "completed" for r in rows) else 1
