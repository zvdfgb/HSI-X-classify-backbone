"""Audit saved confirmation artifacts and summarize paired seeds; no inference."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import torch


LABELS = ["健康草地", "受胁迫草地", "人造草地", "树木", "土壤", "水体", "居民区",
          "商业区", "道路", "高速公路", "铁路", "停车场1", "停车场2", "网球场", "跑道"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def avg(xs):
    return statistics.mean(xs)


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else None


def fmt_sd(xs):
    spread = sd(xs)
    return f"{avg(xs):.3f} ± {spread:.3f}" if spread is not None else f"{avg(xs):.3f}（仅一次）"


def verify_metric(metric, size):
    cm = np.asarray(metric["confusion_matrix"], dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or cm.sum() != size or (cm < 0).any():
        raise ValueError("Invalid saved confusion matrix")
    support = cm.sum(1)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(len(cm)), where=support > 0)
    oa = np.trace(cm) / size
    chance = np.dot(cm.sum(0).astype(float), support.astype(float)) / size**2
    expected = {"oa": oa*100, "aa": recall.mean()*100, "kappa": (oa-chance)/(1-chance)*100}
    for key, value in expected.items():
        if not math.isclose(metric[key], value, abs_tol=1e-8):
            raise ValueError(f"Saved metric mismatch: {key}")
    if not np.array_equal(metric["support"], support) or not np.allclose(metric["per_class_recall"], recall*100):
        raise ValueError("Per-class support/recall mismatch")


def analyze_confirmation(folder, cfg, rows):
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    expected = {(d, v, s) for d in cfg["datasets"] for v in cfg["variants"] for s in cfg["seeds"]}
    keys = {(r["dataset"], r["variant"], r["seed"]) for r in rows}
    if len(rows) != len(expected) or keys != expected or any(r["status"] != "completed" for r in rows):
        raise ValueError("Incomplete/duplicate confirmation jobs")
    if not cfg["evaluate_test"]:
        raise ValueError("Confirmation summary requires saved final test metrics")
    runtime = read(folder / "runtime.json")
    fingerprints = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
                    for name, digest in runtime["source_sha256"].items()}
    histories, manifests, spatial = {}, {}, {}
    for dataset in cfg["datasets"]:
        info = manifests[dataset] = read(folder / dataset / "data_manifest.json")
        if hashlib.sha256((folder / dataset / "split.npz").read_bytes()).hexdigest() != info["split_sha256"]:
            raise ValueError("Saved split fingerprint mismatch")
        with np.load(folder / dataset / "split.npz") as split:
            centers = {k: np.asarray(split[k]) for k in ("train", "validation", "test")}
        center_sets = {k: set(map(tuple, value)) for k, value in centers.items()}
        if any(len(center_sets[k]) != len(centers[k]) for k in centers):
            raise ValueError("Duplicate centers")
        if center_sets["train"] & center_sets["validation"] or center_sets["train"] & center_sets["test"] or center_sets["validation"] & center_sets["test"]:
            raise ValueError("Center splits overlap")
        if any(len(centers[k]) != info[k] for k in centers):
            raise ValueError("Manifest center counts mismatch")
        tree = cKDTree(centers["train"])
        spatial[dataset] = {}
        for split_name in ("validation", "test"):
            distance, _ = tree.query(centers[split_name], p=np.inf)
            spatial[dataset][split_name] = {
                "median_nearest_chebyshev_distance": float(np.median(distance)),
                "within_training_window_pct": float(np.mean(distance <= cfg["window_size"]//2)*100),
                "overlapping_training_window_pct": float(np.mean(distance <= cfg["window_size"]-1)*100),
            }
    for row in rows:
        dataset, variant, seed = row["dataset"], row["variant"], row["seed"]
        info = manifests[dataset]
        job = folder / dataset / f"{variant}_seed{seed}"
        if read(job / "result.json") != row or row["selection_split"] != "validation":
            raise ValueError(f"Result metadata mismatch: {job}")
        if row["split_sha256"] != info["split_sha256"] or row["epochs"] != cfg["epochs"]:
            raise ValueError("Split/epoch mismatch")
        hist = [json.loads(line) for line in (job / "history.jsonl").read_text(encoding="utf-8").splitlines()]
        histories[dataset, variant, seed] = hist
        if [h["epoch"] for h in hist] != list(range(1, cfg["epochs"]+1)) or any(h["train_samples"] != info["train"] or not math.isfinite(h["train_loss"]) or not all(math.isfinite(h["validation"][k]) for k in ("oa", "aa", "kappa", "loss")) for h in hist):
            raise ValueError(f"Incomplete/non-finite training history: {job}")
        best = max(hist, key=lambda h: (h["validation"]["oa"], -h["validation"]["loss"]))
        if best["epoch"] != row["best_epoch"] or not math.isclose(best["validation"]["oa"], row["validation"]["oa"], abs_tol=1e-8):
            raise ValueError("Checkpoint selection does not match validation history")
        if not (job / "best.pth").is_file():
            raise ValueError("Missing saved checkpoint")
        checkpoint = torch.load(job / "best.pth", map_location="cpu", weights_only=True)
        if checkpoint["epoch"] != row["best_epoch"] or checkpoint["selection_split"] != "validation" or checkpoint["validation"] != row["validation"]:
            raise ValueError("Checkpoint metadata differs from saved result")
        if checkpoint["model_args"] != read(job / "config.json")["model_args"] or any(not torch.isfinite(tensor).all() for tensor in checkpoint["state_dict"].values()):
            raise ValueError("Checkpoint model arguments or finite-weight check failed")
        del checkpoint
        verify_metric(row["validation"], info["validation"])
        verify_metric(row["test"], info["test"])
        with (job / "validation_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        correct = sum(r["true_class_1based"] == r["predicted_class_1based"] for r in predictions)
        if len(predictions) != info["validation"] or not math.isclose(correct/len(predictions)*100, row["validation"]["oa"], abs_tol=1e-8):
            raise ValueError("Validation prediction/score mismatch")
    audit = {"completed_jobs": len(rows), "epochs_per_job": cfg["epochs"],
             "all_histories_complete": True, "all_centers_disjoint": True,
             "checkpoint_selection_matches_validation_history": True,
             "checkpoint_metadata_matches_results_and_weights_are_finite": True,
             "saved_confusion_matrices_match_reported_metrics": True,
             "validation_predictions_match_reported_metrics": True,
             "training_source_sha256_matches": fingerprints,
             "spatial_neighborhood_audit": spatial,
             "scope": "Checks saved artifacts and current source; does not rerun inference or establish how many historical invocations occurred."}
    write(folder / "正式验收核对.json", audit)
    aggregates, pairs = [], []
    lines = ["# TransMixer 双模态交互：正式实验复盘", "",
             f"批次：`{folder.name}`；每组{cfg['epochs']}轮，种子{cfg['seeds']}，共{len(rows)}次训练全部完成。",
             "使用保存的最终测试指标汇总，没有重新训练、重新选checkpoint或重复测试推理。",
             "checkpoint按验证OA选取，同分时按较低验证交叉熵选取。±为训练种子的样本标准差，不是置信区间。", ""]
    for dataset in cfg["datasets"]:
        info = manifests[dataset]
        jobs = {(r["variant"], r["seed"]): r for r in rows if r["dataset"] == dataset}
        variants, seeds = cfg["variants"], cfg["seeds"]
        if "F0" not in variants:
            raise ValueError("Paired baseline F0 is required")
        lines.extend([f"## {dataset} 测试结果", "",
                      f"训练{info['train']}、验证{info['validation']}、官方测试{info['test']}个中心。", "",
                      "| 组 | 方式 | " + " | ".join(f"seed{s} OA%" for s in seeds) + " | 测试OA均值±SD | 相对F0平均百分点 | 配对差值SD | 胜/平/负 | 参数 |",
                      "|---|---|" + "---:|"*len(seeds) + "---:|---:|---:|---:|---:|"])
        for variant in variants:
            rs = [jobs[variant, s] for s in seeds]
            oa = [r["test"]["oa"] for r in rs]
            delta = [jobs[variant, s]["test"]["oa"]-jobs["F0", s]["test"]["oa"] for s in seeds]
            wins, ties = sum(x > 1e-10 for x in delta), sum(abs(x) <= 1e-10 for x in delta)
            aggregate = {"dataset": dataset, "variant": variant, "label": rs[0]["label"], "n": len(rs),
                         "seeds": seeds, "test_oa": oa, "test_oa_mean": avg(oa), "test_oa_sample_sd": sd(oa),
                         "paired_delta_vs_F0": delta, "paired_delta_mean": avg(delta), "paired_delta_sample_sd": sd(delta),
                         "wins": wins, "ties": ties, "losses": len(delta)-wins-ties,
                         "test_aa_mean": avg([r["test"]["aa"] for r in rs]),
                         "test_kappa_mean": avg([r["test"]["kappa"] for r in rs]),
                         "parameters": rs[0]["parameters"], "training_seconds_mean": avg([r["training_seconds"] for r in rs]),
                         "test_per_class_recall_mean": np.mean([r["test"]["per_class_recall"] for r in rs], axis=0).tolist()}
            aggregates.append(aggregate)
            spread = f"{sd(delta):.3f}" if sd(delta) is not None else "—"
            lines.append(f"| {variant} | {rs[0]['label']} | " + " | ".join(f"{x:.3f}" for x in oa) +
                         f" | {fmt_sd(oa)} | {avg(delta):+.3f} | {spread} | {wins}/{ties}/{len(delta)-wins-ties} | {rs[0]['parameters']:,} |")
        lines.extend(["", "### 机制对照：不能只看谁的均值最高", "",
                      "| 比较 | 问题 | 各种子测试OA差值（百分点） | 平均±差值SD | 正差次数 |",
                      "|---|---|---|---:|---:|"])
        comparisons = [("F4", "F1", "双向交互 vs 辅助自增强容量对照"), ("F4", "F2", "双向 vs 辅助指导HSI"),
                       ("F4", "F3", "双向 vs HSI指导辅助"), ("F4", "F5", "整体双向 vs 分工双向"),
                       ("F5", "F6", "分工匹配 vs 交换分工"), ("F6", "F4", "均值最高的对照 vs 整体双向")]
        for left, right, question in comparisons:
            if left not in variants or right not in variants:
                continue
            delta = [jobs[left, s]["test"]["oa"]-jobs[right, s]["test"]["oa"] for s in seeds]
            pair = dict(dataset=dataset, left=left, right=right, deltas=delta, mean=avg(delta), sample_sd=sd(delta), wins=sum(d > 0 for d in delta))
            pairs.append(pair)
            lines.append(f"| {left}−{right} | {question} | " + "、".join(f"{d:+.3f}" for d in delta) + f" | {fmt_sd(delta)} | {pair['wins']}/{len(seeds)} |")
        lines.extend(["", "### 如何用于你的科研", ""])
        dataset_pairs = {(p["left"], p["right"]): p for p in pairs if p["dataset"] == dataset}
        dataset_aggregates = {a["variant"]: a for a in aggregates if a["dataset"] == dataset}
        if "F4" in dataset_aggregates:
            a = dataset_aggregates["F4"]
            lines.append(f"- F4相对F0平均{a['paired_delta_mean']:+.3f}个百分点，在{a['wins']}/{len(seeds)}个种子上提高。配对方向是否一致比只看某次最高分更有解释力，但少量种子不能证明统计显著。")
        if ("F4", "F1") in dataset_pairs:
            p = dataset_pairs["F4", "F1"]
            lines.append(f"- F4相对辅助自增强F1平均{p['mean']:+.3f}个百分点，在{p['wins']}/{len(seeds)}个种子上提高。需要超过这个容量对照，才能更有把握把收益归因于跨模态交流；目前不宜直接下结论。")
        if ("F5", "F6") in dataset_pairs:
            p = dataset_pairs["F5", "F6"]
            verdict = "本批不支持分工匹配优于交换分工。" if p["mean"] <= 0 else "均值为正，还需结合种子一致性及额外数据确认分工匹配是否有效。"
            lines.append(f"- F5相对F6平均{p['mean']:+.3f}个百分点，在{p['wins']}/{len(seeds)}个种子上提高。{verdict}")
        if ("F4", "F5") in dataset_pairs:
            p = dataset_pairs["F4", "F5"]
            lines.append(f"- 整体双向F4相对分工双向F5平均{p['mean']:+.3f}个百分点，{p['wins']}/{len(seeds)}个种子为正；没有依据因为结构更复杂就优先采用F5。")
        highest = max(dataset_aggregates.values(), key=lambda a: a["test_oa_mean"])
        lines.append(f"- 测试均值最高的是{highest['variant']}（{highest['test_oa_mean']:.3f}%）；这只是本批描述性排名，不能仅凭最高均值认定它稳定最好。交换分工对照若更高，也不能替匹配分工组证明设计有效。")
        lines.extend(["- 10轮的短程趋势不能替代完整对照；不要根据已看过的测试成绩反复调模块。", "",
                      "以上解释仅针对本批数据和预设种子；更换划分或候选必须重新核对，不应当作普遍结论。", ""])
        # Keep the strongest caveat near the conclusions, supported by saved indices.
        valid = [r["validation"]["oa"] for r in jobs.values()]
        sp = spatial[dataset]
        lines.extend(["### 为什么验证集接近100%，测试集却明显低", "",
                      f"{sum(abs(v-100)<1e-8 for v in valid)}/{len(valid)}个最佳checkpoint的验证OA达到100%；验证集已明显饱和。",
                      f"空间索引核对：{sp['validation']['within_training_window_pct']:.2f}%的验证中心落在至少一个训练中心的{cfg['window_size']}×{cfg['window_size']}窗口内；测试中心对应比例仅{sp['test']['within_training_window_pct']:.2f}%。",
                      f"与训练窗口有交集的验证窗口比例为{sp['validation']['overlapping_training_window_pct']:.2f}%，测试窗口为{sp['test']['overlapping_training_window_pct']:.2f}%；最近训练中心的切比雪夫距离中位数分别为{sp['validation']['median_nearest_chebyshev_distance']:.1f}和{sp['test']['median_nearest_chebyshev_distance']:.1f}像素。",
                      "训练、验证、官方测试中心确实互不重复；邻域重叠不等于标签泄漏，但说明本批验证更像在熟悉区域附近判断，可能高估对较远区域的泛化。不能仅凭这些数字判定代码有错误。",
                      "全图无标签PCA/标准化还属于传导式预处理。当前排名必须与这个同场景协议一起报告。",
                      "验证OA都相同时，checkpoint主要由验证交叉熵决定。它与官方测试域的关系仍不确定，不能根据测试成绩重选epoch。", ""])
        lines.extend(["### 测试逐类召回率（三种子均值）", "",
                      "| 类别 | " + " | ".join(variants) + " | F4−F0百分点 |",
                      "|---|" + "---:|"*(len(variants)+1)])
        for c in range(info["classes"]):
            recalls = {v: avg([jobs[v, s]["test"]["per_class_recall"][c] for s in seeds]) for v in variants}
            label = f"{c+1} {LABELS[c]}" if dataset == "Houston2013" else str(c+1)
            difference = f"{recalls['F4']-recalls['F0']:+.3f}" if "F4" in recalls else "—"
            lines.append(f"| {label} | " + " | ".join(f"{recalls[v]:.3f}" for v in variants) + f" | {difference} |")
        # Paired plots expose seed sensitivity instead of treating seeds as datasets.
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
        colors = plt.get_cmap("tab10")
        for i, seed in enumerate(seeds):
            absolute = [jobs[v, seed]["test"]["oa"] for v in variants]
            difference = [value-jobs["F0", seed]["test"]["oa"] for value in absolute]
            axes[0].plot(variants, absolute, marker="o", label=f"seed {seed}", color=colors(i))
            axes[1].plot(variants, difference, marker="o", label=f"seed {seed}", color=colors(i))
        axes[0].set(title="Saved test OA by training seed", ylabel="Test OA (%)")
        axes[1].set(title="Paired change against the same-seed F0", ylabel="OA change (percentage points)")
        axes[1].axhline(0, color="black", linewidth=.8)
        for ax in axes:
            ax.grid(alpha=.2); ax.legend()
        figure = f"{dataset}_test_paired_seeds.png"
        fig.savefig(folder / figure, dpi=180); plt.close(fig)
        lines.extend(["", f"![测试结果与配对差值]({figure})", ""])
    hours = sum(r["training_seconds"] for r in rows)/3600
    lines.extend(["## 下一步实验应先改什么", "",
                  "1. 先固定F0、F1、F4这组最简单且有解释力的比较，验证交互是否超过辅助自增强；若要控制两分支同时增强，再补参数/计算预算尽量接近的双分支自增强对照。",
                  "2. 从官方训练区域构造空间分块验证，并设置与11×11图块相容的隔离带；先检查是否仍有足够类别与训练样本。不要把当前官方测试集挪来调验证方案。",
                  "3. 固定新协议与超参数，用验证集筛查并预先确定下一次测试安排；不要根据已看过的测试成绩反复调模块。可用更强空间留出或独立场景确认泛化。",
                  "4. 补HSI单模态、LiDAR单模态诊断，判断究竟哪个分支贡献不足；当前实验还没有回答这个问题。",
                  "5. 暂缓围绕F5堆叠蒸馏、原型或超图。现有证据优先指向验证协议和基础对照，不宜直接包装复杂创新。", "",
                  "## 验收与证据边界", "",
                  f"共{len(rows)}条完整{cfg['epochs']}轮日志；每轮训练样本数与清单相符，验证选模轮次与日志相符，混淆矩阵可重算OA/AA/Kappa，验证预测文件与分数相符。",
                  f"当前源码指纹与训练时记录{'全部相同' if all(fingerprints.values()) else '存在不同，见正式验收核对.json'}。",
                  f"模型训练时长合计{hours:.2f}小时，不含数据处理、最终测试和本次汇总。",
                  "日志与当前源码支持上述流程核对；不能据此证明过去绝没有额外运行过任何代码。没有重新读取测试影像或重跑模型。",
                  "仅Houston2013的HSI＋LiDAR；没有Berlin/SAR训练结果，也不能直接与旧协议下约88.97%的历史单种子结果比较。", ""])
    write(folder / "正式汇总.json", {"aggregates": aggregates, "paired_comparisons": pairs})
    fields = ["dataset", "variant", "label", "n", "test_oa_mean", "test_oa_sample_sd", "paired_delta_mean",
              "paired_delta_sample_sd", "wins", "ties", "losses", "test_aa_mean", "test_kappa_mean", "parameters", "training_seconds_mean"]
    with (folder / "正式汇总.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(aggregates)
    report = folder / "正式实验复盘.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)
    return report
