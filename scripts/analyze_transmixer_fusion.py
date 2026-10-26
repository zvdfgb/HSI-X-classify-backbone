"""Plot saved fusion results without model inference or test-set access."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same label order as src/metrics.py; Chinese translations for the local guide.
HOUSTON_LABELS = ["健康草地", "受胁迫草地", "人造草地", "树木", "土壤", "水体", "居民区",
                  "商业区", "道路", "高速公路", "铁路", "停车场1", "停车场2", "网球场", "跑道"]

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(folder):
    cfg = read_json(folder / "batch_config.json")
    rows = read_json(folder / "summary.json")
    expected = len(cfg["datasets"]) * len(cfg["seeds"]) * len(cfg["variants"])
    if len(rows) != expected or any(row["status"] != "completed" for row in rows):
        raise ValueError("Analyze only a complete batch; incomplete/failed rows are not treated as scores")
    if cfg["stage"] == "confirm":
        from summarize_fusion_confirmation import analyze_confirmation
        return analyze_confirmation(folder, cfg, rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"], row["seed"]].append(row)
    lines = ["# 双模态交互：实测观察", "", "以下数值来自本目录保存的结果，不重新推理模型。",
             f"阶段：{cfg['stage']}；每组{cfg['epochs']}轮；训练种子：{cfg['seeds']}。",
             "所有checkpoint按验证集选取。10轮单种子观察不能确定最终优劣。", ""]
    for (dataset, seed), jobs in grouped.items():
        jobs.sort(key=lambda row: row["variant"])
        info = read_json(folder / dataset / "data_manifest.json")
        baseline = next((r for r in jobs if r["variant"] == "F0"), None)
        baseline_oa = baseline["validation"]["oa"] if baseline else None
        lines.extend([f"## {dataset} / seed {seed}", "",
                      f"训练{info['train']}中心，验证{info['validation']}中心；每个验证样本对应{100/info['validation']:.4f}个百分点。",
                      f"测试中心{info['test']}个；本批测试评分{'已开启' if cfg['evaluate_test'] else '关闭'}。", "",
                      "| 组 | 验证OA% | 相对F0百分点 | 验证AA% | 最佳轮 | 参数 | 训练秒 | 峰值MiB | 前向ms/32样本 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
        colors = plt.get_cmap("tab10")
        predictions, curves = {}, {}
        for i, row in enumerate(jobs):
            variant = row["variant"]
            history_path = folder / dataset / f"{variant}_seed{seed}" / "history.jsonl"
            history = [json.loads(s) for s in history_path.read_text(encoding="utf-8").splitlines() if s]
            if len(history) != cfg["epochs"] or any(r["train_samples"] != info["train"] for r in history):
                raise ValueError(f"Incomplete history or incorrect training sample count: {history_path}")
            x = [r["epoch"] for r in history]
            curves[variant] = [r["validation"]["oa"] for r in history]
            axes[0].plot(x, [r["validation"]["oa"] for r in history], label=variant, color=colors(i), marker=".")
            axes[2].plot(x, [r["train_loss"] for r in history], label=variant, color=colors(i))
            csv_path = history_path.parent / "validation_predictions.csv"
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                predictions[variant] = list(csv.DictReader(stream))
            correct = sum(p["true_class_1based"] == p["predicted_class_1based"] for p in predictions[variant])
            if len(predictions[variant]) != info["validation"] or abs(correct/info["validation"]*100-row["validation"]["oa"]) > 1e-8:
                raise ValueError(f"Prediction count/score mismatch: {csv_path}")
            if row["split_sha256"] != info["split_sha256"]:
                raise ValueError("Results do not share the recorded data split")
            val = row["validation"]
            delta = f"{val['oa']-baseline_oa:+.3f}" if baseline else "—"
            ms = row["benchmark"]["forward_ms_per_batch"]
            if row["benchmark"]["batch_size"] != 32:
                ms = float("nan")
            peak = row["peak_train_memory_mb"]
            peak_text = f"{peak:.1f}" if peak is not None else "—"
            lines.append(f"| {variant} {row['label']} | {val['oa']:.3f} | {delta} | {val['aa']:.3f} | "
                         f"{row['best_epoch']} | {row['parameters']:,} | {row['training_seconds']:.1f} | {peak_text} | {ms:.2f} |")
        axes[0].set(xlabel="Epoch", ylabel="Validation OA (%)", title="Same split, paired initialization")
        if "F0" in curves:
            for i, (variant, curve) in enumerate(curves.items()):
                axes[1].plot(x, [a-b for a, b in zip(curve, curves["F0"])], label=variant, color=colors(i), marker=".")
        axes[1].axhline(0, color="black", linewidth=.7)
        axes[1].set(xlabel="Epoch", ylabel="OA difference from F0 (pp)", title="Differences are in percentage points")
        axes[2].set(xlabel="Epoch", ylabel="Training cross-entropy", title="Early optimization")
        for ax in axes:
            ax.legend(ncols=4, fontsize=8)
            ax.grid(alpha=.2)
        figure_name = f"{dataset}_seed{seed}_curves.png"
        fig.suptitle(f"{dataset}, seed {seed}, {cfg['epochs']} epochs; validation only")
        fig.savefig(folder / figure_name, dpi=180)
        plt.close(fig)
        lines.extend(["", f"![验证曲线、相对差值与训练损失]({figure_name})", ""])
        by_id = {r["variant"]: r for r in jobs}
        if cfg["stage"] == "pilot" and baseline:
            top = max(jobs, key=lambda r: r["validation"]["oa"])
            gain = top["validation"]["oa"]-baseline_oa
            count = round(gain*info["validation"]/100)
            lines.extend(["### 这批结果怎么读", "",
                          f"最高验证OA为{top['variant']}的{top['validation']['oa']:.3f}%，比原始拼接净多判对{count}个样本（{gain:+.3f}个百分点）。",
                          "目前不足以选定最终交流方式。先完成固定预算的完整训练与多种子确认，再考虑是否继续做复杂机制。",
                          "下列差值只描述本次观察，不代表统计显著性。", ""])
        for left, right, question in (("F4", "F1", "双向交流相对辅助自增强"),
                                      ("F2", "F3", "辅助指导HSI相对反向指导"),
                                      ("F4", "F2", "双向相对辅助指导HSI"),
                                      ("F4", "F3", "双向相对HSI指导辅助"),
                                      ("F5", "F6", "分工匹配相对交换分工")):
            if left in by_id and right in by_id:
                diff = by_id[left]["validation"]["oa"] - by_id[right]["validation"]["oa"]
                count = round(diff * info["validation"] / 100)
                pairs = list(zip(predictions[left], predictions[right]))
                for a, b in pairs:
                    if any(a[k] != b[k] for k in ("row", "column", "true_class_1based")):
                        raise ValueError("Paired predictions are not aligned")
                wins = sum(a["predicted_class_1based"] == a["true_class_1based"] and b["predicted_class_1based"] != b["true_class_1based"] for a, b in pairs)
                losses = sum(a["predicted_class_1based"] != a["true_class_1based"] and b["predicted_class_1based"] == b["true_class_1based"] for a, b in pairs)
                if wins-losses != count:
                    raise ValueError("Paired correctness counts do not match the OA difference")
                lines.append(f"- {question}（{left}−{right}）：{diff:+.3f}个百分点，净差{count:+d}个验证样本；{left}修正{right}的{wins}个错误，同时引入{losses}个新错误。")
        lines.extend(["", "### 逐类验证召回率", "", "类别编号与原始数据标签一致；单类只有约36–40个验证中心，小幅变化未必稳定。", "",
                      "| 类别 | 样本数 | " + " | ".join(r["variant"] for r in jobs) + " |",
                      "|---|---:|" + "---:|" * len(jobs)])
        for class_id in range(info["classes"]):
            label = f"{class_id+1} {HOUSTON_LABELS[class_id]}" if dataset == "Houston2013" else str(class_id+1)
            lines.append(f"| {label} | {info['class_counts']['validation'][class_id]} | " +
                         " | ".join(f"{r['validation']['per_class_recall'][class_id]:.2f}" for r in jobs) + " |")
        final_oa = {}
        for row in jobs:
            h = folder / dataset / f"{row['variant']}_seed{seed}" / "history.jsonl"
            final_oa[row["variant"]] = json.loads(h.read_text(encoding="utf-8").splitlines()[-1])["validation"]["oa"]
        lines.extend(["", "末轮验证OA（检查最佳轮选择是否改变表面排序）：" +
                      "；".join(f"{k} {v:.3f}%" for k, v in final_oa.items()) + "。", "",
                      "### 交互强度", "", "数值来自验证集最佳checkpoint；它们是带符号的残差系数，不是模态贡献百分比。", ""])
        for row in jobs:
            if row["strengths"]:
                lines.append(f"- {row['variant']}：" + "，".join(f"`{k}={v:+.6f}`" for k, v in row["strengths"].items()))
        lines.extend(["", "### 对科研判断的限制", "",
                      "- 一个训练种子无法估计训练随机性的影响；邻域相关也使简单独立样本统计假设不可靠。",
                      "- 验证分数用于选权重，不能替代最终测试分数；同场景、全图无标签预处理的限制仍然存在。",
                      "- 10轮内消息强度可能尚小，慢热结构不能据此淘汰；需查看完整训练曲线和多种子表现。",
                      "- 参数量和调用次数并非全部相等；时延只是当前GPU、batch32、5次前向的短时测量，不是硬件通用结论。",
                      "- F1只控制辅助分支自增强；若要证明跨模态机制独立有效，后续还需更完整的自增强对照。", ""])
    seconds = sum(r["training_seconds"] for r in rows)
    lines.extend(["## 正式确认的预算参考", "",
                  f"本批模型训练合计{seconds/60:.1f}分钟，不包含前期导入、数据处理和结果分析。"])
    if cfg["epochs"] == 10 and len(cfg["seeds"]) == 1 and len(cfg["variants"]) == 7:
        lines.append(f"100轮×3种子7组按当前速度粗略外推约{seconds*30/3600:.1f}小时；该估计不代表已执行。")
    lines.extend(["正式入口见仓库 docs/transmixer_fusion_experiments.md；候选与预算应在查看正式测试结果之前确定。", ""])
    output = folder / "实验观察.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    analyze(parser.parse_args().folder.resolve())
