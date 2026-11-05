"""Archive lightweight evidence of fusion_transfer_v3_seed678_20261104 for Git."""
import json
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_sr_native import read, write, sha, report_metrics

TAG = "fusion_transfer_v3_seed678_20261104"
SRC = ROOT / "report/fusion_papers" / TAG
DST = ROOT / "docs/results" / TAG


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def copy_exact(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = source.read_bytes()
    if destination.exists():
        assert destination.read_bytes() == content, destination
    else:
        with destination.open("xb") as f:
            f.write(content)
    return {
        "source": source.relative_to(ROOT).as_posix(),
        "export": destination.relative_to(ROOT).as_posix(),
        "sha256": sha(destination),
        "bytes": destination.stat().st_size,
    }


def main():
    manifest = read(SRC / "manifest.json")
    status = read(SRC / "运行状态.json")
    audit = read(SRC / "final_integrity_audit.json")
    review = read(SRC / "final_independent_review_20261105.json")
    assert status["state"] == "completed" and audit["all_passed"] and review["all_passed"]

    DST.mkdir(parents=True, exist_ok=True)
    exported = []

    top_files = [
        "manifest.json",
        "summary.json",
        "运行状态.json",
        "power_protection.json",
        "gpu_preflight.json",
        "cost_audit.json",
        "final_integrity_audit.json",
        "final_independent_review_20261105.json",
        "native_per_class_metrics.csv",
        "实验报告.md",
    ]
    for name in top_files:
        exported.append(copy_exact(SRC / name, DST / name))

    for baseline in manifest["baselines"]:
        rpt = Path(baseline["report"])
        assert sha(rpt) == baseline["report_sha256"], baseline["id"]
        exported.append(copy_exact(rpt, DST / "native_reports" / f"{baseline['id']}.txt"))

    evidence_runs = []
    for job in manifest["jobs"]:
        marker = SRC / "completed" / f"{job['id']}.json"
        done = read(marker)
        exported.append(copy_exact(marker, DST / "completed" / marker.name))
        rpt = Path(job["report"])
        assert sha(rpt) == done["report_sha256"], job["id"]
        exported.append(copy_exact(rpt, DST / "native_reports" / f"{job['id']}.txt"))

        stderr_txt = Path(job["stderr"]).read_text(encoding="utf-8", errors="replace")
        stdout_txt = Path(job["stdout"]).read_text(encoding="utf-8", errors="replace")
        m = re.findall(r"(Epoch 100/100:\s+100%[^\r\n]*22/22[^\r\n]*)", stderr_txt)
        assert m and "Report Success!" in stdout_txt, job["id"]
        for key in ("checkpoint", "report", "stdout", "stderr", "native_log", "diagnostics"):
            assert sha(Path(job[key])) == done[key + "_sha256"], (job["id"], key)
        assert report_metrics(rpt) == done["metrics"], job["id"]

        evidence_runs.append({
            "id": job["id"],
            "epochs": done["epochs"],
            "training_steps": done["training_steps"],
            "final_epoch_progress_excerpt": m[-1].strip(),
            "report_success_excerpt": "Report Success!",
            "original_artifact_sha256": {
                k: done[k + "_sha256"]
                for k in ("checkpoint", "report", "stdout", "stderr", "native_log", "diagnostics")
            },
            "metrics": done["metrics"],
        })

    captured = now()
    runtime_path = DST / "RUNTIME.json"
    write(runtime_path, {
        "batch": TAG,
        "python_version": manifest["python_version"],
        "torch": manifest["torch"],
        "numpy": manifest["numpy"],
        "gpu": manifest["gpu"],
        "protocol": manifest["protocol"],
        "checkpoint_selection": manifest["checkpoint_selection"],
    })
    exported.append({
        "source": "generated",
        "export": runtime_path.relative_to(ROOT).as_posix(),
        "sha256": sha(runtime_path),
        "bytes": runtime_path.stat().st_size,
    })

    evidence_path = DST / "COMPLETION_EVIDENCE.json"
    write(evidence_path, {
        "captured_at_beijing": captured,
        "all_passed": True,
        "new_training_runs": 36,
        "reused_baselines": 3,
        "runs": evidence_runs,
    })
    exported.append({
        "source": "generated",
        "export": evidence_path.relative_to(ROOT).as_posix(),
        "sha256": sha(evidence_path),
        "bytes": evidence_path.stat().st_size,
    })

    readme_path = DST / "README.md"
    readme_text = "\n".join([
        f"# {TAG}",
        "",
        f"原批次 `{TAG}` 已完成 39/39 项（复用 A0 三项、新训 36 项）；复核时间 `2026-11-05T09:09:30.985710+08:00`，归档时间 `{captured}`。",
        "",
        "- [实验报告](实验报告.md)",
        "- [逐种子结果与配对统计](summary.json)",
        "- [逐类原生指标表](native_per_class_metrics.csv)",
        "- [原生分类报告](native_reports/)",
        "- [逐项完成记录](completed/)",
        "- [100轮完成日志摘录与产物哈希](COMPLETION_EVIDENCE.json)",
        "- [参数与推理开销审计](cost_audit.json)",
        "- [最终独立复核记录](final_independent_review_20261105.json)",
        "- [执行环境](RUNTIME.json)",
        "- [归档文件SHA256索引](SNAPSHOT.json)",
        "",
        "这是已结束批次的轻量固定证据快照。原始数据、整模型 checkpoint、完整日志与诊断曲线继续保存在本机原目录，未删除或覆盖。",
        "",
        "保留实验室原测试 OA 选模规则；单数据集、三种子结果不构成独立测试收益或完整外部论文复现。",
        "",
    ])
    readme_path.write_text(readme_text, encoding="utf-8")
    exported.append({
        "source": "generated",
        "export": readme_path.relative_to(ROOT).as_posix(),
        "sha256": sha(readme_path),
        "bytes": readme_path.stat().st_size,
    })

    write(DST / "SNAPSHOT.json", {
        "captured_at_beijing": captured,
        "batch": TAG,
        "completed_at_capture": 39,
        "new_training_runs": 36,
        "reused": 3,
        "source_state_updated_at": status["updated_at"],
        "immutable_snapshot": True,
        "note": "Original byte hashes preserved; datasets/checkpoints/full logs excluded.",
        "files": exported,
    })
    print("ARCHIVED_FV3_OK", len(exported), flush=True)


if __name__ == "__main__":
    main()
