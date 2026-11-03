"""Verify exported fusion evidence without datasets, checkpoints or training.

Only standard-library arithmetic and the original native text-report parser
are used. This is archival verification, never prediction or metric evaluation.
"""
import csv
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_sr_native import read, sha, report_metrics


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8):
        raise AssertionError((actual, expected))


def verify_archive(entry):
    snapshot = read(ROOT / entry["snapshot"])
    folder = ROOT / entry["archive"]
    for item in snapshot["files"]:
        path = (ROOT / item["export"]).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise AssertionError(f"Archive path escapes its directory: {path}")
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], path
    summary, manifest = read(folder / "summary.json"), read(folder / "manifest.json")
    rows = {r["id"]: r for r in summary["runs"]}
    assert len(rows) == summary["completed"] == summary["total"] == entry["results"]
    assert read(folder / "运行状态.json")["state"] == "completed"
    evidence = read(folder / "COMPLETION_EVIDENCE.json")
    assert evidence["all_passed"] and evidence["new_training_runs"] == entry["new_training_runs"]
    markers = list((folder / "completed").glob("*.json"))
    assert len(markers) == entry["new_training_runs"]
    excerpts = {r["id"]: r for r in evidence["runs"]}
    for job in manifest["jobs"]:
        report = folder / "native_reports" / f"{job['id']}.txt"
        metrics = report_metrics(report)
        for key in ("oa", "aa", "kappa"):
            close(metrics[key], rows[job["id"]][key])
        seal = job if job["reused"] else read(folder / "completed" / f"{job['id']}.json")
        assert sha(report) == seal["report_sha256"], job["id"]
        if not job["reused"]:
            excerpt = excerpts[job["id"]]
            assert seal["epochs"] == excerpt["epochs"] == 100
            assert "Epoch 100/100:" in excerpt["final_epoch_progress_excerpt"]
            assert "22/22" in excerpt["final_epoch_progress_excerpt"]
            assert excerpt["report_success_excerpt"] == "Report Success!"
            for key in ("checkpoint", "report", "stdout", "stderr", "native_log"):
                digest = excerpt["original_artifact_sha256"][key]
                assert len(digest) == 64
                if key + "_sha256" in seal:
                    assert digest == seal[key + "_sha256"]
            for key in ("oa", "aa", "kappa"):
                close(metrics[key], seal["metrics"][key])
    for group in summary.get("aggregates", []):
        assert group["complete"] and group["n"] == 3
        values = [r for r in rows.values() if r["group"] == group["group"]]
        assert len(values) == 3
        for key in ("oa", "aa", "kappa"):
            close(statistics.mean(r[key] for r in values), group[key]["mean"])
            close(statistics.stdev(r[key] for r in values), group[key]["sample_std"])
    for comparison in summary.get("paired_comparisons", []):
        if "comparison" in comparison and "per_seed_delta_pp" in comparison:
            left, right = comparison["comparison"].split("-")
            for key in ("oa", "aa", "kappa"):
                deltas = []
                for item in comparison["per_seed_delta_pp"]:
                    seed = item["seed"]
                    delta = rows[f"{left}_seed{seed}"][key] - rows[f"{right}_seed{seed}"][key]
                    close(delta, item[key]); deltas.append(delta)
                close(statistics.mean(deltas), comparison["mean_delta_pp"][key])
                assert sum(d > 0 for d in deltas) == comparison["positive_counts"][key]
        elif "differences" in comparison:
            deltas = []
            for item in comparison["differences"]:
                seed = item["seed"]
                delta = rows[f"{comparison['candidate']}_seed{seed}"]["oa"] - rows[f"{comparison['control']}_seed{seed}"]["oa"]
                close(delta, item["oa_delta_pp"]); deltas.append(delta)
            close(statistics.mean(deltas), comparison["mean_oa_delta_pp"])
            assert sum(d > 0 for d in deltas) == comparison["positive_seeds"]
    for name in ("final_integrity_audit.json", "final_audit.json"):
        if (folder / name).exists():
            audit = read(folder / name)
            assert audit.get("all_passed", audit.get("passed")) is True
    print(f"PASS {entry['batch']}: {len(rows)} reports, {len(markers)} new-run completion records")
    return rows


def main():
    index = read(ROOT / "docs/results/fusion_20261102_03_index.json")
    current = {}
    for entry in index["batches"]:
        rows = verify_archive(entry)
        if entry["role"] == "current_window":
            for key, row in rows.items():
                if key in current:
                    for metric in ("oa", "aa", "kappa"):
                        close(current[key][metric], row[metric])
                current[key] = row
    assert len(current) == index["unique_current_results"] == 45
    assert sum(r["group"] != "A00" for r in current.values()) == index["current_new_training_runs"] == 42
    consolidated = index["consolidated_csv"]
    assert sha(ROOT / consolidated["path"]) == consolidated["sha256"]
    with (ROOT / consolidated["path"]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(current)
    assert len({(r["group"], r["seed"]) for r in rows}) == 45
    for row in rows:
        key = f"{row['group']}_seed{row['seed']}"
        for metric in ("oa", "aa", "kappa"):
            close(float(row[metric]), current[key][metric])
        close(float(row["oa_delta_vs_A00_pp"]), current[key]["oa"] - current[f"A00_seed{row['seed']}"]["oa"])
    print("PASS consolidated results: 45 unique rows = 3 reused A baselines + 42 new runs")


if __name__ == "__main__":
    main()
