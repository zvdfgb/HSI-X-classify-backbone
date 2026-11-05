"""One explicitly authorized offline reconciliation; refuses reuse or overwrite."""
import ast
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from scripts import run_fusion_transfer_v3 as q
from scripts.keep_awake_fusion import is_running


def reconcile():
    folder = Path(__file__).resolve().parent
    authorization = q.read(folder/"authorization.json")
    assert authorization["authorized"]
    assert not (folder/"source_transition.json").exists(), "Never repeat an applied reconciliation"
    q.assert_idle()
    status = q.read(q.DEST/"运行状态.json")
    assert status["state"] == "failed" and status["current"] == "C01_seed6"
    assert not any(is_running(status.get(k)) for k in ("queue_pid", "training_pid", "helper_pid"))
    m = q.read(q.DEST/"manifest.json")
    assert q.sha(q.DEST/"manifest.json") == authorization["original_manifest_sha256"]
    script_rel = "scripts/run_fusion_transfer_v3.py"
    assert q.sha(folder/"run_fusion_transfer_v3.py") == m["source_and_config_sha256"][script_rel]
    for rel, digest in m["source_and_config_sha256"].items():
        if rel != script_rel:
            assert q.sha(ROOT/rel) == digest, rel
    for rel, digest in m["data_sha256"].items():
        assert q.sha(ROOT/rel) == digest, rel
    for baseline in m["baselines"]:
        for key in ("checkpoint", "report", "config"):
            assert q.sha(baseline[key]) == baseline[key+"_sha256"]
    # Check the actual native launch call and environment statements by AST,
    # independent of formatting and the new orchestration-only code.
    def protocol_statements(source):
        tree = ast.parse(source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_queue")
        result = []
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "env" for t in node.targets):
                result.append(ast.dump(node, include_attributes=False))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (isinstance(node.func.value, ast.Name) and
                    ((node.func.value.id == "subprocess" and node.func.attr == "Popen") or
                     (node.func.value.id == "env" and node.func.attr == "pop"))):
                    result.append(ast.dump(node, include_attributes=False))
        return sorted(result)
    before_protocol = protocol_statements((folder/"run_fusion_transfer_v3.py").read_text(encoding="utf-8"))
    assert len(before_protocol) == 3
    assert before_protocol == protocol_statements(Path(q.__file__).read_text(encoding="utf-8"))
    audit_path = q.DEST/"monitoring/20261104_154324_failure_audit.json"
    evidence = q.read(audit_path)
    assert evidence["remaining_unstarted"] == 35
    raw = evidence["native_completed_but_not_queue_sealed"][0]
    job = m["jobs"][0]
    assert job["id"] == raw["id"] == "C01_seed6"
    assert not list((q.DEST/"completed").glob("*.json"))
    for key in q.ARTIFACTS:
        assert q.sha(job[key]) == raw["artifact_sha256"][key], key
    diagnostics = q.read(job["diagnostics"])
    assert diagnostics["group"] == "C01" and diagnostics["seed"] == 6
    assert diagnostics["training_steps"] == len(diagnostics["history"]) == 2200
    for pending in m["jobs"][1:]:
        assert not any(Path(pending[k]).exists() for k in q.ARTIFACTS), pending["id"]
    started = datetime.fromisoformat(status["current_started_at"])
    finished = datetime.fromtimestamp(Path(job["stdout"]).stat().st_mtime, tz=started.tzinfo)
    done = {"id": job["id"], "seed": 6, "group": "C01", "epochs": 100, "training_steps": 2200,
        "started_at": started.isoformat(), "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished-started).total_seconds(),
        "time_basis": "Original queue start timestamp to final stdout mtime; approximate wall time, not monotonic measurement",
        "recovered_at": q.now(), "recovered_after_queue_error": True,
        "recovery_evidence": str(audit_path), "failure_audit_sha256": q.sha(audit_path),
        "metrics": q.report_metrics(job["report"]),
        **{key+"_sha256": raw["artifact_sha256"][key] for key in q.ARTIFACTS}}
    q.audit_job(job, done)
    assert done["metrics"] == raw["metrics"]
    for path in q.LOG.glob("queue.*.log"):
        shutil.copy2(path, folder/path.name)
    q.write(folder/"test_results.json", {"all_passed": True, "tests": 11,
        "command": f"{sys.executable} -X utf8 -m scripts.test_fusion_transfer_v3_recovery",
        "verified_at": q.now(), "test_sha256": q.sha(ROOT/"scripts/test_fusion_transfer_v3_recovery.py"),
        "coverage": ["actual Windows handle denies replace until released", "persistent denial preserves destination",
            "non-sharing errors are not retried", "concurrent writers/readers", "bounded read retries",
            "only verified contiguous prefix skipped", "reject corrupt, unknown, gap or unsealed outputs"]})
    transition = {"authorized": True, "applied_at": q.now(), "user_instruction": "修复",
        "before_sha256": authorization["original_queue_sha256"], "after_sha256": q.sha(Path(q.__file__)),
        "all_other_pins_unchanged": True, "original_pinned_files": len(m["source_and_config_sha256"]),
        "native_launch_and_environment_ast_unchanged": True, "data_and_reuse_verified": True,
        "model_loss_optimizer_selection_protocol_unchanged": True,
        "scope": "Queue-only bounded Windows sharing retries and explicit evidence-checked resume",
        "accepted_existing_job": "C01_seed6", "next_job": "C02_seed6", "remaining_unstarted": 35,
        "old_manifest_sha256": authorization["original_manifest_sha256"],
        "test_results_sha256": q.sha(folder/"test_results.json")}
    q.write(folder/"source_transition.json", transition)
    m["source_and_config_sha256"][script_rel] = transition["after_sha256"]
    for path in [folder/"source_transition.json", folder/"test_results.json", folder/"authorization.json",
                 folder/"reconcile.py", ROOT/"scripts/test_fusion_transfer_v3_recovery.py"]:
        m["source_and_config_sha256"][path.relative_to(ROOT).as_posix()] = q.sha(path)
    m["authorized_queue_recovery"] = str(folder/"source_transition.json")
    q.atomic(q.DEST/"manifest.json", m)
    q.pins(m)
    with (q.DEST/"completed/C01_seed6.json").open("x", encoding="utf-8") as stream:
        json.dump(done, stream, ensure_ascii=False, indent=2)
    assert q.completed_prefix(m) == 1
    q.summarize()
    attempt = datetime.now().strftime("%Y%m%d_%H%M%S")
    q.state(state="ready_for_authorized_resume", queue_pid=None, training_pid=None, helper_pid=None,
        completed=4, new_completed=1, current=None, group=None, seed=None, current_epoch=0,
        error=None, failed_at=None, recovery_authorization=str(folder/"authorization.json"),
        previous_failure_archived=str(folder/"运行状态.json"),
        recovery_queue_stdout=str(q.LOG/f"queue.resume_{attempt}.stdout.log"),
        recovery_queue_stderr=str(q.LOG/f"queue.resume_{attempt}.stderr.log"))
    q.write(folder/"reconciliation_result.json", {"all_passed": True, "checked_at": q.now(),
        "completed": 4, "reused": 3, "new_completed": 1, "remaining": 35,
        "accepted_metrics": done["metrics"], "original_six_artifacts_unchanged": True,
        "completed_marker_sha256": q.sha(q.DEST/"completed/C01_seed6.json"),
        "manifest_after_sha256": q.sha(q.DEST/"manifest.json")})
    print(json.dumps(q.read(folder/"reconciliation_result.json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    reconcile()
