"""Queue the six approved runs through scripts/train.py; never train here.

Only artifact hashing, configuration, process orchestration and saved-report
arithmetic live here. Original data, optimizer, selection and metrics own all
scientific execution. Existing batches are read-only.
"""
import argparse
import gc
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_sr_native import read, write, sha, report_metrics
from scripts.run_fusion_papers import verify_pins, PROTECTED
from scripts.keep_awake_fusion import kernel, is_running

TAG = "sr_lg_prior_seed678_20261103"
DEST = ROOT / "report/fusion_papers" / TAG
LOGS = ROOT / "log/fusion_papers" / TAG
ARCHIVE = ROOT / "docs/experiments/provenance" / TAG
PLAN = ROOT / "docs/experiments/20261103_SR局部全局协同_实验矩阵_已批准.json"
NATIVE = ROOT / "report/sr_native/sr_native_20261029"
PREVIOUS = ROOT / "report/fusion_papers/fusion_transfer_seed78_20261102"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def state_write(value):
    write(DEST / "运行状态.tmp", {**value, "updated_at": now()})
    os.replace(DEST / "运行状态.tmp", DEST / "运行状态.json")


def check_no_training():
    # Read the OS process inventory; no command-line shell composition from files.
    query = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' "
             "-and $_.CommandLine -match '(scripts[/\\\\]train\\.py|train_all\\.py|run_batch\\.py)' } "
             "| Select-Object -ExpandProperty ProcessId")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", query],
        capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.stdout.strip():
        raise RuntimeError(f"Existing native training process: {result.stdout.strip()}")


def audit_job(job, seal):
    for kind in ("checkpoint", "report"):
        assert sha(job[kind]) == seal[f"{kind}_sha256"], (job["id"], kind)
    if not job["reused"]:
        assert seal["epochs"] == 100
        for kind in ("stdout", "stderr", "native_log"):
            assert sha(job[kind]) == seal[f"{kind}_sha256"], (job["id"], kind)
        assert "Report Success!" in Path(job["stdout"]).read_text(encoding="utf-8", errors="replace")
        assert re.search(r"Epoch 100/100:\s+100%[^\r\n]*22/22",
            Path(job["stderr"]).read_text(encoding="utf-8", errors="replace"))
    return report_metrics(job["report"])


def prepare():
    import numpy as np
    import torch
    from src.config import ExperimentConfig
    from scripts.verify_sr_lg_prior import preflight

    check_no_training()
    plan = read(PLAN)
    assert plan["training_authorized"] and plan["new_training"] == 6 and plan["total"] == 9
    assert plan["protocol"]["seeds"] == [6, 7, 8]
    if DEST.exists(): raise FileExistsError(f"Refusing to overwrite batch {DEST}")
    old = read(PREVIOUS / "manifest.json"); before = read(ARCHIVE / "pre_edit_audit.json")
    assert read(PREVIOUS / "运行状态.json")["state"] == "completed"
    assert read(PREVIOUS / "final_integrity_audit.json")["all_passed"]
    assert sha(PREVIOUS/"manifest.json") == before["previous_manifest_sha256"]
    assert sha(PREVIOUS/"final_integrity_audit.json") == before["previous_final_audit_sha256"]
    for category in ("source_and_config_sha256", "data_sha256"):
        for rel, digest in old[category].items():
            path = ARCHIVE / "model_registry_before.py.txt" if rel=="src/model_registry.py" else ROOT / rel
            assert sha(path) == digest, rel
    for rel, digest in before["protected_lab_files"].items(): assert sha(ROOT/rel)==digest, rel
    # Confirm the ONLY migration is the two registered names; no old mapping changes.
    registry = (ROOT/"src/model_registry.py").read_text(encoding="utf-8")
    insertion = '    **{f"TransMixer_SRP_P{i:02d}": "src.models.transmixer_sr_prior.adapter" for i in (1, 2)},\n'
    archived = (ARCHIVE/"model_registry_before.py.txt").read_text(encoding="utf-8")
    assert registry.replace(insertion, "", 1) == archived
    write(ARCHIVE / "source_migration.json", {"applied_at": now(), "authorized": True,
        "reason": "User-approved SR attention-prior experiment; two registry entries only",
        "before_sha256": sha(ARCHIVE/"model_registry_before.py.txt"),
        "after_sha256": sha(ROOT/"src/model_registry.py"), "protected_lab_files_unchanged": True})
    assert Path(sys.executable).resolve()==Path(old["python"]).resolve()
    assert platform.python_version()==old["python_version"]
    assert str(torch.__version__)==old["torch"] and np.__version__==old["numpy"]
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0)==old["gpu"]
    DEST.mkdir(parents=True); LOGS.mkdir(parents=True, exist_ok=True)
    native, sealed = read(NATIVE/"manifest.json"), read(NATIVE/"final_integrity_audit.json")
    defaults, stats = read(ROOT/"configs/train.json"), read(NATIVE/"training_input_statistics.json")
    jobs, reused = [], []
    for seed in (6, 7, 8):
        j = next(j for j in native["jobs"] if j["id"]==f"M3_seed{seed}")
        seal = next(j for j in sealed["outputs"] if j["id"]==f"M3_seed{seed}")
        cfg = read(j["config"])
        for key, value in defaults.items():
            if key not in ("experiment_name", "model_name", "random_seed"): assert cfg[key]==value, key
        assert cfg["random_seed"]==seed and cfg["ablation_mode"]=="decoupled"
        assert cfg["sr"]=={"spatial_sigma":3.0,"spectral_scale":stats[str(seed)]["spectral_scale"]}
        assert sha(j["config"])==native["source_and_config_sha256"][Path(j["config"]).relative_to(ROOT).as_posix()]
        metrics = audit_job({**j,"reused":True}, seal)
        reused.append({"id":f"A00_seed{seed}","metrics":metrics,"native_id":j["id"]})
        jobs.append({**j,"id":f"A00_seed{seed}","group":"A00","seed":seed,"reused":True,
            "reuse_source":j["id"],"checkpoint_sha256":seal["checkpoint_sha256"],"report_sha256":seal["report_sha256"]})
    for seed in (6, 7, 8):
        for group in ("P01", "P02"):
            cfg = {**defaults,"experiment_name":f"{TAG}_{group}_seed{seed}",
                "model_name":f"TransMixer_SRP_{group}","random_seed":seed,"ablation_mode":"decoupled",
                "sr":{"spatial_sigma":3.0,"spectral_scale":stats[str(seed)]["spectral_scale"]}}
            exp = ExperimentConfig(**cfg); path = ROOT/"configs/runs"/TAG/f"{group}_seed{seed}.json"
            if path.exists(): raise FileExistsError(path)
            write(path,cfg)
            jobs.append({"id":f"{group}_seed{seed}","group":group,"seed":seed,"reused":False,
                "model":exp.model_name,"config":str(path),"checkpoint":exp.model_save_path,
                "report":exp.report_path,"native_log":exp.log_path,
                "stdout":str(LOGS/f"{group}_seed{seed}.stdout.log"),"stderr":str(LOGS/f"{group}_seed{seed}.stderr.log")})
    assert [j["id"] for j in jobs if not j["reused"]]==plan["new_job_order"]
    for j in jobs:
        if not j["reused"]:
            for key in ("checkpoint","report","native_log","stdout","stderr"):
                if Path(j[key]).exists(): raise FileExistsError(j[key])
    write(DEST/"reuse_audit.json",{"all_passed":True,"results":reused})
    preflight(jobs, DEST)
    sources = list((ROOT/"src").rglob("*.py"))
    sources += [ROOT/p for p in ("scripts/train.py","scripts/prepare_sr_native.py","scripts/run_fusion_papers.py",
        "scripts/keep_awake_fusion.py","scripts/keep_awake_sr_prior.py","scripts/run_sr_lg_prior.py",
        "scripts/verify_sr_lg_prior.py","configs/train.json","AGENTS.md","docs/specs/sr-lg-prior-20261103.md")]
    sources += [PLAN, NATIVE/"training_input_statistics.json", DEST/"preflight.json", DEST/"reuse_audit.json"]
    sources += list(ARCHIVE.glob("*")) + [Path(j["config"]) for j in jobs]
    manifest = {"run_tag":TAG,"created_at":now(),"python":sys.executable,"python_version":platform.python_version(),
        "torch":str(torch.__version__),"numpy":np.__version__,"gpu":torch.cuda.get_device_name(0),
        "total":9,"reused":3,"new_training":6,"jobs":jobs,
        "source_and_config_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        "data_sha256":old["data_sha256"],"training_entry":"scripts/train.py",
        "checkpoint_selection":"native_test_OA_best","additional_losses":False}
    write(DEST/"manifest.json",manifest); verify_pins(manifest)
    state_write({"state":"prepared","total":9,"reused":3,"completed":3,"new_completed":0,
        "new_total":6,"current":None,"queue_pid":None,"training_pid":None,"helper_pid":None})
    summarize()
    print("Prepared 9 results: 3 reused + 6 new native runs",flush=True)


def summarize():
    manifest = read(DEST/"manifest.json")
    rows = [{k:j[k] for k in ("id","group","seed","reused","report")} | report_metrics(j["report"])
        for j in manifest["jobs"] if j["reused"] or (DEST/"completed"/f"{j['id']}.json").exists()]
    table = {r["id"]:r for r in rows}; aggregates=[]; comparisons=[]
    for group in ("A00","P01","P02"):
        r = [r for r in rows if r["group"]==group]
        a = {"group":group,"n":len(r),"complete":len(r)==3}
        if len(r)==3:
            a.update({k:{"mean":statistics.mean(x[k] for x in r),"sample_std":statistics.stdev(x[k] for x in r)} for k in ("oa","aa","kappa")})
        aggregates.append(a)
    for pair in read(PLAN)["comparisons"]:
        a,b=pair.split("-"); deltas=[]
        for seed in (6,7,8):
            if f"{a}_seed{seed}" in table and f"{b}_seed{seed}" in table:
                deltas.append({"seed":seed,**{k:table[f"{a}_seed{seed}"][k]-table[f"{b}_seed{seed}"][k] for k in ("oa","aa","kappa")}})
        comparisons.append({"comparison":pair,"n":len(deltas),"per_seed_delta_pp":deltas,
            "mean_delta_pp":{k:statistics.mean(d[k] for d in deltas) for k in ("oa","aa","kappa")} if deltas else None,
            "positive_counts":{k:sum(d[k]>0 for d in deltas) for k in ("oa","aa","kappa")}})
    write(DEST/"summary.json",{"total":9,"completed":len(rows),"reused":3,"new_completed":len(rows)-3,
        "new_total":6,"seeds":[6,7,8],"runs":rows,"aggregates":aggregates,"paired_comparisons":comparisons,
        "limitations":read(PLAN)["limitations"]})
    lines=["# SR与局部/全局分工协同实验", "",f"已完成{len(rows)}/9：复用3项，新训{len(rows)-3}/6项。", "",
        "A00为原L+SR；P01为正确SR注意力先验；P02只错置注意力先验，末端SR仍正确。",
        "每项100轮，seed6/7/8，原测试OA选模；不构成独立测试或统计显著性证明。", "",
        "| 组别 | seed | OA % | AA % | Kappa % | 来源 |", "|---|---:|---:|---:|---:|---|"]
    for r in rows: lines.append(f"| {r['group']} | {r['seed']} | {r['oa']:.4f} | {r['aa']:.4f} | {r['kappa']:.4f} | {'复用' if r['reused'] else '新训'} |")
    lines += ["","## 均值±样本标准差（ddof=1）","","| 组别 | OA % | AA % | Kappa % |","|---|---:|---:|---:|"]
    for a in aggregates:
        if a["complete"]: lines.append(f"| {a['group']} | " + " | ".join(f"{a[k]['mean']:.4f} ± {a[k]['sample_std']:.4f}" for k in ("oa","aa","kappa")) + " |")
    lines += ["","## 配对差值（百分点）","","| 比较 | 指标 | seed6 | seed7 | seed8 | 平均差值 | 正向次数 |","|---|---|---:|---:|---:|---:|---:|"]
    for c in comparisons:
        if not c["n"]: continue
        for k in ("oa","aa","kappa"):
            ds={d["seed"]:d[k] for d in c["per_seed_delta_pp"]}
            values=[f"{ds[s]:+.4f}" if s in ds else "待完成" for s in (6,7,8)]
            lines.append(f"| {c['comparison']} | {k} | " + " | ".join(values) + f" | {c['mean_delta_pp'][k]:+.4f} | {c['positive_counts'][k]}/{c['n']} |")
    lines += ["","新增参数仅每个原始block一个零初始化beta；其符号允许为负。错位控制使用一个固定置换，不代表所有错位情形。",
        "预检、封存与完成证据：preflight.json、manifest.json、completed、final_integrity_audit.json。",""]
    (DEST/"实验报告.md").write_text("\n".join(lines),encoding="utf-8")


def final_audit():
    manifest=read(DEST/"manifest.json"); verify_pins(manifest); results=[]
    assert len(list((DEST/"completed").glob("*.json")))==6
    for j in manifest["jobs"]:
        seal=j if j["reused"] else read(DEST/"completed"/f"{j['id']}.json")
        metrics=audit_job(j,seal)
        results.append({"id":j["id"],"metrics":metrics,
            **{f"{k}_sha256":sha(j[k]) for k in ("checkpoint","report")}})
    summarize()
    assert read(DEST/"summary.json")["completed"]==9
    write(DEST/"final_integrity_audit.json",{"all_passed":True,"verified_at":now(),"reused":3,
        "new_training_runs":6,"source_config_data_pins_verified":True,"results":results})


def run():
    import msvcrt
    manifest=read(DEST/"manifest.json")
    assert read(DEST/"运行状态.json")["state"]=="prepared", "No automatic resume/retraining"
    check_no_training(); verify_pins(manifest)
    lock=(DEST/"queue.lock").open("xb+"); lock.write(b"0"); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    state={"state":"starting","total":9,"reused":3,"completed":3,"new_completed":0,"new_total":6,
        "queue_pid":os.getpid(),"training_pid":None,"helper_pid":None,"current":None,"started_at":now()}
    child=None; helper=None
    env=dict(os.environ,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",PYTHONIOENCODING="utf-8",PYTHONUNBUFFERED="1")
    env.pop("CUBLAS_WORKSPACE_CONFIG",None)
    try:
        (DEST/"completed").mkdir(exist_ok=True)
        if not kernel.SetThreadExecutionState(0x80000001): raise RuntimeError("Native sleep protection failed")
        state_write(state)
        with (LOGS/"power.stdout.log").open("xb") as out, (LOGS/"power.stderr.log").open("xb") as err:
            helper=subprocess.Popen([manifest["python"],"-X","utf8","-u","scripts/keep_awake_sr_prior.py"],
                cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        state["helper_pid"]=helper.pid; state_write(state)
        # Protection must be verified before the first scientific process starts.
        for _ in range(30):
            if helper.poll() is not None: raise RuntimeError("Power helper exited before training")
            if (DEST/"power_protection.json").exists(): break
            time.sleep(1)
        assert read(DEST/"power_protection.json")["state"]=="active"
        for job in manifest["jobs"]:
            if job["reused"]: continue
            verify_pins(manifest); check_no_training()
            if helper.poll() is not None: raise RuntimeError("Power helper stopped; refusing next run")
            for key in ("stdout","stderr","checkpoint","report","native_log"):
                if Path(job[key]).exists(): raise FileExistsError(job[key])
            start,clock=now(),time.monotonic()
            state.update(state="launching",current=job,training_pid=None,current_epoch=None,current_started_at=start)
            state_write(state)
            with Path(job["stdout"]).open("xb") as out,Path(job["stderr"]).open("xb") as err:
                child=subprocess.Popen([manifest["python"],"-X","utf8","-u","scripts/train.py","--config",job["config"]],
                    cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                state.update(state="training",training_pid=child.pid); state_write(state)
                print(f"Started {job['id']} pid={child.pid}",flush=True)
                while child.poll() is None:
                    time.sleep(10)
                    if not kernel.SetThreadExecutionState(0x80000001): raise RuntimeError("Sleep protection lost")
                    if helper.poll() is not None: raise RuntimeError("Power helper stopped during training; active child is preserved")
                    with Path(job["stderr"]).open("rb") as stream:
                        stream.seek(max(0,Path(job["stderr"]).stat().st_size-16000))
                        epochs=re.findall(r"Epoch (\d+)/100",stream.read().decode("utf-8",errors="replace"))
                    state.update(current_epoch=int(epochs[-1]) if epochs else None,
                        current_elapsed_seconds=round(time.monotonic()-clock,1)); state_write(state)
            if child.returncode!=0: raise RuntimeError(f"Native training failed: {job['id']} exit={child.returncode}")
            metrics=report_metrics(job["report"])
            seal={"id":job["id"],"seed":job["seed"],"epochs":100,"started_at":start,"finished_at":now(),
                "elapsed_seconds":time.monotonic()-clock,"metrics":metrics,
                **{f"{k}_sha256":sha(job[k]) for k in ("checkpoint","report","stdout","stderr","native_log")}}
            audit_job(job,seal); verify_pins(manifest)
            import torch
            net=torch.load(job["checkpoint"],map_location="cpu",weights_only=False)
            seal["best_checkpoint_prior_diagnostics"]=net.diagnostics(); del net; gc.collect()
            write(DEST/"completed"/f"{job['id']}.json",seal)
            summarize(); state.update(completed=state["completed"]+1,new_completed=state["new_completed"]+1,training_pid=None)
            state_write(state); print(f"Completed {job['id']}: OA={metrics['oa']:.4f}",flush=True)
        final_audit(); state.update(state="completed",current=None,training_pid=None,finished_at=now()); state_write(state)
        print("All 9 results complete: 3 reused + 6 new.",flush=True)
    except BaseException as exc:
        state.update(state="failed",error=repr(exc),training_pid=child.pid if child and child.poll() is None else None)
        state_write(state); raise
    finally:
        # On an orchestration failure preserve a live native child and its guard.
        while child and child.poll() is None:
            kernel.SetThreadExecutionState(0x80000001); time.sleep(20)
        kernel.SetThreadExecutionState(0x80000000)
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("action",choices=["prepare","run","summary","audit"])
    action=parser.parse_args().action
    {"prepare":prepare,"run":run,"summary":summarize,"audit":final_audit}[action]()
