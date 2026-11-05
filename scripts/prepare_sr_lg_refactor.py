"""Prepare reviewable native configs; register only after approval and idle checks.

Never launches training, evaluates a model, or changes the native protocol.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_sr_native import read, sha
from src.config import ExperimentConfig
from src.models.transmixer_sr_lg_v2.adapter import NAMES

TAG = 'sr_lg_refactor_20261104'
PLAN = ROOT/'docs/experiments/20261104_SR局部全局重构_待审核.json'
REVIEW = ROOT/'docs/experiments/20261104_SR配合局部全局重构_分析与审核稿.md'
PREFLIGHT = ROOT/f'report/{TAG}/preflight_cpu.json'
REGISTRY = ROOT/'src/model_registry.py'
GROUPS = dict(zip(('R00','R01','R02','R03'), NAMES))
PROVENANCE = ROOT/f'docs/experiments/provenance/{TAG}'


def now():
    return datetime.now().astimezone().isoformat()


def exclusive_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def verify(hashes):
    for name, digest in hashes.items():
        if sha(ROOT/name) != digest:
            raise RuntimeError(f'Sealed file changed: {name}')


def baselines():
    """Reuse sealed native outputs, not replays or substitute metrics."""
    manifest = read(ROOT/'docs/results/sr_native_20261029/manifest.json')
    controls = read(ROOT/'report/sr_lg_operator_controls_20261104/manifest.json')
    audit = read(ROOT/'docs/results/sr_native_20261029/final_integrity_audit.json')
    if audit['mismatches']:
        raise RuntimeError('Historical audit has unresolved mismatches')
    seals = {item['id']:item for item in audit['outputs']}
    result = []
    for job in manifest['jobs']:
        if job['group'] not in ('M0','M2','M3'):
            continue
        # The completed native batch audit seals original checkpoints/reports.
        entry = dict(job, alias=job['group'])
        entry['sha256'] = {k: sha(job[k]) for k in ('config','checkpoint','report')}
        config_rel = Path(job['config']).relative_to(ROOT).as_posix()
        if manifest['source_and_config_sha256'][config_rel] != entry['sha256']['config']:
            raise RuntimeError(f'Historical config changed: {config_rel}')
        for key in ('checkpoint','report'):
            if seals[job['id']][f'{key}_sha256'] != entry['sha256'][key]:
                raise RuntimeError(f'Historical output changed: {job[key]}')
        result.append(entry)
    for job in controls['jobs']:
        if job['group'] != 'GG':
            continue
        marker = read(ROOT/f"report/sr_lg_operator_controls_20261104/completed/{job['id']}.json")
        entry = dict(job, alias='GG_SR')
        entry['sha256'] = {k: sha(job[k]) for k in ('config','checkpoint','report')}
        config_rel = Path(job['config']).relative_to(ROOT).as_posix()
        if controls['source_and_config_sha256'][config_rel] != entry['sha256']['config']:
            raise RuntimeError(f'Historical control config changed: {config_rel}')
        for key in ('checkpoint','report'):
            if entry['sha256'][key] != marker[f'{key}_sha256']:
                raise RuntimeError(f'Control output changed: {job[key]}')
        result.append(entry)
    if len(result) != 12 or {(j['alias'], j['seed']) for j in result} != {
            (g,s) for g in ('M0','M2','M3','GG_SR') for s in (6,7,8)}:
        raise RuntimeError('Expected exactly twelve sealed baselines')
    return result


def prepare():
    if PLAN.exists():
        raise FileExistsError(f'Preserve existing review plan: {PLAN}')
    preflight = read(PREFLIGHT)
    if preflight['all_passed'] is not True or preflight['registry_modified']:
        raise RuntimeError('CPU preflight required')
    verify(preflight['protected_sources_sha256'])
    verify(preflight['candidate_source_sha256'])
    active = read(ROOT/'report/fusion_papers/fusion_transfer_v3_seed678_20261104/manifest.json')
    verify(active['source_and_config_sha256'])
    # Read/hash only; never edit the other batch's files or touch its GPU.
    reused = baselines()
    planned = []
    for group, model in GROUPS.items():
        for seed in (6,7,8):
            source = ROOT/f'configs/runs/sr_native_20261029/M3_seed{seed}.json'
            cfg = read(source)
            cfg.update(model_name=model, experiment_name=f'{TAG}_{group}_seed{seed}')
            exp = ExperimentConfig(**cfg)
            target = ROOT/f'configs/runs/{TAG}/{group}_seed{seed}.json'
            outputs = {k: getattr(exp,k) for k in ('model_save_path','report_path','log_path','image_path','image_path_web')}
            if target.exists() or any(Path(p).exists() for p in outputs.values()):
                raise FileExistsError(f'Refuse overwriting config/output: {target}')
            planned.append((target,cfg,dict(id=f'{group}_seed{seed}',group=group,seed=seed,
                model=model,config=str(target),source_config=str(source),
                source_config_sha256=sha(source),outputs=outputs)))
    jobs = []
    for target,cfg,job in planned:
        exclusive_json(target,cfg)
        job['config_sha256'] = sha(target)
        jobs.append(job)
    plan = dict(tag=TAG,created_at=now(),implementation_authorized=True,
        training_authorized=False,approval_note=None,new_training_runs=12,seeds=[6,7,8],
        review=str(REVIEW),review_sha256=sha(REVIEW),preflight_cpu=str(PREFLIGHT),
        preflight_cpu_sha256=sha(PREFLIGHT),candidate_source_sha256=preflight['candidate_source_sha256'],
        protected_sources_sha256=preflight['protected_sources_sha256'],
        active_batch_seal_verified=True,registry_modified=False,
        jobs=jobs,baselines=reused,
        protocol=dict(dataset='Houston2013 official split',pca=30,window=11,epochs=100,
            batch_size=128,optimizer='native AdamW',learning_rate=1e-4,weight_decay=.01,
            split_channels=[32,32],checkpoint_selection='unchanged native test OA',
            training_entry='scripts/train.py -> src/runner.py -> src/trainer.py'),
        pending=['user review of these twelve runs','other training controllers idle',
            'guarded registry insertion','native real-data batch128 and GPU preflight',
            'training-source/config/data sealing and sequential native training',
            'native final reports and full-test-set gain diagnostics'])
    exclusive_json(PLAN,plan)
    print(json.dumps(dict(plan=str(PLAN),new_configs=len(jobs),training_started=False,
        baselines_reused=len(reused),registry_modified=False),ensure_ascii=False))


def assert_idle():
    command = ('Get-CimInstance Win32_Process | Where-Object { '
        '$_.Name -match "^python(w)?(.exe)?$" -and $_.CommandLine -match '
        '"(train[.]py|run_fusion_transfer_v3[.]py|run_lg_operator_controls[.]py|launch_lg_controls_after_analysis[.]py)" '
        '} | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress')
    found = subprocess.run(['powershell','-NoProfile','-Command',command],
        check=True,capture_output=True,text=True,encoding='utf-8',errors='replace').stdout.strip()
    if found:
        raise RuntimeError('Training process/controller still active; preserve its sealed registry')


def register():
    plan = read(PLAN)
    if plan.get('training_authorized') is not True or not str(plan.get('approval_note') or '').strip():
        raise RuntimeError('Review matrix approval and recorded approval_note required')
    if sha(REVIEW) != plan['review_sha256'] or sha(PREFLIGHT) != plan['preflight_cpu_sha256']:
        raise RuntimeError('Reviewed document or CPU preflight changed')
    verify(plan['protected_sources_sha256'])
    verify(plan['candidate_source_sha256'])
    for job in plan['jobs']:
        if sha(job['config']) != job['config_sha256']:
            raise RuntimeError('Reviewed config changed')
    assert_idle()
    original = REGISTRY.read_bytes()
    namespace = {}
    exec(compile(original,str(REGISTRY),'exec'),namespace)
    for model in NAMES:
        if any(model in namespace[k] for k in ('MODEL_REGISTRY','LEGACY_MODEL_REGISTRY','MODEL_ALIASES')):
            raise RuntimeError(f'Existing model name: {model}')
    newline = '\r\n' if b'\r\n' in original else '\n'
    insertion = ''.join(f'    "{name}": "src.models.transmixer_sr_lg_v2.adapter",{newline}' for name in NAMES)
    oldtext = original.decode('utf-8')
    anchor = 'MODEL_REGISTRY = {'+newline
    if oldtext.count(newline+anchor) != 1:
        raise RuntimeError('Cannot locate a unique native registry')
    proposed = oldtext.replace(newline+anchor,newline+anchor+insertion,1).encode('utf-8')
    changed = {}
    exec(compile(proposed,str(REGISTRY),'exec'),changed)
    if any(changed[k] != namespace[k] for k in ('LEGACY_MODEL_REGISTRY','MODEL_ALIASES')):
        raise RuntimeError('Historical registry changed')
    if changed['MODEL_REGISTRY'] != {**namespace['MODEL_REGISTRY'],**{
            name:'src.models.transmixer_sr_lg_v2.adapter' for name in NAMES}}:
        raise RuntimeError('Unexpected registry edit')
    PROVENANCE.mkdir(parents=True,exist_ok=True)
    backup = PROVENANCE/'model_registry.before.py'
    with backup.open('xb') as stream:
        stream.write(original)
    temporary = REGISTRY.with_name('model_registry.sr_lg_refactor.tmp')
    with temporary.open('xb') as stream:
        stream.write(proposed)
    try:
        assert_idle()
        if REGISTRY.read_bytes() != original:
            raise RuntimeError('Registry changed concurrently')
        os.replace(temporary,REGISTRY)
    finally:
        if temporary.exists():
            temporary.unlink()
    exclusive_json(PROVENANCE/'registry_transition.json',dict(time=now(),approval_note=plan['approval_note'],
        plan_sha256=sha(PLAN),before=hashlib.sha256(original).hexdigest(),after=sha(REGISTRY),
        training_started=False,added=list(NAMES)))
    print('Registered reviewed model names; no training launched. Native data/GPU preflight still required.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','register'))
    action = parser.parse_args().action
    {'prepare':prepare,'register':register}[action]()
