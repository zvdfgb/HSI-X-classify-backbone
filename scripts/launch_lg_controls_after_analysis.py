"""Finish diagnostic audit before a documented, additive registry transition."""
import hashlib
import json
import runpy
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'report/sr_lg_mechanisms_20261104'
ARCHIVE=ROOT/'docs/experiments/provenance/sr_lg_operator_controls_20261104'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

if __name__=='__main__':
    deadline=time.monotonic()+900
    while read(DEST/'progress.json').get('state')!='completed':
        if time.monotonic()>deadline:raise TimeoutError('Analysis not completed; no training started')
        time.sleep(5)
    manifest=read(DEST/'manifest.json')
    for rel,digest in manifest['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    registry=ROOT/'src/model_registry.py'
    original=registry.read_bytes();old=runpy.run_path(str(registry))
    text=original.decode('utf-8')
    anchor='MODEL_REGISTRY = {\n'
    assert text.count(anchor)==2  # primary and LEGACY_MODEL_REGISTRY
    addition=('    "TransMixer_LG_LL_SR2": "src.models.transmixer_lg_controls.adapter",\n'
              '    "TransMixer_LG_GG_SR2": "src.models.transmixer_lg_controls.adapter",\n')
    assert 'TransMixer_LG_LL_SR2' not in text
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    backup=ARCHIVE/'model_registry.before.py'
    with backup.open('xb') as f:f.write(original)
    registry.write_bytes(text.replace(anchor,anchor+addition,1).encode('utf-8'))
    current=runpy.run_path(str(registry))
    for k in ('LEGACY_MODEL_REGISTRY','MODEL_ALIASES'):assert current[k]==old[k]
    assert current['MODEL_REGISTRY']=={**old['MODEL_REGISTRY'],**{f'TransMixer_LG_{g}_SR2':'src.models.transmixer_lg_controls.adapter' for g in ('LL','GG')}}
    transition={'old_sha256':sha(backup),'new_sha256':sha(registry),'only_added_names':['TransMixer_LG_LL_SR2','TransMixer_LG_GG_SR2'],'diagnostic_source_audit_passed_before_transition':True,'historical_mappings_unchanged':True}
    (ARCHIVE/'registry_transition.json').write_text(json.dumps(transition,indent=2),encoding='utf-8')
    for action in ('prepare','run'):
        subprocess.run([sys.executable,'-X','utf8','-u','scripts/run_lg_operator_controls.py',action],cwd=ROOT,check=True)
