"""Six authorized controls; scientific execution ONLY through scripts/train.py."""
import argparse
import copy
import ctypes
import gc
import io
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_sr_native import read,write,sha,report_metrics

TAG='sr_lg_operator_controls_20261104'
DEST=ROOT/'report'/TAG
LOG=ROOT/'log'/TAG
ARCHIVE=ROOT/'docs/experiments/provenance'/TAG
NATIVE=ROOT/'report/sr_native/sr_native_20261029'
PLAN=ROOT/'docs/experiments/20261104_局部全局算子对照_已批准.json'
PROTECTED=['configs/train.json','scripts/train.py','src/runner.py','src/trainer.py','src/data.py',
           'src/evaluator.py','src/metrics.py','src/config.py','src/models/transmixer_net/model.py',
           'src/models/transmixer_paper/model.py','src/models/transmixer_sr/adapter.py']

def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()

def state(value):
    value['updated_at']=now();tmp=DEST/'state.tmp';write(tmp,value);os.replace(tmp,DEST/'state.json')

def check_idle():
    command="Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' -and $_.CommandLine -match '(scripts[/\\\\]train\\.py|train_all\\.py|run_batch\\.py)' } | Select-Object -ExpandProperty ProcessId"
    result=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode or result.stdout.strip():raise RuntimeError('Another training process exists: '+result.stdout)

def pins(manifest):
    for category in ('source_and_config_sha256','data_sha256'):
        for rel,digest in manifest[category].items():assert sha(ROOT/rel)==digest,rel
    for j in manifest['baselines']:
        for k in ('checkpoint','report'):assert sha(j[k])==j[k+'_sha256'],(j['id'],k)

def prepare():
    import numpy as np
    import torch
    from src import config
    from src.config import ExperimentConfig,ConfigManager
    from src.data import getMyData,set_random_seed
    from src.model_registry import get_model_adapter
    from src.models.transmixer_paper.model import NeighborTransMixer
    assert read(PLAN)['training_authorized'] and read(PLAN)['new_training_runs']==6
    assert read(ROOT/'report/sr_lg_mechanisms_20261104/progress.json')['state']=='completed'
    check_idle()
    if DEST.exists():raise FileExistsError(DEST)
    DEST.mkdir(parents=True);LOG.mkdir(parents=True);ARCHIVE.mkdir(parents=True,exist_ok=True)
    old=read(NATIVE/'manifest.json');sealed=read(NATIVE/'final_integrity_audit.json')['outputs']
    assert str(torch.__version__)==old['torch'] and np.__version__==old['numpy']
    assert Path(sys.executable).resolve()==Path(old['python']).resolve()
    assert torch.cuda.get_device_name(0)==old['gpu']
    for rel in PROTECTED:assert sha(ROOT/rel)==old['source_and_config_sha256'][rel],rel
    for rel,digest in old['data_sha256'].items():assert sha(ROOT/rel)==digest,rel
    baselines=[]
    for seed in (6,7,8):
        j=next(j for j in old['jobs'] if j['id']==f'M3_seed{seed}')
        seal=next(s for s in sealed if s['id']==j['id'])
        baselines.append({**j,**{k+'_sha256':seal[k+'_sha256'] for k in ('checkpoint','report')}})
    # Native data only, used solely for gradient/interface preflight.
    torch.set_num_threads(4);set_random_seed(6)
    loaders=getMyData(0,30,11,128,0)
    hsi,raw,aux,labels=next(iter(loaders[0]))
    batch={'hsi_pca':hsi.to('cuda:0'),'aux':aux.to('cuda:0'),'label':labels.to('cuda:0')}
    del loaders,hsi,raw,aux,labels;gc.collect()
    jobs,checks=[],[]
    for seed in (6,7,8):
        baseline_config=read(ROOT/'configs/runs/sr_native_20261029'/f'M3_seed{seed}.json')
        for kind in ('LL','GG'):
            cfg={**baseline_config,'experiment_name':f'{TAG}_{kind}_seed{seed}',
                 'model_name':f'TransMixer_LG_{kind}_SR2'}
            config_path=ROOT/'configs/runs'/TAG/f'{kind}_seed{seed}.json';write(config_path,cfg)
            exp=ExperimentConfig(**cfg);config._init(exp)
            set_random_seed(seed)
            baseline=NeighborTransMixer('SR2',**baseline_config['sr'],hsi_channels=30,aux_channels=1,
                num_classes=15,hidden_dim=64,num_blocks=2,window_size=11,mode='decoupled').to('cuda:0')
            rng_cpu=torch.get_rng_state().clone();rng_gpu=torch.cuda.get_rng_state().clone()
            set_random_seed(seed);adapter=get_model_adapter(cfg['model_name']);bundle=adapter.build_model(config,0,'cuda:0');net=bundle['net']
            assert torch.equal(rng_cpu,torch.get_rng_state()) and torch.equal(rng_gpu,torch.cuda.get_rng_state())
            excluded='.global_attn.' if kind=='LL' else '.local_dw.'
            original=baseline.state_dict();current=net.state_dict()
            shared=[k for k in original if excluded not in k]
            assert all(torch.equal(original[k],current[k]) for k in shared)
            net.eval();baseline.eval()
            # Restore ONLY substituted operators: full output must match the original.
            saved=[]
            for a,b in zip(net.blocks,baseline.blocks):
                attr='global_attn' if kind=='LL' else 'local_dw'
                saved.append(getattr(a,attr));setattr(a,attr,copy.deepcopy(getattr(b,attr)))
            with torch.no_grad():
                reference=baseline(batch['hsi_pca'],batch['aux'])
                restored=net(batch['hsi_pca'],batch['aux'])
            torch.testing.assert_close(reference,restored,atol=0,rtol=0)
            for block,module in zip(net.blocks,saved):setattr(block,'global_attn' if kind=='LL' else 'local_dw',module)
            net.train();net.zero_grad(set_to_none=True)
            output=adapter.forward_train(bundle,batch);loss=torch.nn.functional.cross_entropy(output,batch['label']);loss.backward()
            assert output.shape==(128,15) and torch.isfinite(loss)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
            changed_prefix='blocks.0.global_attn.' if kind=='LL' else 'blocks.0.local_dw.'
            grad=sum(float(p.grad.abs().sum()) for k,p in net.named_parameters() if k.startswith(changed_prefix) and p.grad is not None)
            assert grad>0
            net.eval()
            with torch.no_grad():expected=adapter.forward_eval(bundle,batch)
            stream=io.BytesIO();torch.save(net,stream);stream.seek(0)
            restored_net=torch.load(stream,weights_only=False).eval()
            with torch.no_grad():actual=restored_net(batch['hsi_pca'],batch['aux'])
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            count=sum(p.numel() for p in net.parameters());base_count=sum(p.numel() for p in baseline.parameters())
            checks.append({'seed':seed,'kind':kind,'parameters':count,'baseline_parameters':base_count,
                'shared_parameters_exact':True,'base_rng_exact':True,'restored_base_forward_exact':True,
                'native_batch128_finite_gradient':True,'replacement_gradient_l1':grad,'whole_model_roundtrip_exact':True})
            identifier=f'{kind}_seed{seed}'
            job={'id':identifier,'group':kind,'seed':seed,'model':cfg['model_name'],'config':str(config_path),
                 'checkpoint':exp.model_save_path,'report':exp.report_path,'native_log':exp.log_path,
                 'stdout':str(LOG/(identifier+'.stdout.log')),'stderr':str(LOG/(identifier+'.stderr.log'))}
            assert not any(Path(job[k]).exists() for k in ('checkpoint','report','native_log','stdout','stderr'))
            jobs.append(job)
            print('PREFLIGHT',identifier,'parameters',count,flush=True)
            del baseline,net,bundle,restored_net,output,loss,expected,actual,stream,original,current,saved
            gc.collect();torch.cuda.empty_cache()
    write(DEST/'preflight.json',checks)
    files=list((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'scripts/keep_awake_lg_controls.py',ROOT/'scripts/train.py',
        ROOT/'configs/train.json',ROOT/'AGENTS.md',PLAN,DEST/'preflight.json']
    files += [Path(j['config']) for j in jobs]
    manifest={'created_at':now(),'tag':TAG,'python':sys.executable,'torch':str(torch.__version__),'numpy':np.__version__,
        'gpu':torch.cuda.get_device_name(0),'new_training_runs':6,'seeds':[6,7,8], 'jobs':jobs,'baselines':baselines,
        'source_and_config_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files},'data_sha256':old['data_sha256'],
        'checkpoint_selection':'unchanged native test OA best','training_entry':'scripts/train.py',
        'capacity_matched':False,'parameters_from_preflight':checks,
        'authorization':'User asked to execute the preceding explicitly described two controls x seeds6/7/8; no additional combinations'}
    write(DEST/'manifest.json',manifest);pins(manifest)
    state({'state':'prepared','new_completed':0,'new_total':6,'queue_pid':None,'training_pid':None})

def summary():
    import numpy as np
    manifest=read(DEST/'manifest.json');rows=[]
    for j in manifest['baselines']:
        rows.append({'id':j['id'],'group':'LG','seed':j['seed'],'reused':True,**report_metrics(j['report'])})
    for j in manifest['jobs']:
        marker=DEST/'completed'/f"{j['id']}.json"
        if marker.exists():rows.append({'id':j['id'],'group':j['group'],'seed':j['seed'],'reused':False,**read(marker)['metrics']})
    stats={}
    for g in ('LG','LL','GG'):
        items=[r for r in rows if r['group']==g]
        if len(items)==3:stats[g]={m:{'mean':float(np.mean([r[m] for r in items])), 'std':float(np.std([r[m] for r in items],ddof=1))} for m in ('oa','aa','kappa')}
    comparisons=[]
    for g in ('LL','GG'):
        for seed in (6,7,8):
            a=next((r for r in rows if r['group']==g and r['seed']==seed),None)
            b=next(r for r in rows if r['group']=='LG' and r['seed']==seed)
            if a:comparisons.append({'group':g,'seed':seed,**{m+'_delta_pp':a[m]-b[m] for m in ('oa','aa','kappa')}})
    write(DEST/'summary.json',{'completed':len(rows),'total':9,'new_completed':len(rows)-3,'new_total':6,'runs':rows,'statistics':stats,'comparisons':comparisons})

def run():
    from scripts.keep_awake_fusion import kernel,is_running
    manifest=read(DEST/'manifest.json');assert read(DEST/'state.json')['state']=='prepared'
    check_idle();pins(manifest)
    status={'state':'starting','queue_pid':os.getpid(),'training_pid':None,'new_completed':0,'new_total':6,'started_at':now()}
    (DEST/'completed').mkdir();state(status)
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
    env.pop('CUBLAS_WORKSPACE_CONFIG',None)
    child=None;helper=None
    try:
        assert kernel.SetThreadExecutionState(0x80000001)
        with (LOG/'power.stdout.log').open('xb') as out,(LOG/'power.stderr.log').open('xb') as err:
            helper=subprocess.Popen([manifest['python'],'-X','utf8','-u','scripts/keep_awake_lg_controls.py'],cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        status['helper_pid']=helper.pid;state(status)
        for _ in range(30):
            if (DEST/'power.json').exists():break
            if helper.poll() is not None:raise RuntimeError('Power guard failed')
            time.sleep(1)
        assert read(DEST/'power.json')['state']=='active'
        for j in manifest['jobs']:
            check_idle();pins(manifest)
            assert helper.poll() is None
            for k in ('checkpoint','report','native_log','stdout','stderr'):
                if Path(j[k]).exists():raise FileExistsError(j[k])
            started=now();clock=time.monotonic()
            status.update(state='launching',current=j['id'],current_epoch=None,current_started_at=started);state(status)
            with Path(j['stdout']).open('xb') as out,Path(j['stderr']).open('xb') as err:
                child=subprocess.Popen([manifest['python'],'-X','utf8','-u','scripts/train.py','--config',j['config']],cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                status.update(state='training',training_pid=child.pid);state(status)
                while child.poll() is None:
                    time.sleep(10)
                    assert kernel.SetThreadExecutionState(0x80000001)
                    if helper.poll() is not None:raise RuntimeError('Guard failed; active scientific child preserved')
                    with Path(j['stderr']).open('rb') as f:
                        f.seek(max(0,Path(j['stderr']).stat().st_size-16000));tail=f.read().decode('utf-8',errors='replace')
                    found=re.findall(r'Epoch (\d+)/100',tail)
                    status.update(current_epoch=int(found[-1]) if found else None,elapsed_seconds=round(time.monotonic()-clock,1));state(status)
            if child.returncode:raise RuntimeError(f"Training failed {j['id']} code{child.returncode}")
            assert 'Report Success!' in Path(j['stdout']).read_text(encoding='utf-8',errors='replace')
            assert re.search(r'Epoch 100/100:\s+100%[^\r\n]*22/22',Path(j['stderr']).read_text(encoding='utf-8',errors='replace'))
            done={'id':j['id'],'seed':j['seed'],'epochs':100,'started_at':started,'finished_at':now(),
                'elapsed_seconds':time.monotonic()-clock,'metrics':report_metrics(j['report']),
                **{k+'_sha256':sha(j[k]) for k in ('checkpoint','report','native_log','stdout','stderr')}}
            pins(manifest);write(DEST/'completed'/f"{j['id']}.json",done);summary()
            status.update(new_completed=status['new_completed']+1,training_pid=None);state(status)
            print('COMPLETED',j['id'],done['metrics'],flush=True)
        pins(manifest);summary()
        write(DEST/'final_integrity_audit.json',{'all_passed':True,'checked_at':now(),'completed_new':6,
            'protected_sources_unchanged':True,'epochs100_and_native_reports':True,'source_config_data_and_reused_outputs_verified':True})
        status.update(state='completed',current=None,training_pid=None,finished_at=now());state(status)
    except BaseException as exc:
        status.update(state='failed',error=repr(exc));state(status);raise
    finally:
        while child and child.poll() is None:
            kernel.SetThreadExecutionState(0x80000001);time.sleep(15)
        kernel.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','summary']);args=p.parse_args()
    {'prepare':prepare,'run':run,'summary':summary}[args.action]()
