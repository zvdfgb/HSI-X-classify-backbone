"""Queue seven jobs through existing train.py after the old fusion batch ends.

No data processing, optimizer, training loop, inference, or checkpoint selection
is implemented here. Model losses are supplied through optional adapter hooks.
"""
import argparse,hashlib,json,os,re,subprocess,sys,time,traceback
from datetime import datetime,timezone,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)

def wait_and_seal(plan,folder):
    import psutil
    legacy=ROOT/plan['legacy_run'];pid=plan['legacy_process_id']
    while psutil.pid_exists(pid):
        proc=psutil.Process(pid)
        if 'run_lab_configs.ps1' not in ' '.join(proc.cmdline()):break
        write(folder/'运行状态.json',{'state':'waiting_for_legacy','updated_at':now(),'runner_pid':os.getpid(),'legacy_process_id':pid,'legacy_run':str(legacy),'message':'Waiting for the already running three-method native lab batch; no new GPU work yet.'})
        time.sleep(20)
    # Finalize with the already validated report parser before any core edits.
    code=subprocess.call([sys.executable,'-X','utf8',str(ROOT/'scripts/compare_lab_fusion.py'),'--run',str(legacy)],cwd=ROOT)
    if code:raise RuntimeError('Original fusion batch did not finish cleanly; no distillation started')
    for proc in psutil.process_iter(['name','cmdline']):
        try:
            command=' '.join(proc.info.get('cmdline') or []).replace('\\','/').lower()
            if 'compare_lab_fusion.py' in command and str(legacy).replace('\\','/').lower() in command:
                proc.wait(timeout=60)
        except psutil.NoSuchProcess:pass
    summary=read(legacy/'summary.json');manifest=read(legacy/'manifest.json')
    if summary['status']!='completed':raise RuntimeError('Legacy batch incomplete')
    if not all(sha(ROOT/p)==v for p,v in manifest['source_sha256'].items()):raise RuntimeError('Legacy core changed before dependency audit')
    audits=[]
    for job in manifest['models']:
        cfg=job['config'];console=legacy/(job['model']+'.stderr.log')
        if not re.search(r'Epoch 100/100:\s*100%.*22/22',console.read_text(encoding='utf-8',errors='replace')):raise RuntimeError(f"Missing final epoch for {job['model']}")
        audits.append({'model':job['model'],'report':cfg['report_path'],'checkpoint':cfg['model_save_path'],'report_sha256':sha(cfg['report_path']),'checkpoint_sha256':sha(cfg['model_save_path']),'all_100_epochs_complete':True})
    sealed={'sealed_at':now(),'legacy_run':str(legacy),'summary':summary,'source_sha256_at_completion':manifest['source_sha256'],'runs':audits,'note':'Frozen BEFORE adding optional KD hooks; later intentional source edits do not retroactively change these completed jobs.'}
    write(folder/'原三方法最终验收.json',sealed);write(legacy/'最终验收封存.json',sealed)
    return summary

def install_hooks(folder):
    record=folder/'蒸馏接口接入记录.json'
    if record.exists():
        if all(sha(ROOT/c['file'])==c['after'] for c in read(record)['changes']):return
        raise RuntimeError('Existing hook installation differs from recorded source; audit before recovery')
    files=['src/config.py','src/trainer.py','src/model_registry.py']
    backup=folder/'接入前原代码';backup.mkdir(exist_ok=True)
    changes=[]
    for rel in files:
        path=ROOT/rel;original=path.read_bytes();(backup/path.name).write_bytes(original)
        text=original.decode('utf-8');nl='\r\n' if '\r\n' in text else '\n'
        if rel.endswith('config.py'):
            old='    random_seed: int = 6'
            new=old+nl+'    distillation: Optional[Dict[str, Any]] = None  # optional training-only adapter settings'
        elif rel.endswith('model_registry.py'):
            old='MODEL_REGISTRY = {'
            names=('TransMixer_KD_B0_RSC','TransMixer_KD_F4_RSC','TransMixer_KD_F4_CNN','TransMixer_KD_F4_RSC_Confidence','TransMixer_Control_F4_LabelSmooth')
            new=old+nl+nl.join(f'    "{n}": "src.models.transmixer_distill.adapter",' for n in names)
        else:
            old='            loss = criterion(outputs, batch["label"])'
            new=nl.join(['            compute_loss = getattr(model_adapter, "compute_loss", None)',
                         '            loss = (compute_loss(model_bundle, outputs, batch, criterion)',
                         '                    if compute_loss is not None else criterion(outputs, batch["label"]))'])
        if text.count(old)!=1:raise RuntimeError(f'Unexpected source context in {rel}; refusing automatic rewrite')
        updated=text.replace(old,new,1)
        if rel.endswith('trainer.py'):
            anchor="    getLog(log_path, '-------------------Finished Training-------------------')"
            if updated.count(anchor)!=1:raise RuntimeError('Missing finish callback anchor')
            updated=updated.replace(anchor,anchor+nl+nl+'    on_training_complete = getattr(model_adapter, "on_training_complete", None)'+nl+'    if on_training_complete is not None:'+nl+'        on_training_complete(model_bundle)',1)
        path.write_bytes(updated.encode('utf-8'))
        changes.append({'file':rel,'before':hashlib.sha256(original).hexdigest(),'after':sha(path)})
    write(folder/'蒸馏接口接入记录.json',{'time':now(),'changes':changes,'scope':'Optional loss and completion hooks; original CE fallback, optimizer, data, checkpoint and evaluation remain.'})

def validate(folder):
    env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONIOENCODING':'utf-8'}
    with (folder/'接口测试.log').open('w',encoding='utf-8') as log:
        code=subprocess.call([sys.executable,'-X','utf8','-m','unittest','src.models.transmixer_distill.test_distillation','-v'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if code:raise RuntimeError('Distillation tests failed; no new training launched')
    from src.config import ExperimentConfig,ConfigManager
    from src.model_registry import get_model_adapter
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    checks=[]
    for name in ('RSCNet','CNN','TransMixer_B0_Decoupled','TransMixer_Fusion_F4'):
        cfg=ExperimentConfig(model_name=name,dataset_type=0,enable_visualization=False)
        adapter=get_model_adapter(name);torch.manual_seed(6)
        bundle=adapter.build_model(ConfigManager(cfg),0,torch.device('cuda:0'));bundle['net'].to('cuda:0').train()
        batch={'hsi_pca':torch.randn(2,1,30,11,11,device='cuda'),'hsi':torch.randn(2,1,144,11,11,device='cuda'),'aux':torch.randn(2,1,11,11,device='cuda'),'label':torch.tensor([0,1],device='cuda')}
        logits=adapter.forward_train(bundle,batch);loss=torch.nn.functional.cross_entropy(logits,batch['label']);loss.backward()
        if logits.shape!=(2,15) or not torch.isfinite(logits).all() or not all(p.grad is None or torch.isfinite(p.grad).all() for p in bundle['net'].parameters()):raise RuntimeError('GPU adapter validation failed')
        checks.append({'model':name,'forward_backward':'passed','parameters':sum(p.numel() for p in bundle['net'].parameters())})
        del bundle,logits,loss,batch;torch.cuda.empty_cache()
    write(folder/'预检核验.json',{'tests':'unittest passed; see 接口测试.log','GPU':checks,'torch':torch.__version__,'device':torch.cuda.get_device_name(0)})

def snapshot_sources():
    paths=list((ROOT/'src').rglob('*.py'))+[ROOT/'scripts/train.py',Path(__file__),ROOT/'scripts/train_transmixer_ablation.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def run_jobs(plan,folder,legacy):
    from src.config import ExperimentConfig
    from scripts.train_transmixer_ablation import execute_plan
    base=read(ROOT/plan['base_config']);results=[];by_id={}
    for item in plan['jobs']:
        job=dict(item);jobfolder=folder/job['id'];jobfolder.mkdir(exist_ok=False)
        values={**base,'experiment_name':f"{plan['run_tag']}_{job['id']}_seed{plan['seed']}",'model_name':job['model'],'random_seed':plan['seed'],'epochs':plan['epochs']}
        if job['role']!='teacher':
            settings={'temperature':plan['temperature'],'weight':plan['weight'],'diagnostics_path':str(jobfolder/'损失与教师核验.json')}
            if job['role']=='label_smoothing':settings['label_smoothing']=plan['label_smoothing']
            else:
                teacher=by_id[job['teacher']]
                if teacher['status']!='success':raise RuntimeError('Teacher dependency did not finish')
                settings.update(teacher_checkpoint=teacher['checkpoint'],teacher_sha256=sha(teacher['checkpoint']))
            if job['role']=='confidence':settings['confidence_threshold']=plan['confidence_threshold']
            values['distillation']=settings
        cfg=ExperimentConfig(**values)
        job.update(status='running',started_at=now(),config=values,job_folder=str(jobfolder))
        results.append(job)
        write(folder/'运行状态.json',{'state':'training','updated_at':now(),'runner_pid':os.getpid(),'current_job':job['id'],'model':job['model'],'completed_jobs':sum(j['status']=='success' for j in results),'total_jobs':len(plan['jobs'])})
        write(folder/'results.json',{'jobs':results,'updated_at':now(),'status':'running'})
        code=execute_plan([cfg],jobfolder)
        row=read(jobfolder/'summary.json')[0]
        job.update(row,finished_at=now());by_id[job['id']]=job
        write(folder/'results.json',{'jobs':results,'updated_at':now(),'status':'running' if not code else 'failed'})
        if code:raise RuntimeError(f"Job {job['id']} failed; see {jobfolder}")
    write(folder/'results.json',{'jobs':results,'updated_at':now(),'status':'completed'})
    return results

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=ROOT/'configs/weekend_distillation.json');args=parser.parse_args()
    plan=read(args.config);folder=ROOT/'report/weekend_distillation'/plan['run_tag'];folder.mkdir(parents=True,exist_ok=True)
    lock=folder/'runner.lock'
    try:
        with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    except FileExistsError:raise SystemExit('Run lock already exists; audit before retrying')
    write(folder/'plan.json',plan)
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
    try:
        legacy=wait_and_seal(plan,folder)
        write(folder/'运行状态.json',{'state':'installing_optional_kd_hooks','updated_at':now(),'runner_pid':os.getpid()})
        install_hooks(folder);validate(folder)
        manifest={'created_at':now(),'plan':plan,'source_sha256':snapshot_sources(),'python':sys.executable,'legacy_final_audit':str(folder/'原三方法最终验收.json'),'training_entry':'scripts/train.py through existing execute_plan','checkpoint_selection':'legacy_test_accuracy','environment':{k:os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONIOENCODING')}}
        write(folder/'manifest.json',manifest)
        run_jobs(plan,folder,legacy)
        if not all(sha(ROOT/p)==v for p,v in manifest['source_sha256'].items()):raise RuntimeError('Sources changed during distillation jobs')
        subprocess.check_call([sys.executable,'-X','utf8',str(ROOT/'scripts/summarize_weekend_distillation.py'),'--run',str(folder)],cwd=ROOT)
        write(folder/'运行状态.json',{'state':'completed','updated_at':now(),'runner_pid':os.getpid(),'completed_jobs':len(plan['jobs']),'total_jobs':len(plan['jobs']),'message':'All seven native-lab jobs completed; final comprehensive report awaits audit and rendering.'})
    except Exception as exc:
        write(folder/'运行状态.json',{'state':'failed','updated_at':now(),'runner_pid':os.getpid(),'error':str(exc),'traceback':traceback.format_exc()})
        traceback.print_exc();return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
