"""Use the lab optimizer/data/checkpoint pipeline; teachers live outside net."""
import hashlib,json,math
from pathlib import Path
import torch
from torch.nn import functional as F
from ..transmixer_net.model import TransMixerNet
from ..transmixer_fusion.model import FusionTransMixerNet
from .loss import temperature_kl

VARIANTS={
    'TransMixer_KD_B0_RSC':('B0','RSCNet','kd'),
    'TransMixer_KD_F4_RSC':('F4','RSCNet','kd'),
    'TransMixer_KD_F4_CNN':('F4','CNN','kd'),
    'TransMixer_KD_F4_RSC_Confidence':('F4','RSCNet','confidence'),
    'TransMixer_Control_F4_LabelSmooth':('F4',None,'label_smoothing'),
}

def state_hash(net):
    h=hashlib.sha256()
    for name,tensor in net.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def build_model(config,dataset_type,device):
    from src.model_registry import get_model_adapter
    manager=getattr(config,'_global_config_manager',None)
    cfg=manager.config if manager is not None else getattr(config,'config',config)
    student,teacher_name,mode=VARIANTS[cfg.model_name]
    settings=dict(getattr(cfg,'distillation',None) or {})
    allowed={'temperature','weight','confidence_threshold','label_smoothing','teacher_checkpoint','teacher_sha256','diagnostics_path'}
    if set(settings)-allowed:raise ValueError(f'Unknown distillation fields: {set(settings)-allowed}')
    weight=float(settings.get('weight',.3));temp=float(settings.get('temperature',4))
    if not math.isfinite(weight) or weight<0 or not math.isfinite(temp) or temp<=0:raise ValueError('Invalid distillation weight/temperature')
    epsilon=float(settings.get('label_smoothing',.1))
    if not 0<=epsilon<1:raise ValueError('Invalid label smoothing')
    if mode=='confidence':
        threshold=float(settings.get('confidence_threshold',.7))
        if not math.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Invalid confidence threshold')
    kwargs=dict(hsi_channels=config.get_value('channels'),aux_channels=config.get_value('lidar_or_sar_channels')[dataset_type],num_classes=config.get_value('out_features')[dataset_type],window_size=config.get_value('windowSize'))
    net=TransMixerNet(**kwargs) if student=='B0' else FusionTransMixerNet('F4',**kwargs)
    net=net.to(device)
    bundle={'net':net,'mode':mode,'settings':settings,'weight':weight,'temperature':temp,'epsilon':epsilon}
    if teacher_name is not None and weight>0:
        path=Path(settings['teacher_checkpoint'])
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=settings.get('teacher_sha256'):raise ValueError('Teacher checkpoint SHA256 mismatch')
        # Only the explicitly hashed, locally trained checkpoint is loaded.
        teacher=torch.load(path,map_location=device,weights_only=False)
        teacher.requires_grad_(False);teacher.eval()
        bundle.update(teacher=teacher,teacher_adapter=get_model_adapter(teacher_name),teacher_checkpoint_sha256=digest,teacher_initial_state_sha256=state_hash(teacher))
    bundle['statistics']={k:torch.zeros((),device=device,dtype=torch.float64) for k in ('ce_sum','kd_sum','total_sum','samples','accepted','teacher_correct')}
    bundle['steps']=0
    return bundle

def forward_train(bundle,batch):
    return bundle['net'](batch['hsi_pca'],batch['aux'])

def forward_eval(bundle,batch):
    # Native evaluator only loads the saved student and never needs a teacher.
    return forward_train(bundle,batch)

def compute_loss(bundle,logits,batch,criterion):
    labels=batch['label'];zero=logits.sum()*0
    if bundle['mode']=='label_smoothing':
        ce=F.cross_entropy(logits,labels,label_smoothing=bundle['epsilon']);kd=zero
        accepted=torch.zeros(len(labels),device=logits.device,dtype=torch.bool);correct=zero
    else:
        ce=criterion(logits,labels);kd=zero
        accepted=torch.zeros(len(labels),device=logits.device,dtype=torch.bool);correct=zero
        if bundle['weight']>0:
            teacher=bundle['teacher'];teacher.eval()
            with torch.no_grad():
                teacher_logits=bundle['teacher_adapter'].forward_eval({'net':teacher},batch)
            if not torch.isfinite(teacher_logits).all():raise FloatingPointError('Nonfinite teacher logits')
            threshold=bundle['settings'].get('confidence_threshold',.7) if bundle['mode']=='confidence' else None
            kd,accepted=temperature_kl(logits,teacher_logits,bundle['temperature'],threshold)
            correct=(teacher_logits.argmax(1)==labels).sum()
    loss=ce+bundle['weight']*kd
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite distillation objective')
    stats=bundle['statistics'];size=len(labels)
    with torch.no_grad():
        for key,value in [('ce_sum',ce.detach()*size),('kd_sum',kd.detach()*size),('total_sum',loss.detach()*size),('samples',size),('accepted',accepted.sum()),('teacher_correct',correct)]:stats[key].add_(value)
    bundle['steps']+=1
    return loss

def on_training_complete(bundle):
    stats={k:float(v.item()) for k,v in bundle['statistics'].items()};samples=max(stats['samples'],1)
    report={'steps':bundle['steps'],'samples':int(stats['samples']),'teacher_acceptance_fraction':stats['accepted']/samples,'mean_ce':stats['ce_sum']/samples,'mean_kd':stats['kd_sum']/samples,'mean_total_loss':stats['total_sum']/samples,'mode':bundle['mode'],'settings':bundle['settings'],'student_parameters':sum(p.numel() for p in bundle['net'].parameters()),'teacher_in_student_state_dict':any(k.startswith('teacher') for k in bundle['net'].state_dict()),'scope':'Training-batch diagnostics on augmented patches, not test performance'}
    if 'teacher' in bundle:
        teacher=bundle['teacher'];after=state_hash(teacher)
        unchanged=after==bundle['teacher_initial_state_sha256'];no_grad=all(p.grad is None and not p.requires_grad for p in teacher.parameters())
        report.update(teacher_checkpoint_sha256=bundle['teacher_checkpoint_sha256'],teacher_initial_state_sha256=bundle['teacher_initial_state_sha256'],teacher_final_state_sha256=after,teacher_unchanged=unchanged,teacher_has_no_grad=no_grad,teacher_training_accuracy=stats['teacher_correct']/samples)
        if not unchanged or not no_grad:raise RuntimeError('Frozen teacher state/gradient audit failed')
    if report['teacher_in_student_state_dict']:raise RuntimeError('Teacher leaked into student checkpoint')
    path=bundle['settings'].get('diagnostics_path')
    if path:
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
