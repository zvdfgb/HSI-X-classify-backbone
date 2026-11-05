"""CPU-only implementation checks, not scientific training or accuracy tests."""
import copy
import io
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from src import config
from src.config import ExperimentConfig
from src.models.transmixer_paper.model import NeighborTransMixer
from src.models.transmixer_lg_controls.model import replace_operators
from src.models.transmixer_sr_lg_v2 import adapter
from src.models.transmixer_sr_lg_v2.model import SRBalancedTransMixer
from scripts.prepare_sr_native import read,write,sha

DEST=ROOT/'report/sr_lg_refactor_20261104'
PROTECTED=['src/model_registry.py','src/config.py','src/data.py','src/runner.py','src/trainer.py',
    'src/evaluator.py','src/metrics.py','src/models/transmixer_net/model.py',
    'src/models/transmixer_paper/model.py','src/models/transmixer_sr/adapter.py','scripts/train.py','configs/train.json','AGENTS.md']
KW=dict(hsi_channels=30,aux_channels=1,num_classes=15,hidden_dim=64,num_blocks=2,window_size=11,mode='decoupled')
SR=read(ROOT/'configs/runs/sr_native_20261029/M3_seed6.json')['sr']

def count(net):return sum(p.numel() for p in net.parameters())

def main():
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized(), 'CPU preflight must not initialize CUDA'
    locked={p:sha(ROOT/p) for p in PROTECTED}
    results=[]
    # Original complete backbone and the added modules preserve caller RNG.
    for seed in (6,7,8):
        torch.manual_seed(seed);base=NeighborTransMixer('SR2',**SR,**KW)
        expected_rng=torch.get_rng_state().clone()
        torch.manual_seed(seed);candidate=SRBalancedTransMixer('aligned',**SR,**KW)
        assert torch.equal(expected_rng,torch.get_rng_state())
        assert all(torch.equal(v,candidate.state_dict()[k]) for k,v in base.state_dict().items())
        results.append({'check':'shared_initialization_and_rng','seed':seed,'passed':True})
    variants={}
    for mode in ('uniform','aligned','misaligned'):
        torch.manual_seed(6);variants[mode]=SRBalancedTransMixer(mode,**SR,**KW).eval()
    aligned=variants['aligned']
    for net in variants.values():
        assert count(net)==count(aligned)
        assert all(torch.equal(v,net.state_dict()[k]) for k,v in aligned.state_dict().items())
    torch.manual_seed(876);hsi=torch.randn(3,30,11,11);aux=torch.randn(3,1,11,11)
    with torch.no_grad():initial={m:n(hsi,aux) for m,n in variants.items()}
    for mode in variants:
        torch.testing.assert_close(initial['aligned'],initial[mode],atol=0,rtol=0)
        assert all(torch.equal(b.last_gains,torch.ones(3,2)) for b in variants[mode].blocks)
    reference=NeighborTransMixer.neighbor_weights(base,hsi,aux)
    for mode,net in variants.items():
        torch.testing.assert_close(reference,net.gap.weights,atol=0,rtol=0)
        expected=reference.flatten(1)
        if mode=='misaligned':expected=expected[:,net.summary_permutation]
        assert all(torch.equal(b.summary_weights,expected) for b in net.blocks)
    assert (aligned.summary_permutation!=torch.arange(121)).all()
    results.append({'check':'equal_capacity_initial_outputs_and_original_SR_readout','passed':True,'parameters':count(aligned),'baseline_parameters':count(base)})
    # A nonzero gain head stands in for a learned state; no optimizer is used.
    for net in variants.values():
        with torch.no_grad():
            for b in net.blocks:
                b.branch_gate[-1].weight.copy_(torch.linspace(-.25,.25,32).reshape(2,16))
    with torch.no_grad():changed={m:n(hsi,aux) for m,n in variants.items()}
    contrasts={m:float((changed['aligned']-changed[m]).abs().max()) for m in ('uniform','misaligned')}
    assert all(v>1e-8 for v in contrasts.values())
    assert any(float(b.last_gains.std(0).max())>1e-8 for b in aligned.blocks)
    results.append({'check':'SR_context_changes_nonzero_head_outputs','passed':True,'max_logit_differences':contrasts})
    # Direct comparison with the previous GG: only the readout changes to mean.
    torch.manual_seed(6);gg_sr=replace_operators(NeighborTransMixer('SR2',**SR,**KW),'GG').eval()
    cfg=read(ROOT/'configs/runs/sr_native_20261029/M3_seed6.json')
    cfg.update(model_name='TransMixer_LGR2_GG_Avg',experiment_name='sr_lg_refactor_cpu_preflight')
    config._init(ExperimentConfig(**cfg));torch.manual_seed(6)
    gg_bundle=adapter.build_model(config,0,'cpu');gg=gg_bundle['net'].eval()
    assert all(torch.equal(v,gg_sr.state_dict()[k]) for k,v in gg.state_dict().items())
    gg_sr.gap=nn.AdaptiveAvgPool2d(1)
    with torch.no_grad():torch.testing.assert_close(gg(hsi,aux),gg_sr(hsi,aux),atol=0,rtol=0)
    results.append({'check':'GG_without_SR_matches_prior_GG_shared_state_and_mean_readout','passed':True,'parameters':count(gg)})
    # Native config/adapter with a full-sized synthetic batch, backward only.
    cfg.update(model_name='TransMixer_LGR2_SR')
    config._init(ExperimentConfig(**cfg));torch.manual_seed(6)
    bundle=adapter.build_model(config,0,'cpu');net=bundle['net'].train()
    batch={'hsi_pca':torch.randn(128,30,11,11),'aux':torch.randn(128,1,11,11),'label':torch.arange(128)%15}
    logits=adapter.forward_train(bundle,batch);assert logits.shape==(128,15)
    loss=torch.nn.functional.cross_entropy(logits,batch['label']);loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    gradients={}
    for i,b in enumerate(net.blocks):
        g=float(b.branch_gate[-1].weight.grad.abs().sum());assert g>0
        gradients[f'block{i}_gain_head_l1']=g
        assert (b.last_gains>=0).all() and (b.last_gains<=2).all()
    results.append({'check':'native_adapter_batch128_finite_backward','passed':True,'synthetic_inputs_only':True,'gradients':gradients})
    net.eval();sample={k:v[:2] for k,v in batch.items()}
    with torch.no_grad():before=adapter.forward_eval(bundle,sample)
    stream=io.BytesIO();torch.save(net,stream);stream.seek(0);restored=torch.load(stream,weights_only=False).eval()
    with torch.no_grad():after=restored(hsi[:1],aux[:1]);expected=net(hsi[:1],aux[:1])
    torch.testing.assert_close(after,expected,atol=0,rtol=0)
    assert restored.gap.weights.shape[0]==1 and all(b.summary_weights.shape==(1,121) for b in restored.blocks)
    with torch.no_grad():
        sample['label']=torch.tensor([14,14]);torch.testing.assert_close(before,adapter.forward_eval(bundle,sample),atol=0,rtol=0)
    results.append({'check':'whole_model_roundtrip_new_batch_context_and_no_label_dependency','passed':True})
    invalid=[{'random_seed':9},{'window_size':9},{'channels':64},{'ablation_mode':'pure_ssm'},
             {'sr':{'spatial_sigma':-1,'spectral_scale':.2}},{'model_name':'unknown_refactor'}]
    for change in invalid:
        bad={**cfg,**change};config._init(ExperimentConfig(**bad))
        try:adapter.build_model(config,0,'cpu')
        except ValueError:pass
        else:raise AssertionError(change)
    config._init(ExperimentConfig(**cfg))
    try:adapter.build_model(config,1,'cpu')
    except ValueError:pass
    else:raise AssertionError('wrong dataset accepted')
    block=copy.deepcopy(aligned.blocks[0]);block.summary_weights=None
    try:block.summarize(torch.zeros(1,121,32),torch.zeros(1,121,32))
    except RuntimeError:pass
    else:raise AssertionError('stale context accepted')
    results.append({'check':'invalid_scope_configuration_and_missing_context_rejected','passed':True})
    assert not torch.cuda.is_initialized()
    assert locked=={p:sha(ROOT/p) for p in PROTECTED}
    payload={'all_passed':True,'time':datetime.now().astimezone().isoformat(),'device':'cpu',
        'torch':str(torch.__version__),'new_training_runs':0,'real_data_preflight':False,
        'gpu_preflight':False,'registry_modified':False,'checks':results,'protected_sources_sha256':locked,
        'candidate_source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'src/models/transmixer_sr_lg_v2').glob('*.py')},
        'checker_sha256':sha(Path(__file__))}
    write(DEST/'preflight_cpu.json',payload)
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(DEST/f'preflight_failure_{datetime.now():%Y%m%d_%H%M%S}.json',{'error':repr(exc)})
        raise
