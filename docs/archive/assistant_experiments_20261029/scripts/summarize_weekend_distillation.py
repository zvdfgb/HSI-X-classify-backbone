"""Read completed backbone reports and training diagnostics; no inference."""
import argparse,csv,hashlib,json,re,sys
from datetime import datetime,timezone,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_transmixer_ablation import read_metrics
LABELS={'T1':'RSCNet 教师','T2':'CNN 教师','K1':'B0 + RSC 蒸馏','K2':'F4 + RSC 蒸馏','K3':'F4 + CNN 蒸馏','K4':'F4 + RSC 置信过滤','C1':'F4 + 标签平滑'}
CLASSES=['健康草地','受胁草地','合成草地','树木','土壤','水体','住宅','商业区','道路','高速公路','铁路','停车场1','停车场2','网球场','跑道']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def parse_report(path):
    import numpy as np
    text=Path(path).read_text(encoding='utf-8');metrics=read_metrics(path)
    match=re.search(r'^\s*\[\[.*?\]\]',text,re.M|re.S)
    if not match:raise ValueError(f'Missing complete confusion matrix: {path}')
    values=list(map(int,re.findall(r'\d+',match.group())))
    if len(values)!=225:raise ValueError(f'Expected 225 matrix values, got {len(values)}')
    cm=np.array(values).reshape(15,15);n=int(cm.sum());assert n==12197
    oa=float(np.trace(cm)/n*100);aa=float(np.mean(np.diag(cm)/cm.sum(1))*100);pe=float(np.dot(cm.sum(1),cm.sum(0))/n**2);k=float((oa/100-pe)/(1-pe)*100)
    assert max(abs(oa-metrics['oa']),abs(aa-metrics['aa']),abs(k-metrics['kappa']))<1e-8
    return {**metrics,'test_samples':n,'per_class_recall':(np.diag(cm)/cm.sum(1)*100).tolist(),'confusion_matrix':cm.tolist(),'report_sha256':sha(path)}

def build(folder):
    import torch
    results=read(folder/'results.json');manifest=read(folder/'manifest.json');seal=read(folder/'原三方法最终验收.json')
    if results['status']!='completed' or len(results['jobs'])!=7:raise RuntimeError('All seven new jobs must finish before final summary')
    checks={p:sha(ROOT/p)==v for p,v in manifest['source_sha256'].items()}
    if not all(checks.values()):raise RuntimeError('New experiment source hashes changed')
    original=[]
    for old in seal['summary']['runs']:
        row={**old,**parse_report(old['report']),'protocol':'legacy','id':{'TransMixer_B0_Decoupled':'F0','TransMixer_Fusion_F1':'F1','TransMixer_Fusion_F4':'F4'}[old['model']]}
        assert all(row[k]==old[k] for k in ('oa','aa','kappa'))
        original.append(row)
    baselines={r['model']:r for r in original};audits=[];jobs=[]
    preflight={r['model']:r['parameters'] for r in read(folder/'预检核验.json')['GPU']}
    for job in results['jobs']:
        if job['status']!='success':raise RuntimeError(f"Incomplete {job['id']}")
        row={**job,**parse_report(job['report']),'method':LABELS[job['id']],'checkpoint_sha256':sha(job['checkpoint']),'protocol':'legacy'}
        text=Path(job['console']).read_text(encoding='utf-8',errors='replace')
        if not re.search(r'Epoch 100/100:\s*100%.*22/22',text):raise RuntimeError('Final epoch absent')
        if 'baseline' in job:row['delta_oa_vs_student']=row['oa']-baselines[job['baseline']]['oa']
        net=torch.load(job['checkpoint'],map_location='cpu',weights_only=False)
        count=sum(p.numel() for p in net.parameters())
        if not all(torch.isfinite(p).all() for p in net.parameters()):raise RuntimeError('Nonfinite saved model')
        row['parameters']=count
        if job['role']!='teacher':
            cfg=read(job['config']);diag=read(cfg['distillation']['diagnostics_path'])
            if count!=baselines[job['baseline']]['parameters']:raise RuntimeError('Student inference parameter count changed')
            if hasattr(net,'teacher') or any('teacher' in k for k in net.state_dict()):raise RuntimeError('Teacher stored in student')
            if diag['steps']!=2200 or diag['samples']!=281600:raise RuntimeError('Training objective did not cover 100 full epochs')
            if job['role']!='label_smoothing':
                if not diag['teacher_unchanged'] or not diag['teacher_has_no_grad']:raise RuntimeError('Teacher freezing audit failed')
                if diag['teacher_checkpoint_sha256']!=sha(cfg['distillation']['teacher_checkpoint']):raise RuntimeError('Teacher source mismatch')
            row['diagnostics']=diag
        else:assert count==preflight[job['model']]
        del net
        audits.append({'id':job['id'],'all_100_epochs_complete':True,'matrix_metrics_recomputed':True,'parameters':count,'checkpoint_sha256':row['checkpoint_sha256']})
        row['duration_seconds']=(datetime.fromisoformat(job['finished_at'])-datetime.fromisoformat(job['started_at'])).total_seconds()
        jobs.append(row)
    allrows=original+jobs;byid={r['id']:r for r in allrows}
    pairs=[('F1','F0'),('F4','F0'),('F4','F1'),('K1','F0'),('K2','F4'),('K3','F4'),('K4','K2'),('K2','C1'),('K2','K1'),('K2','K3'),('C1','F4')]
    contrasts=[{'left':l,'right':r,'oa_delta_pp':byid[l]['oa']-byid[r]['oa'],'aa_delta_pp':byid[l]['aa']-byid[r]['aa']} for l,r in pairs]
    formal_path=ROOT/'report/fusion_ablation/fusion_20261024_210925_749584/正式汇总.json';formal=read(formal_path)
    summary={'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'status':'completed','legacy_methods':allrows,'contrasts':contrasts,'validation_protocol_21_jobs':formal,'legacy_baseline_delta_vs_historical_pp':byid['F0']['oa']-88.9726982044765,'audit':{'new_job_checks':audits,'new_sources_unchanged':checks,'legacy_frozen_audit':str(folder/'原三方法最终验收.json')},'scope':'31 formal runs grouped by protocol; seed6 exploratory distillation, not independent validation or statistical significance.'}
    (folder/'周末实验总汇.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    fields=['id','method','model','oa','aa','kappa','parameters','delta_oa_vs_student','duration_seconds','report','checkpoint']
    with (folder/'原流程十组结果.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(allrows)
    lines=['# TransMixer 周末实验全面复盘','',f"汇总时间：{summary['created_at']}。正式训练共31次：验证选模融合21次、原实验室流程融合3次、新增教师与蒸馏对照7次。试启动与合成输入测试不计入正式训练。",'',
           '## 原实验室流程下的十组结果','',
           'Houston2013，seed6，100轮，官方2832训练中心，PCA30，窗口11，batch128，AdamW lr0.0001。所有方法沿用旧测试准确率选模，不能视为独立测试确认。','',
           '| 组 | 方法 | OA % | AA % | Kappa ×100 | 推理参数 |','|---|---|---:|---:|---:|---:|']
    for r in allrows:lines.append(f"| {r['id']} | {r['method']} | {r['oa']:.4f} | {r['aa']:.4f} | {r['kappa']:.4f} | {r['parameters']:,} |")
    lines+=['','## 预先固定的关键对照','','| 比较 | OA差值 百分点 |','|---|---:|']
    for c in contrasts:lines.append(f"| {c['left']} − {c['right']} | {c['oa_delta_pp']:+.4f} |")
    lines+=['',f"本轮原版直接融合与历史88.9727%的差值为 {summary['legacy_baseline_delta_vs_historical_pp']:+.4f} 个百分点，历史checkpoint未找回，因此不要求逐位复现。",'',
           '## 蒸馏设置与诊断','', '温度4，CE加0.3倍温度缩放KL；置信过滤阈值0.7；标签平滑0.1。教师独立训练后固定，同一增强batch生成软目标；学生均从头100轮，推理网络及参数量不变。所有值预先固定，未用学生测试成绩搜索。RSCNet额外接收原始144波段HSI，CNN只接收PCA HSI和辅助模态，因此教师比较也涉及信息量差异。','']
    for r in jobs:
        if 'diagnostics' in r:
            d=r['diagnostics'];lines.append(f"- {r['id']}：2200个优化步骤，平均CE {d['mean_ce']:.5f}，平均KD {d['mean_kd']:.5f}；教师接受比例 {d['teacher_acceptance_fraction']*100:.2f}%；学生参数 {r['parameters']:,}。"+('已核验教师参数与BN状态不变且无梯度。' if 'teacher_unchanged' in d else '无教师的平滑对照。'))
    lines+=['','## 之前验证选模的三种子融合实验','','该批使用2267训练／565验证／12197测试，按验证OA及损失保存模型。与上表分开解释，不把差额归因于模块。','',
           '| 组 | 测试OA 均值±样本SD | 相对F0平均差值 |','|---|---:|---:|']
    for r in formal['aggregates']:lines.append(f"| {r['variant']} | {r['test_oa_mean']:.3f} ± {r['test_oa_sample_sd']:.3f} | {r['paired_delta_mean']:+.3f} |")
    lines+=['','## 逐类召回率','','| 类别 | F0 | F4 | K1 | K2 | K3 | K4 | C1 |','|---|---:|---:|---:|---:|---:|---:|']
    for i,name in enumerate(CLASSES):lines.append('| '+name+' | '+' | '.join(f"{byid[k]['per_class_recall'][i]:.3f}" for k in ('F0','F4','K1','K2','K3','K4','C1'))+' |')
    lines+=['','## 证据与限制','', '原三方法在接入蒸馏接口前已封存源码、报告及checkpoint哈希；新七组与各自manifest匹配，所有OA／AA／Kappa均由原报告混淆矩阵复核。蒸馏教师未进入学生checkpoint。结果仍是单种子、单数据集、测试选模的探索性结果；常规KL蒸馏与双向交互本身不是新方法，不能据此宣称显著或SOTA。平滑组只是实用正则化对照，并非严格匹配所有梯度的因果对照。', '',
           '来源：蒸馏基本形式 https://arxiv.org/abs/1503.02531 ；KL接口 https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html 。','']
    for r in allrows:lines.append(f"- {r['id']}：[原始标准报告](<{r['report']}>)；[checkpoint](<{r['checkpoint']}>)。")
    (folder/'周末实验全面报告.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'output':str(folder),'methods':len(allrows),'formal_runs':31},ensure_ascii=False))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args();build(args.run.resolve())
