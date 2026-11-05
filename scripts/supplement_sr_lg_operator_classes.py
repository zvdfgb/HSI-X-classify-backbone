"""Class-level diagnostic counts from existing native reports; no inference."""
import csv
import json
import re
import statistics as st
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_sr_native import read,write,sha,report_metrics
A=ROOT/'docs/results/sr_lg_mechanisms_20261104'
T=ROOT/'report/sr_lg_operator_controls_20261104'
CN=['健康草地','受胁草地','人造草地','树木','土壤','水体','住宅区','商业区','道路','高速公路','铁路','停车场1','停车场2','网球场','跑道']

def matrix(path):
    t=path.read_text(encoding='utf-8-sig');start=t.rindex('[[');end=t.index(']]',start)+2
    v=list(map(int,re.findall(r'\d+',t[start:end])));assert len(v)==225
    return [v[i:i+15] for i in range(0,225,15)]

if __name__=='__main__':
    assert read(T/'independent_final_audit.json')['all_passed']
    manifest=read(T/'manifest.json');matrices={};official={};digests={}
    for group,jobs in [('new',manifest['jobs']),('baseline',manifest['baselines'])]:
        for j in jobs:
            g=j['group'] if group=='new' else 'LG';p=Path(j['report'])
            seal=read(T/'completed'/f"{j['id']}.json") if group=='new' else j
            assert sha(p)==seal['report_sha256']
            matrices[g,j['seed']]=matrix(p);official[g,j['seed']]=report_metrics(p)
            digests[f'{g}_seed{j["seed"]}']=sha(p)
    rows=[];summary=[]
    for g in ('LL','GG'):
        for seed in (6,7,8):
            left=matrices[g,seed];right=matrices['LG',seed]
            assert list(map(sum,left))==list(map(sum,right))
            net=sum(left[c][c]-right[c][c] for c in range(15))
            assert abs(net*100/12197-(official[g,seed]['oa']-official['LG',seed]['oa']))<1e-10
            for c in range(15):
                n=sum(right[c]);delta=left[c][c]-right[c][c]
                rows.append({'group':g,'seed':seed,'class_cn':CN[c],'n':n,'net_correct_vs_LG':delta,'recall_delta_pp':delta*100/n})
        for c in CN:
            z=[r for r in rows if r['group']==g and r['class_cn']==c]
            summary.append({'group':g,'class_cn':c,'n':z[0]['n'],'mean_recall_delta_pp':st.mean(r['recall_delta_pp'] for r in z),
                'positive_seeds':sum(r['net_correct_vs_LG']>0 for r in z),
                **{f'seed{r["seed"]}_net':r['net_correct_vs_LG'] for r in z}})
    initial=A/'SNAPSHOT.before_class_supplement.json'
    if not initial.exists():initial.write_bytes((A/'SNAPSHOT.json').read_bytes())
    for name,values in [('算子对照_逐种子类别变化.csv',rows),('算子对照_类别汇总.csv',summary)]:
        with (A/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    text='# 算子对照的类别补充分析\n\n本表只读取已经封存的九份原生报告，不新增训练或推理。数值为统一算子相对混合分工LG的类别召回率变化，单位百分点；正数表示统一算子更好。三个种子不能作统计显著性证明。\n\n'
    text+='| 类别 | LL平均变化 | LL正向种子数 | GG平均变化 | GG正向种子数 |\n|---|---:|---:|---:|---:|\n'
    for c in CN:
        l=next(r for r in summary if r['group']=='LL' and r['class_cn']==c);g=next(r for r in summary if r['group']=='GG' and r['class_cn']==c)
        text+=f'| {c} | {l["mean_recall_delta_pp"]:+.4f} | {l["positive_seeds"]}/3 | {g["mean_recall_delta_pp"]:+.4f} | {g["positive_seeds"]}/3 |\n'
    road={g:next(r for r in summary if r['group']==g and r['class_cn']=='道路') for g in ('LL','GG')}
    with (A/'diagnostics/class_effects_summary.csv').open(encoding='utf-8-sig',newline='') as f:old_classes=list(csv.DictReader(f))
    before=next(r for r in old_classes if r['comparison']=='L_alone' and r['class_cn']=='道路')
    after=next(r for r in old_classes if r['comparison']=='SR_on_L' and r['class_cn']=='道路')
    text+=f'\n最明确的类别线索在道路：混合分工相对LL平均提高 {-road["LL"]["mean_recall_delta_pp"]:.4f} 个百分点，相对GG提高 {-road["GG"]["mean_recall_delta_pp"]:.4f}，两个比较均三个种子同向。整体OA持平不能概括为局部分支完全没有作用。\n\n'
    text+=f'结合原有诊断，分工单独相对纯SSM使道路类平均变化 {float(before["recall_delta_mean_pp"]):+.4f} 个百分点；在分工上加SR后，道路类平均变化 {float(after["recall_delta_mean_pp"]):+.4f}。目前可以提出“SR与混合算子的配合在道路类上更明显”这一研究线索，但不能由一个类别推出普遍的边界改善，更不能认定已证明因果机制。\n\n'
    text+='\n整体OA几乎持平，不代表各类别表现相同。类别改变还可能受到参数量、训练随机性及分支归一化／残差结构影响；该表不证明局部分支没有作用，也不证明两路全局在其他数据集更好。两路全局三个种子的OA样本标准差较小，只是本次观察。\n\n所有新对照均保留SR；不能仅用这些对照认定混合算子的优势由SR产生。1∶1比例没有与其他比例比较，也未证明最优。\n\n[详细报告](分析报告.md) · [逐种子类别计数](算子对照_逐种子类别变化.csv) · [类别汇总CSV](算子对照_类别汇总.csv)\n'
    (A/'算子对照类别补充分析.md').write_text(text,encoding='utf-8')
    index=A/'README.md';old=index.read_text(encoding='utf-8')
    link='- [算子对照类别补充分析](算子对照类别补充分析.md)'
    if link not in old:index.write_text(old.replace('- [分析报告](分析报告.md)','- [分析报告](分析报告.md)\n'+link),encoding='utf-8')
    snap=read(A/'SNAPSHOT.json');snap['supplementary_class_analysis']={'new_training_or_inference':False,'source_report_sha256':digests,'script_sha256':sha(Path(__file__)),'previous_snapshot':initial.name}
    snap['files']={p.relative_to(A).as_posix():sha(p) for p in A.rglob('*') if p.is_file() and p.name!='SNAPSHOT.json'}
    write(A/'SNAPSHOT.json',snap)
    for g in ('LL','GG'):
        ranked=sorted([r for r in summary if r['group']==g],key=lambda r:r['mean_recall_delta_pp'])
        print(g,'lowest',ranked[:3],'highest',ranked[-3:])
