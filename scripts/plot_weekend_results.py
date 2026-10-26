"""Plots from audited saved summaries; no model loading or inference."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main(folder):
    d=json.loads((folder/'周末实验总汇.json').read_text(encoding='utf-8'));rows=d['legacy_methods'];by={r['id']:r for r in rows}
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#8794a4' if r['id'] in ('F0','F1','F4') else '#805e94' if r['id'].startswith('T') else '#c28c42' if r['id']=='C1' else '#245c9e' for r in rows]
    fig,ax=plt.subplots(figsize=(8.2,3.2),layout='constrained');x=np.arange(len(rows));ax.bar(x,[r['oa'] for r in rows],color=colors)
    ax.set_xticks(x,[r['id'] for r in rows]);ax.set_ylim(0,103);ax.set_ylabel('测试 OA（%）');ax.grid(axis='y',alpha=.2)
    for i,r in enumerate(rows):ax.text(i,r['oa']+1.1,f"{r['oa']:.2f}",ha='center',fontsize=8)
    fig.savefig(folder/'原流程十组OA.png',dpi=220);plt.close(fig)
    keep=[c for c in d['contrasts'] if (c['left'],c['right']) in [('F4','F0'),('F4','F1'),('K1','F0'),('K2','F4'),('K3','F4'),('K4','K2'),('K2','C1')]]
    fig,ax=plt.subplots(figsize=(8.2,3.7),layout='constrained');vals=[c['oa_delta_pp'] for c in keep];y=np.arange(len(keep))
    ax.barh(y,vals,color=['#245c9e' if v>=0 else '#ad6375' for v in vals]);ax.axvline(0,color='#555',linewidth=.8)
    ax.set_yticks(y,[f"{c['left']} − {c['right']}" for c in keep]);ax.invert_yaxis();ax.set_xlabel('OA 差值（百分点）');ax.grid(axis='x',alpha=.2)
    bound=max(max(map(abs,vals)),.1);ax.set_xlim(min(0,min(vals))-.3*bound,max(0,max(vals))+.3*bound)
    for i,v in enumerate(vals):ax.text(v+(.025*bound if v>=0 else -.025*bound),i,f'{v:+.3f}',va='center',ha='left' if v>=0 else 'right',fontsize=9)
    fig.savefig(folder/'关键对照差值.png',dpi=220);plt.close(fig)
    ids=['K1','K2','K3','K4','C1'];matrix=np.array([np.array(by[k]['per_class_recall'])-np.array(by['F0' if k=='K1' else 'F4']['per_class_recall']) for k in ids]);limit=max(abs(matrix.min()),abs(matrix.max()),.1)
    labels=['健康草地','受胁草地','合成草地','树木','土壤','水体','住宅','商业区','道路','高速公路','铁路','停车场1','停车场2','网球场','跑道']
    fig,ax=plt.subplots(figsize=(8.2,4.4),layout='constrained');im=ax.imshow(matrix,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_xticks(range(15),labels,rotation=55,ha='right',fontsize=8);ax.set_yticks(range(5),['K1−F0','K2−F4','K3−F4','K4−F4','C1−F4']);fig.colorbar(im,ax=ax,label='召回率变化（百分点）')
    fig.savefig(folder/'逐类召回率变化.png',dpi=220);plt.close(fig)
    formal=d['validation_protocol_21_jobs']['aggregates'];fig,ax=plt.subplots(figsize=(8.2,3.2),layout='constrained')
    for r in formal:ax.plot([6,7,8],r['test_oa'],marker='o',label=r['variant'],linewidth=1.4)
    ax.set_xticks([6,7,8]);ax.set_xlabel('训练随机种子');ax.set_ylabel('测试 OA（%）');ax.grid(axis='y',alpha=.2);ax.legend(ncol=7,fontsize=8,loc='upper center');ax.set_ylim(80,90)
    fig.savefig(folder/'验证选模三种子.png',dpi=220);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();main(args.run.resolve())
