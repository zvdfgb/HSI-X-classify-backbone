"""Summarize existing lab reports; never trains, selects, or evaluates a model.

Optional waiting only watches the supplied lab training process and refreshes
partial summaries. All training is performed by unchanged scripts/train_all.py.
"""
import argparse,csv,hashlib,json,re,sys,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_transmixer_ablation import read_metrics

LABELS={'TransMixer_B0_Decoupled':'F0 原版直接融合','TransMixer_Fusion_F1':'F1 辅助分支自增强','TransMixer_Fusion_F4':'F4 整体双向交互'}
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def save(folder,manifest,alive):
 rows=[]
 for model in manifest['models']:
  cfg=model['config']; report=Path(cfg['report_path']);log=Path(cfg['log_path'])
  row={'model':cfg['model_name'],'method':LABELS[cfg['model_name']],'seed':cfg['random_seed'],'epochs':cfg['epochs'],'status':'pending','report':str(report),'checkpoint':cfg['model_save_path'],'parameters':model['parameters']}
  if log.is_file():row['status']='training_or_evaluating'
  if report.is_file():
   import numpy as np
   text=report.read_text(encoding='utf-8')
   metrics=read_metrics(report)
   # The task header contains depth=[[...], [...], 2]. Match only a
   # matrix beginning on its own line, otherwise the header is consumed too.
   cm_text=re.search(r'^\s*\[\[.*?\]\]',text,re.S|re.M)
   if cm_text:
    numbers=list(map(int,re.findall(r'\d+',cm_text.group())))
    assert len(numbers)==225,(report,len(numbers))
    cm=np.array(numbers).reshape(15,15);n=int(cm.sum());assert n==12197,n
    oa=float(np.trace(cm)/n*100);aa=float(np.mean(np.diag(cm)/cm.sum(1))*100)
    pe=float(np.dot(cm.sum(1),cm.sum(0))/n**2);kappa=float((oa/100-pe)/(1-pe)*100)
    assert max(abs(oa-metrics['oa']),abs(aa-metrics['aa']),abs(kappa-metrics['kappa']))<1e-8
    row.update(metrics,status='metrics_complete',test_samples=n,confusion_matrix=cm.tolist(),per_class_recall=(np.diag(cm)/cm.sum(1)*100).tolist())
    row['visualization_complete']=Path(cfg['image_path']).is_file()
    if log.is_file():
     match=re.findall(r'max_acc:\s*([-+0-9.eE]+)',log.read_text(encoding='utf-8'))
     if match:row['best_training_test_oa']=float(match[-1])*100
   elif not alive:row['status']='incomplete_report'
  if not alive and row['status'] in ('pending','training_or_evaluating'):row['status']='failed_or_interrupted'
  rows.append(row)
 core={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in manifest['source_sha256'].items()}
 assert all(core.values()),'Source changed during training; preserve outputs and audit before interpreting.'
 stdout_path=folder/('batch.stdout.log' if (folder/'batch.stdout.log').exists() else 'train_all.stdout.log')
 stdout=stdout_path.read_text(encoding='utf-8',errors='replace') if stdout_path.exists() else ''
 complete=all(r['status']=='metrics_complete' for r in rows)
 status='running' if alive else ('completed' if complete and 'Failed: 0' in stdout else 'needs_attention')
 state_path=folder/'运行状态.json'
 state=json.loads(state_path.read_text(encoding='utf-8-sig')) if state_path.exists() else {}
 state.update(state=status,updated_at=now(),metrics_completed=sum('oa' in r for r in rows),total_jobs=len(rows))
 state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
 summary={'status':status,'updated_at':now(),'checkpoint_selection':manifest['checkpoint_selection'],'core_source_unchanged':core,'runs':rows,'interpretation_scope':'one seed; original laboratory test-selected checkpoint protocol; no independent validation claim'}
 base=next((r for r in rows if r['model']=='TransMixer_B0_Decoupled' and 'oa' in r),None)
 for r in rows:
  if base and 'oa' in r:r['delta_oa_vs_F0']=r['oa']-base['oa']
 f1=next((r for r in rows if r['model']=='TransMixer_Fusion_F1' and 'oa' in r),None)
 f4=next((r for r in rows if r['model']=='TransMixer_Fusion_F4' and 'oa' in r),None)
 if f1 and f4:summary['F4_minus_F1_oa']=f4['oa']-f1['oa']
 (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 fields=['method','model','seed','epochs','status','oa','aa','kappa','delta_oa_vs_F0','parameters','test_samples','report','checkpoint']
 with (folder/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 lines=['# 原实验室 backbone 下的特征融合对比','',f'更新时间：{now()}。状态：{status}。','',
 '本轮读取 configs/train.json 的旧设置，分别调用已有 scripts/train.py → ExperimentRunner → src/trainer.py 和 src/evaluator.py。外层 PowerShell 仅依次启动三个原训练命令。未使用 src/fusion_experiment.py，未修改原数据、训练、测试或模型保存流程。','',
 '设置：Houston2013，原版 B0 直接融合、F1 辅助自增强、F4 双向交互；seed 6；100 轮；学习率 0.0001；AdamW 权重衰减 0.01；batch 128；PCA 30；窗口 11×11；官方训练中心 2832、测试中心 12197；不另划验证集；沿用 train.json 的 visualization=false。所有方法均将 OMP/MKL CPU 线程数设为 4，减少线程争用；这是运行环境设置，未改变训练算法。','',
 '**评价口径：沿用旧训练器按测试准确率选择最好 checkpoint。该规则用于衔接历史约 89% 的记录，不能称为验证集选模后的独立测试结果。三个方法均只有 seed6，不能据此声称跨种子稳定或统计显著。**','',
 '| 方法 | 状态 | OA % | AA % | Kappa ×100 | 相对本轮直接融合的 OA 百分点 | 参数 |','|---|---|---:|---:|---:|---:|---:|']
 for r in rows:
  val=lambda k:f"{r[k]:.4f}" if k in r else '待完成'
  lines.append(f"| {r['method']} | {r['status']} | {val('oa')} | {val('aa')} | {val('kappa')} | {val('delta_oa_vs_F0')} | {r['parameters']:,} |")
 if base:
  delta=base['oa']-88.9726982044765
  lines+=['',f"本轮原版直接融合 OA 为 {base['oa']:.4f}%，与历史归档 88.9727% 相差 {delta:+.4f} 个百分点。历史 checkpoint 未找回，不能保证当前软件和预处理版本与历史运行完全一致；差异只作为复现衔接情况。"]
 if base and f4:
  delta=f4['oa']-base['oa']
  lines+=['',f"F4 相对本轮原版直接融合：{delta:+.4f} 个百分点。"+('本次 seed6 下有数值提高。' if delta>0 else '本次 seed6 下没有数值提高。')]
 if f1 and f4:
  delta=f4['oa']-f1['oa']
  lines+=['',f"F4 相对近似等参数 F1：{delta:+.4f} 个百分点。"+('本次交互高于辅助自增强，但独立机制收益仍需多种子确认。' if delta>0 else '本次没有超过辅助自增强，不支持将提升归因于跨模态交互。')]
 if complete:
  lines+=['','## 逐类召回率','', '| 类别 | F0 | F1 | F4 | F4−F0 |','|---|---:|---:|---:|---:|']
  names=['健康草地','受胁草地','合成草地','树木','土壤','水体','住宅','商业区','道路','高速公路','铁路','停车场1','停车场2','网球场','跑道']
  for i,n in enumerate(names):lines.append(f"| {n} | {base['per_class_recall'][i]:.3f} | {f1['per_class_recall'][i]:.3f} | {f4['per_class_recall'][i]:.3f} | {f4['per_class_recall'][i]-base['per_class_recall'][i]:+.3f} |")
 lines+=['','## 原始结果与复核','', '本报告直接解析原 backbone 的标准报告并用 15×15 混淆矩阵复算 OA、AA、Kappa，不重复运行测试。源文件 SHA256 与启动时记录一致。完整参数、入口命令和运行环境见 manifest.json。']
 for r in rows:lines+=['',f"- {r['method']}：[标准报告](<{r['report']}>)；[模型](<{r['checkpoint']}>)。"]
 (folder/'对比报告.md').write_text('\n'.join(lines),encoding='utf-8')
 return summary

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path);parser.add_argument('--wait-for-pid',type=int);args=parser.parse_args()
 folder=args.run.resolve();manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
 process=None
 if args.wait_for_pid:
  import psutil
  try:process=psutil.Process(args.wait_for_pid);created=process.create_time()
  except psutil.NoSuchProcess:pass
 while process is not None:
  try:alive=process.is_running() and process.create_time()==created
  except psutil.NoSuchProcess:alive=False
  if not alive:break
  save(folder,manifest,True);time.sleep(20)
 summary=save(folder,manifest,False)
 (folder/'比较已结束.json').write_text(json.dumps({'status':summary['status'],'finished_at':now(),'summary':str(folder/'summary.json'),'report':str(folder/'对比报告.md')},ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'status':summary['status'],'report':str(folder/'对比报告.md')},ensure_ascii=False))
 return 0 if summary['status']=='completed' else 1
if __name__=='__main__':raise SystemExit(main())
