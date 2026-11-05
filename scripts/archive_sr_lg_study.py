"""Archive lightweight evidence only after independent final verification."""
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_sr_native import read,write,sha
D=ROOT/'report/sr_lg_mechanisms_20261104'
T=ROOT/'report/sr_lg_operator_controls_20261104'
A=ROOT/'docs/results/sr_lg_mechanisms_20261104'

def copy(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    content=source.read_bytes()
    if destination.exists():assert destination.read_bytes()==content,destination
    else:
        with destination.open('xb') as f:f.write(content)

if __name__=='__main__':
    deadline=time.monotonic()+12*3600
    while not (T/'independent_final_audit.json').exists():
        if read(T/'state.json')['state']=='failed':raise RuntimeError('Queue failed; no completed archive created')
        if time.monotonic()>deadline:raise TimeoutError('Final audit unavailable')
        time.sleep(15)
    audit=read(T/'independent_final_audit.json');assert audit['all_passed']
    for rel,digest in audit['artifact_sha256'].items():assert sha(ROOT/rel)==digest,rel
    A.mkdir(parents=True,exist_ok=True)
    for p in D.glob('*.csv'):copy(p,A/'diagnostics'/p.name)
    for p in D.glob('*.png'):copy(p,A/'figures'/p.name)
    for p in D.glob('*.pdf'):copy(p,A/'figures'/p.name)
    for name in ('analysis_summary.json','manifest.json','illustration_samples.json'):
        copy(D/name,A/'diagnostics'/name)
    for name in ('manifest.json','preflight.json','summary.json','final_integrity_audit.json','independent_final_audit.json'):
        copy(T/name,A/'training'/name)
    for p in (T/'completed').glob('*.json'):copy(p,A/'training/completed'/p.name)
    tm=read(T/'manifest.json')
    for j in tm['jobs']+tm['baselines']:
        group=j['group'] if j in tm['jobs'] else 'LG'
        copy(Path(j['report']),A/'native_reports'/f'{group}_seed{j["seed"]}.txt')
        copy(Path(j['config']),A/'configs'/f'{group}_seed{j["seed"]}.json')
    source_report=ROOT/'docs/experiments/20261104_局部全局与SR机制分析.md'
    text=source_report.read_text(encoding='utf-8').replace('../../report/sr_lg_mechanisms_20261104/','figures/').replace('../results/sr_lg_prior_seed678_20261103/','../sr_lg_prior_seed678_20261103/')
    (A/'分析报告.md').write_text(text,encoding='utf-8')
    (A/'README.md').write_text('# 局部／全局与SR机制研究证据\n\n已完成15个历史checkpoint的只读诊断及LL／GG各seed6/7/8共6次新训。历史LG三项复用。\n\n- [分析报告](分析报告.md)\n- [三种子原始训练汇总](training/summary.json)\n- [独立最终核验](training/independent_final_audit.json)\n- [原生分类报告](native_reports/)\n- [机制诊断表格](diagnostics/)\n- [图表](figures/)\n\n原训练、数据、测试OA选模规则不变。测试参与选模，不构成独立测试证据。三个种子、单数据集；冻结干预不等于重新训练贡献，参数量不同。详见报告限制。\n\n数据、checkpoint、完整日志及逐样本预测保留于本地；此目录只收录轻量证据。manifest中的绝对路径为原机追溯路径。归档报告只调整相对链接，未改动分析文字和数值。\n',encoding='utf-8')
    write(A/'SNAPSHOT.json',{'all_passed':True,'source_report_sha256':sha(source_report),
        'files':{p.relative_to(A).as_posix():sha(p) for p in A.rglob('*') if p.is_file() and p.name!='SNAPSHOT.json'}})
    print('ARCHIVED',A,flush=True)
