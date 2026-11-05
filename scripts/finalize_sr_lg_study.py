"""Wait for the authorized queue, independently seal outputs, finish the report."""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_lg_operator_controls import DEST,ARCHIVE,pins
from scripts.prepare_sr_native import read,write,sha
D=ROOT/'report/sr_lg_mechanisms_20261104'

if __name__=='__main__':
    deadline=time.monotonic()+12*3600
    last=None
    while True:
        status=read(DEST/'state.json')
        if status['state']=='failed':raise RuntimeError(status)
        if status['state']=='completed':break
        if time.monotonic()>deadline:raise TimeoutError('Queue incomplete; no result manufactured')
        key=(status.get('current'),status.get('new_completed'),status.get('current_epoch',0))
        if key!=last:print(key,flush=True);last=key
        time.sleep(15)
    manifest=read(DEST/'manifest.json');pins(manifest)
    assert len(manifest['jobs'])==6
    for j in manifest['jobs']:
        marker=read(DEST/'completed'/f"{j['id']}.json")
        assert marker['epochs']==100
        for k in ('checkpoint','report','native_log','stdout','stderr'):assert sha(j[k])==marker[k+'_sha256']
        assert re.search(r'Epoch 100/100:\s+100%[^\r\n]*22/22',Path(j['stderr']).read_text(encoding='utf-8',errors='replace'))
    dm=read(D/'manifest.json');transition=read(ARCHIVE/'registry_transition.json')
    for rel,digest in dm['source_sha256'].items():
        if Path(rel).as_posix().replace('\\','/')=='src/model_registry.py':
            assert sha(ARCHIVE/'model_registry.before.py')==digest==transition['old_sha256']
            assert sha(ROOT/rel)==transition['new_sha256']
        else:assert sha(ROOT/rel)==digest,rel
    assert dm['data_sha256']==manifest['data_sha256']
    subprocess.run([sys.executable,'-X','utf8','scripts/write_sr_lg_report.py'],cwd=ROOT,check=True)
    t=read(DEST/'summary.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'axes.unicode_minus':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for g,name in [('LG','局部＋全局（复用）'),('LL','两路局部'),('GG','两路全局')]:
        values=[next(r['oa'] for r in t['runs'] if r['group']==g and r['seed']==s) for s in (6,7,8)]
        axes[0].plot([6,7,8],values,'o-',label=name)
    axes[0].set(title='保持SSM和SR的算子对照',xlabel='种子',ylabel='原报告OA（%）',xticks=[6,7,8]);axes[0].legend();axes[0].grid(alpha=.2)
    for g,dx in [('LL',-.15),('GG',.15)]:
        values=[next(r['oa_delta_pp'] for r in t['comparisons'] if r['group']==g and r['seed']==s) for s in (6,7,8)]
        bars=axes[1].bar([s+dx for s in (6,7,8)],values,width=.28,label=g)
        axes[1].bar_label(bars,fmt='%.2f',padding=3)
    axes[1].axhline(0,color='black',lw=.7);axes[1].set(title='统一算子相对混合分工',xlabel='种子',ylabel='OA差值（百分点）',xticks=[6,7,8]);axes[1].legend()
    fig.savefig(D/'operator_controls.png',dpi=200);fig.savefig(D/'operator_controls.pdf');plt.close(fig)
    conclusions=[]
    for g in ('LL','GG'):
        diff=[-r['oa_delta_pp'] for r in t['comparisons'] if r['group']==g]
        conclusions.append(f'混合分工相对 {g} 的平均 OA 增益为 {sum(diff)/3:+.4f} 个百分点，三个种子中 {sum(v>0 for v in diff)} 个为正。')
    stable=all(r['oa_delta_pp']<0 for r in t['comparisons'])
    verdict=('当前三个种子初步支持混合算子优于两种统一算子，仍需保留参数量与单数据集限制。' if stable else '混合分工未能在三个种子中稳定胜过两种统一算子；当前不宜将固定1∶1混合作为已被充分验证的独立贡献。')
    report=ROOT/'docs/experiments/20261104_局部全局与SR机制分析.md'
    with report.open('a',encoding='utf-8') as f:
        f.write('\n\n## 8. 六次对照完成后的判断\n\n'+' '.join(conclusions)+'\n\n'+verdict+'\n\n![新算子对照结果](../../report/sr_lg_mechanisms_20261104/operator_controls.png)\n\n')
        f.write('汇报可以这样说：“这次我把两个点拆开分析了。SR单独使用的收益比较稳定，平均提高约1.09个百分点，作用主要是根据中心相似性筛选邻域。局部全局分工单独使用的均值接近零，并且改变了类别间的取舍。两者组合相对纯SSM平均提高约2.34个百分点，但互补效果会随种子变化；不能声称稳定改善边界。保持SSM和SR的六项算子对照已经跑完，'+verdict+'”\n\n')
        f.write('下一步判断：优先保留有稳定独立收益的SR；对分工根据本次算子对照决定保留为核心创新还是作为待改进结构。任何新比例、模块、数据集或种子矩阵都需另列方案，不自动扩展本批次。当前按测试OA选模，不能作为独立测试或SOTA证据。\n')
    files=[p for p in D.rglob('*') if p.is_file() and 'attempts' not in p.parts]
    files+=[report,ROOT/'scripts/summarize_sr_lg_mechanisms.py',ROOT/'scripts/write_sr_lg_report.py',Path(__file__)]
    write(DEST/'independent_final_audit.json',{'all_passed':True,'new_training_runs':6,'epochs_each':100,
        'all_completed_outputs_rehashed':True,'diagnostic_sources_checked_with_explicit_registry_transition':True,
        'interpretation':conclusions+[verdict],'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}})
    print('FINALIZED',conclusions,verdict,flush=True)
