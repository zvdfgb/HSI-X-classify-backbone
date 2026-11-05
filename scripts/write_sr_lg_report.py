"""Generate reviewable evidence tables from sealed native/diagnostic outputs."""
import csv
import json
import re
import statistics as st
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'report/sr_lg_mechanisms_20261104'
T=ROOT/'report/sr_lg_operator_controls_20261104'
OUT=ROOT/'docs/experiments/20261104_局部全局与SR机制分析.md'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def metric(p):
    t=p.read_text(encoding='utf-8-sig')
    return {k:float(re.search(rf'(?m)^([\d.eE+-]+) {v} accuracy \(%\)',t)[1]) for k,v in [('oa','Overall'),('aa','Average'),('kappa','Kappa')]}
def fmt(a):return f'{st.mean(a):.4f} ± {st.stdev(a):.4f}'
def table(headers,rs):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rs])

if __name__=='__main__':
    assert read(D/'progress.json')['state']=='completed'
    training_state=read(T/'state.json')
    complete=training_state['state']=='completed'
    if complete:assert read(T/'final_integrity_audit.json')['all_passed']
    a=read(D/'analysis_summary.json');pre=read(T/'preflight.json')
    t=read(T/'summary.json') if (T/'summary.json').exists() else {'runs':[], 'statistics':{},'comparisons':[]}
    manifest=read(ROOT/'report/sr_native/sr_native_20261029/manifest.json')
    native={(j['group'],j['seed']):metric(Path(j['report'])) for j in manifest['jobs'] if j['group'] in ('M0','M1','M2','M3','M4')}
    labels={'M0':'纯 SSM','M1':'仅分工','M2':'纯 SSM＋SR','M3':'分工＋SR','M4':'分工＋仅距离'}
    report=OUT.read_text(encoding='utf-8').split('\n\n## 1. 历史完整训练')[0]
    report=re.sub(r'日期：2026-11-04。[^\n]*',f'日期：2026-11-04。15 个历史模型的只读诊断已完成；新对照训练已完成 {training_state["new_completed"]}/6。'+('全部产物已核验。' if complete else '训练结果尚未齐全，相关结论待完成后填写。'),report,count=1)
    report+='\n\n## 1. 历史完整训练：两个点的独立作用与组合收益\n\n'
    report+='下表始终采用历史原报告，未用重新推理的微小差异改写历史成绩。标准差为三个种子的样本标准差；±标准差不是置信区间。\n\n'
    report+=table(['组别','含义','OA：seed6 / 7 / 8（%）','OA均值±标准差','AA均值±标准差','Kappa均值±标准差'],[
        [g,labels[g],' / '.join(f'{native[g,s]["oa"]:.4f}' for s in (6,7,8)),*[fmt([native[g,s][m] for s in (6,7,8)]) for m in ('oa','aa','kappa')]] for g in labels])
    pairs=[('分工独立贡献','M1','M0'),('SR独立贡献','M2','M0'),('分工上加SR','M3','M1'),('SR上加分工','M3','M2'),('光谱项相对仅距离','M3','M4'),('组合相对纯SSM','M3','M0')]
    report+='\n\n'+table(['比较','seed6增益','seed7增益','seed8增益','平均增益（百分点）','正增益种子数'],[
        [name,*[f'{native[l,s]["oa"]-native[r,s]["oa"]:+.4f}' for s in (6,7,8)],f'{st.mean([native[l,s]["oa"]-native[r,s]["oa"] for s in (6,7,8)]):+.4f}',sum(native[l,s]['oa']>native[r,s]['oa'] for s in (6,7,8))] for name,l,r in pairs])
    inter=[native['M3',s]['oa']-native['M1',s]['oa']-native['M2',s]['oa']+native['M0',s]['oa'] for s in (6,7,8)]
    report+='\n\n历史交互增益（seed6／7／8）为 '+ '、'.join(f'{v:+.4f}' for v in inter)+f' 个百分点，均值 {st.mean(inter):+.4f}。两个种子为正，一个为负，因此是互补线索，尚不足以称为稳定协同。分工单独平均约等于零；SR 单独三个种子均为正，是目前更稳定的一项。\n'
    report+='\n## 2. 逐样本得失：收益是否抵消了新错误\n\n以下使用本次推理，属于已有 checkpoint 的诊断。seed8 的 M2 与历史报告存在一个正确样本的微小差异；因此这些样本计数不替代上面的历史成绩。\n\n'
    paired=rows(D/'paired_errors.csv')
    report+=table(['比较','种子','纠正旧错误','引入新错误','净增加正确数'],[[r['comparison'],r['seed'],r['gained'],r['lost'],r['net']] for r in paired])
    report+='\n\n'+table(['比较','三个种子都纠正的样本','三个种子都改错的样本','至少一次纠正','至少一次改错','不同种子出现相反得失'],[[r['comparison'],r['gained_in_all_three'],r['lost_in_all_three'],r['gained_in_at_least_one'],r['lost_in_at_least_one'],r['gain_one_seed_loss_another']] for r in a['cross_seed_consistency']])
    report+='\n\n'+table(['种子','仅组合判对的样本','两项单独都判对、组合反而判错'],[[r['seed'],r['only_combined_correct'],r['both_single_correct_combination_wrong']] for r in a['interactions']])
    report+='\n\n“仅组合判对”说明存在互补样本；同时存在组合改错的样本，需要连同净收益一起解释，不能只挑成功案例。\n'
    report+='\n## 3. 哪些类别受益，哪些类别受损\n\n以下是类别召回率变化，三种子平均，单位为百分点。括号给出正收益种子数；它不是统计显著性。\n\n'
    cs=rows(D/'class_effects_summary.csv')
    report+=table(['类别','分工独立','SR独立','分工上加SR','SR上加分工'],[[c,*[f'{float(next(r for r in cs if r["class_cn"]==c and r["comparison"]==k)["recall_delta_mean_pp"]):+.3f} ({next(r for r in cs if r["class_cn"]==c and r["comparison"]==k)["positive_seeds"]}/3)' for k in ('L_alone','SR_alone','SR_on_L','L_on_SR')]] for c in dict.fromkeys(r['class_cn'] for r in cs)])
    report+='\n## 4. SR 实际在选择什么位置\n\n有效邻居数定义为 `1 / sum(w²)`，是权重分布的等效数量，不是实际保留下来的硬邻居个数。异类质量只在已知标签的非中心邻居内重新归一化，并仅统计标签覆盖≥30／120、观察到异类的样本。标签只用于事后解释，没有用于生成权重。\n\n'
    wr=[]
    for seed,w in a['weight_summaries'].items():
        wr.append([seed,w['eligible_mixed_n'],f'{w["center_weight"]["mean"]*100:.2f}%',f'{w["effective_neighbors"]["mean"]:.2f}',f'{w["other_weight_fraction_uniform_eligible_mixed_mean"]*100:.2f}%',f'{w["other_weight_fraction_distance_eligible_mixed_mean"]*100:.2f}%',f'{w["other_weight_fraction_sr_eligible_mixed_mean"]*100:.2f}%'])
    report+=table(['种子','支持分析的混合邻域数','平均中心权重','平均有效邻居','均匀的异类质量','仅距离的异类质量','SR的异类质量'],wr)
    report+='\n\n这支持“SR 抑制与中心不一致的邻居”这一解释。它不是只取中心：中心权重远低于 100%，有效邻居仍有几十个。空间距离本身已能减少混合，光谱项进一步筛选；但这种权重统计不自动等于分类提升，更不证明每种复杂边界都受益。\n'
    ss=rows(D/'stratified_effects.csv');rs=[]
    for k in ('SR_alone','SR_on_L','L_on_SR'):
        for group in ('observed_mixed_supported','no_other_observed_supported','low_neighbor_label_coverage'):
            z=[r for r in ss if r['comparison']==k and r['stratum']==group]
            rs.append([k,group,' / '.join(r['n'] for r in z),f'{st.mean(float(r["delta_pp"]) for r in z):+.3f}'])
    report+='\n\n'+table(['比较','邻域分层','样本数seed6/7/8','平均OA变化（百分点）'],rs)
    report+='\n\n分层中文解释：`observed_mixed_supported` 为标签覆盖足够且观察到异类；`no_other_observed_supported` 为覆盖足够但没有观察到异类；`low_neighbor_label_coverage` 为标签覆盖不足。分层阈值是诊断约定，不能把它们直接写成真实边界／内部区域。\n'
    report+='\n## 5. 固定模型干预：模型在依赖什么\n\n下表均为“改动后的 OA 减去该 checkpoint 原 OA”，单位为百分点。没有重新训练，不能把掉点幅度当成独立模块贡献。\n\n'
    report+=table(['模型','干预','seed6','seed7','seed8','平均变化'],[[r['group'],r['intervention'],*[f'{r[k]:+.3f}' for k in ('seed6_delta','seed7_delta','seed8_delta','delta_mean_pp')]] for r in a['interventions'] if r['group'] in ('M1','M3') and r['intervention']!='native'])
    report+='\n\n关闭分支可以判断当前模型是否依赖该支路；两路都会掉点仍不证明它们不可替代。全局支路原有残差与局部支路原有归一化方式不同，置零干预也不能用于公平排名两路重要性。\n\n![机制诊断汇总](../../report/sr_lg_mechanisms_20261104/mechanism_overview.png)\n\n![固定规则抽取的SR权重示例](../../report/sr_lg_mechanisms_20261104/sr_weight_examples.png)\n'
    report+='\n## 6. 六次算子对照\n\n'
    report+=('六次均走原入口训练至100轮，原报告覆盖12197个测试样本，原测试OA选模规则不变。LG为历史M3复用，LL与GG为本次新训。原报告、checkpoint、日志和配置逐项封存。' if complete else f'当前完成 {training_state["new_completed"]}/6；当前任务 {training_state.get("current")}。未完成结果不参与最终比较。全部计划使用原入口100轮，原测试OA选模规则不变。')+'\n\n'
    report+=table(['结构','种子','OA（%）','AA（%）','Kappa（%）','来源'],[[r['group'],r['seed'],*[f'{r[m]:.4f}' for m in ('oa','aa','kappa')],'历史复用' if r['reused'] else '本次100轮'] for r in t['runs']])
    report+='\n\n'+table(['结构','OA均值±样本标准差','AA均值±样本标准差','Kappa均值±样本标准差'],[[g,*[f'{t["statistics"][g][m]["mean"]:.4f} ± {t["statistics"][g][m]["std"]:.4f}' for m in ('oa','aa','kappa')]] for g in ('LG','LL','GG') if g in t['statistics']])
    report+='\n\n'+table(['对照相对LG','种子','OA变化（百分点）'],[[r['group'],r['seed'],f'{r["oa_delta_pp"]:+.4f}'] for r in t['comparisons']])
    report+='\n\n'+table(['结构','参数量'],[['LG',pre[0]['baseline_parameters']]]+[[g,next(p['parameters'] for p in pre if p['kind']==g)] for g in ('LL','GG')])
    report+='\n\n预检通过：共享参数初始化相同；全局随机数状态相同；临时恢复原算子后完整输出完全相同；原数据batch128梯度有限且新算子梯度非零；整模型保存／加载后输出相同。新算子参数量不同，分支外围结构也有差异，因此结果只能支持当前算子替换下的结论。\n'
    report+='\n## 7. 复现偏差与完整性\n\n'
    replay=[]
    for seed in (6,7,8):
        rr=read(D/f'seed{seed}/metrics.json')
        for g in labels:
            v=next(r for r in rr if r['group']==g and r['intervention']=='native')
            replay.append([g,seed,*[f'{v[m]-native[g,seed][m]:+.8f}' for m in ('oa','aa','kappa')]])
    report+=table(['模型','种子','重推OA减历史OA','重推AA减历史AA','重推Kappa减历史Kappa'],replay)
    report+='\n\n所有历史文件未修改；本次分析门槛与解析修订、失败／中止的只读尝试均有留档。该复现偏差表是实际结果，不宣称逐像素完全一致；不把微小推理差异包装成模型提升。训练所需核心源码、原训练链、数据、基线checkpoint与报告均通过哈希核验；模型注册表只新增两个名称，旧映射未变。\n'
    OUT.write_text(report,encoding='utf-8')
    print(OUT)
