"""Posthoc paired-error/weight analysis; native reports remain metric authority."""
import csv
import itertools
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_sr_lg_mechanisms import DEST, SEEDS, GROUPS, read, write, csv_write, sha, now

CLASSES = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees', 'Soil', 'Water',
           'Residential', 'Commercial', 'Road', 'Highway', 'Railway', 'Parking lot 1',
           'Parking lot 2', 'Tennis court', 'Running track']
CN = ['健康草地','受胁草地','人造草地','树木','土壤','水体','住宅区','商业区','道路','高速公路','铁路','停车场1','停车场2','网球场','跑道']
PAIRS = [('L_alone','M1','M0'),('SR_alone','M2','M0'),('SR_on_L','M3','M1'),
         ('L_on_SR','M3','M2'),('spectral_vs_distance','M3','M4'),('combined','M3','M0')]


def statistics(values):
    a = np.asarray(values, dtype=float)
    return {'mean':float(a.mean()),'std':float(a.std(ddof=1)),'values':a.tolist(),
            'positive':int((a>0).sum())}


def main():
    paired, classes, strata, patterns, energy, interactions, weight_rows = [],[],[],[],[],[],[]
    strata_classes=[]
    native, intervention = {}, {}
    per_seed, prediction_seeds, metadata = {}, {}, {}
    for seed in SEEDS:
        folder=DEST/f'seed{seed}'
        with np.load(folder/'predictions.npz') as data:
            pred={key:data[key] for key in data.files}
        with np.load(folder/'neighbor_metadata.npz') as data:
            meta={key:data[key] for key in data.files}
        prediction_seeds[seed]=pred; metadata[seed]=meta
        labels=pred['labels']; n=len(labels)
        correct={g:pred[g]==labels for g in GROUPS}
        masks={'all':np.ones(n,dtype=bool),'observed_mixed_supported':meta['mixed'] & (meta['known_neighbors']>=30),
               'no_other_observed_supported':meta['no_observed_other_sufficient'],
               'low_neighbor_label_coverage':meta['low_coverage']}
        assert sum(int(masks[k].sum()) for k in list(masks)[1:])==n
        reports=read(folder/'metrics.json')
        for row in reports:
            key=(row['group'],row['intervention'])
            intervention.setdefault(key,[]).append(row)
            if row['intervention']=='native':native[(seed,row['group'])]=row
        for name,left,right in PAIRS:
            gain=(correct[left]&~correct[right]); loss=(~correct[left]&correct[right])
            paired.append({'seed':seed,'comparison':name,'left':left,'right':right,'gained':int(gain.sum()),
                           'lost':int(loss.sum()),'net':int(gain.sum()-loss.sum()),'delta_pp':float((gain.sum()-loss.sum())*100/n)})
            assert abs(paired[-1]['delta_pp']-(native[(seed,left)]['oa']-native[(seed,right)]['oa']))<1e-8
            for c in range(15):
                mask=labels==c
                classes.append({'seed':seed,'comparison':name,'class':CLASSES[c],'class_cn':CN[c],
                    'n':int(mask.sum()),'gained':int((gain&mask).sum()),'lost':int((loss&mask).sum()),
                    'net':int((gain&mask).sum()-(loss&mask).sum()),
                    'recall_delta_pp':float((gain[mask].sum()-loss[mask].sum())*100/mask.sum())})
            for stratum,mask in masks.items():
                strata.append({'seed':seed,'comparison':name,'stratum':stratum,'n':int(mask.sum()),
                    'gained':int((gain&mask).sum()),'lost':int((loss&mask).sum()),
                    'delta_pp':float((gain[mask].sum()-loss[mask].sum())*100/mask.sum()) if mask.any() else None})
                if stratum!='all':
                    for c in range(15):
                        cm=mask & (labels==c)
                        if cm.any():
                            strata_classes.append({'seed':seed,'comparison':name,'stratum':stratum,
                                'class':CLASSES[c],'class_cn':CN[c],'n':int(cm.sum()),
                                'gained':int((gain&cm).sum()),'lost':int((loss&cm).sum()),
                                'net':int((gain&cm).sum()-(loss&cm).sum())})
        code=sum(correct[g].astype(np.int8)*(1<<i) for i,g in enumerate(GROUPS[:4]))
        for bits in itertools.product((0,1),repeat=4):
            v=sum(b<<i for i,b in enumerate(bits))
            patterns.append({'seed':seed,'M0':bits[0],'M1':bits[1],'M2':bits[2],'M3':bits[3],
                             'n':int((code==v).sum())})
        interaction=correct['M3'].astype(int)-correct['M1'].astype(int)-correct['M2'].astype(int)+correct['M0'].astype(int)
        interactions.append({'seed':seed,'interaction_pp':float(interaction.mean()*100),
            'sum_correctness_interaction':int(interaction.sum()),
            'positive_sample_n':int((interaction>0).sum()),'negative_sample_n':int((interaction<0).sum()),
            'only_combined_correct':int((~correct['M0']&~correct['M1']&~correct['M2']&correct['M3']).sum()),
            'both_single_correct_combination_wrong':int((correct['M1']&correct['M2']&~correct['M3']).sum())})
        for r in read(folder/'branch_energy.json'):
            key=(r['group'],r['block'])
            per_seed.setdefault((seed,*key),{'n':0,'global':0.,'local':0.})
            dest=per_seed[(seed,*key)]; dest['n']+=r['batch_n']
            dest['global']+=r['global_energy_sum'];dest['local']+=r['local_energy_sum']
        for c in range(15):
            mask=(labels==c)&masks['observed_mixed_supported']
            if mask.any():
                weight_rows.append({'seed':seed,'class':CLASSES[c],'class_cn':CN[c],'n':int(mask.sum()),
                    'other_mass_uniform':float(meta['other_weight_fraction_uniform'][mask].mean()),
                    'other_mass_distance':float(meta['other_weight_fraction_distance'][mask].mean()),
                    'other_mass_sr':float(meta['other_weight_fraction_sr'][mask].mean()),
                    'center_weight':float(meta['center_weight'][mask].mean())})
        csv_write(folder/'paired_errors.csv',[r for r in paired if r['seed']==seed])
        sample_rows=[]
        for i in range(n):
            sample_rows.append({'index':i,'row':int(pred['row'][i]),'col':int(pred['col'][i]),
                'label':int(labels[i]),'class_cn':CN[labels[i]],
                **{g:int(pred[g][i]) for g in GROUPS},
                'correctness_pattern_M0_M1_M2_M3':''.join(str(int(correct[g][i])) for g in GROUPS[:4]),
                'known_neighbors':int(meta['known_neighbors'][i]),'observed_other_fraction':float(meta['observed_other_fraction'][i]),
                'center_weight':float(meta['center_weight'][i]),'effective_neighbors':float(meta['effective_neighbors'][i])})
        csv_write(folder/'sample_outcomes.csv',sample_rows)
    for (seed,group,block),r in per_seed.items():
        energy.append({'seed':seed,'group':group,'block':block,'global_rms':float(np.sqrt(r['global']/r['n'])),
                       'local_rms':float(np.sqrt(r['local']/r['n']))})
    pair_summary={name:statistics([r['delta_pp'] for r in paired if r['comparison']==name]) for name,_,_ in PAIRS}
    intervention_summary=[]
    for (group,variant),rows in intervention.items():
        values=[r['oa']-native[(r['seed'],group)]['oa'] for r in rows]
        intervention_summary.append({'group':group,'intervention':variant,
            'oa_mean':float(np.mean([r['oa'] for r in rows])),
            'delta_mean_pp':statistics(values)['mean'],'delta_std_pp':statistics(values)['std'],
            'seed6_delta':values[0],'seed7_delta':values[1],'seed8_delta':values[2]})
    class_summary=[]
    for name,_,_ in PAIRS:
        for c in CLASSES:
            rows=[r for r in classes if r['comparison']==name and r['class']==c]
            class_summary.append({'comparison':name,'class':c,'class_cn':rows[0]['class_cn'],
                'n':rows[0]['n'],'recall_delta_mean_pp':float(np.mean([r['recall_delta_pp'] for r in rows])),
                'net_correct_mean':float(np.mean([r['net'] for r in rows])),
                'seed6_net':rows[0]['net'],'seed7_net':rows[1]['net'],'seed8_net':rows[2]['net'],
                'positive_seeds':sum(r['net']>0 for r in rows)})
    cross_seed=[]
    for name,left,right in PAIRS:
        gains=np.stack([(prediction_seeds[s][left]==prediction_seeds[s]['labels']) &
                         (prediction_seeds[s][right]!=prediction_seeds[s]['labels']) for s in SEEDS])
        losses=np.stack([(prediction_seeds[s][left]!=prediction_seeds[s]['labels']) &
                         (prediction_seeds[s][right]==prediction_seeds[s]['labels']) for s in SEEDS])
        cross_seed.append({'comparison':name,'gained_in_all_three':int(gains.all(0).sum()),
            'lost_in_all_three':int(losses.all(0).sum()),'gained_in_at_least_one':int(gains.any(0).sum()),
            'lost_in_at_least_one':int(losses.any(0).sum()),
            'gain_one_seed_loss_another':int((gains.any(0)&losses.any(0)).sum())})
    outputs={'paired_errors':paired,'class_effects':classes,'class_effects_summary':class_summary,
             'stratified_effects':strata,'correctness_patterns':patterns,'branch_energy':energy,
             'interventions_summary':intervention_summary,'class_neighbor_weights':weight_rows,
             'cross_seed_consistency':cross_seed,'interactions':interactions,'stratified_class_effects':strata_classes}
    for name,rows in outputs.items():csv_write(DEST/(name+'.csv'),rows)
    summary={'created_at':now(),'new_training_runs':0,'native_replays':15,
        'pair_summary':pair_summary,'interactions':interactions,
        'interaction_summary':statistics([r['interaction_pp'] for r in interactions]),
        'interventions':intervention_summary,'cross_seed_consistency':cross_seed,
        'weight_summaries':{str(s):read(DEST/f'seed{s}'/'weight_summary.json') for s in SEEDS},
        'strata_definition':'posthoc labels; 30/120 observed non-center neighbors; mixed means >=1 observed different-label neighbor; unknown is never another class',
        'diagnostic_limits':['test-OA checkpoint selection','single dataset, three seeds',
            'no independent spatial samples or significance claim','fixed-model interventions induce distribution shift',
            'activation magnitude is not causal importance','observed-label strata are not complete true object boundaries',
            'one fixed pooling permutation does not cover all misalignment distributions']}
    write(DEST/'analysis_summary.json',summary)
    print(json.dumps({'pair_summary':pair_summary,'interaction_summary':summary['interaction_summary'],
                      'interventions':intervention_summary,'weight_summaries':summary['weight_summaries']},ensure_ascii=False,indent=2))
    render(summary,class_summary,intervention_summary,metadata,prediction_seeds)


def render(summary,classes,interventions,metadata,predictions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(1,4,figsize=(19,5.3),layout='constrained')
    for g,label,color in [('M0','纯SSM','#758396'),('M1','仅局部/全局','#d69c48'),('M2','仅SR','#4c9d9b'),('M3','分工＋SR','#5d7ebc')]:
        a=[read(DEST/f'seed{s}'/'metrics.json') for s in SEEDS]
        values=[next(r['oa'] for r in rows if r['group']==g and r['intervention']=='native') for rows in a]
        axes[0].plot(SEEDS,values,'o-',label=label,color=color,lw=2)
    axes[0].set(title='四组配对结果',xlabel='随机种子',ylabel='OA（%）',xticks=list(SEEDS))
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    names=['L_alone','SR_alone','SR_on_L','L_on_SR']
    a=np.array([[next(r['recall_delta_mean_pp'] for r in classes if r['comparison']==name and r['class']==c) for name in names] for c in CLASSES])
    bound=max(abs(a).max(),1)
    im=axes[1].imshow(a,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto')
    axes[1].set(yticks=range(15),yticklabels=CN,xticks=range(4),xticklabels=['分工独立','SR独立','分工上加SR','SR上加分工'],title='类别召回率变化（三种子平均）')
    axes[1].tick_params(axis='x',rotation=25,labelsize=8);axes[1].tick_params(axis='y',labelsize=8)
    fig.colorbar(im,ax=axes[1],label='百分点',shrink=.7)
    variants=['pool_uniform','pool_distance_only','pool_misaligned_sr','pool_center_only','mute_global','mute_local']
    names_cn=['改为平均','仅距离','SR固定错位','仅中心','关闭全局支路','关闭局部支路']
    values=[next(r['delta_mean_pp'] for r in interventions if r['group']=='M3' and r['intervention']==v) for v in variants]
    for ax,sl,title,color in [(axes[2],slice(0,4),'固定组合模型：汇总干预','#5d7ebc'),
                             (axes[3],slice(4,6),'固定组合模型：分支干预','#d69c48')]:
        bars=ax.barh(names_cn[sl],values[sl],color=color)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
        ax.axvline(0,c='black',lw=.7)
        ax.set_xlim(min(values[sl])*1.25,.15)
        ax.set(title=title,xlabel='OA变化（百分点）')
    fig.suptitle('局部／全局通道分工与SR：已有模型机制诊断',fontsize=14)
    fig.savefig(DEST/'mechanism_overview.png',dpi=200)
    fig.savefig(DEST/'mechanism_overview.pdf');plt.close(fig)
    meta=metadata[6];pred=predictions[6];truth=pred['labels']
    with np.load(DEST/'seed6/sr_weights.npz') as d:w=d['weights']
    cases=[('仅组合判对',(pred['M0']!=truth)&(pred['M1']!=truth)&(pred['M2']!=truth)&(pred['M3']==truth)),
           ('SR纠正分工错误',(pred['M1']!=truth)&(pred['M3']==truth)),
           ('SR引入错误',(pred['M1']==truth)&(pred['M3']!=truth))]
    fig,axes=plt.subplots(3,3,figsize=(10,10),layout='constrained')
    selected=[]
    for row,(name,mask) in enumerate(cases):
        # Predefined first positions, no visual/cherry-picked score selection.
        idx=np.flatnonzero(mask)[:3]
        for col,ax in enumerate(axes[row]):
            if col>=len(idx):ax.axis('off');continue
            i=int(idx[col]);im=ax.imshow(w[i],cmap='viridis',vmin=0,vmax=.12);ax.scatter([5],[5],c='red',s=12)
            ax.set(title=f'{name}\n({pred["row"][i]},{pred["col"][i]}) {CN[truth[i]]}\n中心权重{meta["center_weight"][i]:.3f}，有效邻居{meta["effective_neighbors"][i]:.1f}',xticks=[],yticks=[])
            selected.append({'selection':'first three test indices per predefined outcome','case':name,'index':i,
                             'row':int(pred['row'][i]),'col':int(pred['col'][i]),'class':CN[truth[i]]})
    fig.suptitle('固定规则抽取的SR权重示例（seed6；红点为中心）',fontsize=14)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='位置权重（统一色标；上限0.12）',shrink=.6)
    fig.savefig(DEST/'sr_weight_examples.png',dpi=180);plt.close(fig)
    write(DEST/'illustration_samples.json',selected)


if __name__=='__main__':main()
