"""Locked-result diagnostics and figures. No selection or estimator fitting."""
import csv,platform,re
from env import *
from fast_refined import run
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=BUNDLE/'results';FIG=BUNDLE/'figures'
def read(name):
    with (OUT/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))
def label(k):
    if k=='S0_CMA':return 'CMA E2'
    if k=='S1_legacy':return 'Previous ratio + EMA'
    return k.replace('matched_','Matched: ').replace('centered_ratio_','Centered ').replace('gated_ratio_','Gated ').replace('learned_slope_','Slope model ').replace('_m',' / mean ')
def savefig(name):
    plt.savefig(FIG/(name+'.png'),dpi=165,bbox_inches='tight')
    svg=FIG/(name+'.svg');plt.savefig(svg,bbox_inches='tight')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close()
def main():
    assert rt.read_json(OUT/'audit.json')['status']=='passed'
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
    sel=rt.read_json(OUT/'selection.json');cal=rt.read_json(OUT/'calibration.json');specs=sel['selected_specs']
    summaries={r['model']:{m:float(r[m]) for m in METRICS} for r in read('fresh_summary.csv')}
    pairs={(r['new'],r['baseline'],r['metric']):r for r in read('fresh_paired.csv')}
    mainkey=sel['primary']['diagnostic'];neutral=[f+'_p0.25_m1' for f in ['centered_ratio','gated_ratio','learned_slope']]
    ds=[];profile=[]
    for k,sp in specs.items():
        a=np.stack([run(d,sp)[1] for d in FRESH]);f=a[:,:,10];q=a[:,:,17]
        ds.append(dict(model=k,mean_factor=float(f.mean()),factor_sd=float(f.std()),
           factor_q05=float(np.quantile(f,.05)),factor_q50=float(np.median(f)),factor_q95=float(np.quantile(f,.95)),
           factor_above_one=float(np.mean(f>1.00001)),factor_below_one=float(np.mean(f<.99999)),
           factor_tv=float(np.abs(np.diff(f,axis=1)).mean()),mean_gate=float(q.mean()),mean_Ki=float(a[:,:,4].mean()),
           fallback_fraction=float(a[:,:,11].mean()),factor_clip_fraction=float(a[:,:,15].mean()),
           saturation_fraction=float(a[:,:,13].mean()),Ki_cap_fraction=float(np.mean(a[:,:,4]>=6000))))
        for t in range(288):
            profile.append(dict(model=k,tick=t,hour=(t+1)/12,tracking_error_pct=float(np.abs(a[:,t,1]).mean()*100),
                mean_Ki=float(a[:,t,4].mean()),mean_factor=float(f[:,t].mean()),mean_gate=float(q[:,t].mean())))
    rt.write_csv(OUT/'factor_diagnostics.csv',ds);rt.write_csv(OUT/'time_profiles.csv',profile)
    dindex={r['model']:r for r in ds}
    pi={(r['model'],r['tick']):r for r in profile};segments=[]
    for key in specs:
        if key not in cal['matched']:continue
        for baseline in ['S0_CMA',cal['matched'][key]]:
            for name,lo,hi in [('warmup',0,6),('early',6,24),('middle',24,240),('late',240,288)]:
                differences=[pi[key,t]['tracking_error_pct']-pi[baseline,t]['tracking_error_pct'] for t in range(lo,hi)]
                segments.append(dict(model=key,baseline=baseline,segment=name,ticks=hi-lo,
                    TE_difference_pp=float(np.mean(differences)),whole_day_contribution_pp=float(sum(differences)/288)))
    rt.write_csv(OUT/'tracking_segments.csv',segments)
    neutral_table=[]
    for k in neutral:
        s=summaries[k];b=summaries['S0_CMA'];c=summaries[cal['matched'][k]]
        neutral_table.append(dict(model=k,matched=cal['matched'][k],fresh_mean_factor=dindex[k]['mean_factor'],
            constant_factor=specs[cal['matched'][k]]['constant'],
            **{m+'_relative_to_CMA_pct':(s[m]/b[m]-1)*100 for m in METRICS},
            **{m+'_relative_to_matched_pct':(s[m]/c[m]-1)*100 for m in METRICS}))
    rt.write_csv(OUT/'neutral_ablation.csv',neutral_table)

    pred=read('calibration_slope_forecasts.csv');true=np.array([float(r['actual']) for r in pred]);learned=np.array([float(r['learned']) for r in pred])
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    hb=axes[0].hexbin(true,learned,gridsize=40,mincnt=1,bins='log',cmap='Blues')
    axes[0].plot([0,.09],[0,.09],'--',color='#555',lw=1)
    axes[0].axhline(B_SLOPE,color='#d58538',label='B constant slope')
    axes[0].set(xlabel='Offline auction slope',ylabel='Predicted slope',title='Held-out calibration observations')
    axes[0].legend(frameon=False);fig.colorbar(hb,ax=axes[0],label='Interval count')
    val=rt.read_json(OUT/'slope_validation.json');x=np.arange(2);w=.36
    axes[1].bar(x-w/2,[val['B_mae'],val['B_rmse']],w,label='B constant',color='#d58538')
    axes[1].bar(x+w/2,[val['learned_mae'],val['learned_rmse']],w,label='Learned slope',color='#4c78a8')
    axes[1].set_xticks(x,['MAE','RMSE']);axes[1].set(ylabel='Normalized spend / multiplier',title='Predictive improvement is not control improvement')
    axes[1].legend(frameon=False);fig.tight_layout();savefig('slope-forecast')

    keys=['S1_legacy']+neutral;fig,ax=plt.subplots(figsize=(9.5,4.2))
    for i,k in enumerate(keys):
        d=dindex[k];med=d['factor_q50'];lo=d['factor_q05'];hi=d['factor_q95']
        ax.errorbar(med,i,xerr=[[med-lo],[hi-med]],fmt='o',color='#d58538' if i==0 else '#4c78a8',capsize=4)
        ax.scatter(d['mean_factor'],i,marker='x',color='#222',zorder=3)
    ax.axvline(1,color='#555',ls='--');ax.set_yticks(range(len(keys)),[label(k) for k in keys]);ax.invert_yaxis()
    ax.set(xlabel='Ki factor: dot = median; x = mean; range = 5th to 95th percentile',title='Equal power p=0.25; corrected arms target mean 1 on calibration')
    fig.tight_layout();savefig('factor-centering')

    diagnostic=[sel['families'][f]['diagnostic'] for f in ['centered_ratio','gated_ratio','learned_slope','constant']]
    keys=list(dict.fromkeys(['S1_legacy']+diagnostic));fig,axes=plt.subplots(1,3,figsize=(13,4.6),sharey=True)
    for ax,m,title in zip(axes,['tracking_error','unspent','multiplier_tv'],['Tracking error','Final unspent','Multiplier variation']):
        for i,k in enumerate(keys):
            r=pairs[k,'S0_CMA',m];scale=100/summaries['S0_CMA'][m]
            mean=float(r['mean_difference'])*scale;lo=float(r['ci95_low'])*scale;hi=float(r['ci95_high'])*scale
            ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=3,color='#d58538' if k==mainkey else '#4c78a8')
        ax.axvline(0,color='#555',lw=1);ax.set(title=title,xlabel='Difference / CMA mean (%)')
        ax.set_yticks(range(len(keys)),[label(k) for k in keys])
    axes[0].invert_yaxis();fig.suptitle('400 new days: locked diagnostic picks; paired bootstrap 95% intervals',y=1.02)
    fig.tight_layout();savefig('fresh-differences')

    keys=list(dict.fromkeys(neutral+diagnostic[:3]));fig,axes=plt.subplots(1,3,figsize=(13,5),sharey=True)
    for ax,m,title in zip(axes,['tracking_error','unspent','multiplier_tv'],['Tracking error','Final unspent','Multiplier variation']):
        for i,k in enumerate(keys):
            ctrl=cal['matched'][k];r=pairs[k,ctrl,m];scale=100/summaries[ctrl][m]
            mean=float(r['mean_difference'])*scale;lo=float(r['ci95_low'])*scale;hi=float(r['ci95_high'])*scale
            ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=3,color='#d58538' if k==mainkey else '#4c78a8')
        ax.axvline(0,color='#555',lw=1);ax.set(title=title,xlabel='Difference / matched constant mean (%)')
        ax.set_yticks(range(len(keys)),[label(k) for k in keys])
    axes[0].invert_yaxis();fig.suptitle('Does adaptation outperform its calibration-matched constant?',y=1.02)
    fig.tight_layout();savefig('matched-differences')

    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for k,color in [('S0_CMA','#203d5b'),('S1_legacy','#999999'),(mainkey,'#d58538')]:
        rows=[r for r in profile if r['model']==k];h=[r['hour'] for r in rows]
        axes[0].plot(h,[r['tracking_error_pct'] for r in rows],label=label(k),color=color)
        axes[1].plot(h,[r['mean_Ki'] for r in rows],color=color)
        axes[2].plot(h,[r['mean_factor'] for r in rows],color=color)
    axes[0].set(ylabel='Abs. error (% budget)',title='400-day profile: fixed primary-family diagnostic')
    axes[0].legend(frameon=False);axes[1].set(ylabel='Mean Ki');axes[2].set(ylabel='Mean factor',xlabel='Hour')
    axes[2].set_xticks(range(0,25,2));fig.tight_layout();savefig('time-profile')

    lines=['| 모델 | TE (%) | UR (%) | J (%) | V |','|---|---:|---:|---:|---:|']
    table_keys=list(dict.fromkeys(['S0_CMA','S1_legacy']+diagnostic+[p['primary'] for p in sel['families'].values()]+neutral+[cal['matched'][k] for k in diagnostic[:3]]))
    for k in table_keys:
        s=summaries[k];lines.append(f"| {k} | {s['tracking_error']*100:.6f} | {s['unspent']*100:.6f} | {s['objective']*100:.6f} | {s['multiplier_tv']:.6f} |")
    (OUT/'report_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    rt.write_json(OUT/'report_metadata.json',dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,
      script_sha256=rt.sha(Path(__file__)),extra_descriptive_rollouts=len(specs)*len(FRESH),final_test_used=False))
    print('Figures and descriptive diagnostics complete',flush=True)
if __name__=='__main__':main()
