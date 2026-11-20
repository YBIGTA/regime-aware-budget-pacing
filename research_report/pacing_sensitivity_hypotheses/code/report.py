"""Describe locked results; never change policies or choose a winner."""
from env import *
from experiment import read_csv,OUT,MAIN,PAIRS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
FIG=BUNDLE/'figures'
LABELS={'CMA':'CMA E2','G':'Gated (old init)','C_all':'Constant: always',
 'C_warm':'Constant: 30-min warmup','C_rule':'Constant: same fallback',
 'C_matched':'Constant: same fallback + mean',
 'G_neutral':'Gated: neutral init','G_first':'Gated: first-ratio init',
 'R':'Centered (old init)','R_neutral':'Centered: neutral init','R_first':'Centered: first-ratio init',
 'L':'Learned slope (old EMA)','L_noema':'Learned: no added EMA','L_wide':'Learned: wider factor bounds'}

def figsave(name):
    plt.savefig(FIG/(name+'.png'),dpi=165,bbox_inches='tight')
    svg=FIG/(name+'.svg');plt.savefig(svg,bbox_inches='tight')
    svg.write_text('\n'.join(l.rstrip() for l in svg.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
    plt.close()

def main():
    assert rt.read_json(OUT/'audit.json')['status']=='passed'
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    specs=rt.read_json(OUT/'calibration.json')['specs']
    scores={r['model']:{m:float(r[m]) for m in METRICS} for r in read_csv(OUT/'fresh_summary.csv')}
    ds={r['model']:{m:float(v) for m,v in r.items() if m!='model'} for r in read_csv(OUT/'diagnostics_summary.csv')}
    pairs={(r['new'],r['baseline'],r['metric']):r for r in read_csv(OUT/'fresh_paired.csv')}
    dp={(r['new'],r['baseline'],r['metric']):r for r in read_csv(OUT/'diagnostics_paired.csv')}
    profiles=read_csv(OUT/'time_profiles.csv')
    def curve(k,m):return np.array([float(r[m]) for r in profiles if r['model']==k])
    checks=[]
    for k in specs:
        b=scores['CMA'];s=scores[k]
        checks.append(dict(model=k,**{m+'_relative_pct':(s[m]/b[m]-1)*100 for m in METRICS},
          strict_mean_pass=k!='CMA' and s['tracking_error']<=b['tracking_error'] and s['unspent']<=b['unspent'] and s['multiplier_tv']<b['multiplier_tv']))
    rt.write_csv(OUT/'CMA_comparisons.csv',checks)
    hypotheses=[]
    for i,(a,b) in enumerate(MAIN,1):
        row=pairs[a,b,'tracking_error']
        hypotheses.append(dict(hypothesis=f'H{i}',new=a,baseline=b,TE_difference=float(row['mean_difference']),
          TE_ci98_75=[float(row['ci98_75_low']),float(row['ci98_75_high'])],
          additional_TE_improvement_supported=float(row['ci98_75_high'])<0,
          TE_worsening_supported=float(row['ci98_75_low'])>0,
          UR_one_sided95_upper=float(pairs[a,b,'unspent']['one_sided95_upper'])))
    rt.write_json(OUT/'conclusion.json',dict(hypotheses=hypotheses,CMA_decision='retain_CMA_no_winner_selected',final_test_used=False,
      warning='Main TE intervals adjusted for 4 fixed contrasts; other intervals descriptive. Strict sample means are not statistical noninferiority.'))
    lines=['# 동결한 800일 결과표','', '|정책|TE %|UR %|J %|V|평균 factor|평균 Ki|factor 제한 %|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for k in specs:
        s=scores[k];d=ds[k]
        lines.append(f"|{k}|{100*s['tracking_error']:.6f}|{100*s['unspent']:.6f}|{100*s['objective']:.6f}|{s['multiplier_tv']:.6f}|{d['mean_factor']:.6f}|{d['mean_ki']:.2f}|{100*d['factor_clip_rate']:.3f}|")
    lines+=['','## 사전 고정 주 비교','', '|비교|TE 차이 %p (98.75% 구간)|UR 차이 %p (95% 구간)|V 차이 (95% 구간)|','|---|---|---|---|']
    def interval(a,b,m,wide=False):
        row=pairs[a,b,m];scale=1 if m=='multiplier_tv' else 100;tag='98_75' if wide else '95'
        return f"{scale*float(row['mean_difference']):+.7f} [{scale*float(row['ci'+tag+'_low']):+.7f}, {scale*float(row['ci'+tag+'_high']):+.7f}]"
    for a,b in MAIN:lines.append(f'|{a} − {b}|{interval(a,b,"tracking_error",True)}|{interval(a,b,"unspent")}|{interval(a,b,"multiplier_tv")}|')
    lines+=['','## 모든 사전 비교 (95% 기술적 구간)','', '|비교|TE 차이 %p|UR 차이 %p|V 차이|','|---|---|---|---|']
    for a,b in PAIRS:lines.append(f'|{a} − {b}|{interval(a,b,"tracking_error")}|{interval(a,b,"unspent")}|{interval(a,b,"multiplier_tv")}|')
    (OUT/'report_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

    hour=(np.arange(288)+1)/12
    fig,axes=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
    for k in ['G','C_all','C_warm','C_matched']:
        axes[0,0].plot(hour[:24],curve(k,'factor')[:24],label=LABELS[k])
        axes[0,1].plot(hour[:24],100*curve(k,'absolute_error')[:24],label=LABELS[k])
        if k!='G':axes[1,0].plot(hour,100*(curve('G','absolute_error')-curve(k,'absolute_error')),label='G minus '+k,color={'C_all':'C1','C_warm':'C2','C_matched':'C3'}[k])
    axes[0,0].set(title='H1: initial Ki factor',ylabel='Factor',xlabel='Hour')
    axes[0,1].set(title='First 2 hours',ylabel='Mean absolute tracking error (%)',xlabel='Hour')
    axes[1,0].axhline(0,color='black',lw=.8);axes[1,0].set(title='Whole-day tracking difference',ylabel='Difference (percentage points)',xlabel='Hour')
    for i,b in enumerate(['C_all','C_warm','C_rule','C_matched']):
        v=[100*float(dp['G',b,m]['mean_difference']) for m in ['TE_0_30','TE_30_120','TE_120_1200','TE_1200_1440']]
        axes[1,1].bar(np.arange(4)+(i-1.5)*.19,v,width=.19,label='G minus '+b,color=['C1','C2','C4','C3'][i])
    axes[1,1].axhline(0,color='black',lw=.8);axes[1,1].set_xticks(range(4),['0–0.5h','0.5–2h','2–20h','20–24h'])
    axes[1,1].set(title='Contribution to whole-day TE difference',ylabel='Percentage points')
    for ax in axes.flat:ax.legend(fontsize=8)
    figsave('warmup-controls')

    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for col,keys in enumerate([['R','R_neutral','R_first'],['G','G_neutral','G_first']]):
        for k in keys:
            axes[0,col].plot(hour[:24],curve(k,'factor')[:24],label=LABELS[k])
            axes[1,col].plot(hour[:24],100*curve(k,'absolute_error')[:24],label=LABELS[k])
        axes[0,col].set(title='H2: '+('centered ratio' if col==0 else 'gated ratio'),ylabel='Mean factor',xlabel='Hour')
        axes[1,col].set(ylabel='Mean absolute tracking error (%)',xlabel='Hour')
        for ax in axes[:,col]:ax.legend(fontsize=8)
    figsave('initialization')

    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,m,scale,title in zip(axes,['tracking_error','unspent','multiplier_tv'],[100,100,1],['Tracking: 98.75% adjusted CI (pp)','Unspent: 95% descriptive CI (pp)','Action variation: 95% CI']):
        for j,(a,b) in enumerate(MAIN):
            row=pairs[a,b,m];tag='98_75' if m=='tracking_error' else '95'
            x=scale*float(row['mean_difference']);lo=scale*float(row['ci'+tag+'_low']);hi=scale*float(row['ci'+tag+'_high'])
            ax.errorbar(x,j,xerr=[[x-lo],[hi-x]],fmt='o',capsize=4,color=f'C{j}')
        ax.axvline(0,color='black',lw=.9);ax.set(title=title,yticks=range(4),yticklabels=['H1 G − matched','H2 neutral − old init','H3 no EMA − old','H4 wide − old']);ax.invert_yaxis()
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)
    figsave('hypothesis-contrasts')

    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for k in ['L','L_noema','L_wide']:
        axes[0].plot(hour,curve(k,'factor'),label=LABELS[k])
        axes[1].plot(hour,100*curve(k,'factor_clip'),label=LABELS[k])
        if k!='L':axes[2].plot(hour,100*(curve(k,'absolute_error')-curve('L','absolute_error')),label=k+' minus L',color={'L_noema':'C1','L_wide':'C2'}[k])
    axes[0].set(title='H3/H4: Ki factor',ylabel='Mean factor',xlabel='Hour')
    axes[1].set(title='Factor clipping',ylabel='Clipped intervals (%)',xlabel='Hour')
    axes[2].set(title='Tracking difference vs L',ylabel='Difference (percentage points)',xlabel='Hour');axes[2].axhline(0,color='black',lw=.8)
    for ax in axes:ax.legend(fontsize=8)
    figsave('lag-and-bounds')
    rt.write_json(OUT/'report_manifest.json',dict(source_sha256=rt.sha(Path(__file__)),
      input_sha256={p.name:rt.sha(p) for p in OUT.glob('*.csv')},no_new_rollouts=True))
    print('Tables and four figures written; no selection performed')

if __name__=='__main__':main()
