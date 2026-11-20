"""Read frozen results to produce tables and figures; no new policy evaluation."""
from env import *
from experiment import OUT,read_csv,MAIN,PAIRS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
FIG=BUNDLE/'figures'
LABELS={'CMA':'CMA','CMA_T05':'CMA + target 50%','CMA_T1':'CMA + target 100%',
 'G':'Gated','G_T05':'Gated + target 50%','G_T1':'Gated + target 100%',
 'G_R':'Gated + restore','G_T05_R':'Gated + target 50% + restore',
 'G_T1_R':'Gated + target 100% + restore'}
COLORS={'CMA':'#555555','G':'#0072B2','G_T05':'#009E73','G_R':'#E69F00','G_T05_R':'#D55E00','G_T1_R':'#CC79A7'}

def savefig(name):
    plt.savefig(FIG/(name+'.png'),dpi=165,bbox_inches='tight')
    path=FIG/(name+'.svg');plt.savefig(path,bbox_inches='tight')
    path.write_text('\n'.join(l.rstrip() for l in path.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
    plt.close()

def main():
    assert rt.read_json(OUT/'audit.json')['status']=='passed'
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    specs=rt.read_json(OUT/'protocol.json')['design']['policies']
    scores={r['model']:{m:float(r[m]) for m in METRICS} for r in read_csv(OUT/'fresh_summary.csv')}
    diag={r['model']:{m:float(v) for m,v in r.items() if m!='model'} for r in read_csv(OUT/'diagnostics_summary.csv')}
    pairs={(r['new'],r['baseline'],r['metric']):r for r in read_csv(OUT/'fresh_paired.csv')}
    slope=float(rt.linear.coef_[0]*rt.y_std/rt.scaler.scale_[0])
    rt.write_json(OUT/'control_geometry.json',dict(B_unclipped_slope=slope,
      identity='q_target=1/N+lambda*error/(N-t), while 0<=S/B<=1',
      approximation='delta_base ~= lambda*error/(B_slope*(N-t)), ignoring candidate quantization, prediction zero floor and action limits',
      extra_proportional_gain_last_tick_half=.5/slope,extra_proportional_gain_last_tick_full=1/slope,
      caveat='This describes the base multiplier adjustment on the same state, not the total closed-loop gain. Network outputs and later observations can also change.'))
    profiles=read_csv(OUT/'time_profiles.csv');daily=read_csv(OUT/'fresh_daily.csv')
    def curve(k,m):return np.array([float(r[m]) for r in profiles if r['model']==k])
    def interval(a,b,m,adjust=False):
        r=pairs[a,b,m];s=1 if m=='multiplier_tv' else 100;tag='98_75' if adjust else '95'
        return f"{s*float(r['mean_difference']):+.7f} [{s*float(r['ci'+tag+'_low']):+.7f}, {s*float(r['ci'+tag+'_high']):+.7f}]"
    lines=['# 새 1000일 동결 결과','', '|정책|TE %|UR %|J %|V|평균 factor|복원 가동 %|','|---|---:|---:|---:|---:|---:|---:|']
    for k in specs:
        s=scores[k];d=diag[k]
        lines.append(f"|{k}|{s['tracking_error']*100:.6f}|{s['unspent']*100:.6f}|{s['objective']*100:.6f}|{s['multiplier_tv']:.6f}|{d['mean_factor']:.6f}|{d['restoration_rate']*100:.3f}|")
    lines+=['','## 주 비교','', '|비교|UR 차이 %p (98.75% 구간)|TE 차이 %p (95% 구간)|V 차이 (95% 구간)|','|---|---|---|---|']
    for a,b in MAIN:lines.append(f'|{a} − {b}|{interval(a,b,"unspent",True)}|{interval(a,b,"tracking_error")}|{interval(a,b,"multiplier_tv")}|')
    lines+=['','## 모든 고정 비교: 95% 기술적 구간','', '|비교|UR 차이 %p|TE 차이 %p|V 차이|','|---|---|---|---|']
    for a,b in PAIRS:lines.append(f'|{a} − {b}|{interval(a,b,"unspent")}|{interval(a,b,"tracking_error")}|{interval(a,b,"multiplier_tv")}|')
    lines+=['','## 마감 기전','', '|정책|마지막 2시간 TE %|마지막 30분 TE %|최종 구간 직전 잔액 %|최종 구간 지출 %|최종 UR %|','|---|---:|---:|---:|---:|---:|']
    for k in specs:
        d=diag[k];lines.append(f"|{k}|{d['last2h_TE']*100:.6f}|{d['last30m_TE']*100:.6f}|{d['remaining_pre_final']*100:.6f}|{d['final_spend']*100:.6f}|{scores[k]['unspent']*100:.6f}|")
    (OUT/'report_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

    fig,axes=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
    for ax,m,s,title in zip(axes,['unspent','tracking_error','multiplier_tv'],[100,100,1],
      ['Unspent: 98.75% adjusted CI (pp)','Tracking: 95% descriptive CI (pp)','Action variation: 95% CI']):
        for j,(a,b) in enumerate(MAIN):
            r=pairs[a,b,m];tag='98_75' if m=='unspent' else '95'
            x=s*float(r['mean_difference']);lo=s*float(r['ci'+tag+'_low']);hi=s*float(r['ci'+tag+'_high'])
            ax.errorbar(x,j,xerr=[[x-lo],[hi-x]],fmt='o',capsize=4,color=f'C{j}')
        ax.axvline(0,color='black',lw=.8);ax.set(title=title,yticks=range(4),yticklabels=['CMA: target 50%','Gated: target 50%','Gated: restore','Gated: both'])
        ax.invert_yaxis();ax.xaxis.set_major_locator(MaxNLocator(4));ax.ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)
    savefig('main-contrasts')

    hour=(np.arange(288)+1)/12;keys=['CMA','G','G_T05','G_R','G_T05_R']
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for k in keys:
        color=COLORS[k];label=LABELS[k]
        axes[0,0].plot(hour,100*curve(k,'absolute_error'),color=color,label=label)
        axes[0,1].plot(hour[-24:],100*curve(k,'absolute_error')[-24:],color=color,label=label)
        axes[1,0].plot(hour[-24:],100*curve(k,'target_rate')[-24:],color=color,label=label)
        axes[1,1].plot(hour[-24:],curve(k,'base')[-24:],color=color,label=label)
    axes[0,0].set(title='Original cumulative target: full day',ylabel='Mean absolute error (%)')
    axes[0,1].set(title='Last 2 hours',ylabel='Mean absolute error (%)')
    axes[1,0].set(title='Feedforward spend target: last 2 hours',ylabel='Budget per interval (%)')
    axes[1,1].set(title='Chosen base multiplier: last 2 hours',ylabel='Mean base multiplier')
    for ax in axes.flat:ax.set_xlabel('Hour (interval end)');ax.legend(fontsize=8)
    savefig('target-and-tracking')

    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for k in ['G_R','G_T05_R','G_T1_R']:
        color=COLORS[k];label=LABELS[k]
        axes[0,0].plot(hour,100*curve(k,'restoration_applied'),color=color,label=label)
        axes[0,1].plot(hour,curve(k,'risk_weight'),color=color,label=label)
        axes[1,0].plot(hour,curve(k,'factor'),color=color,label=label)
    axes[1,0].plot(hour,curve('G','factor'),color=COLORS['G'],ls='--',label='Gated: no restoration')
    for k in keys:
        y=np.sort([100*float(r['unspent']) for r in daily if r['model']==k])
        axes[1,1].plot(y,np.arange(1,len(y)+1)/len(y),color=COLORS[k],label=LABELS[k])
    axes[0,0].set(title='Restoration actually raises Ki',ylabel='Intervals (%)',xlabel='Hour (interval end)')
    axes[0,1].set(title='Risk restoration weight',ylabel='Mean weight',xlabel='Hour (interval end)')
    axes[1,0].set(title='Resulting Ki factor',ylabel='Mean factor',xlabel='Hour (interval end)')
    axes[1,1].set(title='Final unspent distribution',xlabel='Unspent budget (%) — symlog',ylabel='Fraction of days')
    axes[1,1].set_xscale('symlog',linthresh=.001)
    for ax in axes.flat:ax.legend(fontsize=8)
    savefig('risk-restoration')

    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,m,scale,title in zip(axes,['tracking_error','unspent','multiplier_tv'],[100,100,1],['Tracking (%)','Unspent (%)','Action variation']):
        for parent,color in [('CMA','#555555'),('G','#0072B2')]:
            ks=[parent,parent+'_T05',parent+'_T1'];ax.plot([0,.5,1],[scores[k][m]*scale for k in ks],'-o',color=color,label=parent)
        ks=['G_R','G_T05_R','G_T1_R'];ax.plot([0,.5,1],[scores[k][m]*scale for k in ks],'-o',color='#D55E00',label='G + restoration')
        ax.set(title=title,xlabel='Remaining-budget target weight',xticks=[0,.5,1]);ax.legend(fontsize=9)
    savefig('target-strength')
    rt.write_json(OUT/'report_manifest.json',dict(source_sha256=rt.sha(Path(__file__)),
      input_sha256={p.name:rt.sha(p) for p in OUT.glob('*.csv')},no_new_rollouts=True))
    print('Tables and four figures saved; no fresh-data selection')

if __name__=='__main__':main()
