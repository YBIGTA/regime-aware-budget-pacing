"""Descriptive analysis after locked confirmation; cannot select or train a model."""
import csv
import platform
from pathlib import Path
from bridge import *
from fast_sensitivity import run, TRACE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=BUNDLE/'results'
FIG=BUNDLE/'figures'
LABEL={'S0_CMA':'CMA E2', 'F0_fixed':'Fixed PI', 'state_derivative':'B slope',
       'fixed_derivative':'Fixed + B slope'}

def label(key):
    if key in LABEL:return LABEL[key]
    return key.replace('state_cumulative_p','Cumulative p=').replace('state_window_raw_p','Window p=')\
       .replace('state_window_smooth_p','Window+EMA p=').replace('state_constant_x','Constant x')\
       .replace('fixed_cumulative_p','Fixed cumulative p=').replace('fixed_window_smooth_p','Fixed EMA p=')

def read(name):
    with (OUT/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))

def savefig(name):
    plt.savefig(FIG/(name+'.png'),dpi=170,bbox_inches='tight')
    svg=FIG/(name+'.svg')
    plt.savefig(svg,bbox_inches='tight')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close()

def main():
    assert rt.read_json(OUT/'audit.json')['status']=='passed'
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.alpha':.2,'figure.facecolor':'white','savefig.facecolor':'white'})
    sel=rt.read_json(OUT/'selection.json');specs=sel['selected_specs']
    mainkey=sel['primary']['diagnostic']
    summaries=read('fresh_summary.csv');scores={r['model']:{m:float(r[m]) for m in METRICS} for r in summaries}
    daily=read('fresh_daily.csv');byday={(r['model'],int(r['day'])):r for r in daily}
    delta={d:float(byday[mainkey,d]['unspent'])-float(byday['S0_CMA',d]['unspent']) for d in FRESH}
    cases={'worst_UR':max(delta,key=delta.get),'best_UR':min(delta,key=delta.get)}
    profiles=[];details=[];case_rows=[];traces={}
    for key,sp in specs.items():
        arr=[]
        for day in FRESH:
            _,trace=run(day,sp);arr.append(trace)
            if day in cases.values() and key in ['S0_CMA',mainkey]:
                traces[key,day]=trace
                case_rows.extend({'model':key,'day':day,'tick':t,**dict(zip(TRACE,row))} for t,row in enumerate(trace))
        a=np.stack(arr);f=a[:,:,10];g=a[:,:,9]
        details.append(dict(model=key,mean_factor=float(f.mean()),factor_q05=float(np.quantile(f,.05)),
           factor_q50=float(np.median(f)),factor_q95=float(np.quantile(f,.95)),
           factor_below_one_fraction=float(np.mean(f<1-1e-5)),factor_above_one_fraction=float(np.mean(f>1+1e-5)),
           factor_step_tv=float(np.abs(np.diff(f,axis=1)).mean()),
           gain_q05=float(np.quantile(g,.05)),gain_q50=float(np.median(g)),gain_q95=float(np.quantile(g,.95)),
           mean_ki=float(a[:,:,4].mean()),ki_cap_fraction=float(np.mean(a[:,:,4]>=6000)),
           fallback_fraction=float(a[:,:,11].mean()),factor_clip_fraction=float(a[:,:,15].mean()),
           integral_block_fraction=float(a[:,:,12].mean()),saturation_fraction=float(a[:,:,13].mean()),
           late_factor_mean=float(f[:,240:].mean())))
        for t in range(288):
            profiles.append(dict(model=key,tick=t,hour=(t+1)/12,
              tracking_error_pct=float(np.abs(a[:,t,1]).mean()*100),signed_error_pct=float(a[:,t,1].mean()*100),
              mean_ki=float(a[:,t,4].mean()),mean_factor=float(f[:,t].mean()),mean_gain=float(g[:,t].mean()),
              mean_action=float(a[:,t,0].mean()),mean_integral=float(a[:,t,5].mean())))
    rt.write_csv(OUT/'controller_diagnostics.csv',details)
    rt.write_csv(OUT/'time_profiles.csv',profiles)
    rt.write_csv(OUT/'case_traces.csv',case_rows)
    rt.write_json(OUT/'case_selection.json',dict(method='Post-hoc descriptive extreme UR differences of preselected diagnostic vs CMA; never used for selection.',cases=cases))

    diag=read('B_slope_diagnostic.csv');actual=np.array([float(r['actual_fd_slope']) for r in diag])
    fig,ax=plt.subplots(figsize=(8.4,4.5))
    ax.hist(actual,bins=65,color='#4c78a8',alpha=.85,label='Auction finite difference (offline only)')
    ax.axvline(B_SLOPE,color='#e45756',lw=2.2,label='B predicted slope (effectively constant)')
    ax.set(xlabel='Normalized 5-minute spend change / multiplier change',ylabel='5-minute intervals',
       title='B predictor supplies almost no sensitivity variation (80 development days)')
    ax.legend(frameon=False,fontsize=9)
    savefig('slope-diagnostic')

    keys=['S0_CMA','F0_fixed','state_derivative']+[sel['families'][f]['diagnostic'] for f in
          ['state_cumulative','state_window_raw','state_window_smooth','state_constant']]
    keys=list(dict.fromkeys(keys));y=np.arange(len(keys))
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),sharey=True)
    colors=['#203d5b' if k=='S0_CMA' else '#d58538' if k==mainkey else '#839bb1' for k in keys]
    for ax,m,title,mult in zip(axes,['tracking_error','unspent','multiplier_tv'],
                             ['Tracking error (%)','Final unspent (%)','Multiplier variation'],[100,100,1]):
        ax.barh(y,[scores[k][m]*mult for k in keys],color=colors)
        ax.axvline(scores['S0_CMA'][m]*mult,color='#203d5b',ls='--',lw=1)
        ax.set(title=title,xlabel='Lower is better')
        ax.set_yticks(y,[label(k) for k in keys])
    axes[0].invert_yaxis();fig.suptitle('300 new confirmation days; candidates frozen on a separate 160 days',y=1.02)
    fig.tight_layout();savefig('fresh-comparison')

    paired=read('fresh_paired.csv');pindex={(r['new'],r['baseline'],r['metric']):r for r in paired}
    diffkeys=[k for k in keys if k not in ['S0_CMA','F0_fixed','state_derivative']]
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,m,title in zip(axes,['tracking_error','unspent','multiplier_tv'],
                         ['Tracking error','Final unspent','Multiplier variation']):
        for i,k in enumerate(diffkeys):
            r=pindex[k,'S0_CMA',m];scale=100/scores['S0_CMA'][m]
            mean=float(r['mean_difference'])*scale
            low=float(r['ci95_low'])*scale;high=float(r['ci95_high'])*scale
            ax.errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt='o',capsize=4,
                       color='#d58538' if k==mainkey else '#4c78a8')
        ax.axvline(0,color='#555',lw=1);ax.set(title=title,xlabel='Difference / CMA mean (%)')
        ax.set_yticks(np.arange(len(diffkeys)),[label(k) for k in diffkeys])
    axes[0].invert_yaxis();fig.suptitle('Paired daily bootstrap 95% intervals (descriptive; no multiplicity correction)',y=1.02)
    fig.tight_layout();savefig('paired-differences')

    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for key,color in [('S0_CMA','#203d5b'),(mainkey,'#d58538')]:
        rows=[r for r in profiles if r['model']==key];hours=[r['hour'] for r in rows]
        axes[0].plot(hours,[r['tracking_error_pct'] for r in rows],label=label(key),color=color)
        axes[1].plot(hours,[r['mean_ki'] for r in rows],color=color)
        axes[2].plot(hours,[r['mean_factor'] for r in rows],color=color)
    axes[0].set(ylabel='Abs. tracking error (%)',title='Locked diagnostic candidate vs CMA: 300-day time profile')
    axes[0].legend(frameon=False);axes[1].set(ylabel='Mean Ki');axes[2].set(ylabel='Mean Ki correction',xlabel='Hour')
    axes[2].set_xticks(range(0,25,2));fig.tight_layout();savefig('time-profile')

    fig,axes=plt.subplots(3,2,figsize=(12,8),sharex=True)
    for col,(reason,day) in enumerate(cases.items()):
        for key,color in [('S0_CMA','#203d5b'),(mainkey,'#d58538')]:
            t=traces[key,day];h=(np.arange(288)+1)/12
            axes[0,col].plot(h,t[:,1]*100,label=label(key),color=color)
            axes[1,col].plot(h,t[:,0],color=color,alpha=.85)
            axes[2,col].plot(h,t[:,10],color=color)
        axes[0,col].axhline(0,color='#555',lw=.6)
        axes[0,col].set(title=f'{reason}: day {day}',ylabel='Target - spend (% budget)')
        axes[1,col].set(ylabel='Final multiplier');axes[2,col].set(ylabel='Ki correction',xlabel='Hour')
        axes[2,col].set_xticks(range(0,25,4))
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle('Post-hoc examples; not evidence for candidate selection',y=1.01)
    fig.tight_layout();savefig('cases')
    paper=ROOT/'outputs/sensitivity-paper/paper.pdf'
    rt.write_json(OUT/'report_metadata.json',dict(python=platform.python_version(),numpy=np.__version__,
        matplotlib=matplotlib.__version__,report_sha256=rt.sha(Path(__file__)),
        svg_whitespace_normalized=True,
        paper_sha256=rt.sha(paper) if paper.exists() else None,
        additional_trace_rollouts=len(specs)*len(FRESH),additional_rollouts_reason='Descriptive analysis only; no model selection or final test.',
        final_test_used=False))
    print('Figures and descriptive diagnostics complete',flush=True)

if __name__=='__main__':main()
