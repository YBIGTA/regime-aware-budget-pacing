"""Figures and tables from frozen CSVs only; no policy selection or rollout."""
from env import *
from experiment import OUT,read_csv,MAIN,PAIRS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
FIG=BUNDLE/'figures'
LABELS={'CMA':'CMA','G':'Gated','G_T05':'Symmetric 50%','G_A05':'Asymmetric 50%','G_A1':'Asymmetric 100%'}
COLORS={'CMA':'#555555','G':'#0072B2','G_T05':'#CC79A7','G_A05':'#009E73','G_A1':'#D55E00'}

def savefig(name):
    plt.savefig(FIG/(name+'.png'),dpi=165,bbox_inches='tight')
    path=FIG/(name+'.svg');plt.savefig(path,bbox_inches='tight')
    path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
    plt.close()

def main():
    assert rt.read_json(OUT/'audit.json')['status']=='passed'
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    specs=rt.read_json(OUT/'protocol.json')['design']['policies']
    scores={r['model']:{m:float(r[m]) for m in METRICS} for r in read_csv(OUT/'fresh_summary.csv')}
    diag={r['model']:{m:float(v) for m,v in r.items() if m!='model'} for r in read_csv(OUT/'diagnostics_summary.csv')}
    pairs={(r['new'],r['baseline'],r['metric']):r for r in read_csv(OUT/'fresh_paired.csv')}
    profiles=read_csv(OUT/'time_profiles.csv');daily=read_csv(OUT/'fresh_daily.csv')
    def curve(k,m):return np.array([float(r[m]) for r in profiles if r['model']==k])
    def interval(a,b,m,adjust=False):
        r=pairs[a,b,m];s=1 if m=='multiplier_tv' else 100;tag='98_75' if adjust else '95'
        return f"{s*float(r['mean_difference']):+.7f} [{s*float(r['ci'+tag+'_low']):+.7f}, {s*float(r['ci'+tag+'_high']):+.7f}]"
    lines=['# 새 1,000일 비대칭 목표 실험','', '|정책|TE %|UR %|J %|V|','|---|---:|---:|---:|---:|']
    for k in specs:
        s=scores[k];lines.append(f"|{k}|{s['tracking_error']*100:.6f}|{s['unspent']*100:.6f}|{s['objective']*100:.6f}|{s['multiplier_tv']:.6f}|")
    lines+=['','## 주 비교','', '|비교|UR 차이 %p (98.75% 구간)|TE 차이 %p (95% 구간)|V 차이 (95% 구간)|','|---|---|---|---|']
    for a,b in MAIN:lines.append(f'|{a} − {b}|{interval(a,b,"unspent",True)}|{interval(a,b,"tracking_error")}|{interval(a,b,"multiplier_tv")}|')
    lines+=['','## 모든 고정 비교: 95% 기술적 구간','', '|비교|UR 차이 %p|TE 차이 %p|V 차이|','|---|---|---|---|']
    for a,b in PAIRS:lines.append(f'|{a} − {b}|{interval(a,b,"unspent")}|{interval(a,b,"tracking_error")}|{interval(a,b,"multiplier_tv")}|')
    lines+=['','## 목표와 기본 승수의 변경 빈도','', '|정책|목표 상향 %|목표 하향 %|기본 승수 상향 %|기본 승수 하향 %|마지막 2시간 기본 승수 상향 %|','|---|---:|---:|---:|---:|---:|']
    for k in specs:
        d=diag[k];lines.append('|'+k+'|'+'|'.join(f'{100*d[m]:.3f}' for m in ['target_raise_rate','target_lower_rate','base_raise_rate','base_lower_rate','last2h_base_raise_rate'])+'|')
    lines+=['','기본 승수의 상향/하향은 같은 상태에서 고정 목표를 썼을 때와의 비교다. 다른 폐루프 경로의 실제 행동과 비교한 값이 아니다.','', '## 마감 잔액 분해','', '|정책|마지막 2시간 TE %|마지막 30분 TE %|23:55 잔액 %|마지막 5분 지출 %|최종 UR %|','|---|---:|---:|---:|---:|---:|---:|']
    for k in specs:
        d=diag[k];lines.append(f"|{k}|{d['last2h_TE']*100:.6f}|{d['last30m_TE']*100:.6f}|{d['remaining_pre_final']*100:.6f}|{d['final_spend']*100:.6f}|{scores[k]['unspent']*100:.6f}|")
    (OUT/'report_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

    fig,axes=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
    labels=['CMA A50 - CMA','G A50 - G','CMA A50 - T50','G A50 - T50']
    for ax,m,s,title in zip(axes,['unspent','tracking_error','multiplier_tv'],[100,100,1],['Unspent: adjusted 98.75% CI (pp)','Tracking: 95% CI (pp)','Action variation: 95% CI']):
        for j,(a,b) in enumerate(MAIN):
            r=pairs[a,b,m];tag='98_75' if m=='unspent' else '95'
            x=s*float(r['mean_difference']);lo=s*float(r['ci'+tag+'_low']);hi=s*float(r['ci'+tag+'_high'])
            ax.errorbar(x,j,xerr=[[x-lo],[hi-x]],fmt='o',capsize=4,color=f'C{j}')
        ax.axvline(0,color='black',lw=.8);ax.set(title=title,yticks=range(4),yticklabels=labels)
        ax.invert_yaxis();ax.xaxis.set_major_locator(MaxNLocator(4));ax.ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)
    savefig('main-contrasts')

    hour=(np.arange(288)+1)/12;keys=['CMA','G','G_T05','G_A05','G_A1']
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for k in keys:
        kwargs=dict(color=COLORS[k],label=LABELS[k])
        axes[0,0].plot(hour,100*curve(k,'absolute_error'),**kwargs)
        axes[0,1].plot(hour[-24:],100*curve(k,'absolute_error')[-24:],**kwargs)
        axes[1,0].plot(hour[-24:],100*curve(k,'target_rate')[-24:],**kwargs)
        axes[1,1].plot(hour[-24:],curve(k,'base')[-24:],**kwargs)
    axes[0,0].set(title='Original cumulative target: full day',ylabel='Mean absolute error (%)')
    axes[0,1].set(title='Last 2 hours',ylabel='Mean absolute error (%)')
    axes[1,0].set(title='Spend target: last 2 hours',ylabel='Budget per interval (%)')
    axes[1,1].set(title='Chosen base multiplier: last 2 hours',ylabel='Mean base multiplier')
    for ax in axes.flat:ax.set_xlabel('Hour (interval end)');ax.legend(fontsize=8)
    savefig('target-and-tracking')

    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,m,scale,title in zip(axes,['tracking_error','unspent','multiplier_tv'],[100,100,1],['Tracking (%)','Unspent (%)','Action variation']):
        for parent,color in [('CMA','#555555'),('G','#0072B2')]:
            for mode,style,label in [('T','--','symmetric'),('A','-','asymmetric')]:
                ks=[parent,parent+'_'+mode+'05',parent+'_'+mode+'1']
                ax.plot([0,.5,1],[scores[k][m]*scale for k in ks],style+'o',color=color,label=parent+' '+label)
        ax.set(title=title,xlabel='Target correction weight',xticks=[0,.5,1]);ax.legend(fontsize=8)
    savefig('target-strength')

    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    idx=np.arange(len(keys));spend=[100*diag[k]['final_spend'] for k in keys];ur=[100*scores[k]['unspent'] for k in keys]
    axes[0].bar(idx,spend,label='Spent in final interval',color='#56B4E9')
    axes[0].bar(idx,ur,bottom=spend,label='Still unspent at midnight',color='#D55E00')
    axes[0].set(xticks=idx,xticklabels=['CMA','G','G T50','G A50','G A100'],title='Budget remaining at 23:55',ylabel='Budget (%)')
    for k in keys:
        y=np.sort([100*float(r['unspent']) for r in daily if r['model']==k])
        axes[1].plot(y,np.arange(1,len(y)+1)/len(y),color=COLORS[k],label=LABELS[k])
    axes[1].set_xscale('symlog',linthresh=.001)
    axes[1].set(title='Final unspent distribution',xlabel='Unspent budget (%) — symlog',ylabel='Fraction of days')
    for ax in axes:ax.legend(fontsize=8)
    savefig('terminal-balance')
    rt.write_json(OUT/'report_manifest.json',dict(source_sha256=rt.sha(Path(__file__)),input_sha256={p.name:rt.sha(p) for p in OUT.glob('*.csv')},no_new_rollouts=True))
    print('Tables and four figures saved; no new rollouts or selection')

if __name__=='__main__':main()
