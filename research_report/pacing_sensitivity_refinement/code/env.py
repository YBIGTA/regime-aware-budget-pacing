"""Frozen CMA environment; separate development, selection and confirmation dates."""
import os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BUNDLE=Path(__file__).resolve().parents[1]
OLD=ROOT/'research_report/pacing_ki_common'
FIRST=ROOT/'research_report/pacing_sensitivity_pi'
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
os.environ['MPLCONFIGDIR']=str(ROOT/'outputs/mplconfig')
sys.path.insert(0,str(OLD/'code'))
import runtime as rt
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
FIT=list(range(400000,400240))
CAL=list(range(430000,430080))
SELECT=list(range(440000,440200))
FRESH=list(range(530000,530400))
PROBE=list(range(600000,600005))
groups=[FIT,CAL,SELECT,FRESH,PROBE,list(rt.cfg.ids('test'))]
assert all(not set(a)&set(b) for i,a in enumerate(groups) for b in groups[:i])
rt.ALLOWED=set(FIT+CAL+SELECT+FRESH+PROBE)
METRICS=rt.METRICS
PRIOR=(1/288)/rt.cfg.initial_multiplier
B_SLOPE=float(rt.linear.coef_[0]*rt.y_std/rt.scaler.scale_[0])
FEATURES=['base','base_squared','log_budget','sin_time','cos_time','mean_action',
 'mean_win_rate','mean_log_count','mean_base_bid','mean_normalized_spend','log_ratio',
 'action_sd','count_cv','base_x_win_rate','base_x_log_count','base_x_log_ratio','base_x_base_bid','error']
def spec(method=0,power=.25,target=1.,center=0.,constant=1.,model=None):
    model=model or dict(mean=[0.]*18,scale=[1.]*18,coef=[0.]*18,intercept=0.)
    return {**rt.baseline_specs()['team_cma_sum'],'method':method,'power':power,'target':target,
      'center':center,'constant':constant,'g_ref':PRIOR,'ema_alpha':1-float(np.exp(-1/6)),
      'slope_model':{k:model[k] for k in ['mean','scale','coef','intercept']}}
