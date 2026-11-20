"""Frozen parent artifacts and disjoint follow-up cohorts."""
import os,sys,math,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BUNDLE=Path(__file__).resolve().parents[1]
OLD=ROOT/'research_report/pacing_ki_common'
PARENT=ROOT/'research_report/pacing_sensitivity_refinement'
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
os.environ['MPLCONFIGDIR']=str(ROOT/'outputs/mplconfig')
sys.path.insert(0,str(OLD/'code'))
import runtime as rt
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
CAL=list(range(450000,450080))
FRESH=list(range(540000,540800))
PROBE=list(range(610000,610005))
rt.ALLOWED=set(CAL+FRESH+PROBE)
assert not rt.ALLOWED.intersection(rt.cfg.ids('test'))
METRICS=rt.METRICS
PRIOR=(1/288)/rt.cfg.initial_multiplier

def amend(sp,**changes):
    return {**copy.deepcopy(sp),'init_mode':0,'constant_mode':0,'factor_low':.5,'factor_high':2.,**changes}

def policies():
    prev=rt.read_json(PARENT/'results/selection.json')['selected_specs']
    g=amend(prev['gated_ratio_p0.25_m0.75'])
    r=amend(prev['centered_ratio_p1_m1'])
    l=amend(prev['learned_slope_p1_m0.75'])
    c=amend(prev['matched_gated_ratio_p0.25_m0.75'])
    return dict(CMA=amend(prev['S0_CMA']),G=g,C_all=c,
      C_warm=amend(c,constant_mode=1),C_rule=amend(c,constant_mode=2),
      C_matched=amend(c,constant_mode=2),
      G_neutral=amend(g,init_mode=1),G_first=amend(g,init_mode=2),
      R=r,R_neutral=amend(r,init_mode=1),R_first=amend(r,init_mode=2),
      L=l,L_noema=amend(l,ema_alpha=1.),L_wide=amend(l,factor_low=.25,factor_high=3.))
