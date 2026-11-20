"""Frozen CMA/G parents and a new, sealed-until-freeze evaluation cohort."""
import os,sys,math,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BUNDLE=Path(__file__).resolve().parents[1]
OLD=ROOT/'research_report/pacing_ki_common'
PARENT=ROOT/'research_report/pacing_sensitivity_hypotheses'
DEADLINE=ROOT/'research_report/pacing_deadline_pi'
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
os.environ['MPLCONFIGDIR']=str(ROOT/'outputs/mplconfig')
sys.path.insert(0,str(OLD/'code'))
import runtime as rt
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
FRESH=list(range(560000,561000))
PROBE=list(range(630000,630005))
rt.ALLOWED=set(FRESH+PROBE)
assert not rt.ALLOWED.intersection(rt.cfg.ids('test'))
METRICS=rt.METRICS
PRIOR=(1/288)/rt.cfg.initial_multiplier

def policies():
    old=rt.read_json(DEADLINE/'results/protocol.json')['design']['policies'];out={}
    for parent in ['CMA','G']:
        for suffix,blend,asymmetric in [('',0.,False),('_T05',.5,False),('_T1',1.,False),('_A05',.5,True),('_A1',1.,True)]:
            out[parent+suffix]={**copy.deepcopy(old[parent]),'target_blend':blend,'asymmetric':asymmetric}
    return out
