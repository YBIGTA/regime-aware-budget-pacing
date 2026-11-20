"""Frozen CMA/G parents and a new, sealed-until-freeze evaluation cohort."""
import os,sys,math,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BUNDLE=Path(__file__).resolve().parents[1]
OLD=ROOT/'research_report/pacing_ki_common'
PARENT=ROOT/'research_report/pacing_sensitivity_hypotheses'
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
os.environ['MPLCONFIGDIR']=str(ROOT/'outputs/mplconfig')
sys.path.insert(0,str(OLD/'code'))
import runtime as rt
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
FRESH=list(range(550000,551000))
PROBE=list(range(620000,620005))
rt.ALLOWED=set(FRESH+PROBE)
assert not rt.ALLOWED.intersection(rt.cfg.ids('test'))
METRICS=rt.METRICS
PRIOR=(1/288)/rt.cfg.initial_multiplier

def policies():
    old=rt.read_json(PARENT/'results/calibration.json')['specs'];out={}
    for parent in ['CMA','G']:
        for blend,label in [(0.,''),(.5,'_T05'),(1.,'_T1')]:
            out[parent+label]={**copy.deepcopy(old[parent]),'target_blend':blend,'risk_restore':False,'risk_ramp':.25}
    for blend,label in [(0.,''),(.5,'_T05'),(1.,'_T1')]:
        out['G'+label+'_R']={**copy.deepcopy(out['G'+label]),'risk_restore':True}
    return out
