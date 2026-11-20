"""Read-only bridge to the frozen B/E2 environment; final test is inaccessible."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = Path(__file__).resolve().parents[1]
OLD = ROOT / 'research_report/pacing_ki_common'
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'outputs/mplconfig')
sys.path.insert(0, str(OLD / 'code'))
import runtime as rt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)

TRAIN = list(range(400000, 400080))
SELECT = list(range(420000, 420160))
FRESH = list(range(520000, 520300))
PROBE = list(range(600000, 600005))
SPLITS = [TRAIN, SELECT, FRESH, PROBE, list(rt.cfg.ids('test'))]
assert all(not set(a) & set(b) for i,a in enumerate(SPLITS) for b in SPLITS[:i])
rt.ALLOWED = set(TRAIN + SELECT + FRESH + PROBE)
METRICS = rt.METRICS
FIXED = rt.read_json(rt.STATE/'code/B/frozen_experiment2.json')['B_fixed_PI']
B_SLOPE = float(rt.linear.coef_[0] * rt.y_std / rt.scaler.scale_[0])
RATIO_PRIOR = (1 / rt.cfg.ticks) / rt.cfg.initial_multiplier

def spec(method=0, strength=1., fixed=False, constant=1.):
    base = dict(rt.baseline_specs()['team_cma_sum'])
    if fixed:
        base.update(anchor=FIXED, weights=[0.]*44)
    return {**base, 'method':method, 'strength':strength, 'constant':constant,
            'g_ref': B_SLOPE if method == 1 else RATIO_PRIOR,
            'ema_alpha':1-float(np.exp(-1/6))}

def candidates():
    specs={'S0_CMA':spec(), 'F0_fixed':spec(fixed=True),
           'state_derivative':spec(1), 'fixed_derivative':spec(1,fixed=True)}
    families={k:[k] for k in specs}
    for family,method,fixed in [('state_cumulative',2,False),('fixed_cumulative',2,True),
                              ('state_window_raw',3,False),('state_window_smooth',4,False),
                              ('fixed_window_smooth',4,True)]:
        families[family]=[]
        for power in [.25,.5,1.]:
            key=f'{family}_p{power:g}'
            specs[key]=spec(method,power,fixed)
            families[family].append(key)
    families['state_constant']=[]
    for factor in [.5,.75,1.25,1.5,2.]:
        key=f'state_constant_x{factor:g}'
        specs[key]=spec(5,constant=factor)
        families['state_constant'].append(key)
    return specs,families
