"""Ridge gain selector: grouped training CV and validation continuous replay."""
import json
import hashlib
import re
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import gain_feasibility_v4 as g
import gain_predictability_v5 as v


def design(x, center, scale, nonlinear):
    z = np.clip((x-center)/scale, -6, 6)
    parts = [np.ones((len(z), 1)), z]
    if nonlinear:
        parts.extend([z*z, z[:, [2]]*z])
    return np.concatenate(parts, axis=1)


def fit(x, y, penalty, nonlinear, weights):
    center = x.mean(0)
    scale = x.std(0)
    scale[scale < 1e-8] = 1
    a = design(x, center, scale, nonlinear)
    reg = np.eye(a.shape[1])*penalty
    reg[0, 0] = 0
    w = weights/weights.mean()
    coef = np.linalg.solve(a.T@(w[:, None]*a)+reg, a.T@(w[:, None]*y))
    return center, scale, coef


def predict(model, x, nonlinear):
    return design(x, model[0], model[1], nonlinear)@model[2]


def pick(pred, baseline, threshold):
    # Always retain the baseline as a zero-improvement alternative.
    pred = pred.copy()
    pred[:, baseline] = 0
    ids = pred.argmin(1)
    ids[pred[np.arange(len(ids)), ids] >= -threshold] = baseline
    return ids


def period_weights(periods):
    return np.array([1/np.sum(periods == p) for p in periods])


def continuous(ep, params, bias, horizon, model, nonlinear,
               threshold, enabled, pairs, baseline):
    state = None
    pieces = []
    decisions = []
    start = 0
    while start < 48:
        if start == 0:
            end, chosen = 6, baseline
        else:
            chosen = baseline
            end = min(48, start+horizon)
            if enabled and end-start == horizon and state[0] >= .1:
                x = np.array([[
                    (48-start)/48, state[0]/ep['budget'],
                    start/48-(ep['budget']-state[0])/ep['budget'],
                    state[2], *ep['ticks'][start-1]['features'],
                ]], dtype=float)
                if not np.isfinite(x).all():
                    raise ValueError('Nonfinite live feature.')
                chosen = int(pick(predict(model, x, nonlinear),
                                  baseline, threshold)[0])
        gains = {'kp': float(pairs.iloc[chosen].kp),
                 'ki': float(pairs.iloc[chosen].ki)}
        f, states = g.roll(ep, gains, bias, start, end, state)
        f['candidate'] = int(pairs.iloc[chosen].candidate)
        pieces.append(f)
        decisions.append(int(chosen != baseline))
        state = states[end]
        start = end
    return pd.concat(pieces, ignore_index=True), decisions


def metrics(f, budget):
    tv = float(np.abs(np.diff(np.log(f.alpha.to_numpy()))).mean())
    unspent = float(f.remaining.iloc[-1]/budget)
    te = float(f.error.abs().mean())
    return dict(te=te, expected_conversions=float(f.value.sum()),
                unspent=unspent, objective=te+.01*tv+.2*unspent)


def main():
    base = ROOT/'runs/global_retune_v3'
    cfg = json.loads((base/'config.json').read_text())
    controller = json.loads((base/'selected_controller.json').read_text())
    if controller['antiwindup'] != 'clamp':
        raise ValueError('Expected clamp controller.')
    sources = [p for p in ROOT.glob('runs/gain_feasibility_v4_*')
               if (p/'candidate_windows.csv').is_file()
               and (p/'protocol.json').is_file()]
    source = max(sources, key=lambda p:p.name)
    protocol = json.loads((source/'protocol.json').read_text())
    if protocol['source_controller'] != controller:
        raise ValueError('v4 source controller differs from current controller.')
    frame = pd.read_csv(source/'candidate_windows.csv')
    pairs = frame[['candidate','kp','ki']].drop_duplicates().sort_values('candidate').reset_index(drop=True)
    if pairs.candidate.duplicated().any():
        raise ValueError('Ambiguous candidate IDs.')
    ids = pairs.candidate.tolist()
    p = controller['parameters']
    matches = np.flatnonzero(np.isclose(pairs.kp,p['kp']) & np.isclose(pairs.ki,p['ki']))
    if len(matches) != 1:
        raise ValueError('Baseline is ambiguous.')
    baseline = int(matches[0])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = ROOT/'runs'/('gain_training_v6_'+stamp)
    out.mkdir(parents=True)
    print('OUTPUT:', out.resolve(), flush=True)
    models, searches, reports, details, settings = {}, [], [], [], {}
    for h in [6,12]:
        b = frame[frame.horizon == h]
        meta, x, a = v.prepare(b[b.split=='train'], ids)
        periods = meta.period.to_numpy()
        if set(periods) != set(cfg['split']['train']):
            raise ValueError('Training periods differ from config.')
        y = (a['search_loss']-a['search_loss'][:,[baseline]])/(a['local_j'][:,[baseline]]+.01)
        best = None
        for nonlinear in [False, True]:
            for penalty in [1.,10.,100.]:
                oof = np.zeros_like(y)
                for held in np.unique(periods):
                    fit_mask = periods != held
                    model = fit(x[fit_mask],y[fit_mask],penalty,nonlinear,
                                period_weights(periods[fit_mask]))
                    oof[~fit_mask] = predict(model,x[~fit_mask],nonlinear)
                for threshold in [.02,.05,.10]:
                    selected = pick(oof,baseline,threshold)
                    score = v.period_mean(y[np.arange(len(y)),selected],periods)
                    key = (score, -threshold, nonlinear, -penalty)
                    searches.append(dict(horizon=h,nonlinear=nonlinear,
                                         penalty=penalty,threshold=threshold,cv_score=score))
                    if best is None or key < best[0]:
                        best = (key,nonlinear,penalty,threshold,selected.copy())
        _, nonlinear, penalty, threshold, selected = best
        # Practical predeclared floor: at least 0.5% normalized penalized improvement.
        enabled = bool(best[0][0] < -.005)
        if not enabled:
            selected[:] = baseline
        model = fit(x,y,penalty,nonlinear,period_weights(periods))
        models[h] = (model,nonlinear,threshold,enabled)
        settings[h] = dict(nonlinear=nonlinear,penalty=penalty,
                           threshold=threshold,enabled=enabled,
                           raw_train_cv_score=best[0][0])
        np.savez_compressed(out/f'model_h{h}.npz',center=model[0],scale=model[1],coef=model[2])
        for split in ['train','validation']:
            m, q, ar = v.prepare(b[b.split==split],ids)
            selected_ids = (selected if split=='train' else
                            pick(predict(model,q,nonlinear),baseline,threshold))
            if not enabled:
                selected_ids = np.full(len(m),baseline)
            for name, choices in [('PI',np.full(len(m),baseline)),('selector',selected_ids)]:
                report, detail = v.evaluate(m,ar,choices,baseline,split,h,name)
                reports.append(report)
                details.append(detail)
        print('TRAINED:',h,settings[h],flush=True)
    pd.DataFrame(searches).to_csv(out/'train_cv_search.csv',index=False)
    pd.DataFrame(reports).to_csv(out/'local_summary.csv',index=False)
    pd.concat(details).to_csv(out/'local_windows.csv',index=False)
    pairs.to_csv(out/'candidates.csv',index=False)
    (out/'protocol.json').write_text(json.dumps({
        'settings':settings,'features':v.FEATURES,'source':str(source),
        'config':cfg,'source_controller':controller,
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'candidate_csv_sha256':hashlib.sha256((source/'candidate_windows.csv').read_bytes()).hexdigest(),
        'scope':'Exploratory; PI preselected with period 10; CV labels share that frozen PI. Periods 11/12 already inspected previously.',
        'continuous_replay':'Validation only; carried budget/integral/previous output; no reset on switch.',
    },indent=2))
    lab = g.d.load_lab()
    g.d.self_test(lab)
    biases = {int(k):v for k,v in controller['initial_bias'].items()}
    episodes, traces, seen = [], [], set()
    for path in sorted((ROOT/'data').glob('pack-*/*.parquet')):
        match = re.fullmatch(r'p(\d+)-a(\d+)\.parquet',path.name)
        if not match or int(match[1]) not in cfg['split']['validation']:
            continue
        ep = lab.prepare_episode(pd.read_parquet(path),cfg['seed'])
        identity = (ep['period'],ep['ad'])
        if identity in seen:
            raise ValueError('Duplicate episode.')
        seen.add(identity)
        bias = float(np.clip((biases[ep['ad']] if controller['calibrated'] else 0)+p['offset'],-6,4))
        full, _ = g.roll(ep,p,bias)
        reference, _ = g.d.replay(ep,p,bias,'clamp')
        if not np.isclose(metrics(full,ep['budget'])['objective'],reference['objective']):
            raise ValueError('Replay parity failed.')
        for name,h in [('PI',6),('matched_P',6),('selector_h6',6),('selector_h12',12)]:
            if name == 'PI':
                f = full.copy()
            elif name == 'matched_P':
                f, _ = g.roll(ep,dict(p,ki=0.),bias)
            else:
                f, _ = continuous(ep,p,bias,h,*models[h],pairs,baseline)
            episodes.append(dict(period=ep['period'],advertiser=ep['ad'],method=name,
                                 **metrics(f,ep['budget'])))
            f['method'],f['period'],f['advertiser'] = name,ep['period'],ep['ad']
            traces.append(f)
        print('REPLAYED:',path.name,flush=True)
    expected = {(period,ad) for period in cfg['split']['validation'] for ad in biases}
    if seen != expected:
        raise ValueError('Incomplete validation coverage.')
    e = pd.DataFrame(episodes)
    e.to_csv(out/'continuous_episodes.csv',index=False)
    pd.concat(traces).to_csv(out/'continuous_traces.csv',index=False)
    summary = e.groupby(['method','period'])[['te','expected_conversions','unspent','objective']].mean().groupby('method').mean()
    summary.to_csv(out/'continuous_summary.csv')
    print('\nCONTINUOUS VALIDATION:\n',summary.to_string(),flush=True)
    (out/'DONE').write_text('Completed training and exploratory validation replay.\n')
    print('\nDONE:',out.resolve(),flush=True)

if __name__ == '__main__':
    main()
