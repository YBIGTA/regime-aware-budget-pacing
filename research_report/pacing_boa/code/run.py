"""BOA action mixture of G and CMA_A05; simulator feedback after each action."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT / 'research_report/pacing_asymmetric_pi'
sys.path.insert(0, str(PARENT / 'code'))
from controller import AsymmetricPI
from env import rt, np, FRESH


class BOA:
    """Fixed-rate Bernstein update, Figure 1 of arXiv:1404.1356v5."""
    def __init__(self, eta=0.5):
        if not 0 < eta <= 0.5:
            raise ValueError('Require 0 < eta <= 0.5 for losses in [0,1]')
        self.eta = eta
        self.logw = np.log(np.array([0.5, 0.5]))

    @property
    def weights(self):
        z = np.exp(self.logw - self.logw.max())
        return z / z.sum()

    def update(self, losses):
        losses = np.asarray(losses, dtype=float)
        if losses.shape != (2,) or not np.isfinite(losses).all() or np.any((losses < 0) | (losses > 1)):
            raise ValueError('Two finite losses in [0,1] required')
        excess = losses - float(self.weights @ losses)
        self.logw -= self.eta * excess + self.eta**2 * excess**2
        self.logw -= self.logw.max()


class Mixture:
    def __init__(self, specs, boa, loss_scale=0.01):
        if loss_scale <= 0:
            raise ValueError('loss_scale must be positive')
        self.experts = [AsymmetricPI(specs[n]) for n in ('G', 'CMA_A05')]
        self.boa, self.loss_scale = boa, loss_scale
        self.pending = None

    def reset(self):
        # PI state restarts daily; BOA weights persist in declared day order.
        for expert in self.experts:
            expert.reset()
        self.pending = None

    def act(self, obs):
        if self.pending is not None:
            raise RuntimeError('Feedback must precede the next action')
        actions = np.array([expert.act(obs)[0] for expert in self.experts])
        weights = self.boa.weights.copy()
        action = float(np.clip(weights @ actions, rt.cfg.multiplier_min, rt.cfg.multiplier_max))
        self.pending = (obs.tick, actions, weights)
        return action, dict(saturated=bool(action <= 0 or action >= 1),
                            action_G=float(actions[0]), action_A05=float(actions[1]),
                            weight_G=float(weights[0]), weight_A05=float(weights[1]))

    def feedback(self, obs, tick):
        if self.pending is None or self.pending[0] != obs.tick:
            raise RuntimeError('Feedback without the matching action')
        # Replay only the JUST COMPLETED window, at the actual pre-action budget.
        # Candidate actions were fixed before this window was revealed.
        _, actions, _ = self.pending
        costs = np.array([rt.core.execute(tick, float(a), max(0., obs.budget-obs.cumulative_spend))[0]
                          for a in actions])
        next_fraction = (obs.cumulative_spend + costs) / obs.budget
        error = np.abs((obs.tick+1)/obs.ticks - next_fraction)
        losses = np.minimum(error/self.loss_scale, 1.)
        if obs.tick == obs.ticks-1:
            losses += np.minimum(np.maximum(0., 1-next_fraction)/self.loss_scale, 1.)
        self.boa.update(losses / 2.)
        self.pending = None


def run_day(day, specs, boa, loss_scale=0.01):
    ep = rt.episode(day)
    model = Mixture(specs, boa, loss_scale)
    def sink(obs, action, cost, censored):
        model.feedback(obs, ep.tape[obs.tick])
    return rt.core.rollout(ep, rt.cfg, model, sink=sink)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', type=int, default=1000)
    parser.add_argument('--eta', type=float, default=0.5)
    parser.add_argument('--loss-scale', type=float, default=0.01)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.days <= len(FRESH):
        parser.error('--days must be between 1 and 1000')
    if args.out.exists():
        parser.error('Output already exists; use a new output directory')
    boa = BOA(args.eta)
    if not np.isfinite(args.loss_scale) or args.loss_scale <= 0:
        parser.error('--loss-scale must be finite and positive')
    days = FRESH[:args.days]
    specs = rt.read_json(PARENT / 'results/protocol.json')['design']['policies']
    args.out.mkdir(parents=True)
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in ('pacing_boa', 'pacing_asymmetric_pi', 'pacing_ki_common')
               for p in (ROOT/'research_report'/folder/'code').glob('*.py')}
    rt.write_json(args.out/'protocol.json', dict(
        days=days, eta=args.eta, loss_scale=args.loss_scale,
        experts=['G', 'CMA_A05'], initialization=[0.5, 0.5],
        weight_reset='once at experiment start; carry across days',
        feedback='after completed window; simulator-only counterfactual replay',
        loss='0.5*min(abs(next_error)/scale,1) + terminal*0.5*min(unspent/scale,1)',
        update='logw -= eta*r + eta**2*r**2; r=loss-dot(w,loss)',
        tv_used_in_update=False, cohort='existing development confirmation; not final test',
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        sources=sources, simulator=rt.fingerprint(), final_test_used=False))
    daily = []
    start = time.monotonic()
    with (args.out/'weights.csv').open('w', newline='', encoding='utf-8') as wf, \
         (args.out/'daily.csv').open('w', newline='', encoding='utf-8') as df:
        weight_writer = csv.DictWriter(wf, fieldnames=['day', 'tick', 'weight_G', 'weight_A05',
                                                      'action_G', 'action_A05', 'action'])
        weight_writer.writeheader()
        day_writer = None
        for i, day in enumerate(days):
            score, trace = run_day(day, specs, boa, args.loss_scale)
            row = dict(model='BOA_G_A05', **score)
            if day_writer is None:
                day_writer = csv.DictWriter(df, fieldnames=list(row))
                day_writer.writeheader()
            day_writer.writerow(row)
            daily.append(row)
            for r in trace:
                weight_writer.writerow({k:r[k] for k in weight_writer.fieldnames})
            df.flush()
            wf.flush()
            rt.write_json(args.out/'checkpoint.json', dict(completed_days=i+1, day=day,
                          next_weights=boa.weights.tolist(), logw=boa.logw.tolist()))
            if i == 0 or (i+1) % 25 == 0 or i+1 == len(days):
                print(f'{i+1}/{len(days)} days; elapsed={time.monotonic()-start:.1f}s; '
                      f'next_weights={boa.weights.round(4).tolist()}', flush=True)
    summary = {'BOA_G_A05': {m:float(np.mean([r[m] for r in daily])) for m in rt.METRICS}}
    rt.write_json(args.out/'summary.json', summary)
    rows = [dict(model=name, tracking_error_pct=100*s['tracking_error'],
                 unspent_pct=100*s['unspent'], multiplier_tv=s['multiplier_tv'],
                 objective_pct=100*s['objective']) for name,s in summary.items()]
    rt.write_csv(args.out/'summary.csv', rows)
    report = ['# BOA 결합 평가', '', f'기존 확인 자료 {len(days)}일. 최종 테스트 미사용.', '',
              '| 모델 | 추종 오차 (%) | 미소진율 (%) | 승수 변동 |',
              '|---|---:|---:|---:|']
    for r in rows:
        report.append(f"| {r['model']} | {r['tracking_error_pct']:.6f} | {r['unspent_pct']:.6f} | {r['multiplier_tv']:.6f} |")
    report += ['', 'BOA는 실제 결합 행동의 경로에서 평가했다. 기존 실험의 지표를 가중 평균한 값이 아니다.',
               '가중치 갱신에는 완료된 경매 구간의 가상 재실행을 사용한다. 실제 광고 로그만으로 가능한 방식은 아니다.',
               '이 갱신 손실에 BOA 공식을 적용한 탐색 실험이며, 제어 성능에 대한 BOA 이론 보장을 주장하지 않는다.']
    (args.out/'README.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    main()

