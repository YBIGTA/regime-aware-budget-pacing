"""Compiled sensitivity replay, derived from frozen fast_eval.kernel.
Only Ki scaling and diagnostic trace columns are added. The reference policy
in controller.py independently implements the estimator and uses core.rollout.
"""
from bridge import rt, np
import math
from numba import njit
from concurrent.futures import ThreadPoolExecutor
from fast_eval import pack

TRACE = ['action','error','cumulative_spend','kp','ki','integral','base','slope',
         'ki_original','g_est','factor','fallback','integral_blocked','saturated',
         'budget_censored','factor_clipped','g_ref']

@njit(cache=True, nogil=True)
def kernel(offset, bids_base, competitors, base_means, budget, initial, scale_mean, scale_sd,
           coef, intercept, y_mean, y_sd, anchor, gains_scale, weights, behind_input, method, power, constant, g_ref, ema_alpha):
    ticks = 288
    n_input = 8 if behind_input else 7
    hidden = weights[:(n_input + 1) * 4].reshape((n_input + 1, 4))
    head = weights[(n_input + 1) * 4:].reshape((4, 2))
    history = np.zeros((ticks, 7))
    trace = np.zeros((ticks, 17))
    batch = np.empty((101, 89), dtype=np.float32)
    grid = np.linspace(0., 1., 101)
    x = np.zeros(89)
    integral = spent = te = tv = max_ahead = saturated = 0.
    sum_spend = sum_action = 0.
    smooth_g = g_ref
    for tick in range(ticks):
        error = tick / ticks - spent / budget
        if tick == 0:
            base, slope = initial, 0.
        else:
            x[1] = tick / ticks
            x[2] = math.log(budget)
            x[3] = spent / budget
            x[4] = error
            for h in range(12):
                past = tick - 12 + h
                for k in range(7):
                    x[5 + h * 7 + k] = history[past, k] if past >= 0 else 0.
            for k in range(1, 89):
                v = np.float32((x[k] - scale_mean[k]) / scale_sd[k])
                for g in range(101):
                    batch[g, k] = v
            for g in range(101):
                batch[g, 0] = np.float32((grid[g] - scale_mean[0]) / scale_sd[0])
            predicted = batch @ coef
            chosen = 0
            best = 1e100
            for g in range(101):
                z = np.float32(predicted[g] + intercept)
                p = np.float32(np.float32(z * np.float32(y_sd)) + np.float32(y_mean))
                predicted[g] = max(p, np.float32(0.))
                dist = abs(float(predicted[g]) - 1 / ticks)
                if dist < best:
                    best, chosen = dist, g
            base = grid[chosen]
            left, right = max(0, chosen - 5), min(100, chosen + 5)
            slope = (float(predicted[right]) - float(predicted[left])) / (grid[right] - grid[left])
        recent_n = min(6, tick)
        recent_spend = traffic_sum = 0.
        for h in range(tick - recent_n, tick):
            recent_spend += history[h, 2] * budget
            traffic_sum += offset[h + 1] - offset[h]
        traffic_mean = traffic_sum / recent_n if recent_n else 0.
        variance = 0.
        for h in range(tick - recent_n, tick):
            variance += (offset[h + 1] - offset[h] - traffic_mean) ** 2
        variation = math.sqrt(variance / recent_n) / max(1., traffic_mean) if recent_n else 0.
        state = np.empty(n_input)
        state[0] = 2 * tick / ticks - 1
        state[1] = 2 * (1 - spent / budget) - 1
        state[2] = np.tanh(20 * error)
        state[3] = np.tanh(50 * (recent_n / ticks - recent_spend / budget))
        state[4] = np.tanh(variation)
        state[5] = 2 * base - 1
        state[6] = np.tanh(slope * ticks / 10)
        if behind_input:
            state[7] = 1. if error >= .005 else -1.
        hval = np.empty(4)
        for h in range(4):
            value = hidden[n_input, h]
            for k in range(n_input):
                value += state[k] * hidden[k, h]
            hval[h] = np.tanh(value)
        gains = np.empty(2)
        for k in range(2):
            value = 0.
            for h in range(4):
                value += hval[h] * head[h, k]
            gains[k] = min(2 * gains_scale[k], max(0., anchor[k] + np.tanh(value) * gains_scale[k]))
        kp, ki = gains[0], gains[1]
        ki_original = ki
        if tick:
            if history[tick-1,6] == 0 and history[tick-1,1] > 1e-4:
                sum_spend += history[tick-1,2]
                sum_action += history[tick-1,1]
        gain = g_ref
        valid = True
        if method == 1:
            gain = slope
            valid = tick > 0
        elif method == 2:
            weight = 1 - tick/ticks
            gain = weight*g_ref + (1-weight)*sum_spend/(1e-6+sum_action)
            valid = sum_action > 1e-4
        elif method == 3 or method == 4:
            numerator = denominator = 0.
            count_valid = 0
            for h in range(max(0,tick-12),tick):
                if history[h,6] == 0 and history[h,1] > 1e-4:
                    numerator += history[h,2]
                    denominator += history[h,1]
                    count_valid += 1
            valid = count_valid >= 6 and denominator > 1e-4
            if valid:
                gain = numerator/(1e-6+denominator)
                if method == 4:
                    smooth_g += (gain-smooth_g)*ema_alpha
                    gain = smooth_g
        factor = 1.
        clipped = False
        fallback = not valid or not math.isfinite(gain) or gain < 0
        if not fallback:
            raw_factor = (g_ref/max(gain,.1*g_ref))**power
            factor = min(2.,max(.5,raw_factor))
            clipped = raw_factor < .5 or raw_factor > 2.
        if method == 0:
            factor = 1.
            fallback = clipped = False
        if method == 5:
            factor = constant
            fallback = clipped = False
        ki = min(2*gains_scale[1],max(0.,ki_original*factor))
        candidate = integral + ki * error / ticks
        raw = base + kp * error + candidate
        blocked = (raw > 1 and error > 0) or (raw < 0 and error < 0)
        if blocked:
            raw = base + kp * error + integral
        else:
            integral = candidate
        saturated += 1. if raw < 0 or raw > 1 else 0.
        action = min(1., max(0., raw))
        cost = 0.
        wins = 0
        for index in range(offset[tick], offset[tick + 1]):
            payment = bids_base[index] * action
            if payment > competitors[index] and payment > 0:
                cost += payment
                wins += 1
        censored = cost > max(0., budget - spent)
        if censored:
            cost = 0.
            wins = 0
            for index in range(offset[tick], offset[tick + 1]):
                payment = bids_base[index] * action
                if payment > competitors[index] and payment > 0 and payment <= max(0., budget - spent) - cost:
                    cost += payment
                    wins += 1
        spent += cost
        end_error = (tick + 1) / ticks - spent / budget
        history[tick, 0] = 1.
        history[tick, 1] = action
        history[tick, 2] = cost / budget
        count = offset[tick + 1] - offset[tick]
        history[tick, 3] = math.log1p(count)
        history[tick, 4] = wins / max(1, count)
        history[tick, 5] = base_means[tick]
        history[tick, 6] = 1. if censored else 0.
        te += abs(end_error)
        max_ahead = max(max_ahead, -end_error)
        if tick:
            tv += abs(action - trace[tick - 1, 0])
        trace[tick, 0] = action
        trace[tick, 1] = end_error
        trace[tick, 2] = spent
        trace[tick, 3] = kp
        trace[tick, 4] = ki
        trace[tick, 5] = integral
        trace[tick, 6] = base
        trace[tick, 7] = slope
        trace[tick, 8] = ki_original
        trace[tick, 9] = gain
        trace[tick, 10] = factor
        trace[tick, 11] = 1. if fallback else 0.
        trace[tick, 12] = 1. if blocked else 0.
        trace[tick, 13] = 1. if raw < 0 or raw > 1 else 0.
        trace[tick, 14] = 1. if censored else 0.
        trace[tick, 15] = 1. if clipped else 0.
        trace[tick, 16] = g_ref
    te /= ticks
    unspent = max(0., 1 - spent / budget)
    return np.array([te, unspent, te + unspent, tv / (ticks - 1), max_ahead, saturated / ticks]), trace


def run(day,spec):
    if day not in rt.ALLOWED: raise ValueError('Forbidden date')
    p=pack(day)
    metrics,trace=kernel(*p[:5],rt.cfg.initial_multiplier,rt.scaler.mean_,rt.scaler.scale_,
        rt.linear.coef_,rt.linear.intercept_,rt.y_mean,rt.y_std,np.array(spec['anchor']),
        rt.gain_scale,np.array(spec['weights']),True,spec['method'],spec['strength'],
        spec['constant'],spec['g_ref'],spec['ema_alpha'])
    return {'day':day,'tape_sha256':p[5],**dict(zip(rt.METRICS,metrics))},trace

def evaluate(specs,days,workers=8):
    if not set(days)<=rt.ALLOWED: raise ValueError('Forbidden dates')
    for day in days: pack(day)
    items=[(day,k,s) for day in days for k,s in specs.items()]
    def one(item):
        day,key,sp=item
        score,_=run(day,sp)
        return {'model':key,**score}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows=list(pool.map(one,items))
    return sorted(rows,key=lambda r:(r['model'],r['day']))
