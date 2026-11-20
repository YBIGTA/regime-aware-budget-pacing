"""Compiled follow-up hypothesis controls, derived from frozen sensitivity kernel.
Only Ki scaling and diagnostic trace columns are added. The reference policy
in controller.py independently implements the estimator and uses core.rollout.
"""
from env import rt, np
import math
from numba import njit
from concurrent.futures import ThreadPoolExecutor
from fast_eval import pack

TRACE = ['action','error','cumulative_spend','kp','ki','integral','base','slope',
         'ki_original','g_est','factor','fallback','integral_blocked','saturated',
         'budget_censored','factor_clipped','g_ref','gate']

@njit(cache=True)
def make_features(history,tick,budget,base,error):
    n=0;am=wr=lc=bv=ym=count_mean=0.
    for h in range(max(0,tick-12),tick):
        if history[h,6]==0 and history[h,1]>1e-4:
            n+=1;am+=history[h,1];wr+=history[h,4];lc+=history[h,3]
            bv+=history[h,5];ym+=history[h,2];count_mean+=math.expm1(history[h,3])
    if n==0:return np.zeros(18)
    lr=math.log(max(ym/(1e-6+am),1e-8))
    am/=n;wr/=n;lc/=n;bv/=n;ym/=n;count_mean/=n
    av=cv=0.
    for h in range(max(0,tick-12),tick):
        if history[h,6]==0 and history[h,1]>1e-4:
            av+=(history[h,1]-am)**2
            cv+=(math.expm1(history[h,3])-count_mean)**2
    return np.array([base,base*base,math.log(budget),math.sin(2*math.pi*tick/288),
      math.cos(2*math.pi*tick/288),am,wr,lc,bv,ym,lr,math.sqrt(av/n),
      math.sqrt(cv/n)/max(1.,count_mean),base*wr,base*lc,base*lr,base*bv,error])

@njit(cache=True, nogil=True)
def kernel(offset, bids_base, competitors, base_means, budget, initial, scale_mean, scale_sd,
           coef, intercept, y_mean, y_sd, anchor, gains_scale, weights, behind_input, method, power, constant, g_ref, ema_alpha, target, center, model_mean, model_scale, model_coef, model_intercept, init_mode, constant_mode, factor_low, factor_high):
    ticks = 288
    n_input = 8 if behind_input else 7
    hidden = weights[:(n_input + 1) * 4].reshape((n_input + 1, 4))
    head = weights[(n_input + 1) * 4:].reshape((4, 2))
    history = np.zeros((ticks, 7))
    trace = np.zeros((ticks, 18))
    batch = np.empty((101, 89), dtype=np.float32)
    grid = np.linspace(0., 1., 101)
    x = np.zeros(89)
    integral = spent = te = tv = max_ahead = saturated = 0.
    sum_spend = sum_action = 0.
    smooth_g = math.exp(center/power) if init_mode==1 else g_ref
    initialized = False
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
        numerator = denominator = 0.
        count_valid = 0
        for h in range(max(0,tick-12),tick):
            if history[h,6] == 0 and history[h,1] > 1e-4:
                numerator += history[h,2]
                denominator += history[h,1]
                count_valid += 1
        valid = count_valid >= 6
        gain = g_ref
        gate = 1.
        if valid:
            ratio = numerator/(1e-6+denominator)
            raw_gain = ratio
            if method == 8:
                fs = make_features(history,tick,budget,base,error)
                pred = model_intercept
                for j in range(18):
                    pred += min(8.,max(-8.,(fs[j]-model_mean[j])/model_scale[j]))*model_coef[j]
                raw_gain = math.exp(min(math.log(.2),max(math.log(1e-5),pred)))
            if init_mode==2 and not initialized:smooth_g=raw_gain
            smooth_g += (raw_gain-smooth_g)*ema_alpha
            initialized=True
            gain = smooth_g
            if method == 7:
                residual = 0.
                for h in range(max(0,tick-12),tick):
                    if history[h,6] == 0 and history[h,1] > 1e-4:
                        residual += (history[h,2]-ratio*history[h,1])**2
                stderr = math.sqrt(count_valid/(count_valid-1)*residual)/denominator
                gate = 1/(1+(stderr/max(ratio,1e-8)/.25)**2)
        factor = 1.
        clipped = False
        fallback = not valid or not math.isfinite(gain) or gain < 0
        if not fallback:
            if method == 4: raw_factor=(g_ref/max(gain,.1*g_ref))**power
            else: raw_factor=target*math.exp(gate*(center-power*math.log(max(gain,1e-8))))
            factor = min(factor_high,max(factor_low,raw_factor))
            clipped = raw_factor < factor_low or raw_factor > factor_high
        if method == 0:
            factor = 1.
            fallback = clipped = False
        if method == 5:
            fallback = (constant_mode==1 and tick<6) or (constant_mode==2 and fallback)
            factor = 1. if fallback else constant
            clipped = False
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
        trace[tick, 17] = gate
    te /= ticks
    unspent = max(0., 1 - spent / budget)
    return np.array([te, unspent, te + unspent, tv / (ticks - 1), max_ahead, saturated / ticks]), trace


def run(day,spec):
    if day not in rt.ALLOWED: raise ValueError('Forbidden date')
    p=pack(day)
    metrics,trace=kernel(*p[:5],rt.cfg.initial_multiplier,rt.scaler.mean_,rt.scaler.scale_,
        rt.linear.coef_,rt.linear.intercept_,rt.y_mean,rt.y_std,np.array(spec['anchor']),
        rt.gain_scale,np.array(spec['weights']),True,spec['method'],spec['power'],
        spec['constant'],spec['g_ref'],spec['ema_alpha'],spec['target'],spec['center'],
        np.array(spec['slope_model']['mean']),np.array(spec['slope_model']['scale']),
        np.array(spec['slope_model']['coef']),spec['slope_model']['intercept'],
        spec['init_mode'],spec['constant_mode'],spec['factor_low'],spec['factor_high'])
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
