"""
Step 1: Audit Log Generator
============================
Runs the 6-step math engine across a grid of parameter combinations
for every round in pancake_1year.jsonl.

Strategy: instead of saving one record per combo (216 × 105k = 22M records / 4 GB),
we save ONE record per round containing:
  - X_t: market state features
  - best_theta: the parameter combo with highest PnL that round (the training target)
  - all_pnls: PnL for every combo (compact array, not full records)

This produces ~105k records / ~150 MB — fast to generate and easy to upload.

Run:  python3 dwt_denoising/audit_log_generator.py
Time: ~30 min on CPU
Size: ~150 MB
"""

import json
import numpy as np
import pywt
from datetime import datetime
from itertools import product
import sys, os

sys.path.insert(0, "dwt_denoising")
from wavelet_denoise import wavelet_denoise

# ── Parameter grid ─────────────────────────────────────────────────────────────
PARAM_GRID = {
    "wavelet_level":    [1, 2, 3],
    "hurst_window":     [400],          # fixed — short windows always give H=0.5 (broken)
    "fuzzy_threshold":  [0.30, 0.45, 0.60],
    "slope_thresh":     [0.03, 0.07, 0.12],
    "lookback":         [2, 4],
}

COMBOS = list(product(
    PARAM_GRID["wavelet_level"],
    PARAM_GRID["hurst_window"],
    PARAM_GRID["fuzzy_threshold"],
    PARAM_GRID["slope_thresh"],
    PARAM_GRID["lookback"],
))
print(f"Parameter combinations: {len(COMBOS)}")

# ── 6-Step math engine ─────────────────────────────────────────────────────────

def run_6step(prices, wavelet_level, hurst_window, fuzzy_threshold,
              slope_thresh, lookback, markov_outcomes):
    """
    Run the full 6-step engine with given parameters.
    Returns dict with all intermediate values and final P_up.
    """
    n = len(prices)
    if n < max(hurst_window, 20):
        return None

    # Step 1: DWT denoising
    smoothed = wavelet_denoise(prices, wavelet="db4", level=wavelet_level)
    slope    = np.diff(smoothed, prepend=smoothed[0])
    avg_slope = float(np.mean(slope[-lookback:]))

    # Step 2: Hurst exponent
    # Hurst on RAW prices — smoothed prices destroy mean-reversion signal (H always > 0.5)
    raw_arr = np.array(prices)
    H = _hurst(raw_arr[-hurst_window:] if len(raw_arr) >= hurst_window else raw_arr)

    # Step 3: Fuzzy inference
    P_up, P_down, conf = _fuzzy(avg_slope, H, fuzzy_threshold,
                                 _markov_p_up(markov_outcomes))

    # Step 4: Signal direction
    if avg_slope > slope_thresh:
        signal = "UP"
    elif avg_slope < -slope_thresh:
        signal = "DOWN"
    else:
        signal = "NEUTRAL"

    return {
        "avg_slope":  round(avg_slope, 6),
        "H":          round(H, 4),
        "P_up":       round(P_up, 4),
        "P_down":     round(P_down, 4),
        "confidence": round(conf, 4),
        "signal":     signal,
        "smoothed_last": round(float(smoothed[-1]), 4),
    }


def _hurst(series):
    n = len(series)
    if n < 30: return 0.5
    returns = np.diff(series)
    if len(returns) < 40: return 0.5
    lags, rs_vals = [], []
    for lag in range(10, len(returns) // 4, 10):
        sub_rs = []
        for start in range(0, len(returns) - lag, lag):
            chunk = returns[start:start + lag]
            devs  = np.cumsum(chunk - np.mean(chunk))
            R = np.max(devs) - np.min(devs)
            S = np.std(chunk, ddof=1)
            if S > 0: sub_rs.append(R / S)
        if len(sub_rs) >= 2:
            lags.append(np.log(lag))
            rs_vals.append(np.log(np.mean(sub_rs)))
    if len(lags) < 4: return 0.5
    H, _ = np.polyfit(lags, rs_vals, 1)
    return float(np.clip(H, 0.05, 0.95))


def _trap(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    elif b <= x <= c:    return 1.0
    elif a < x < b:      return (x - a) / (b - a)
    else:                return (d - x) / (d - c)


def _fuzzy(slope_val, H, fuzzy_threshold, markov_p_up):
    t = fuzzy_threshold
    fs = {
        "strong_down": _trap(slope_val, -99, -99, -t*2, -t),
        "weak_down":   _trap(slope_val, -t*2, -t, -t*0.5, 0.0),
        "neutral":     _trap(slope_val, -t, -t*0.3, t*0.3, t),
        "weak_up":     _trap(slope_val, 0.0, t*0.5, t, t*2),
        "strong_up":   _trap(slope_val, t, t*2, 99, 99),
    }
    fh = {
        "mean_rev": _trap(H, 0.0, 0.0, 0.35, 0.5),
        "random":   _trap(H, 0.35, 0.45, 0.55, 0.65),
        "trending": _trap(H, 0.5, 0.65, 1.0, 1.0),
    }
    fm = {
        "biased_down": _trap(markov_p_up, 0.0, 0.0, 0.35, 0.5),
        "biased_up":   _trap(markov_p_up, 0.5, 0.65, 1.0, 1.0),
    }
    up_act = [
        min(fs["strong_up"],   fh["trending"])  * 1.0,
        min(fs["weak_up"],     fh["trending"])  * 0.7,
        min(fs["strong_down"], fh["mean_rev"])  * 0.8,
        min(fm["biased_up"],   fh["trending"])  * 0.5,
    ]
    dn_act = [
        min(fs["strong_down"], fh["trending"])  * 1.0,
        min(fs["weak_down"],   fh["trending"])  * 0.7,
        min(fs["strong_up"],   fh["mean_rev"])  * 0.8,
        min(fm["biased_down"], fh["trending"])  * 0.5,
    ]
    raw_up   = max(up_act)
    raw_down = max(dn_act)
    total    = raw_up + raw_down
    NEUTRAL  = 0.15
    if total < 1e-6:
        P_up = P_down = 0.5
    else:
        P_up   = (raw_up   / total) * (1 - NEUTRAL) + 0.5 * NEUTRAL
        P_down = (raw_down / total) * (1 - NEUTRAL) + 0.5 * NEUTRAL
    conf = abs(P_up - 0.5) * 2.0 * (1 - fh["random"])
    return float(P_up), float(P_down), float(conf)


def _markov_p_up(outcomes):
    if not outcomes: return 0.5
    last = outcomes[-1]
    counts = {"UP->UP": 0, "UP->DOWN": 0, "DOWN->UP": 0, "DOWN->DOWN": 0}
    for i in range(len(outcomes) - 1):
        k = f"{outcomes[i]}->{outcomes[i+1]}"
        if k in counts: counts[k] += 1
    tot = counts[f"{last}->UP"] + counts[f"{last}->DOWN"]
    return counts[f"{last}->UP"] / tot if tot > 0 else 0.5


# ── Market state features ──────────────────────────────────────────────────────

def compute_market_state(prices, timestamps, round_data):
    """
    Compute X_t: the market state features for round t.
    These are the INPUTS to the LSTM (what the market looks like).
    """
    if len(prices) < 20:
        return None

    p = np.array(prices[-50:] if len(prices) >= 50 else prices)
    returns = np.diff(p)

    # Realized volatility (std of returns, last 20 ticks)
    vol_20  = float(np.std(returns[-20:]))  if len(returns) >= 20 else 0.0
    vol_50  = float(np.std(returns))

    # Trend: EMA slope
    ema_fast = _ema(p, 5)
    ema_slow = _ema(p, 20)
    trend    = float(ema_fast[-1] - ema_slow[-1]) / float(p[-1]) if p[-1] > 0 else 0.0

    # Price momentum (last 5 ticks)
    momentum = float(p[-1] - p[-5]) / float(p[-5]) if len(p) >= 5 and p[-5] > 0 else 0.0

    # Skewness and kurtosis of recent returns
    skew = float(np.mean(returns[-20:]**3) / (np.std(returns[-20:])**3 + 1e-8)) if len(returns) >= 20 else 0.0
    kurt = float(np.mean(returns[-20:]**4) / (np.std(returns[-20:])**4 + 1e-8)) if len(returns) >= 20 else 0.0

    # Pool features
    total = round_data.get("total_bnb", 0)
    bull  = round_data.get("bull_bnb", 0)
    pool_up_pct  = bull / total if total > 0 else 0.5
    pool_imbalance = abs(pool_up_pct - 0.5) * 2  # 0=balanced, 1=fully skewed
    mult_up   = round_data.get("mult_up", 2.0)
    mult_down = round_data.get("mult_down", 2.0)

    # Time features
    ts = timestamps[-1] if timestamps else 0
    dt = datetime.fromtimestamp(ts) if ts > 0 else datetime.now()
    hour_sin = float(np.sin(2 * np.pi * dt.hour / 24))
    hour_cos = float(np.cos(2 * np.pi * dt.hour / 24))
    dow_sin  = float(np.sin(2 * np.pi * dt.weekday() / 7))
    dow_cos  = float(np.cos(2 * np.pi * dt.weekday() / 7))

    return {
        "vol_20":        round(vol_20, 6),
        "vol_50":        round(vol_50, 6),
        "trend":         round(trend, 6),
        "momentum":      round(momentum, 6),
        "skewness":      round(skew, 4),
        "kurtosis":      round(kurt, 4),
        "pool_up_pct":   round(pool_up_pct, 4),
        "pool_imbalance":round(pool_imbalance, 4),
        "mult_up":       round(mult_up, 4),
        "mult_down":     round(mult_down, 4),
        "hour_sin":      round(hour_sin, 4),
        "hour_cos":      round(hour_cos, 4),
        "dow_sin":       round(dow_sin, 4),
        "dow_cos":       round(dow_cos, 4),
    }


def _ema(prices, span):
    alpha = 2 / (span + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema


# ── Main generation loop ───────────────────────────────────────────────────────

def generate_audit_log(
    data_file="dwt_denoising/pancake_1year.jsonl",
    out_file="dwt_denoising/audit_log.jsonl",
    price_window=400,
    markov_window=30,
):
    t_start = datetime.now()
    """
    Compact format: ONE record per round (not one per combo).
    Each record contains:
      - X: market state features (14 values)
      - best_theta: the combo with highest PnL → training target
      - best_pnl: PnL of the best combo
      - combo_pnls: list of PnL for all 216 combos (for analysis)
      - performance of best combo

    Result: ~105k records, ~150 MB, ~30 min on CPU.
    """
    print(f"Loading {data_file}...")
    rounds = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rounds.append(json.loads(line))
    rounds.sort(key=lambda r: r["epoch"])
    print(f"Loaded {len(rounds):,} rounds")
    print(f"Parameter combos: {len(COMBOS)}")
    print(f"Output format: 1 record per round = {len(rounds):,} records")
    print(f"Output: {out_file}")
    print()

    written = 0
    skipped = 0

    with open(out_file, "w") as out_f:
        for i, r in enumerate(rounds):
            if i < price_window:
                skipped += 1
                continue

            price_buf = [rounds[j]["lock_price"] for j in range(i - price_window, i)
                         if rounds[j].get("lock_price")]
            ts_buf    = [rounds[j]["start_ts"]   for j in range(i - price_window, i)
                         if rounds[j].get("lock_price")]

            if len(price_buf) < 50:
                skipped += 1
                continue

            markov_outcomes = [rounds[j]["outcome"] for j in range(max(0, i - markov_window), i)
                               if rounds[j].get("outcome")]

            X = compute_market_state(price_buf, ts_buf, r)
            if X is None:
                skipped += 1
                continue

            actual_outcome = r.get("outcome")
            lock_price     = r.get("lock_price")
            close_price    = r.get("close_price")
            mult_up        = r.get("mult_up", 2.0)
            mult_down      = r.get("mult_down", 2.0)

            best_pnl    = -999.0
            best_idx    = 0
            combo_pnls  = []

            # Cache DWT smoothed arrays and Hurst values per unique (wavelet_level, hurst_window)
            # to avoid recomputing them 216 times per round
            dwt_cache   = {}   # wavelet_level → (smoothed, slope)
            hurst_cache = {}   # (wavelet_level, hurst_window) → H

            for idx, (wl, hw, ft, st, lb) in enumerate(COMBOS):
                # DWT: cache per wavelet_level
                if wl not in dwt_cache:
                    sm = wavelet_denoise(price_buf, wavelet="db4", level=wl)
                    sl = np.diff(sm, prepend=sm[0])
                    dwt_cache[wl] = (sm, sl)
                smoothed, slope_arr = dwt_cache[wl]

                # Hurst: cache per (wavelet_level, hurst_window)
                hk = (wl, hw)
                if hk not in hurst_cache:
                    # Hurst on RAW prices — smoothed destroys mean-reversion signal
                    raw_for_hurst = np.array(price_buf)
                    series = raw_for_hurst[-hw:] if len(raw_for_hurst) >= hw else raw_for_hurst
                    hurst_cache[hk] = _hurst(series)
                H = hurst_cache[hk]

                # Slope value
                avg_slope = float(np.mean(slope_arr[-lb:]))

                # Signal
                if avg_slope > st:    signal = "UP"
                elif avg_slope < -st: signal = "DOWN"
                else:                 signal = "NEUTRAL"

                if signal == "UP":
                    pnl = (mult_up - 1) if actual_outcome == "UP" else -1.0
                elif signal == "DOWN":
                    pnl = (mult_down - 1) if actual_outcome == "DOWN" else -1.0
                else:
                    pnl = 0.0

                combo_pnls.append(round(pnl, 4))
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_idx = idx

            if not combo_pnls:
                skipped += 1
                continue

            # Best combo details
            wl, hw, ft, st, lb = COMBOS[best_idx]
            best_result = run_6step(price_buf, wl, hw, ft, st, lb, markov_outcomes)
            price_error = abs((best_result["smoothed_last"] if best_result else lock_price) - close_price) if close_price else 0.0

            record = {
                "epoch":          r["epoch"],
                "start_ts":       r["start_ts"],
                "actual_outcome": actual_outcome,
                "lock_price":     lock_price,
                "close_price":    close_price,
                # Market state (LSTM input features)
                "X": X,
                # Best parameter combo (LSTM training target)
                "best_theta": {
                    "wavelet_level":   wl,
                    "hurst_window":    hw,
                    "fuzzy_threshold": ft,
                    "slope_thresh":    st,
                    "lookback":        lb,
                },
                "best_pnl":    round(best_pnl, 4),
                "price_error": round(price_error, 4),
                # Compact PnL array for all combos (used for analysis)
                "combo_pnls":  combo_pnls,
            }
            out_f.write(json.dumps(record) + "\n")
            written += 1

            if (i % 2000) == 0:
                pct = i / len(rounds) * 100
                elapsed = (datetime.now() - t_start).total_seconds()
                eta = elapsed / max(pct, 0.1) * (100 - pct) / 60
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"{pct:5.1f}%  round {i:,}/{len(rounds):,}  "
                      f"written={written:,}  ETA={eta:.0f}min")

    print(f"\nDone. Written: {written:,} records  Skipped: {skipped:,}")
    size_mb = os.path.getsize(out_file) / 1024 / 1024
    print(f"File: {out_file}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="dwt_denoising/pancake_1year.jsonl")
    parser.add_argument("--out-file",  default="dwt_denoising/audit_log.jsonl")
    args = parser.parse_args()
    generate_audit_log(data_file=args.data_file, out_file=args.out_file)
