"""
PancakeSwap Prediction — LSTM Adaptive Control System
======================================================
Architecture:
  1. LSTM Controller dynamically tunes 6-step engine params every round.
  2. 6-Step Engine (DWT→Hurst→Fuzzy→Markov→EV→Kelly) runs with LSTM params.
  3. Pool-skew guard: only bet when pool is genuinely skewed (proven edge).
  4. Kelly sizing on the final bet.

Run:  python3 dwt_denoising/server.py
Open: http://localhost:8000
"""

import os, json, time, threading
import numpy as np
import requests as req
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from web3 import Web3

import sys
sys.path.insert(0, "dwt_denoising")
from wavelet_denoise import wavelet_denoise

# ── Try loading LSTM controller ───────────────────────────────────────────────
try:
    from lstm_controller import LSTMInference, X_FEATURES
    _lstm = LSTMInference("dwt_denoising/lstm_controller.pt")
    LSTM_AVAILABLE = True
except Exception as e:
    print(f"[LSTM] Not available: {e}")
    _lstm = None
    LSTM_AVAILABLE = False
    X_FEATURES = []

# ── Config ────────────────────────────────────────────────────────────────────
BSC_RPC            = "https://bsc-dataseed.binance.org/"
CHAINLINK_CONTRACT = "0x0567F2323251f0Aab15c8dFb1967E4e8A7D42aeE"
PANCAKE_CONTRACT   = "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA"
HISTORY            = 400
CHUNK              = 20
PANCAKE_SEC        = 300
MARKOV_HISTORY     = 30
KELLY_FRACTION     = 0.25
MIN_POOL_BNB       = 0.05   # lowered — pool fills gradually, 0.05 BNB is enough to read ratio
MIN_POOL_TIME      = 60    # wait 60s into round before evaluating (pool more stable)
REFRESH_SEC        = 10
# Pool-skew threshold proven profitable in 1-year backtest (105,071 rounds)
# pool_up > 65% → bet DOWN at ~3.1x → WR 32.3% → break-even 32% → profitable
POOL_SKEW_THRESH   = 65.0

DEFAULT_PARAMS = {
    "wavelet_level": 1, "hurst_window": 400,
    "fuzzy_threshold": 0.45, "slope_thresh": 0.07,
    "lookback": 3, "source": "default",
}

LOG_DIR  = "dwt_denoising/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"rounds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

w3           = Web3(Web3.HTTPProvider(BSC_RPC))
PANCAKE_ADDR = Web3.to_checksum_address(PANCAKE_CONTRACT)
app          = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

state = {
    "prices": [], "times": [], "smoothed": [], "slope": [],
    "live_price": 0.0,
    "H": 0.5, "hurst_label": "RANDOM", "hurst_color": "#aaaaaa",
    "rolling_hurst": [],
    "P_up": 0.5, "P_down": 0.5, "confidence": 0.0,
    "round": {},
    "decision": {"action":"SKIP","side":None,"ev_up":0,"ev_down":0,"kelly":0,"reason":"Initializing..."},
    "current_round_decision": {"action":"SKIP","side":None,"ev_up":0,"ev_down":0,"kelly":0,"reason":"Starting up...","epoch":None,"phase":"LIVE"},
    "next_round_decision":    {"action":"SKIP","side":None,"ev_up":0,"ev_down":0,"kelly":0,"reason":"Waiting...","epoch":None,"phase":"LIVE"},
    "markov_matrix": {}, "outcomes": [],
    "win_rate": 0.0, "wins": 0, "losses": 0,
    "trade_log": [], "live_log": [],
    # Virtual bet tracking — shows what would happen if you bet every SKIP round
    "virtual_wins":   0,
    "virtual_losses": 0,
    "virtual_profit": 0.0,   # flat-bet cumulative P&L
    "virtual_log":    [],     # last 20 virtual bet results
    "lstm_params": DEFAULT_PARAMS,
    "lstm_available": LSTM_AVAILABLE,
    "pool_skew_thresh": POOL_SKEW_THRESH,
    "last_update": "", "status": "Starting...", "error": "",
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHERS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_latest_chainlink():
    payload = {"jsonrpc":"2.0","method":"eth_call",
               "params":[{"to":CHAINLINK_CONTRACT,"data":"0xfeaf968c"},"latest"],"id":0}
    r   = req.post(BSC_RPC, json=payload, timeout=5)
    raw = r.json()["result"][2:]
    w   = [raw[i:i+64] for i in range(0, len(raw), 64)]
    return int(w[0],16), int(w[1],16)/1e8, int(w[3],16)


def fetch_chainlink_history(latest_id, n=HISTORY):
    ids = [latest_id - i for i in range(n)]
    prices, timestamps = [], []
    for start in range(0, len(ids), CHUNK):
        chunk = ids[start:start+CHUNK]
        batch = [{"jsonrpc":"2.0","method":"eth_call",
                  "params":[{"to":CHAINLINK_CONTRACT,
                             "data":"0x9a6fc8f5"+format(rid,"064x")},"latest"],
                  "id": start+i}
                 for i, rid in enumerate(chunk)]
        try:
            resp  = req.post(BSC_RPC, json=batch, timeout=15)
            items = resp.json()
            if not isinstance(items, list): continue
            for item in items:
                if item.get("id") is None: continue
                if "result" not in item: continue
                raw = item["result"][2:]
                ww  = [raw[j:j+64] for j in range(0, len(raw), 64)]
                if len(ww) < 4: continue
                p = int(ww[1],16)/1e8
                t = int(ww[3],16)
                if p > 10 and t > 0:
                    prices.append(p)
                    timestamps.append(t)
        except Exception:
            pass
        time.sleep(0.15)
    prices     = list(reversed(prices))
    timestamps = list(reversed(timestamps))
    return np.array(prices), [datetime.fromtimestamp(t).strftime("%H:%M:%S") for t in timestamps], np.array(timestamps, dtype=float)


def _mapping_slot(key, slot):
    encoded = key.to_bytes(32,"big") + slot.to_bytes(32,"big")
    return int(w3.keccak(encoded).hex(), 16)


def fetch_pancake_epoch():
    return int(w3.eth.get_storage_at(PANCAKE_ADDR, 10).hex(), 16)


def fetch_round_data(epoch):
    base  = _mapping_slot(epoch, 14)
    words = [int(w3.eth.get_storage_at(PANCAKE_ADDR, base+i).hex(), 16) for i in range(14)]
    total = words[8]/1e18; bull = words[9]/1e18; bear = words[10]/1e18
    mult_up   = (total/bull*0.97)  if bull  > 0 else 0.0
    mult_down = (total/bear*0.97)  if bear  > 0 else 0.0
    return {
        "epoch":         words[0],
        "start_ts":      words[1],
        "lock_ts":       words[2],
        "close_ts":      words[3],
        "lock_price":    words[4]/1e8 if words[4] else None,
        "close_price":   words[5]/1e8 if words[5] else None,
        "total_bnb":     round(total, 4),
        "bull_bnb":      round(bull,  4),
        "bear_bnb":      round(bear,  4),
        "mult_up":       round(mult_up,   3),
        "mult_down":     round(mult_down, 3),
        "oracle_called": bool(words[13]),
    }


def fetch_markov_history(current_epoch, n=MARKOV_HISTORY):
    outcomes = []
    for ep in range(current_epoch - n, current_epoch):
        try:
            rd = fetch_round_data(ep)
            if rd["lock_price"] and rd["close_price"] and rd["oracle_called"]:
                outcomes.append("UP" if rd["close_price"] > rd["lock_price"] else "DOWN")
        except Exception:
            pass
    return outcomes


# ═══════════════════════════════════════════════════════════════════════════════
# 6-STEP MATH ENGINE  (params tuned by LSTM each round)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dwt(prices_arr, wavelet_level=1):
    smoothed = wavelet_denoise(prices_arr, wavelet="db4", level=wavelet_level)
    slope    = np.diff(smoothed, prepend=smoothed[0])
    return smoothed, slope


def hurst_exponent(series, hurst_window=400):
    n = len(series)
    if n < 30: return 0.5
    series = series[-hurst_window:] if len(series) >= hurst_window else series
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


def hurst_regime(H):
    if H > 0.6:   return "TRENDING",       "#00cc44"
    elif H < 0.4: return "MEAN-REVERTING", "#ff4444"
    else:         return "RANDOM",         "#aaaaaa"


def build_markov_matrix(outcomes):
    counts = {("UP","UP"):0,("UP","DOWN"):0,("DOWN","UP"):0,("DOWN","DOWN"):0}
    for i in range(len(outcomes)-1):
        k = (outcomes[i], outcomes[i+1])
        if k in counts: counts[k] += 1
    matrix = {}
    for fs in ["UP","DOWN"]:
        tot = counts[(fs,"UP")] + counts[(fs,"DOWN")]
        matrix[(fs,"UP")]   = round(counts[(fs,"UP")]  /tot, 3) if tot else 0.5
        matrix[(fs,"DOWN")] = round(counts[(fs,"DOWN")]/tot, 3) if tot else 0.5
    return matrix


def _trap(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    elif b <= x <= c:    return 1.0
    elif a < x < b:      return (x-a)/(b-a)
    else:                return (d-x)/(d-c)


def fuzzy_inference(slope_val, H, markov_p_up, fuzzy_threshold=0.45):
    """Fuzzy inference with LSTM-tuned fuzzy_threshold."""
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
        min(fs["strong_up"],   fh["trending"]) * 1.0,
        min(fs["weak_up"],     fh["trending"]) * 0.7,
        min(fs["strong_down"], fh["mean_rev"]) * 0.8,
        min(fm["biased_up"],   fh["trending"]) * 0.5,
    ]
    dn_act = [
        min(fs["strong_down"], fh["trending"]) * 1.0,
        min(fs["weak_down"],   fh["trending"]) * 0.7,
        min(fs["strong_up"],   fh["mean_rev"]) * 0.8,
        min(fm["biased_down"], fh["trending"]) * 0.5,
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
    return round(float(P_up),4), round(float(P_down),4), round(float(conf),4)


def run_6step_engine(prices_arr, outcomes, params):
    """
    Run the full 6-step engine with LSTM-provided params.
    Returns all intermediate values for logging and UI.
    """
    wl  = params.get("wavelet_level",   1)
    hw  = params.get("hurst_window",    400)
    ft  = params.get("fuzzy_threshold", 0.45)
    st  = params.get("slope_thresh",    0.07)
    lb  = params.get("lookback",        3)

    # Step 1: DWT
    smoothed, slope = compute_dwt(prices_arr, wavelet_level=wl)
    slope_val = float(np.mean(slope[-lb:]))

    # Step 2: Hurst — compute on RAW prices, not smoothed
    # DWT smoothing destroys mean-reversion signal (H always > 0.5 on smoothed)
    # Raw prices correctly show H < 0.5 during mean-reverting regimes
    H = hurst_exponent(np.array(prices_arr), hurst_window=hw)
    h_label, h_color = hurst_regime(H)

    # Step 3 & 4: Markov + Fuzzy
    matrix      = build_markov_matrix(outcomes)
    last_out    = outcomes[-1] if outcomes else "UP"
    markov_p_up = matrix.get((last_out, "UP"), 0.5)
    P_up, P_down, conf = fuzzy_inference(slope_val, H, markov_p_up, fuzzy_threshold=ft)

    return {
        "smoothed":    smoothed,
        "slope":       slope,
        "slope_val":   slope_val,
        "H":           round(H, 4),
        "h_label":     h_label,
        "h_color":     h_color,
        "matrix":      matrix,
        "markov_p_up": markov_p_up,
        "P_up":        P_up,
        "P_down":      P_down,
        "conf":        conf,
    }


def compute_market_state(prices_arr, timestamps_arr, round_data):
    """Compute X_t features for LSTM input."""
    if len(prices_arr) < 20:
        return {}
    p = prices_arr[-50:] if len(prices_arr) >= 50 else prices_arr
    returns = np.diff(p)
    vol_20  = float(np.std(returns[-20:])) if len(returns) >= 20 else 0.0
    vol_50  = float(np.std(returns))
    # EMA trend
    def ema(arr, span):
        a = 2/(span+1); e = np.zeros(len(arr)); e[0] = arr[0]
        for i in range(1, len(arr)): e[i] = a*arr[i] + (1-a)*e[i-1]
        return e
    ema5  = ema(p, 5)
    ema20 = ema(p, 20)
    trend = float(ema5[-1] - ema20[-1]) / float(p[-1]) if p[-1] > 0 else 0.0
    momentum = float(p[-1] - p[-5]) / float(p[-5]) if len(p) >= 5 and p[-5] > 0 else 0.0
    skew = float(np.mean(returns[-20:]**3) / (np.std(returns[-20:])**3 + 1e-8)) if len(returns) >= 20 else 0.0
    kurt = float(np.mean(returns[-20:]**4) / (np.std(returns[-20:])**4 + 1e-8)) if len(returns) >= 20 else 0.0
    total = round_data.get("total_bnb", 0)
    bull  = round_data.get("bull_bnb", 0)
    pool_up_pct   = bull / total if total > 0 else 0.5
    pool_imbalance = abs(pool_up_pct - 0.5) * 2
    ts = timestamps_arr[-1] if len(timestamps_arr) > 0 else 0
    dt = datetime.fromtimestamp(float(ts)) if ts > 0 else datetime.now()
    return {
        "vol_20": round(vol_20, 6), "vol_50": round(vol_50, 6),
        "trend": round(trend, 6), "momentum": round(momentum, 6),
        "skewness": round(skew, 4), "kurtosis": round(kurt, 4),
        "pool_up_pct": round(pool_up_pct, 4),
        "pool_imbalance": round(pool_imbalance, 4),
        "mult_up": round(round_data.get("mult_up", 2.0), 4),
        "mult_down": round(round_data.get("mult_down", 2.0), 4),
        "hour_sin": round(float(np.sin(2*np.pi*dt.hour/24)), 4),
        "hour_cos": round(float(np.cos(2*np.pi*dt.hour/24)), 4),
        "dow_sin":  round(float(np.sin(2*np.pi*dt.weekday()/7)), 4),
        "dow_cos":  round(float(np.cos(2*np.pi*dt.weekday()/7)), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DECISION ENGINE  (pool-skew guard + EV + Kelly)
# ═══════════════════════════════════════════════════════════════════════════════

def make_decision(P_up, P_down, mult_up, mult_down, confidence,
                  total_bnb=0.0, round_start_ts=0, pool_up_pct=50.0):
    """
    Final decision combining 6-step engine output with pool-skew guard.

    virtual_bet: what the FULL signal (6-step engine + LSTM + pool) recommends,
    ignoring only the skew threshold gate. This shows what would happen if you
    bet on every round using the complete strategy signal.

    Three signal sources combined for virtual_bet:
      1. Pool side:   minority side (DOWN if pool_up>50%, UP if pool_up<50%)
      2. EV side:     whichever side has higher expected value
      3. 6-step side: P_up > 0.5 → UP signal, P_up < 0.5 → DOWN signal

    Virtual bet uses majority vote of these three signals.
    """
    ev_up   = round(P_up   * mult_up   - (1-P_up),   4) if mult_up   > 0 else -1.0
    ev_down = round(P_down * mult_down - (1-P_down),  4) if mult_down > 0 else -1.0
    k_up    = round(max(0, (P_up*mult_up-(1-P_up))/mult_up*KELLY_FRACTION),        4) if mult_up   > 1 else 0.0
    k_down  = round(max(0, (P_down*mult_down-(1-P_down))/mult_down*KELLY_FRACTION), 4) if mult_down > 1 else 0.0

    # Virtual bet: ALWAYS use the LSTM-tuned 6-step engine signal.
    # No pool fallback, no neutral — every round gets a bet.
    #
    # Priority order:
    #   1. Engine has real signal (P_up ≠ 0.5, conf > 0) → use P_up
    #   2. Engine neutral but LSTM active → use raw slope direction
    #   3. LSTM warming up (P_up=0.5, conf=0, slope=0) → use EV side
    pool_side   = "DOWN" if pool_up_pct >= 50 else "UP"
    ev_side     = "UP"   if ev_up >= ev_down   else "DOWN"
    engine_side = "UP"   if P_up > 0.5 else ("DOWN" if P_up < 0.5 else None)

    if engine_side is not None and confidence > 0:
        # Priority 1: engine has a real fuzzy signal
        virt_side = engine_side
        virt_src  = "engine"
    elif engine_side is not None:
        # Priority 2: P_up is non-neutral but conf=0 (H near random)
        # Use P_up direction directly — it still reflects slope
        virt_side = engine_side
        virt_src  = "engine_low_conf"
    else:
        # Priority 3: P_up=0.5 exactly (LSTM warming up or truly flat slope)
        # Use EV side — whichever multiplier is higher.
        # NOTE: this has no real edge — it's just the higher-mult side.
        # Label clearly so the dashboard shows "no signal" state.
        virt_side = ev_side
        virt_src  = "no_signal"

    virt_mult = mult_up   if virt_side == "UP"   else mult_down
    virt_ev   = ev_up     if virt_side == "UP"   else ev_down

    virtual_bet = {
        "side":        virt_side,
        "mult":        round(virt_mult, 3),
        "ev":          round(virt_ev, 4),
        "pool_up":     round(pool_up_pct, 1),
        "break_even":  round(1/virt_mult*100, 1) if virt_mult > 0 else 50.0,
        "pool_side":   pool_side,
        "ev_side":     ev_side,
        "engine_side": engine_side or "NEUTRAL",
        "virt_src":    virt_src,
        "signals":     f"pool={pool_side} ev={ev_side} engine={engine_side or 'NEUTRAL'} src={virt_src}",
        "engine_used": True,  # always uses engine pipeline
        "P_up":        P_up,
        "confidence":  confidence,
    }

    result = {"ev_up":ev_up, "ev_down":ev_down, "kelly_up":k_up, "kelly_down":k_down,
              "action":"SKIP", "side":None, "kelly":0.0, "reason":"",
              "virtual_bet": virtual_bet}

    # Guard 1 — pool too small
    if total_bnb < MIN_POOL_BNB:
        result["reason"] = f"Pool too small ({total_bnb:.4f} BNB < {MIN_POOL_BNB})"
        return result

    # Guard 2 — round too new
    elapsed = time.time() - round_start_ts if round_start_ts > 0 else MIN_POOL_TIME
    if elapsed < MIN_POOL_TIME:
        result["reason"] = f"Round too new ({int(elapsed)}s < {MIN_POOL_TIME}s)"
        return result

    # Guard 3 — pool-skew gate (primary signal)
    low_thresh = 100.0 - POOL_SKEW_THRESH
    if pool_up_pct >= POOL_SKEW_THRESH:
        skew_side = "DOWN"
        skew_kelly = k_down
        skew_ev    = ev_down
        skew_mult  = mult_down
    elif pool_up_pct <= low_thresh:
        skew_side = "UP"
        skew_kelly = k_up
        skew_ev    = ev_up
        skew_mult  = mult_up
    else:
        result["reason"] = f"No skew ({pool_up_pct:.1f}% within {low_thresh:.0f}–{POOL_SKEW_THRESH:.0f}%)"
        return result

    result.update({
        "action": "BET",
        "side":   skew_side,
        "kelly":  skew_kelly,
        "reason": (f"Pool skew {pool_up_pct:.1f}% → BET {skew_side} "
                   f"(mult={skew_mult:.2f}x EV={skew_ev:+.3f} "
                   f"P_up={P_up:.2f} conf={confidence:.2f})"),
    })
    return result


def backtest_5min(prices_arr, timestamps_arr, slope, slope_thresh=0.07, lookback=3):
    wins = losses = skips = 0
    trade_log = []
    if len(prices_arr) < 2:
        return 0.0, 0, 0, 0, []
    t_start = timestamps_arr[0]
    t_end   = timestamps_arr[-1]
    bucket_start = (t_start // 300) * 300
    buckets = []
    t = bucket_start
    while t + 300 <= t_end:
        buckets.append((t, t + 300))
        t += 300
    for (b_open, b_close) in buckets:
        idxs = [i for i in range(len(timestamps_arr)) if b_open <= timestamps_arr[i] < b_close]
        if len(idxs) < 2: skips += 1; continue
        signal_idx = idxs[-1]
        if signal_idx < lookback: skips += 1; continue
        avg_slope = np.mean(slope[signal_idx - lookback:signal_idx])
        if avg_slope > slope_thresh:    pred = "UP"
        elif avg_slope < -slope_thresh: pred = "DOWN"
        else: skips += 1; continue
        locked_price = prices_arr[signal_idx]
        lock_ts      = timestamps_arr[signal_idx]
        next_idxs = [i for i in range(len(timestamps_arr)) if timestamps_arr[i] >= b_close]
        if not next_idxs: skips += 1; continue
        close_idx   = next_idxs[0]
        close_price = prices_arr[close_idx]
        close_ts    = timestamps_arr[close_idx]
        if abs(close_ts - lock_ts - 300) > 90 + 300: skips += 1; continue
        actual  = "UP" if close_price > locked_price else "DOWN"
        correct = pred == actual
        if correct: wins += 1
        else:       losses += 1
        trade_log.append({
            "time":    datetime.fromtimestamp(lock_ts).strftime("%H:%M:%S"),
            "pred":    pred, "actual": actual,
            "locked":  round(float(locked_price), 4),
            "close":   round(float(close_price),  4),
            "change":  round(float(close_price - locked_price), 4),
            "correct": correct,
            "bucket":  datetime.fromtimestamp(b_open).strftime("%H:%M"),
        })
    total    = wins + losses
    win_rate = round((wins / total * 100) if total > 0 else 0.0, 1)
    return win_rate, wins, losses, skips, trade_log[-20:]


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

_live_log = []

def log_round(epoch, round_data, decision, engine_out, params,
              X_state, actual_outcome, correct, win_rate, wins, losses, live_price):
    entry = {
        "ts":              datetime.now().isoformat(),
        "epoch":           epoch,
        "round_bucket":    datetime.fromtimestamp(round_data.get("start_ts",0)).strftime("%Y-%m-%d %H:%M") if round_data.get("start_ts") else None,
        "live_price":      live_price,
        "lock_price":      round_data.get("lock_price"),
        "close_price":     round_data.get("close_price"),
        "price_change":    round(round_data["close_price"] - round_data["lock_price"], 4)
                           if round_data.get("lock_price") and round_data.get("close_price") else None,
        "total_bnb":       round_data.get("total_bnb"),
        "bull_bnb":        round_data.get("bull_bnb"),
        "bear_bnb":        round_data.get("bear_bnb"),
        "mult_up":         round_data.get("mult_up"),
        "mult_down":       round_data.get("mult_down"),
        "pool_up_pct":     round(round_data["bull_bnb"] / round_data["total_bnb"] * 100, 1)
                           if round_data.get("total_bnb", 0) > 0 else None,
        # LSTM params
        "lstm_wavelet_level":   params.get("wavelet_level"),
        "lstm_hurst_window":    params.get("hurst_window"),
        "lstm_fuzzy_threshold": params.get("fuzzy_threshold"),
        "lstm_slope_thresh":    params.get("slope_thresh"),
        "lstm_lookback":        params.get("lookback"),
        "lstm_source":          params.get("source"),
        # 6-step engine output
        "dwt_slope":       round(engine_out.get("slope_val", 0), 6),
        "hurst_H":         engine_out.get("H", 0.5),
        "hurst_regime":    engine_out.get("h_label", "RANDOM"),
        "P_up":            engine_out.get("P_up", 0.5),
        "P_down":          engine_out.get("P_down", 0.5),
        "confidence":      engine_out.get("conf", 0.0),
        "markov_p_up":     round(engine_out.get("markov_p_up", 0.5), 4),
        # Decision
        "decision_action": decision.get("action"),
        "decision_side":   decision.get("side"),
        "ev_up":           decision.get("ev_up"),
        "ev_down":         decision.get("ev_down"),
        "kelly_pct":       round(decision.get("kelly", 0) * 100, 2),
        "decision_reason": decision.get("reason"),
        # Outcome
        "actual_outcome":  actual_outcome,
        "correct":         correct,
        # Stats
        "session_win_rate": win_rate,
        "session_wins":     wins,
        "session_losses":   losses,
        # Virtual bet result (what would have happened if skew gate ignored)
        "virtual_side":   decision.get("virtual_bet", {}).get("side"),
        "virtual_mult":   decision.get("virtual_bet", {}).get("mult"),
        "virtual_ev":     decision.get("virtual_bet", {}).get("ev"),
        "virtual_pool_up":decision.get("virtual_bet", {}).get("pool_up"),
        "virtual_break_even": decision.get("virtual_bet", {}).get("break_even"),
        "virtual_signals":decision.get("virtual_bet", {}).get("signals"),
        # virtual_outcome and virtual_won are filled when round closes (actual_outcome known)
        "virtual_outcome": ("WIN" if decision.get("virtual_bet",{}).get("side") == actual_outcome
                            else "LOSS") if actual_outcome and decision.get("virtual_bet",{}).get("side") else None,
        "virtual_won":    (decision.get("virtual_bet",{}).get("side") == actual_outcome)
                          if actual_outcome and decision.get("virtual_bet",{}).get("side") else None,
        "virtual_pnl":    round((decision.get("virtual_bet",{}).get("mult",1)-1)
                          if decision.get("virtual_bet",{}).get("side") == actual_outcome
                          else (-1.0 if actual_outcome and decision.get("virtual_bet",{}).get("side") else 0), 4),
        # Market state
        "market_state":    X_state,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    _live_log.append(entry)
    if len(_live_log) > 200:
        _live_log.pop(0)


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND REFRESH LOOP
# ═══════════════════════════════════════════════════════════════════════════════

_prices_arr     = np.array([])
_timestamps_arr = np.array([])
_times_list     = []
_last_cl_id     = 0
_epoch          = 0
_outcomes       = []
_rolling_hurst  = []
_evaluated_buckets = set()
_persistent_trades = []
_current_round_epoch = None
_epoch_snapshots     = {}
_outcome_logged_epochs = set()
_last_logged_epoch   = None
_current_params      = dict(DEFAULT_PARAMS)
# Virtual bet tracking
_virtual_wins   = 0
_virtual_losses = 0
_virtual_profit = 0.0
_virtual_log    = []   # list of virtual bet result dicts


def init_data():
    global _prices_arr, _timestamps_arr, _times_list, _last_cl_id, _epoch, _outcomes
    state["status"] = "Fetching Chainlink history..."
    _last_cl_id, live_price, _ = fetch_latest_chainlink()
    _prices_arr, _times_list, _timestamps_arr = fetch_chainlink_history(_last_cl_id, n=HISTORY)
    state["status"] = "Fetching PancakeSwap data..."
    _epoch    = fetch_pancake_epoch()
    _outcomes = fetch_markov_history(_epoch, n=MARKOV_HISTORY)
    state["status"] = "Running"
    print(f"Init done: {len(_prices_arr)} rounds, epoch {_epoch}, {len(_outcomes)} outcomes")


def refresh_loop():
    global _prices_arr, _timestamps_arr, _times_list, _last_cl_id
    global _epoch, _outcomes, _rolling_hurst, _evaluated_buckets
    global _persistent_trades, _current_round_epoch, _epoch_snapshots
    global _outcome_logged_epochs, _last_logged_epoch, _current_params
    global _virtual_wins, _virtual_losses, _virtual_profit, _virtual_log

    init_data()

    # Pre-populate backtest
    engine_init = run_6step_engine(_prices_arr, _outcomes, _current_params)
    _, _, _, _, initial_trades = backtest_5min(
        _prices_arr, _timestamps_arr, engine_init["slope"],
        slope_thresh=_current_params["slope_thresh"],
        lookback=_current_params["lookback"],
    )
    for t in initial_trades:
        bk = t["bucket"]
        if bk not in _evaluated_buckets:
            _evaluated_buckets.add(bk)
            _persistent_trades.append(t)
    print(f"Pre-loaded {len(_persistent_trades)} historical trades")
    print(f"Round log: {LOG_FILE}")

    while True:
        try:
            # ── 1. New Chainlink rounds ───────────────────────────────────
            new_id, live_price, _ = fetch_latest_chainlink()
            if new_id > _last_cl_id:
                new_ids = [_last_cl_id + i + 1 for i in range(new_id - _last_cl_id)]
                batch = [{"jsonrpc":"2.0","method":"eth_call",
                          "params":[{"to":CHAINLINK_CONTRACT,
                                     "data":"0x9a6fc8f5"+format(rid,"064x")},"latest"],
                          "id":i}
                         for i, rid in enumerate(new_ids)]
                try:
                    resp  = req.post(BSC_RPC, json=batch, timeout=10)
                    items = resp.json()
                    if isinstance(items, list):
                        for item in items:
                            if item.get("id") is None: continue
                            if "result" not in item: continue
                            raw = item["result"][2:]
                            ww  = [raw[j:j+64] for j in range(0,len(raw),64)]
                            if len(ww) < 4: continue
                            p = int(ww[1],16)/1e8; t = int(ww[3],16)
                            if p > 10 and t > 0:
                                _prices_arr     = np.append(_prices_arr, p)
                                _timestamps_arr = np.append(_timestamps_arr, float(t))
                                _times_list.append(datetime.fromtimestamp(t).strftime("%H:%M:%S"))
                except Exception:
                    pass
                _last_cl_id = new_id

            if len(_prices_arr) > HISTORY:
                _prices_arr     = _prices_arr[-HISTORY:]
                _timestamps_arr = _timestamps_arr[-HISTORY:]
                _times_list     = _times_list[-HISTORY:]

            # ── 2. PancakeSwap round ──────────────────────────────────────
            new_epoch = fetch_pancake_epoch()
            if new_epoch != _epoch:
                _epoch = new_epoch
            round_data = fetch_round_data(_epoch)

            # ── Read previous epoch pool for decision signal ──────────────
            # Current epoch pool is empty at round open (no bets placed yet).
            # Previous epoch pool is fully filled — use it as the pool signal
            # for deciding whether to bet on the current epoch.
            try:
                prev_round_data = fetch_round_data(_epoch - 1)
            except Exception:
                prev_round_data = round_data
            # Always use previous epoch's pool for the decision signal.
            # Current epoch pool fills gradually during the round — at round open
            # it may have only a few early bets (0.05-0.5 BNB), giving an unstable
            # ratio. Previous epoch pool was filled over 5 minutes and is stable.
            # Exception: if prev_round_data has no pool data, fall back to current.
            pool_source = prev_round_data if prev_round_data.get("total_bnb", 0) >= MIN_POOL_BNB \
                          else round_data

            # ── 3. LSTM: get optimized params for this round ──────────────
            X_state = compute_market_state(_prices_arr, _timestamps_arr, pool_source)
            if _lstm is not None and LSTM_AVAILABLE:
                _current_params = _lstm.predict()
            else:
                _current_params = dict(DEFAULT_PARAMS)
            state["lstm_params"] = _current_params

            # ── 4. Run 6-step engine with LSTM params ─────────────────────
            engine_out = run_6step_engine(_prices_arr, _outcomes, _current_params)
            smoothed   = engine_out["smoothed"]
            slope      = engine_out["slope"]
            H          = engine_out["H"]
            h_label    = engine_out["h_label"]
            h_color    = engine_out["h_color"]
            P_up       = engine_out["P_up"]
            P_down     = engine_out["P_down"]
            conf       = engine_out["conf"]
            matrix     = engine_out["matrix"]
            markov_p_up = engine_out["markov_p_up"]
            slope_val  = engine_out["slope_val"]

            _rolling_hurst.append(round(H, 4))
            if len(_rolling_hurst) > HISTORY:
                _rolling_hurst = _rolling_hurst[-HISTORY:]

            # ── 5. Decision (pool-skew gate + EV + Kelly) ─────────────────
            # pool_up_pct for DECISION uses pool_source (prev epoch, fully filled)
            pool_up_pct = pool_source["bull_bnb"] / pool_source["total_bnb"] * 100                           if pool_source.get("total_bnb", 0) > 0 else 50.0
            # pool_up_pct for VIRTUAL BET uses current round (what we're actually betting on)
            current_pool_up_pct = round_data["bull_bnb"] / round_data["total_bnb"] * 100                                    if round_data.get("total_bnb", 0) > 0 else pool_up_pct
            decision = make_decision(
                P_up, P_down,
                pool_source["mult_up"], pool_source["mult_down"],
                conf,
                total_bnb=pool_source.get("total_bnb", 0.0),
                round_start_ts=pool_source.get("start_ts", 0),
                pool_up_pct=current_pool_up_pct,
            )

            # ── 6. Poll outcomes for pending epochs ───────────────────────
            for snap_epoch in list(_epoch_snapshots.keys()):
                if snap_epoch in _outcome_logged_epochs: continue
                if snap_epoch >= _epoch: continue
                try:
                    snap_rd = fetch_round_data(snap_epoch)
                    lp = snap_rd.get("lock_price")
                    cp = snap_rd.get("close_price")
                    if not (lp and cp and snap_rd.get("oracle_called")): continue
                    actual   = "UP" if cp > lp else "DOWN"
                    snap     = _epoch_snapshots[snap_epoch]
                    bet_side = snap["decision"].get("side")
                    result   = ("WIN" if bet_side == actual else "LOSS") if bet_side else "SKIP"
                    correct  = (bet_side == actual) if bet_side else None

                    # ── Virtual bet result ────────────────────────────────
                    # Use the virtual_bet from the snapshot's decision.
                    # The snapshot is updated every refresh, so we need to
                    # read from the log entry's virtual fields directly
                    # (already saved in log_round via decision.virtual_bet).
                    vb        = snap["decision"].get("virtual_bet", {})
                    virt_side = vb.get("side")
                    virt_mult = vb.get("mult", 2.0)
                    # P_up/conf stored in frozen virtual_bet (not engine_out which updates)
                    snap_P_up  = vb.get("P_up", 0.5)
                    snap_conf  = vb.get("confidence", 0.0)
                    if virt_side and actual:
                        virt_won = (virt_side == actual)
                        virt_pnl = round((virt_mult - 1) if virt_won else -1.0, 4)
                        _virtual_wins   += (1 if virt_won else 0)
                        _virtual_losses += (0 if virt_won else 1)
                        _virtual_profit  = round(_virtual_profit + virt_pnl, 4)
                        virt_total = _virtual_wins + _virtual_losses
                        virt_wr    = round(_virtual_wins / virt_total * 100, 1) if virt_total else 0
                        virt_entry = {
                            "epoch":       snap_epoch,
                            "ts":          datetime.now().isoformat(),
                            "real_action": snap["decision"].get("action"),
                            "real_side":   bet_side,
                            "real_result": result,
                            "virt_side":   virt_side,
                            "virt_mult":   virt_mult,
                            "virt_won":    virt_won,
                            "virt_pnl":    virt_pnl,
                            "virt_cumulative_pnl": _virtual_profit,
                            "virt_wr":     virt_wr,
                            "actual":      actual,
                            "pool_up_pct": vb.get("pool_up", 0),
                            "break_even":  vb.get("break_even", 50),
                            "ev":          vb.get("ev", 0),
                            "P_up":        snap_P_up,
                            "confidence":  snap_conf,
                            "signals":     vb.get("signals", ""),
                            "pool_side":   vb.get("pool_side", ""),
                            "ev_side":     vb.get("ev_side", ""),
                            "engine_side": vb.get("engine_side", ""),
                            "virt_src":    vb.get("virt_src", ""),
                        }
                        _virtual_log.append(virt_entry)
                        if len(_virtual_log) > 100:
                            _virtual_log.pop(0)
                        state["virtual_wins"]   = _virtual_wins
                        state["virtual_losses"] = _virtual_losses
                        state["virtual_profit"] = _virtual_profit
                        state["virtual_log"]    = list(_virtual_log[-20:])
                        virt_mark = "✓" if virt_won else "✗"
                        print(f"  🎭 Virtual bet epoch {snap_epoch}: {virt_mark} {virt_side} "
                              f"actual={actual}  pnl={virt_pnl:+.3f}  "
                              f"cumulative={_virtual_profit:+.3f}  WR={virt_wr:.1f}%")

                    # Update Markov
                    _outcomes.append(actual)
                    if len(_outcomes) > MARKOV_HISTORY:
                        _outcomes = _outcomes[-MARKOV_HISTORY:]

                    # Push to LSTM history buffer
                    if _lstm is not None and LSTM_AVAILABLE:
                        price_error = abs(snap.get("smoothed_last", lp) - cp)
                        _lstm.push(
                            X_dict=snap.get("X_state", {}),
                            theta_dict=snap.get("params", DEFAULT_PARAMS),
                            C_dict={
                                "pnl":        (snap_rd["mult_down"]-1 if bet_side=="DOWN" else snap_rd["mult_up"]-1) if correct else -1.0,
                                "price_error": price_error,
                                "dir_correct": correct,
                                "won":         correct,
                            },
                        )

                    # Update current_round_decision result
                    crd = state.get("current_round_decision", {})
                    if crd.get("epoch") == snap_epoch:
                        updated = dict(crd)
                        updated.update({"result": result, "actual": actual,
                                        "lock_price": lp, "close_price": cp})
                        state["current_round_decision"] = updated

                    # Log outcome
                    wins_now   = sum(1 for t in _persistent_trades if t["correct"])
                    losses_now = sum(1 for t in _persistent_trades if not t["correct"])
                    total_now  = wins_now + losses_now
                    wr_now     = round((wins_now/total_now*100) if total_now > 0 else 0.0, 1)
                    log_round(
                        epoch=snap_epoch, round_data=snap_rd,
                        decision=snap["decision"], engine_out=snap.get("engine_out", {}),
                        params=snap.get("params", DEFAULT_PARAMS),
                        X_state=snap.get("X_state", {}),
                        actual_outcome=actual, correct=correct,
                        win_rate=wr_now, wins=wins_now, losses=losses_now,
                        live_price=snap.get("live_price", 0),
                    )
                    _outcome_logged_epochs.add(snap_epoch)
                    mark = "✓ WIN" if result=="WIN" else ("✗ LOSS" if result=="LOSS" else "— SKIP")
                    print(f"  Outcome epoch {snap_epoch}: {mark}  locked=${lp:.4f}  close=${cp:.4f}  bet={bet_side or 'SKIP'}  actual={actual}")
                except Exception:
                    pass

            # ── 7. Incremental backtest ───────────────────────────────────
            _, _, _, _, new_trades = backtest_5min(
                _prices_arr, _timestamps_arr, slope,
                slope_thresh=_current_params["slope_thresh"],
                lookback=_current_params["lookback"],
            )
            for t in new_trades:
                bk = t["bucket"]
                if bk not in _evaluated_buckets:
                    _evaluated_buckets.add(bk)
                    _persistent_trades.append(t)

            wins     = sum(1 for t in _persistent_trades if t["correct"])
            losses   = sum(1 for t in _persistent_trades if not t["correct"])
            total    = wins + losses
            win_rate = round((wins / total * 100) if total > 0 else 0.0, 1)

            # ── 8. Round decision tracking ────────────────────────────────
            now_ts       = time.time()
            lock_ts_val  = round_data.get("lock_ts",  now_ts + 300)
            close_ts_val = round_data.get("close_ts", now_ts + 600)
            until_lock   = lock_ts_val  - now_ts
            until_close  = close_ts_val - now_ts

            # Decision window: 5s before close = start of next round.
            # Pool is fully filled by t=270s (30s before lock).
            # We read the CURRENT round's pool at that point and use it
            # to decide the BET for the NEXT round.
            DECISION_FREEZE = 2    # freeze at 2s before lock — captures whale last-minute bets
            SIGNAL_WINDOW   = 12   # show signal 12s before lock

            if _epoch != _current_round_epoch:
                _current_round_epoch = _epoch
                nrd = state.get("next_round_decision", {})
                if nrd and nrd.get("epoch") == _epoch - 1:
                    promoted = dict(nrd)
                    promoted["result"] = None; promoted["actual"] = None
                    state["current_round_decision"] = promoted
                else:
                    state["current_round_decision"] = {
                        "action": decision["action"], "side": decision["side"],
                        "ev_up": decision["ev_up"], "ev_down": decision["ev_down"],
                        "kelly": decision["kelly"],
                        "reason": decision["reason"] + " (first run)",
                        "P_up": P_up, "P_down": P_down, "confidence": conf,
                        "H": round(H,4), "hurst_label": h_label,
                        "pool_up": round(round_data.get("bull_bnb",0),4),
                        "pool_down": round(round_data.get("bear_bnb",0),4),
                        "mult_up": round(round_data.get("mult_up",0),3),
                        "mult_down": round(round_data.get("mult_down",0),3),
                        "epoch": _epoch, "locked_at": None, "phase": "LIVE",
                        "result": None, "actual": None,
                        "lstm_params": _current_params,
                    }

            # Snapshot at epoch open — saved with current engine output
            if _epoch not in _epoch_snapshots:
                _epoch_snapshots[_epoch] = {
                    "decision":      decision,
                    "engine_out":    {k: (round(float(v),4) if isinstance(v,float) else v)
                                      for k,v in engine_out.items()
                                      if not isinstance(v, np.ndarray)},
                    "params":        dict(_current_params),
                    "X_state":       X_state,
                    "live_price":    live_price,
                    "smoothed_last": round(float(smoothed[-1]),4),
                    "logged_signal": False,
                    "virtual_bet_frozen": False,  # becomes True at lock time
                }
                if len(_epoch_snapshots) > 20:
                    oldest = min(_epoch_snapshots.keys())
                    del _epoch_snapshots[oldest]

            # Update snapshot every refresh EXCEPT the virtual_bet once frozen.
            # virtual_bet is frozen at lock time (pool fully filled) so the
            # outcome polling always uses the correct lock-time pool ratio.
            if _epoch in _epoch_snapshots:
                snap = _epoch_snapshots[_epoch]
                snap["engine_out"]   = {k: (round(float(v),4) if isinstance(v,float) else v)
                                        for k,v in engine_out.items()
                                        if not isinstance(v, np.ndarray)}
                snap["params"]       = dict(_current_params)
                snap["live_price"]   = live_price
                snap["smoothed_last"]= round(float(smoothed[-1]),4)
                # Only update decision (and virtual_bet) if not yet frozen
                if not snap.get("virtual_bet_frozen"):
                    snap["decision"] = decision

            # Log signal at lock window (30s before lock = pool is fully filled)
            nrd_phase = "LIVE"
            nrd_label = f"Live — closes at {datetime.fromtimestamp(close_ts_val).strftime('%H:%M:%S')}"
            if 0 < until_lock <= SIGNAL_WINDOW:
                nrd_phase = "DECIDING"
                nrd_label = f"⚡ Decide now! {int(until_lock)}s to lock"
            if 0 < until_lock <= DECISION_FREEZE:
                nrd_phase = "FINAL"
                nrd_label = f"🔒 FINAL — {int(until_lock)}s to lock"
            elif until_lock <= 0:
                nrd_phase = "LOCKED"
                nrd_label = "Locked — round closing"

            if nrd_phase in ("DECIDING","FINAL","LOCKED") and _epoch in _epoch_snapshots:
                snap = _epoch_snapshots[_epoch]
                # Freeze virtual_bet at lock time — current round pool is now filled
                if not snap.get("virtual_bet_frozen"):
                    snap["virtual_bet_frozen"] = True
                    # Recompute decision with CURRENT round's full pool for virtual bet
                    # At lock time, round_data has the real pool ratio
                    lock_pool_up = round_data["bull_bnb"] / round_data["total_bnb"] * 100 \
                                   if round_data.get("total_bnb", 0) > 0 else 50.0
                    lock_decision = make_decision(
                        P_up, P_down,
                        round_data["mult_up"], round_data["mult_down"],
                        conf,
                        total_bnb=round_data.get("total_bnb", 0.0),
                        round_start_ts=round_data.get("start_ts", 0),
                        pool_up_pct=lock_pool_up,
                    )
                    snap["decision"] = lock_decision  # frozen with lock-time pool
                if not snap.get("logged_signal"):
                    snap["logged_signal"] = True
                    wins_now   = sum(1 for t in _persistent_trades if t["correct"])
                    losses_now = sum(1 for t in _persistent_trades if not t["correct"])
                    total_now  = wins_now + losses_now
                    wr_now     = round((wins_now/total_now*100) if total_now > 0 else 0.0, 1)
                    log_round(
                        epoch=_epoch, round_data=round_data,
                        decision=snap["decision"], engine_out=snap["engine_out"],
                        params=snap["params"], X_state=snap["X_state"],
                        actual_outcome=None, correct=None,
                        win_rate=wr_now, wins=wins_now, losses=losses_now,
                        live_price=live_price,
                    )
                    print(f"  Signal logged epoch {_epoch}: {snap['decision']['action']} {snap['decision']['side'] or ''}  "
                          f"virt={snap['decision'].get('virtual_bet',{}).get('side','?')}  "
                          f"pool={snap['decision'].get('virtual_bet',{}).get('pool_up','?')}%  "
                          f"src={snap['decision'].get('virtual_bet',{}).get('virt_src','?')}")

            state["next_round_decision"] = {
                "action": decision["action"], "side": decision["side"],
                "ev_up": decision["ev_up"], "ev_down": decision["ev_down"],
                "kelly": decision["kelly"], "kelly_up": decision.get("kelly_up",0),
                "kelly_down": decision.get("kelly_down",0),
                "reason": decision["reason"],
                "P_up": P_up, "P_down": P_down, "confidence": conf,
                "H": round(H,4), "hurst_label": h_label,
                "pool_up": round(round_data.get("bull_bnb",0),4),
                "pool_down": round(round_data.get("bear_bnb",0),4),
                "mult_up": round(round_data.get("mult_up",0),3),
                "mult_down": round(round_data.get("mult_down",0),3),
                "epoch": _epoch, "locked_at": datetime.now().strftime("%H:%M:%S") if until_lock <= DECISION_FREEZE else None,
                "phase": nrd_phase, "phase_label": nrd_label,
                "until_lock": int(until_lock),
                "until_close": int(until_close),
                "result": None, "actual": None,
                "lstm_params": _current_params,
            }

            matrix_json = {f"{k[0]}->{k[1]}": v for k, v in matrix.items()}

            state.update({
                "prices":        [round(float(p),4) for p in _prices_arr],
                "times":         list(_times_list),
                "smoothed":      [round(float(s),4) for s in smoothed],
                "slope":         [round(float(s),4) for s in slope],
                "live_price":    round(live_price, 4),
                "H":             round(H, 4),
                "hurst_label":   h_label,
                "hurst_color":   h_color,
                "rolling_hurst": list(_rolling_hurst),
                "P_up":          P_up,
                "P_down":        P_down,
                "confidence":    conf,
                "round":         round_data,
                "pool_source":   pool_source,
                "decision":      decision,
                "markov_matrix": matrix_json,
                "outcomes":      list(_outcomes[-10:]),
                "win_rate":      win_rate,
                "wins":          wins,
                "losses":        losses,
                "trade_log":     _persistent_trades[-20:],
                "live_log":      list(_live_log[-50:]),
                "log_file":      LOG_FILE,
                "lstm_params":   _current_params,
                "lstm_available": LSTM_AVAILABLE,
                "pool_skew_thresh": POOL_SKEW_THRESH,
                # pool_up_pct for display = current round (what Pool Sizes card shows)
                "pool_up_pct":   round(round_data["bull_bnb"] / round_data["total_bnb"] * 100, 1)
                                 if round_data.get("total_bnb", 0) > 0 else 0.0,
                # pool_up_pct used for decision = pool_source (may be prev epoch)
                "decision_pool_up_pct": round(pool_up_pct, 1),
                # Virtual bet stats (what would happen if you bet every round)
                "virtual_wins":   _virtual_wins,
                "virtual_losses": _virtual_losses,
                "virtual_profit": round(_virtual_profit, 4),
                "virtual_log":    list(_virtual_log[-20:]),
                "virtual_wr":     round(_virtual_wins / (_virtual_wins + _virtual_losses) * 100, 1)
                                  if (_virtual_wins + _virtual_losses) > 0 else 0.0,
                "last_update":   datetime.now().strftime("%H:%M:%S"),
                "status":        "Running",
                "error":         "",
            })

            lock_in = int(round_data["lock_ts"] - time.time())
            print(f"[{state['last_update']}] ${live_price:.4f}  H={H:.2f}[{h_label[:4]}]  "
                  f"pool={pool_up_pct:.1f}%  P(UP)={P_up:.2f}  "
                  f"→ {decision['action']} {decision['side'] or ''}  "
                  f"WR={win_rate:.1f}% ({wins}W/{losses}L)  "
                  f"Close:{int(until_close)}s [{nrd_phase}]  "
                  f"LSTM:{_current_params.get('source','?')}")

        except Exception as e:
            state["error"]  = str(e)
            state["status"] = f"Error: {e}"
            print(f"Refresh error: {e}")

        # Adaptive sleep: run every 2s when near decision window,
        # every 10s otherwise. This ensures we catch the 5s window.
        try:
            _until = state.get("next_round_decision", {}).get("until_lock", 999)
        except Exception:
            _until = 999
        if _until is not None and abs(_until) <= 5:
            time.sleep(1)   # 1s cycle in final 5s window
        elif _until is not None and abs(_until) <= 40:
            time.sleep(2)   # 2s cycle near lock
        else:
            time.sleep(REFRESH_SEC)


# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/state")
def get_state():
    return JSONResponse(content=state)


@app.get("/api/logs")
def get_logs(limit: int = 100):
    try:
        with open(LOG_FILE) as f:
            lines = f.readlines()
        entries = [json.loads(l) for l in lines[-limit:] if l.strip()]
        return JSONResponse(content={"file": LOG_FILE, "count": len(entries), "entries": entries})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/logs/files")
def list_log_files():
    files = sorted(os.listdir(LOG_DIR))
    result = []
    for f in files:
        path = os.path.join(LOG_DIR, f)
        size = os.path.getsize(path)
        with open(path) as fh:
            count = sum(1 for _ in fh)
        result.append({"file": f, "path": path, "entries": count, "size_kb": round(size/1024, 1)})
    return JSONResponse(content=result)


@app.get("/api/lstm/status")
def lstm_status():
    return JSONResponse(content={
        "available":   LSTM_AVAILABLE,
        "model_path":  "dwt_denoising/lstm_controller.pt",
        "model_exists": os.path.exists("dwt_denoising/lstm_controller.pt"),
        "current_params": state.get("lstm_params", DEFAULT_PARAMS),
        "history_len": len(_lstm.history) if _lstm else 0,
        "seq_len_needed": 12,
    })


@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open("dwt_denoising/index.html") as f:
        return f.read()


@app.on_event("startup")
def startup_event():
    t = threading.Thread(target=refresh_loop, daemon=True)
    t.start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False,
                app_dir="dwt_denoising")
