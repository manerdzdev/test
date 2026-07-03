"""
PancakeSwap Prediction - Full Strategy Study Tool
==================================================
Implements all 4 layers:
  1. DWT De-noising          - clean Chainlink price signal
  2. Hurst Exponent          - detect trending vs mean-reverting regime
  3. Fuzzy Inference System  - combine signals into win probability P
  4. Markov Chain            - sequence bias from recent round history
  5. EV Calculation          - Expected Value using live pool sizes
  6. Kelly Criterion         - optimal bet sizing (Quarter-Kelly)

Data sources:
  - Chainlink BNB/USD oracle  (price signal)
  - PancakeSwap Prediction contract (pool sizes, epoch, timing)

Press Ctrl+C to stop.
"""

import time
import numpy as np
import requests
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from datetime import datetime
from web3 import Web3
from wavelet_denoise import wavelet_denoise

# ── RPC / Contracts ───────────────────────────────────────────────────────────
BSC_RPC            = "https://bsc-dataseed.binance.org/"
CHAINLINK_CONTRACT = "0x0567F2323251f0Aab15c8dFb1967E4e8A7D42aeE"
PANCAKE_CONTRACT   = "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA"

w3 = Web3(Web3.HTTPProvider(BSC_RPC))
PANCAKE_ADDR = Web3.to_checksum_address(PANCAKE_CONTRACT)

# ── Configuration ─────────────────────────────────────────────────────────────
WAVELET        = "db4"
LEVEL          = 2
REFRESH_SEC    = 10
HISTORY        = 200        # Chainlink rounds to keep
CHUNK          = 20
SLOPE_THRESH   = 0.05
LOOKBACK       = 3
PANCAKE_SEC    = 300        # 5-minute rounds
HURST_WINDOW   = 50         # rounds for Hurst calculation
MARKOV_HISTORY = 30         # past PancakeSwap rounds for Markov matrix
KELLY_FRACTION = 0.25       # Quarter-Kelly safety factor
MIN_EV         = 0.0        # minimum EV to place a bet
MIN_CONFIDENCE = 0.55       # minimum fuzzy P to consider betting
# ──────────────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — CHAINLINK DATA
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_latest_chainlink():
    """Return (round_id, price, timestamp) from Chainlink BNB/USD oracle."""
    payload = {"jsonrpc":"2.0","method":"eth_call",
               "params":[{"to":CHAINLINK_CONTRACT,"data":"0xfeaf968c"},"latest"],"id":0}
    r   = requests.post(BSC_RPC, json=payload, timeout=5)
    raw = r.json()["result"][2:]
    w   = [raw[i:i+64] for i in range(0, len(raw), 64)]
    return int(w[0],16), int(w[1],16)/1e8, int(w[3],16)


def fetch_chainlink_history(latest_id, n=HISTORY):
    """Fetch last n Chainlink rounds. Returns (prices, times_dt, timestamps)."""
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
            resp  = requests.post(BSC_RPC, json=batch, timeout=15)
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
    return (np.array(prices),
            [datetime.fromtimestamp(t) for t in timestamps],
            np.array(timestamps, dtype=float))


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1b — PANCAKESWAP CONTRACT DATA
# ═══════════════════════════════════════════════════════════════════════════════

def _mapping_slot(key: int, slot: int) -> int:
    """Compute storage slot for mapping(uint256 => ...) at base slot."""
    encoded = key.to_bytes(32,"big") + slot.to_bytes(32,"big")
    return int(w3.keccak(encoded).hex(), 16)


def fetch_pancake_epoch() -> int:
    """Read currentEpoch from contract storage slot 10."""
    return int(w3.eth.get_storage_at(PANCAKE_ADDR, 10).hex(), 16)


def fetch_round_data(epoch: int) -> dict:
    """
    Read a PancakeSwap round from contract storage.
    Round struct layout (each word = 32 bytes):
      [0]  epoch
      [1]  startTimestamp
      [2]  lockTimestamp
      [3]  closeTimestamp
      [4]  lockPrice   (int256, /1e8)
      [5]  closePrice  (int256, /1e8)
      [6]  lockOracleId
      [7]  closeOracleId
      [8]  totalAmount (wei)
      [9]  bullAmount  (wei)  ← UP pool
      [10] bearAmount  (wei)  ← DOWN pool
      [11] rewardBaseCalAmount
      [12] rewardAmount
      [13] oracleCalled (bool)
    """
    base = _mapping_slot(epoch, 14)
    words = [int(w3.eth.get_storage_at(PANCAKE_ADDR, base+i).hex(), 16)
             for i in range(14)]

    total = words[8]  / 1e18
    bull  = words[9]  / 1e18
    bear  = words[10] / 1e18

    mult_up   = (total / bull  * 0.97) if bull  > 0 else 0.0
    mult_down = (total / bear  * 0.97) if bear  > 0 else 0.0

    return {
        "epoch":          words[0],
        "start_ts":       words[1],
        "lock_ts":        words[2],
        "close_ts":       words[3],
        "lock_price":     words[4]/1e8 if words[4] else None,
        "close_price":    words[5]/1e8 if words[5] else None,
        "total_bnb":      total,
        "bull_bnb":       bull,
        "bear_bnb":       bear,
        "mult_up":        mult_up,
        "mult_down":      mult_down,
        "oracle_called":  bool(words[13]),
    }


def fetch_markov_history(current_epoch: int, n=MARKOV_HISTORY) -> list:
    """
    Fetch last n completed rounds and return list of outcomes: 'UP' or 'DOWN'.
    A round is UP if closePrice > lockPrice.
    """
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
# LAYER 2 — DWT DE-NOISING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dwt(prices_arr):
    smoothed = wavelet_denoise(prices_arr, wavelet=WAVELET, level=LEVEL)
    slope    = np.diff(smoothed, prepend=smoothed[0])
    return smoothed, slope


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — HURST EXPONENT  (Rescaled Range / R/S Analysis)
# ═══════════════════════════════════════════════════════════════════════════════

def hurst_exponent(series: np.ndarray) -> float:
    """
    Compute Hurst Exponent via R/S analysis on the DWT-smoothed price series.

    H > 0.5  →  Trending (persistent)      → trust DWT slope direction
    H < 0.5  →  Mean-reverting             → consider fading DWT slope
    H ≈ 0.5  →  Random walk               → skip bet

    Uses log-log regression of R/S statistic across multiple sub-window sizes.
    """
    n = len(series)
    if n < 20:
        return 0.5  # not enough data

    lags  = []
    rs_vals = []

    for lag in range(10, n // 2, max(1, n // 20)):
        sub_rs = []
        for start in range(0, n - lag, lag):
            chunk = series[start:start + lag]
            mean  = np.mean(chunk)
            devs  = np.cumsum(chunk - mean)
            R     = np.max(devs) - np.min(devs)
            S     = np.std(chunk, ddof=1)
            if S > 0:
                sub_rs.append(R / S)
        if sub_rs:
            lags.append(np.log(lag))
            rs_vals.append(np.log(np.mean(sub_rs)))

    if len(lags) < 2:
        return 0.5

    H, _ = np.polyfit(lags, rs_vals, 1)
    return float(np.clip(H, 0.0, 1.0))


def hurst_regime(H: float) -> tuple:
    """Return (label, color, description) for a given Hurst value."""
    if H > 0.6:
        return "TRENDING",       "#00cc44", f"H={H:.2f} — trust DWT slope"
    elif H < 0.4:
        return "MEAN-REVERTING", "#ff3333", f"H={H:.2f} — fade DWT slope"
    else:
        return "RANDOM",         "#aaaaaa", f"H={H:.2f} — skip bet"


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — MARKOV CHAIN  (sequence bias in round outcomes)
# ═══════════════════════════════════════════════════════════════════════════════

def build_markov_matrix(outcomes: list) -> dict:
    """
    Build a 2-state Markov transition matrix from round outcome history.
    States: 'UP', 'DOWN'
    Returns dict: {('UP','UP'): p, ('UP','DOWN'): p, ('DOWN','UP'): p, ('DOWN','DOWN'): p}
    """
    counts = {("UP","UP"):0, ("UP","DOWN"):0, ("DOWN","UP"):0, ("DOWN","DOWN"):0}
    for i in range(len(outcomes)-1):
        key = (outcomes[i], outcomes[i+1])
        if key in counts:
            counts[key] += 1

    matrix = {}
    for from_state in ["UP","DOWN"]:
        total = counts[(from_state,"UP")] + counts[(from_state,"DOWN")]
        if total > 0:
            matrix[(from_state,"UP")]   = counts[(from_state,"UP")]   / total
            matrix[(from_state,"DOWN")] = counts[(from_state,"DOWN")] / total
        else:
            matrix[(from_state,"UP")]   = 0.5
            matrix[(from_state,"DOWN")] = 0.5
    return matrix


def markov_bias(outcomes: list, matrix: dict) -> tuple:
    """
    Given the last outcome and transition matrix, return (P_up, P_down) bias.
    If no history, returns (0.5, 0.5).
    """
    if not outcomes or not matrix:
        return 0.5, 0.5
    last = outcomes[-1]
    p_up   = matrix.get((last,"UP"),   0.5)
    p_down = matrix.get((last,"DOWN"), 0.5)
    return p_up, p_down

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — FUZZY INFERENCE SYSTEM  (combine all signals → P)
# ═══════════════════════════════════════════════════════════════════════════════

def _trapezoid(x, a, b, c, d) -> float:
    """Trapezoidal membership function. Returns degree in [0,1]."""
    if x <= a or x >= d:
        return 0.0
    elif b <= x <= c:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:
        return (d - x) / (d - c)


def fuzzify_slope(slope_val: float) -> dict:
    """Map DWT slope to fuzzy membership degrees."""
    return {
        "strong_down": _trapezoid(slope_val, -99, -99, -0.5, -0.1),
        "weak_down":   _trapezoid(slope_val, -0.5, -0.2, -0.05, 0.0),
        "neutral":     _trapezoid(slope_val, -0.1, -0.02, 0.02, 0.1),
        "weak_up":     _trapezoid(slope_val, 0.0, 0.05, 0.2, 0.5),
        "strong_up":   _trapezoid(slope_val, 0.1, 0.5, 99, 99),
    }


def fuzzify_hurst(H: float) -> dict:
    """Map Hurst exponent to fuzzy membership degrees."""
    return {
        "mean_reverting": _trapezoid(H, 0.0, 0.0, 0.35, 0.5),
        "random":         _trapezoid(H, 0.35, 0.45, 0.55, 0.65),
        "trending":       _trapezoid(H, 0.5, 0.65, 1.0, 1.0),
    }


def fuzzify_markov(p_up: float) -> dict:
    """Map Markov P(UP) to fuzzy membership degrees."""
    return {
        "biased_down": _trapezoid(p_up, 0.0, 0.0, 0.35, 0.5),
        "neutral":     _trapezoid(p_up, 0.35, 0.45, 0.55, 0.65),
        "biased_up":   _trapezoid(p_up, 0.5, 0.65, 1.0, 1.0),
    }


def fuzzy_inference(slope_val: float, H: float,
                    markov_p_up: float, markov_p_down: float) -> tuple:
    """
    Fuzzy rule engine. Returns (P_up, P_down, confidence).

    Rules (Mamdani-style, min-AND aggregation):
      R1: IF slope=strong_up   AND hurst=trending      → UP   weight 1.0
      R2: IF slope=weak_up     AND hurst=trending      → UP   weight 0.7
      R3: IF slope=strong_down AND hurst=trending      → DOWN weight 1.0
      R4: IF slope=weak_down   AND hurst=trending      → DOWN weight 0.7
      R5: IF slope=strong_up   AND hurst=mean_reverting → DOWN weight 0.8
      R6: IF slope=strong_down AND hurst=mean_reverting → UP   weight 0.8
      R7: IF hurst=random                              → NEUTRAL (skip)
      R8: IF markov=biased_up  AND hurst=trending      → UP   weight 0.5
      R9: IF markov=biased_down AND hurst=trending     → DOWN weight 0.5
    """
    fs = fuzzify_slope(slope_val)
    fh = fuzzify_hurst(H)
    fm = fuzzify_markov(markov_p_up)

    up_activations   = []
    down_activations = []

    # R1
    up_activations.append(min(fs["strong_up"],   fh["trending"]) * 1.0)
    # R2
    up_activations.append(min(fs["weak_up"],     fh["trending"]) * 0.7)
    # R3
    down_activations.append(min(fs["strong_down"], fh["trending"]) * 1.0)
    # R4
    down_activations.append(min(fs["weak_down"],   fh["trending"]) * 0.7)
    # R5 — fade strong up in mean-reverting regime
    down_activations.append(min(fs["strong_up"],   fh["mean_reverting"]) * 0.8)
    # R6 — fade strong down in mean-reverting regime
    up_activations.append(min(fs["strong_down"],   fh["mean_reverting"]) * 0.8)
    # R8 Markov UP bias
    up_activations.append(min(fm["biased_up"],     fh["trending"]) * 0.5)
    # R9 Markov DOWN bias
    down_activations.append(min(fm["biased_down"], fh["trending"]) * 0.5)

    raw_up   = max(up_activations)   if up_activations   else 0.0
    raw_down = max(down_activations) if down_activations else 0.0
    total    = raw_up + raw_down

    if total < 1e-6:
        return 0.5, 0.5, 0.0

    # Defuzzification: weighted average → probability
    P_up   = raw_up   / total
    P_down = raw_down / total

    # Confidence = how far from 50/50
    confidence = abs(P_up - 0.5) * 2.0   # 0 = random, 1 = certain

    # Blend with random-walk penalty
    random_weight = fh["random"]
    P_up   = P_up   * (1 - random_weight) + 0.5 * random_weight
    P_down = P_down * (1 - random_weight) + 0.5 * random_weight
    confidence *= (1 - random_weight)

    return float(P_up), float(P_down), float(confidence)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 6 — EV CALCULATION & KELLY CRITERION
# ═══════════════════════════════════════════════════════════════════════════════

def expected_value(P: float, multiplier: float) -> float:
    """EV = P × Multiplier − (1 − P)"""
    if multiplier <= 0:
        return -1.0
    return P * multiplier - (1.0 - P)


def kelly_fraction(P: float, multiplier: float) -> float:
    """
    Kelly f* = (P × Multiplier − (1−P)) / Multiplier
    Returns Quarter-Kelly fraction of bankroll to bet.
    Clamped to [0, 0.25] for safety.
    """
    if multiplier <= 1.0:
        return 0.0
    f_star = (P * multiplier - (1.0 - P)) / multiplier
    return float(np.clip(f_star * KELLY_FRACTION, 0.0, 0.25))


def make_decision(P_up: float, P_down: float,
                  mult_up: float, mult_down: float,
                  confidence: float) -> dict:
    """
    Apply EV rules and return final decision dict.
    Rule 1: Only bet if EV > MIN_EV
    Rule 2: Only bet if confidence > MIN_CONFIDENCE
    Rule 3: Bet the side with higher positive EV
    """
    ev_up   = expected_value(P_up,   mult_up)
    ev_down = expected_value(P_down, mult_down)
    k_up    = kelly_fraction(P_up,   mult_up)
    k_down  = kelly_fraction(P_down, mult_down)

    result = {
        "ev_up":    ev_up,
        "ev_down":  ev_down,
        "kelly_up": k_up,
        "kelly_down": k_down,
        "action":   "SKIP",
        "side":     None,
        "kelly":    0.0,
        "reason":   "",
    }

    if confidence < MIN_CONFIDENCE:
        result["reason"] = f"Low confidence ({confidence:.2f} < {MIN_CONFIDENCE})"
        return result

    if ev_up <= MIN_EV and ev_down <= MIN_EV:
        result["reason"] = "Both EV ≤ 0 — no edge"
        return result

    if ev_up > ev_down and ev_up > MIN_EV:
        result["action"] = "BET"
        result["side"]   = "UP"
        result["kelly"]  = k_up
        result["reason"] = f"EV_UP={ev_up:+.3f} > EV_DOWN={ev_down:+.3f}"
    elif ev_down > ev_up and ev_down > MIN_EV:
        result["action"] = "BET"
        result["side"]   = "DOWN"
        result["kelly"]  = k_down
        result["reason"] = f"EV_DOWN={ev_down:+.3f} > EV_UP={ev_up:+.3f}"
    else:
        result["reason"] = "EV tied or insufficient"

    return result

# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST  (5-minute PancakeSwap direction)
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_5min(prices_arr, timestamps_arr, slope):
    """Backtest: predict 5-min direction using DWT slope, check actual outcome."""
    wins = losses = skips = 0
    for i in range(LOOKBACK, len(prices_arr)):
        avg = np.mean(slope[i-LOOKBACK:i])
        if avg > SLOPE_THRESH:    pred = "UP"
        elif avg < -SLOPE_THRESH: pred = "DOWN"
        else: skips += 1; continue

        entry_ts  = timestamps_arr[i]
        target_ts = entry_ts + PANCAKE_SEC
        future_idx, min_diff = None, float("inf")
        for j in range(i+1, len(prices_arr)):
            d = abs(timestamps_arr[j] - target_ts)
            if d < min_diff: min_diff = d; future_idx = j
            if timestamps_arr[j] > target_ts + 60: break

        if future_idx is None or min_diff > 90: skips += 1; continue

        actual = "UP" if prices_arr[future_idx] > prices_arr[i] else "DOWN"
        if pred == actual: wins += 1
        else:              losses += 1

    total    = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0.0
    return win_rate, wins, losses, skips


# ═══════════════════════════════════════════════════════════════════════════════
# CHART
# ═══════════════════════════════════════════════════════════════════════════════

def setup_figure():
    plt.ion()
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0f0f0f")
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    axes = {
        "price":   fig.add_subplot(gs[0, :]),   # full width — price + DWT
        "slope":   fig.add_subplot(gs[1, 0]),   # DWT slope
        "hurst":   fig.add_subplot(gs[1, 1]),   # Hurst rolling
        "pool":    fig.add_subplot(gs[2, 0]),   # pool sizes bar
        "ev":      fig.add_subplot(gs[2, 1]),   # EV bars
        "markov":  fig.add_subplot(gs[3, 0]),   # Markov matrix heatmap
        "summary": fig.add_subplot(gs[3, 1]),   # text summary
    }
    for ax in axes.values():
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
    return fig, axes


def update_chart(fig, axes, times, prices, smoothed, slope,
                 H, hurst_label, hurst_color,
                 round_data, decision,
                 P_up, P_down, confidence,
                 markov_matrix, outcomes,
                 win_rate, wins, losses,
                 live_price, rolling_hurst):

    for ax in axes.values():
        ax.cla()
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    now_str  = datetime.now().strftime("%H:%M:%S")
    action   = decision["action"]
    side     = decision["side"] or ""
    act_color = "#00cc44" if side=="UP" else ("#ff3333" if side=="DOWN" else "#aaaaaa")

    fig.suptitle(
        f"PancakeSwap Prediction Study  |  BNB/USD: ${live_price:.4f}  |  "
        f"Epoch: {round_data['epoch']}  |  "
        f"5-min WinRate: {win_rate:.1f}% ({wins}W/{losses}L)  |  {now_str}",
        fontsize=11, fontweight="bold", color="white"
    )

    # ── Price + DWT ───────────────────────────────────────────────────────────
    ax = axes["price"]
    ax.plot(times, prices,   color="#ffffff", lw=1,   alpha=0.6, label="Raw Chainlink")
    ax.plot(times, smoothed, color="#00aaff", lw=2.5, label=f"DWT ({WAVELET} lvl {LEVEL})")
    ax.fill_between(times, prices, smoothed,
                    where=(prices>smoothed), alpha=0.12, color="#00cc44")
    ax.fill_between(times, prices, smoothed,
                    where=(prices<smoothed), alpha=0.12, color="#ff3333")
    ax.scatter([times[-1]], [live_price], color=act_color, s=80, zorder=5,
               label=f"Live ${live_price:.4f}")
    ax.set_ylabel("Price (USD)", color="white")
    ax.legend(loc="upper left", fontsize=8, facecolor="#222", labelcolor="white", ncol=3)
    ax.grid(True, alpha=0.12, color="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right", fontsize=7, color="white")
    ax.set_title(f"Price vs DWT  |  Hurst: {hurst_label} ({hurst_color})  |  "
                 f"P(UP)={P_up:.2f}  P(DOWN)={P_down:.2f}  Conf={confidence:.2f}",
                 color=hurst_color, fontsize=9)

    # ── DWT Slope ─────────────────────────────────────────────────────────────
    ax = axes["slope"]
    bar_c = ["#00cc44" if s > 0 else "#ff3333" for s in slope]
    ax.bar(times, slope, color=bar_c, alpha=0.8, width=0.015)
    ax.axhline(0,             color="white",   lw=0.8, ls="--", alpha=0.4)
    ax.axhline( SLOPE_THRESH, color="#00cc44", lw=1,   ls=":",  alpha=0.7)
    ax.axhline(-SLOPE_THRESH, color="#ff3333", lw=1,   ls=":",  alpha=0.7)
    ax.set_title("DWT Slope", color="white", fontsize=9)
    ax.set_ylabel("Slope", color="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right", fontsize=7, color="white")
    ax.grid(True, alpha=0.12, color="white")

    # ── Rolling Hurst ─────────────────────────────────────────────────────────
    ax = axes["hurst"]
    if len(rolling_hurst) > 1:
        h_times = times[-len(rolling_hurst):]
        h_arr   = np.array(rolling_hurst)
        ax.plot(h_times, h_arr, color="#ffaa00", lw=2)
        ax.fill_between(h_times, h_arr, 0.5,
                        where=(h_arr > 0.5), alpha=0.2, color="#00cc44")
        ax.fill_between(h_times, h_arr, 0.5,
                        where=(h_arr < 0.5), alpha=0.2, color="#ff3333")
    ax.axhline(0.5, color="white",   lw=1, ls="--", alpha=0.6, label="Random (0.5)")
    ax.axhline(0.6, color="#00cc44", lw=1, ls=":",  alpha=0.5, label="Trending (0.6)")
    ax.axhline(0.4, color="#ff3333", lw=1, ls=":",  alpha=0.5, label="MeanRev (0.4)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Hurst Exponent  (current: {H:.3f})", color="white", fontsize=9)
    ax.set_ylabel("H", color="white")
    ax.legend(fontsize=7, facecolor="#222", labelcolor="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right", fontsize=7, color="white")
    ax.grid(True, alpha=0.12, color="white")

    # ── Pool Sizes ────────────────────────────────────────────────────────────
    ax = axes["pool"]
    bull = round_data["bull_bnb"]
    bear = round_data["bear_bnb"]
    bars = ax.bar(["UP Pool", "DOWN Pool"], [bull, bear],
                  color=["#00cc44", "#ff3333"], alpha=0.85)
    for bar, val in zip(bars, [bull, bear]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f} BNB", ha="center", va="bottom",
                color="white", fontsize=8)
    ax.set_title(f"Pool Sizes  |  Total: {round_data['total_bnb']:.3f} BNB  |  "
                 f"Epoch {round_data['epoch']}", color="white", fontsize=9)
    ax.set_ylabel("BNB", color="white")
    ax.grid(True, alpha=0.12, color="white", axis="y")

    # ── EV Bars ───────────────────────────────────────────────────────────────
    ax = axes["ev"]
    ev_up   = decision["ev_up"]
    ev_down = decision["ev_down"]
    ev_colors = ["#00cc44" if ev_up > 0 else "#ff3333",
                 "#00cc44" if ev_down > 0 else "#ff3333"]
    bars2 = ax.bar([f"EV UP\n(x{round_data['mult_up']:.2f})",
                    f"EV DOWN\n(x{round_data['mult_down']:.2f})"],
                   [ev_up, ev_down], color=ev_colors, alpha=0.85)
    for bar, val in zip(bars2, [ev_up, ev_down]):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + (0.01 if val >= 0 else -0.03),
                f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top",
                color="white", fontsize=9, fontweight="bold")
    ax.axhline(0, color="white", lw=1, ls="--", alpha=0.5)
    ax.set_title(f"Expected Value  |  Decision: {action} {side}",
                 color=act_color, fontsize=9)
    ax.set_ylabel("EV", color="white")
    ax.grid(True, alpha=0.12, color="white", axis="y")

    # ── Markov Heatmap ────────────────────────────────────────────────────────
    ax = axes["markov"]
    if markov_matrix:
        mat = np.array([
            [markov_matrix.get(("UP","UP"),0.5),   markov_matrix.get(("UP","DOWN"),0.5)],
            [markov_matrix.get(("DOWN","UP"),0.5), markov_matrix.get(("DOWN","DOWN"),0.5)],
        ])
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks([0,1]); ax.set_xticklabels(["→ UP","→ DOWN"], color="white")
        ax.set_yticks([0,1]); ax.set_yticklabels(["From UP","From DOWN"], color="white")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        color="black", fontsize=11, fontweight="bold")
        last_out = outcomes[-1] if outcomes else "?"
        ax.set_title(f"Markov Transition  |  Last: {last_out}  |  "
                     f"P(next UP)={markov_matrix.get((last_out,'UP'),0.5):.2f}",
                     color="white", fontsize=9)
    else:
        ax.text(0.5, 0.5, "Collecting round history...",
                ha="center", va="center", color="white", transform=ax.transAxes)
        ax.set_title("Markov Transition", color="white", fontsize=9)

    # ── Text Summary ──────────────────────────────────────────────────────────
    ax = axes["summary"]
    ax.axis("off")
    lock_in = round_data["lock_ts"] - time.time()
    lock_str = f"{int(lock_in)}s" if lock_in > 0 else "LOCKED"

    lines = [
        ("DECISION SUMMARY", "#ffffff", 13, True),
        ("", "#ffffff", 9, False),
        (f"Epoch:        {round_data['epoch']}", "#aaaaaa", 9, False),
        (f"Lock in:      {lock_str}", "#ffaa00", 9, False),
        ("", "#ffffff", 9, False),
        (f"DWT Slope:    {np.mean(slope[-LOOKBACK:]):+.4f}", "#00aaff", 9, False),
        (f"Hurst H:      {H:.3f}  [{hurst_label}]", hurst_color, 9, False),
        (f"P(UP):        {P_up:.3f}", "#00cc44", 9, False),
        (f"P(DOWN):      {P_down:.3f}", "#ff3333", 9, False),
        (f"Confidence:   {confidence:.3f}", "#ffffff", 9, False),
        ("", "#ffffff", 9, False),
        (f"Mult UP:      {round_data['mult_up']:.2f}x", "#00cc44", 9, False),
        (f"Mult DOWN:    {round_data['mult_down']:.2f}x", "#ff3333", 9, False),
        (f"EV UP:        {decision['ev_up']:+.4f}", "#00cc44" if decision['ev_up']>0 else "#ff3333", 9, False),
        (f"EV DOWN:      {decision['ev_down']:+.4f}", "#00cc44" if decision['ev_down']>0 else "#ff3333", 9, False),
        ("", "#ffffff", 9, False),
        (f"Kelly bet:    {decision['kelly']*100:.1f}% of bankroll", "#ffaa00", 9, False),
        ("", "#ffffff", 9, False),
        (f"ACTION: {action} {side}", act_color, 12, True),
        (f"{decision['reason']}", "#aaaaaa", 8, False),
    ]
    y = 0.97
    for text, color, size, bold in lines:
        weight = "bold" if bold else "normal"
        ax.text(0.05, y, text, transform=ax.transAxes,
                color=color, fontsize=size, fontweight=weight, va="top")
        y -= 0.048

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()
    fig.canvas.flush_events()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  PancakeSwap Prediction — Full Strategy Study Tool")
    print("  Layers: DWT → Hurst → Fuzzy Logic → Markov → EV → Kelly")
    print("  Press Ctrl+C to stop")
    print("=" * 70)

    # ── Initial data fetch ────────────────────────────────────────────────────
    print("Fetching Chainlink history...")
    latest_id, live_price, _ = fetch_latest_chainlink()
    prices_arr, times_list, timestamps_arr = fetch_chainlink_history(latest_id, n=HISTORY)
    print(f"  Got {len(prices_arr)} rounds  |  "
          f"{times_list[0].strftime('%H:%M')} → {times_list[-1].strftime('%H:%M')}  |  "
          f"${prices_arr.min():.4f} – ${prices_arr.max():.4f}")

    print("Fetching PancakeSwap round data...")
    epoch      = fetch_pancake_epoch()
    round_data = fetch_round_data(epoch)
    print(f"  Epoch {epoch}  |  "
          f"Lock in {int(round_data['lock_ts'] - time.time())}s  |  "
          f"Pool: {round_data['total_bnb']:.3f} BNB  "
          f"(UP={round_data['bull_bnb']:.3f} / DOWN={round_data['bear_bnb']:.3f})")

    print("Fetching Markov history...")
    outcomes = fetch_markov_history(epoch, n=MARKOV_HISTORY)
    print(f"  Got {len(outcomes)} completed rounds: {outcomes[-5:]}")

    fig, axes     = setup_figure()
    last_id       = latest_id
    rolling_hurst = []
    refresh_count = 0

    try:
        while True:
            refresh_count += 1
            now = datetime.now()

            try:
                # ── Fetch new Chainlink rounds ─────────────────────────────
                new_id, live_price, live_ts = fetch_latest_chainlink()
                if new_id > last_id:
                    new_ids = [last_id + i + 1 for i in range(new_id - last_id)]
                    batch = [{"jsonrpc":"2.0","method":"eth_call",
                              "params":[{"to":CHAINLINK_CONTRACT,
                                         "data":"0x9a6fc8f5"+format(rid,"064x")},"latest"],
                              "id":i}
                             for i, rid in enumerate(new_ids)]
                    try:
                        resp  = requests.post(BSC_RPC, json=batch, timeout=10)
                        items = resp.json()
                        if isinstance(items, list):
                            for item in items:
                                if item.get("id") is None: continue
                                if "result" not in item: continue
                                raw = item["result"][2:]
                                ww  = [raw[j:j+64] for j in range(0,len(raw),64)]
                                if len(ww) < 4: continue
                                p = int(ww[1],16)/1e8
                                t = int(ww[3],16)
                                if p > 10 and t > 0:
                                    prices_arr     = np.append(prices_arr, p)
                                    timestamps_arr = np.append(timestamps_arr, float(t))
                                    times_list.append(datetime.fromtimestamp(t))
                    except Exception:
                        pass
                    last_id = new_id

                # Keep last HISTORY points
                if len(prices_arr) > HISTORY:
                    prices_arr     = prices_arr[-HISTORY:]
                    timestamps_arr = timestamps_arr[-HISTORY:]
                    times_list     = times_list[-HISTORY:]

                # ── Fetch PancakeSwap round ────────────────────────────────
                new_epoch = fetch_pancake_epoch()
                if new_epoch != epoch:
                    # New round started — update Markov history
                    try:
                        prev_rd = fetch_round_data(epoch)
                        if prev_rd["lock_price"] and prev_rd["close_price"]:
                            outcome = "UP" if prev_rd["close_price"] > prev_rd["lock_price"] else "DOWN"
                            outcomes.append(outcome)
                            if len(outcomes) > MARKOV_HISTORY:
                                outcomes = outcomes[-MARKOV_HISTORY:]
                    except Exception:
                        pass
                    epoch = new_epoch

                round_data = fetch_round_data(epoch)

                # ── Layer 2: DWT ───────────────────────────────────────────
                smoothed, slope = compute_dwt(prices_arr)

                # ── Layer 3: Hurst ─────────────────────────────────────────
                hurst_input = smoothed[-HURST_WINDOW:] if len(smoothed) >= HURST_WINDOW else smoothed
                H           = hurst_exponent(hurst_input)
                rolling_hurst.append(H)
                if len(rolling_hurst) > HISTORY:
                    rolling_hurst = rolling_hurst[-HISTORY:]
                hurst_label, hurst_color, hurst_desc = hurst_regime(H)

                # ── Layer 4: Markov ────────────────────────────────────────
                markov_matrix          = build_markov_matrix(outcomes)
                markov_p_up, markov_p_down = markov_bias(outcomes, markov_matrix)

                # ── Layer 5: Fuzzy Inference ───────────────────────────────
                slope_val = float(np.mean(slope[-LOOKBACK:]))
                P_up, P_down, confidence = fuzzy_inference(
                    slope_val, H, markov_p_up, markov_p_down
                )

                # ── Layer 6: EV + Kelly ────────────────────────────────────
                decision = make_decision(
                    P_up, P_down,
                    round_data["mult_up"], round_data["mult_down"],
                    confidence
                )

                # ── Backtest ───────────────────────────────────────────────
                win_rate, wins, losses, skips = backtest_5min(
                    prices_arr, timestamps_arr, slope
                )

                # ── Terminal output ────────────────────────────────────────
                lock_in = int(round_data["lock_ts"] - time.time())
                print(
                    f"[{now.strftime('%H:%M:%S')}] #{refresh_count:04d}  "
                    f"Live:${live_price:.4f}  "
                    f"H={H:.2f}[{hurst_label[:4]}]  "
                    f"P(UP)={P_up:.2f} P(DN)={P_down:.2f}  "
                    f"EV_UP={decision['ev_up']:+.3f} EV_DN={decision['ev_down']:+.3f}  "
                    f"→ {decision['action']} {decision['side'] or ''}  "
                    f"Kelly={decision['kelly']*100:.1f}%  "
                    f"WR={win_rate:.1f}%  Lock:{lock_in}s"
                )

                # ── Update chart ───────────────────────────────────────────
                update_chart(
                    fig, axes,
                    times_list, prices_arr, smoothed, slope,
                    H, hurst_label, hurst_color,
                    round_data, decision,
                    P_up, P_down, confidence,
                    markov_matrix, outcomes,
                    win_rate, wins, losses,
                    live_price, rolling_hurst
                )

            except requests.exceptions.RequestException as e:
                print(f"[{now.strftime('%H:%M:%S')}] Network error: {e}")
            except Exception as e:
                print(f"[{now.strftime('%H:%M:%S')}] Error: {e}")

            time.sleep(REFRESH_SEC)

    except KeyboardInterrupt:
        print("\nStopped.")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
