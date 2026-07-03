"""
BNB/USD Real-Time DWT De-noising Study Tool
============================================
Data source : Chainlink BNB/USD oracle on BSC (exact PancakeSwap feed)
History     : Last 100 real Chainlink price rounds (~55 min at ~33s/round)
Refresh     : Every 10 seconds — fetches latest round and appends to buffer

Raw ticks vs DWT smoothed are genuinely different because Chainlink
updates every ~30s with real price changes, not smooth candle closes.

Press Ctrl+C to stop.
"""

import sys
import time
import requests
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from wavelet_denoise import wavelet_denoise


# ── Configuration ─────────────────────────────────────────────────────────────
WAVELET        = "db4"
LEVEL          = 2
REFRESH_SEC    = 10
SLOPE_THRESH   = 0.05   # USD/round — Chainlink moves ~0.1-1.0 per update
LOOKBACK       = 3      # rounds to average slope over for signal
HISTORY        = 200    # rounds of history to keep (need more for 5-min backtest)
CHUNK          = 20     # batch size per RPC call

# PancakeSwap Prediction: each round is 5 minutes.
# Chainlink updates every ~30s → 5 min = ~10 Chainlink rounds ahead.
# We predict at round i, then check price at round i + PANCAKE_ROUNDS_AHEAD.
CHAINLINK_SEC_PER_ROUND = 30        # approximate
PANCAKE_ROUND_SEC       = 300       # 5 minutes
PANCAKE_ROUNDS_AHEAD    = round(PANCAKE_ROUND_SEC / CHAINLINK_SEC_PER_ROUND)  # = 10
# ──────────────────────────────────────────────────────────────────────────────

BSC_RPC   = "https://bsc-dataseed.binance.org/"
CONTRACT  = "0x0567F2323251f0Aab15c8dFb1967E4e8A7D42aeE"


# ── Chainlink helpers ──────────────────────────────────────────────────────────

def _decode_round(hex_result):
    """Decode a getRoundData / latestRoundData hex result into (price, timestamp)."""
    raw   = hex_result[2:]
    words = [raw[i:i+64] for i in range(0, len(raw), 64)]
    if len(words) < 4:
        return None, None
    price = int(words[1], 16) / 1e8
    ts    = int(words[3], 16)
    return price, ts


def fetch_latest_round():
    """Return (round_id, price, timestamp) for the most recent Chainlink round."""
    payload = {
        "jsonrpc": "2.0", "method": "eth_call",
        "params": [{"to": CONTRACT, "data": "0xfeaf968c"}, "latest"],
        "id": 0
    }
    r   = requests.post(BSC_RPC, json=payload, timeout=5)
    raw = r.json()["result"][2:]
    w   = [raw[i:i+64] for i in range(0, len(raw), 64)]
    round_id = int(w[0], 16)
    price    = int(w[1], 16) / 1e8
    ts       = int(w[3], 16)
    return round_id, price, ts


def fetch_rounds_batch(round_ids):
    """
    Fetch multiple Chainlink rounds in one HTTP request (batch JSON-RPC).
    Returns list of (price, timestamp) in the same order as round_ids.
    """
    batch = [
        {
            "jsonrpc": "2.0", "method": "eth_call",
            "params": [{"to": CONTRACT,
                        "data": "0x9a6fc8f5" + hex(rid)[2:].zfill(64)}, "latest"],
            "id": i
        }
        for i, rid in enumerate(round_ids)
    ]
    resp  = requests.post(BSC_RPC, json=batch, timeout=15)
    items = sorted(resp.json(), key=lambda x: x["id"])
    results = []
    for item in items:
        if "result" not in item:
            results.append((None, None))
            continue
        price, ts = _decode_round(item["result"])
        results.append((price, ts))
    return results


def fetch_history(latest_round_id, n=HISTORY):
    """
    Fetch the last n Chainlink rounds in chunks with rate-limit handling.
    Returns (prices_arr, times_list, timestamps_arr) oldest-first.
    """
    all_prices = []
    all_times  = []

    ids = [latest_round_id - i for i in range(n)]

    for start in range(0, len(ids), CHUNK):
        chunk_ids = ids[start:start + CHUNK]
        batch = [
            {
                "jsonrpc": "2.0", "method": "eth_call",
                "params": [{"to": CONTRACT,
                            "data": "0x9a6fc8f5" + format(rid, "064x")}, "latest"],
                "id": start + i   # globally unique ID
            }
            for i, rid in enumerate(chunk_ids)
        ]
        try:
            resp  = requests.post(BSC_RPC, json=batch, timeout=15)
            items = resp.json()
            if not isinstance(items, list):
                continue
            for item in items:
                if item.get("id") is None:
                    continue   # rate-limited item
                if "result" not in item:
                    continue
                price, ts = _decode_round(item["result"])
                if price and price > 10 and ts and ts > 0:
                    all_prices.append(price)
                    all_times.append(ts)
        except Exception:
            pass
        time.sleep(0.15)   # small delay to avoid rate limiting

    # Reverse so oldest is first
    all_prices = list(reversed(all_prices))
    all_times  = list(reversed(all_times))
    times_list = [datetime.fromtimestamp(t) for t in all_times]
    return np.array(all_prices), times_list, np.array(all_times, dtype=float)


# ── DWT & signal ──────────────────────────────────────────────────────────────

def compute_dwt(prices_arr):
    smoothed = wavelet_denoise(prices_arr, wavelet=WAVELET, level=LEVEL)
    slope    = np.diff(smoothed, prepend=smoothed[0])
    return smoothed, slope


def get_signal(slope):
    recent = np.mean(slope[-LOOKBACK:])
    if recent > SLOPE_THRESH:
        return "▲ UP",      recent, "#00cc44"
    elif recent < -SLOPE_THRESH:
        return "▼ DOWN",    recent, "#ff3333"
    else:
        return "◆ NEUTRAL", recent, "#aaaaaa"


def backtest_5min(prices_arr, timestamps_arr, slope):
    """
    Backtest simulating PancakeSwap Prediction rules:

    - At each Chainlink round i, read the DWT slope → predict UP or DOWN
    - The 'locked price' is prices_arr[i]  (what PancakeSwap locks at round start)
    - The 'close price'  is the price ~5 minutes later
      → we find the actual round closest to i's timestamp + 300 seconds
    - If prediction matches direction: WIN, else: LOSS
    - NEUTRAL signals (slope within threshold) are skipped — no bet placed

    Returns: win_rate, wins, losses, skips, trade_log
    """
    wins = losses = skips = 0
    trade_log = []

    for i in range(LOOKBACK, len(prices_arr)):
        avg_slope = np.mean(slope[i - LOOKBACK:i])

        # Determine prediction
        if avg_slope > SLOPE_THRESH:
            pred = "UP"
        elif avg_slope < -SLOPE_THRESH:
            pred = "DOWN"
        else:
            skips += 1
            continue

        # Find the round ~5 minutes ahead using timestamps
        entry_ts = timestamps_arr[i]
        target_ts = entry_ts + PANCAKE_ROUND_SEC

        # Find the index with timestamp closest to target_ts
        future_idx = None
        min_diff = float('inf')
        for j in range(i + 1, len(prices_arr)):
            diff = abs(timestamps_arr[j] - target_ts)
            if diff < min_diff:
                min_diff = diff
                future_idx = j
            elif timestamps_arr[j] > target_ts + 60:
                # Gone past target by >1 min, stop searching
                break

        # Need a future round within 90 seconds of the 5-min target
        if future_idx is None or min_diff > 90:
            skips += 1
            continue

        locked_price = prices_arr[i]
        close_price  = prices_arr[future_idx]
        actual       = "UP" if close_price > locked_price else "DOWN"
        correct      = (pred == actual)

        if correct:
            wins += 1
        else:
            losses += 1

        trade_log.append({
            "i":            i,
            "entry_time":   datetime.fromtimestamp(entry_ts).strftime("%H:%M:%S"),
            "locked_price": locked_price,
            "close_price":  close_price,
            "change":       close_price - locked_price,
            "prediction":   pred,
            "actual":       actual,
            "correct":      correct,
        })

    total    = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0.0
    return win_rate, wins, losses, skips, trade_log


# ── Chart ─────────────────────────────────────────────────────────────────────

def setup_figure():
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor('#0f0f0f')
    for ax in axes:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes


def update_chart(fig, axes, times, prices, smoothed, slope,
                 signal_label, slope_val, sig_color,
                 win_rate, wins, losses, skips, live_price, trade_log):

    for ax in axes:
        ax.cla()
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    now_str  = datetime.now().strftime("%H:%M:%S")
    live_str = f"${live_price:.4f}" if live_price else "N/A"
    total    = wins + losses

    fig.suptitle(
        f"BNB/USD  [Chainlink — PancakeSwap oracle]  |  Live: {live_str}  |  "
        f"{WAVELET} lvl {LEVEL}  |  "
        f"5-min WinRate: {win_rate:.1f}% ({wins}W / {losses}L / {skips} skipped)  |  {now_str}",
        fontsize=10, fontweight='bold', color='white'
    )

    # ── Panel 1: Raw Chainlink ticks vs DWT smoothed ──────────────────────────
    ax1 = axes[0]
    ax1.plot(times, prices,   color='#ffffff', linewidth=1,   alpha=0.7,
             label="Raw Chainlink ticks", zorder=2)
    ax1.plot(times, smoothed, color='#00aaff', linewidth=2.5, alpha=1.0,
             label=f"DWT Smoothed ({WAVELET} lvl {LEVEL})", zorder=3)
    # Live price dot
    ax1.scatter([times[-1]], [live_price], color=sig_color, s=80, zorder=5,
                label=f"Live {live_str}")

    # Shade area between raw and smoothed to make difference obvious
    ax1.fill_between(times, prices, smoothed,
                     where=(prices > smoothed), alpha=0.15, color='#00cc44', label='Raw > DWT')
    ax1.fill_between(times, prices, smoothed,
                     where=(prices < smoothed), alpha=0.15, color='#ff3333', label='Raw < DWT')

    ax1.set_ylabel("Price (USD)", color='white')
    ax1.legend(loc='upper left', fontsize=8, facecolor='#222', labelcolor='white', ncol=2)
    ax1.grid(True, alpha=0.15, color='white')
    ax1.set_title(
        f"Raw Chainlink Ticks vs DWT Smoothed  |  "
        f"Signal: {signal_label}  (slope {slope_val:+.4f})",
        color=sig_color, fontsize=10
    )

    # ── Panel 2: DWT Slope ────────────────────────────────────────────────────
    ax2 = axes[1]
    bar_colors = ['#00cc44' if s > 0 else '#ff3333' for s in slope]
    ax2.bar(times, slope, color=bar_colors, alpha=0.8, width=0.015)
    ax2.axhline(0,              color='white',   linewidth=0.8, linestyle='--', alpha=0.4)
    ax2.axhline( SLOPE_THRESH,  color='#00cc44', linewidth=1,   linestyle=':',  alpha=0.7,
                label=f'UP  (+{SLOPE_THRESH})')
    ax2.axhline(-SLOPE_THRESH,  color='#ff3333', linewidth=1,   linestyle=':',  alpha=0.7,
                label=f'DOWN (-{SLOPE_THRESH})')
    ax2.set_ylabel("DWT Slope", color='white')
    ax2.legend(loc='upper left', fontsize=9, facecolor='#222', labelcolor='white')
    ax2.grid(True, alpha=0.15, color='white')
    ax2.set_title("DWT Slope — Direction Indicator", color='white', fontsize=10)

    # ── Panel 3: Raw price change per Chainlink round ─────────────────────────
    ax3 = axes[2]
    price_change = np.diff(prices, prepend=prices[0])
    bar_colors3  = ['#00cc44' if c > 0 else '#ff3333' for c in price_change]
    ax3.bar(times, price_change, color=bar_colors3, alpha=0.8, width=0.015)
    ax3.axhline(0, color='white', linewidth=0.8, alpha=0.4)
    ax3.set_ylabel("Δ Price (USD)", color='white')
    ax3.set_xlabel("Time", color='white')
    ax3.grid(True, alpha=0.15, color='white')
    ax3.set_title("Raw Price Change per Chainlink Round (~30s each)", color='white', fontsize=10)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right',
                 fontsize=8, color='white')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()
    fig.canvas.flush_events()


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  BNB/USD Real-Time DWT Study")
    print("  Source  : Chainlink BNB/USD oracle (exact PancakeSwap feed)")
    print(f"  History : Last {HISTORY} real Chainlink rounds (~{HISTORY*CHAINLINK_SEC_PER_ROUND//60} min)")
    print(f"  Backtest: 5-minute direction prediction ({PANCAKE_ROUNDS_AHEAD} rounds ahead)")
    print("  Refresh : every 10s  |  Press Ctrl+C to stop")
    print("=" * 65)

    print("Fetching Chainlink history...")
    latest_round_id, live_price, _ = fetch_latest_round()
    prices_arr, times_list, timestamps_arr = fetch_history(latest_round_id, n=HISTORY)
    print(f"Got {len(prices_arr)} rounds  |  "
          f"{times_list[0].strftime('%H:%M')} → {times_list[-1].strftime('%H:%M')}  |  "
          f"${prices_arr.min():.4f} – ${prices_arr.max():.4f}")

    fig, axes     = setup_figure()
    last_round_id = latest_round_id
    refresh_count = 0

    try:
        while True:
            refresh_count += 1
            now = datetime.now()

            try:
                new_round_id, live_price, live_ts = fetch_latest_round()

                # Append any new rounds since last refresh
                if new_round_id > last_round_id:
                    new_ids  = [last_round_id + i + 1 for i in range(new_round_id - last_round_id)]
                    new_data = fetch_rounds_batch(new_ids)
                    for price, ts in new_data:
                        if price and price > 10 and ts:
                            prices_arr     = np.append(prices_arr, price)
                            timestamps_arr = np.append(timestamps_arr, float(ts))
                            times_list.append(datetime.fromtimestamp(ts))
                    # Keep only last HISTORY points
                    if len(prices_arr) > HISTORY:
                        prices_arr     = prices_arr[-HISTORY:]
                        timestamps_arr = timestamps_arr[-HISTORY:]
                        times_list     = times_list[-HISTORY:]
                    last_round_id = new_round_id

                smoothed, slope = compute_dwt(prices_arr)
                signal_label, slope_val, sig_color = get_signal(slope)

                # 5-minute backtest
                win_rate, wins, losses, skips, trade_log = backtest_5min(
                    prices_arr, timestamps_arr, slope
                )

                diff_avg = np.abs(prices_arr - smoothed).mean()

                print(
                    f"[{now.strftime('%H:%M:%S')}] #{refresh_count:04d}  "
                    f"Live: ${live_price:.4f}  |  "
                    f"DWT: {smoothed[-1]:.4f}  |  "
                    f"Slope: {slope_val:+.4f}  |  "
                    f"Signal: {signal_label}  |  "
                    f"5min WinRate: {win_rate:.1f}% ({wins}W/{losses}L/{skips}skip)  |  "
                    f"Rounds: {len(prices_arr)}"
                )

                # Print last 3 completed trades for visibility
                if trade_log:
                    for t in trade_log[-3:]:
                        mark = "✓" if t["correct"] else "✗"
                        print(f"  {mark} [{t['entry_time']}] "
                              f"Pred:{t['prediction']:4s}  "
                              f"Locked:${t['locked_price']:.4f}  "
                              f"Close:${t['close_price']:.4f}  "
                              f"Δ{t['change']:+.4f}  "
                              f"Actual:{t['actual']}")

                update_chart(fig, axes, times_list, prices_arr, smoothed, slope,
                             signal_label, slope_val, sig_color,
                             win_rate, wins, losses, skips, live_price, trade_log)

            except requests.exceptions.RequestException as e:
                print(f"[{now.strftime('%H:%M:%S')}] Fetch error: {e}")

            time.sleep(REFRESH_SEC)

    except KeyboardInterrupt:
        print("\nStopped.")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
