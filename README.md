# PancakeSwap Prediction — LSTM Adaptive Control System

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Every 5 Minutes                          │
│                                                             │
│  1. Collect last K=12 rounds (X, θ, C)                     │
│         ↓                                                   │
│  2. LSTM Controller → optimized params θ*                   │
│         ↓                                                   │
│  3. 6-Step Engine (DWT→Hurst→Fuzzy→Markov→EV→Kelly)        │
│         ↓                                                   │
│  4. Pool-Skew Gate (proven edge from 1-year backtest)       │
│         ↓                                                   │
│  5. BET / SKIP decision                                     │
│         ↓                                                   │
│  6. Wait for outcome → push (X, θ, C) back to LSTM         │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `server.py` | FastAPI backend — live loop, LSTM integration, dashboard API |
| `lstm_controller.py` | LSTM model definition + training + inference |
| `audit_log_generator.py` | Step 1: generate training data from 1-year history |
| `train_on_lightning.py` | Step 2: training script for Lightning.ai GPU |
| `wavelet_denoise.py` | DWT denoising utility |
| `index.html` | Web dashboard (shows LSTM params + pool skew) |
| `pancake_1year.jsonl` | 1-year PancakeSwap round data (105,071 rounds) |
| `audit_log.jsonl` | Generated training data (created by audit_log_generator.py) |
| `lstm_controller.pt` | Trained model weights (download from Lightning.ai) |

---

## Quick Start (Live Dashboard)

```bash
pip install -r requirements.txt
python3 dwt_denoising/server.py
# Open http://localhost:8000
```

The server runs immediately with **default parameters** (no LSTM needed).
The pool-skew strategy is active from the first round.

---

## Training the LSTM (Full Pipeline)

### Step 1 — Generate audit log (run locally, ~15 min)

```bash
python3 dwt_denoising/audit_log_generator.py
# Output: dwt_denoising/audit_log.jsonl (~500 MB)
# Runs 216 parameter combos × 105,071 rounds
```

### Step 2 — Train on Lightning.ai (free GPU)

1. Go to **https://lightning.ai** → Create free account
2. **New Studio** → Blank → Select **GPU** (A10G free tier)
3. In the terminal:
   ```bash
   pip install torch numpy
   ```
4. Upload these files to Lightning.ai:
   - `audit_log.jsonl`
   - `lstm_controller.py`
   - `train_on_lightning.py`
5. Run:
   ```bash
   python train_on_lightning.py
   ```
6. Wait ~20 minutes (A10G), then download `lstm_controller.pt`
7. Put `lstm_controller.pt` in `dwt_denoising/`

### Step 3 — Run with LSTM active

```bash
python3 dwt_denoising/server.py
# Dashboard shows "LSTM: ACTIVE" once 12 rounds of history are collected
```

---

## What the LSTM Learns

For each round, the LSTM sees the last 12 rounds of:
- **X** (market state): volatility, trend, pool skew, time features
- **θ** (parameters used): wavelet level, Hurst window, fuzzy threshold, slope threshold, lookback
- **C** (performance): PnL, price error, directional accuracy, win/loss

It outputs the optimal parameter set for the **next** round.

### Parameters the LSTM tunes

| Parameter | Range | Effect |
|-----------|-------|--------|
| `wavelet_level` | 1, 2, 3 | DWT smoothing depth (1=reactive, 3=smooth) |
| `hurst_window` | 50–400 | Lookback for regime detection |
| `fuzzy_threshold` | 0.30–0.60 | Sensitivity of fuzzy membership functions |
| `slope_thresh` | 0.03–0.12 | Minimum slope to generate UP/DOWN signal |
| `lookback` | 2, 4 | Rounds to average slope over |

---

## Pool-Skew Strategy (Primary Gate)

From 1-year backtest (105,071 rounds):

| Pool condition | Actual outcome | Rounds/year | Bet side | Mult | Break-even | Profitable? |
|---|---|---|---|---|---|---|
| pool_up > 65% | UP 67.7% | 5,125 | DOWN | ~3.1x | 32% | ✓ WR 32.3% |
| pool_up > 70% | UP 69.1% | 1,329 | DOWN | ~3.7x | 27% | ✓ WR 30.9% |
| pool_up < 35% | DOWN 67.1% | 2,044 | UP | ~3.1x | 32% | ✓ WR 32.3% |

**The LSTM does NOT override the pool-skew gate.** It only tunes the 6-step engine
parameters that compute P_up/confidence shown in the dashboard.

---

## Dashboard

- **LSTM Controller card**: shows current params (source: `lstm` or `default`)
- **Pool UP%**: red if ≥65% (BET DOWN), green if ≤35% (BET UP), grey otherwise
- **Next Round Signal**: BET UP / BET DOWN / SKIP
- **Round Status Banner**: current round decision + result

---

## Log Format

Each line in `logs/rounds_*.jsonl`:
```json
{
  "epoch": 485214,
  "actual_outcome": "DOWN",
  "correct": true,
  "lstm_wavelet_level": 1,
  "lstm_hurst_window": 200,
  "lstm_fuzzy_threshold": 0.42,
  "lstm_slope_thresh": 0.06,
  "lstm_lookback": 3,
  "lstm_source": "lstm",
  "decision_action": "BET",
  "decision_side": "DOWN",
  "pool_up_pct": 68.4,
  "mult_down": 3.12,
  "ev_down": 1.847,
  "kelly_pct": 8.2,
  "hurst_H": 0.71,
  "P_up": 0.075,
  "confidence": 0.85
}
```
