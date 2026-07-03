# PancakeSwap Prediction — Full Strategy Theory & Mathematics
## Complete Reference Document

---

## Overview

This system predicts the direction of BNB/USD price movement for PancakeSwap
Prediction rounds using a 6-layer mathematical pipeline:

```
Layer 1: DWT De-noising          → clean the raw price signal
Layer 2: Hurst Exponent          → identify market regime
Layer 3: Fuzzy Inference System  → combine signals into probability P
Layer 4: Markov Chain            → sequence bias from round history
Layer 5: Expected Value (EV)     → find mathematically mispriced bets
Layer 6: Kelly Criterion         → optimal bet sizing
```

---

## PancakeSwap Prediction — How It Works

Each round lasts exactly 5 minutes (300 seconds):

```
t=0s    Round N opens   → betting is OPEN
t=270s  Decision window → last 30s to decide bet for Round N+1
t=300s  Round N LOCKS   → betting closes, lock price recorded
t=600s  Round N CLOSES  → close price recorded, result determined

Result: UP   if close_price > lock_price
        DOWN if close_price < lock_price
```

The price oracle is Chainlink BNB/USD on BSC:
  Contract: 0x0567F2323251f0Aab15c8dFb1967E4e8A7D42aeE
  Updates every ~30 seconds on-chain

The prediction contract is PancakeSwap Prediction on BSC:
  Contract: 0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA

---

## Layer 1: Discrete Wavelet Transform (DWT) De-noising

### Theory

A raw price series contains two components:
  Price(t) = True_Trend(t) + Noise(t)

DWT separates these by decomposing the signal into frequency bands.
The Daubechies 4 (db4) wavelet is used because it captures sudden
price shifts without over-smoothing.

### Mathematics

**Step 1 — Decomposition:**
  wavedec(signal, wavelet='db4', level=2)
  Returns: [cA2, cD2, cD1]
    cA2 = approximation coefficients (low frequency = trend)
    cD2 = detail level 2 (mid frequency = short-term wiggles)
    cD1 = detail level 1 (high frequency = noise/jitter)

**Step 2 — Noise Estimation (MAD):**
  σ = MAD(cD1) / 0.6745
  where MAD = Median Absolute Deviation
  The constant 0.6745 scales MAD to match normal distribution std dev

**Step 3 — Universal Threshold (VisuShrink):**
  λ = σ × √(2 × ln(n))
  where n = number of data points
  This is the mathematically optimal threshold for Gaussian noise

**Step 4 — Soft Thresholding:**
  For each detail coefficient d:
    d_clean = sign(d) × max(0, |d| - λ)
  Soft thresholding shrinks coefficients toward zero smoothly,
  avoiding sharp discontinuities in the reconstructed signal.
  The approximation coefficients (cA2) are NOT thresholded.

**Step 5 — Reconstruction (IDWT):**
  smoothed = waverec([cA2, cD2_clean, cD1_clean], wavelet='db4')

### DWT Slope

The direction indicator is the first difference of the smoothed signal:
  slope[i] = smoothed[i] - smoothed[i-1]

A rolling average over LOOKBACK=3 points reduces false signals:
  avg_slope = mean(slope[-3:])

### Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| wavelet   | db4   | Daubechies 4 — good for financial data |
| level     | 2     | Decomposition depth (1=reactive, 3=smooth) |
| LOOKBACK  | 3     | Rounds to average slope over |
| SLOPE_THRESH | 0.05 | Minimum slope to generate UP/DOWN signal |

### Interpretation

  avg_slope > +0.05  → UP signal  (price trending upward)
  avg_slope < -0.05  → DOWN signal (price trending downward)
  |avg_slope| < 0.05 → NEUTRAL (no clear direction)

---

## Layer 2: Hurst Exponent (Market Regime Detection)

### Theory

The Hurst Exponent H measures the long-term memory of a time series.
It answers: "Is the current market trending or mean-reverting?"

  H > 0.6  → Persistent / Trending
             Price is likely to CONTINUE in its current direction
             → Trust the DWT slope signal

  H < 0.4  → Anti-persistent / Mean-reverting
             Price is likely to REVERSE its current direction
             → Fade (bet against) the DWT slope signal

  H ≈ 0.5  → Random walk (Brownian motion)
             Price movement is unpredictable
             → Skip the bet

### Mathematics (Rescaled Range / R/S Analysis)

For a time series X of length n, at each lag τ:

1. Divide X into non-overlapping sub-windows of size τ
2. For each sub-window:
   a. Compute mean: μ = mean(X_sub)
   b. Compute cumulative deviation: Y_t = Σ(X_i - μ) for i=1..t
   c. Range: R = max(Y) - min(Y)
   d. Standard deviation: S = std(X_sub)
   e. Rescaled range: R/S

3. Average R/S across all sub-windows for this lag τ
4. Repeat for multiple lag values τ

5. Log-log regression:
   log(R/S) = H × log(τ) + constant
   H = slope of the regression line

### Reliability Constraints

  Minimum lag points required: 4 (otherwise return H=0.5)
  H is clamped to [0.05, 0.95] to avoid artifacts:
    H=1.0 artifact occurs when price series is too flat or too short

### Parameters

| Parameter    | Value | Meaning |
|--------------|-------|---------|
| HURST_WINDOW | 50    | Chainlink rounds used for H calculation (~25 min) |
| H_trending   | >0.6  | Threshold for trending regime |
| H_random     | ~0.5  | Threshold for random walk |
| H_meanrev    | <0.4  | Threshold for mean-reverting regime |

---

## Layer 3: Fuzzy Inference System (FIS)

### Theory

Instead of hard binary rules ("if slope > 0.05 then BET UP"),
fuzzy logic assigns partial membership degrees to each input.
This allows weak signals to combine into strong decisions.

### Fuzzification — Trapezoidal Membership Functions

Each input is mapped to fuzzy sets using trapezoid(x, a, b, c, d):
  = 0        if x ≤ a or x ≥ d
  = (x-a)/(b-a)  if a < x < b  (rising slope)
  = 1        if b ≤ x ≤ c  (flat top)
  = (d-x)/(d-c)  if c < x < d  (falling slope)

**DWT Slope fuzzy sets:**
  strong_down: trapezoid(slope, -∞, -∞, -0.8, -0.15)
  weak_down:   trapezoid(slope, -0.8, -0.3, -0.08, 0.0)
  neutral:     trapezoid(slope, -0.15, -0.03, 0.03, 0.15)
  weak_up:     trapezoid(slope, 0.0, 0.08, 0.3, 0.8)
  strong_up:   trapezoid(slope, 0.15, 0.8, +∞, +∞)

**Hurst fuzzy sets:**
  mean_reverting: trapezoid(H, 0.0, 0.0, 0.35, 0.5)
  random:         trapezoid(H, 0.35, 0.45, 0.55, 0.65)
  trending:       trapezoid(H, 0.5, 0.65, 1.0, 1.0)

**Markov fuzzy sets:**
  biased_down: trapezoid(P_up, 0.0, 0.0, 0.35, 0.5)
  biased_up:   trapezoid(P_up, 0.5, 0.65, 1.0, 1.0)

### Rule Engine (Mamdani-style, min-AND)

UP activation rules:
  R1: min(strong_up,   trending)   × 1.0  → strong trend continuation
  R2: min(weak_up,     trending)   × 0.7  → weak trend continuation
  R3: min(strong_down, mean_rev)   × 0.8  → mean reversion (fade down)
  R4: min(biased_up,   trending)   × 0.5  → Markov sequence bias

DOWN activation rules:
  R5: min(strong_down, trending)   × 1.0  → strong trend continuation
  R6: min(weak_down,   trending)   × 0.7  → weak trend continuation
  R7: min(strong_up,   mean_rev)   × 0.8  → mean reversion (fade up)
  R8: min(biased_down, trending)   × 0.5  → Markov sequence bias

### Defuzzification

  raw_up   = max(R1, R2, R3, R4)
  raw_down = max(R5, R6, R7, R8)
  total    = raw_up + raw_down

Neutral baseline blend (prevents extreme 0/1 outputs):
  NEUTRAL = 0.15
  P_up   = (raw_up   / total) × (1 - NEUTRAL) + 0.5 × NEUTRAL
  P_down = (raw_down / total) × (1 - NEUTRAL) + 0.5 × NEUTRAL

Confidence (how far from 50/50):
  confidence = |P_up - 0.5| × 2.0 × (1 - random_membership)

### Parameters

| Parameter      | Value | Meaning |
|----------------|-------|---------|
| NEUTRAL        | 0.15  | Baseline blend — prevents 100/0 extremes |
| MIN_CONFIDENCE | 0.52  | Minimum confidence to place a bet |

---

## Layer 4: Markov Chain (Sequence Bias)

### Theory

A Markov Chain models the probability of transitioning between states.
For PancakeSwap, the states are: UP, DOWN

The transition matrix T is built from historical round outcomes:
  T[from_state][to_state] = count(from→to) / count(from)

Example:
  After 30 rounds: UP→UP=12, UP→DOWN=8, DOWN→UP=9, DOWN→DOWN=11
  T[UP][UP]   = 12/20 = 0.60
  T[UP][DOWN] = 8/20  = 0.40
  T[DOWN][UP] = 9/20  = 0.45
  T[DOWN][DOWN] = 11/20 = 0.55

If the last round was UP, P(next=UP) = 0.60 from the matrix.

### How It Feeds Into FIS

The Markov P(UP) is fuzzified and used as input R4/R8 in the rule engine.
It adds a small bias (weight 0.5) toward the historically likely direction.

### Parameters

| Parameter      | Value | Meaning |
|----------------|-------|---------|
| MARKOV_HISTORY | 30    | Past rounds used to build transition matrix |

---

## Layer 5: Expected Value (EV) Calculation

### Theory

Winning >50% of the time is NOT enough to be profitable.
PancakeSwap takes a 3% house fee, so you need >51.5% just to break even.

The real edge comes from finding MISPRICED bets — when the crowd
over-allocates to one side, the other side's multiplier becomes very high.

### Mathematics

**Multipliers (after 3% house fee):**
  Multiplier_UP   = (Total_Pool / UP_Pool)   × 0.97
  Multiplier_DOWN = (Total_Pool / DOWN_Pool) × 0.97

**Expected Value:**
  EV_UP   = P_up   × Multiplier_UP   - (1 - P_up)
  EV_DOWN = P_down × Multiplier_DOWN - (1 - P_down)

**Interpretation:**
  EV > 0 → positive edge (bet is mathematically profitable long-term)
  EV < 0 → negative edge (bet loses money long-term)
  EV = 0 → break-even

**Example (EV Arbitrage):**
  Total Pool = 1.0 BNB
  UP Pool    = 0.87 BNB  (87% of people bet UP)
  DOWN Pool  = 0.13 BNB  (13% of people bet DOWN)

  Multiplier_UP   = (1.0/0.87) × 0.97 = 1.115x
  Multiplier_DOWN = (1.0/0.13) × 0.97 = 7.462x

  If P_up = 0.55, P_down = 0.45:
  EV_UP   = 0.55 × 1.115 - 0.45 = +0.163  (positive but small)
  EV_DOWN = 0.45 × 7.462 - 0.55 = +2.808  (strongly positive!)

  → BET DOWN even though P(DOWN) < P(UP)
  → The crowd mispriced the pool — DOWN is the arbitrage

### Decision Rules

  Rule 1: Only bet if EV > 0 for at least one side
  Rule 2: Bet the side with higher positive EV
  Rule 3: If both EV ≤ 0, skip the round
  Rule 4: If confidence < MIN_CONFIDENCE (0.52), skip
           EXCEPTION: If EV > 1.0, override confidence check
           (crowd mispricing is worth betting even with neutral signal)

### Guards (Skip Conditions)

  Pool too small:  total_bnb < 0.005 BNB → skip (pool not filled)
  Round too new:   elapsed < 30s → skip (multipliers unstable)
  Low confidence:  conf < 0.52 AND EV < 1.0 → skip
  Both EV ≤ 0:    no mathematical edge → skip

### Parameters

| Parameter           | Value | Meaning |
|---------------------|-------|---------|
| MIN_EV              | 0.0   | Minimum EV to place a bet |
| MIN_CONFIDENCE      | 0.52  | Minimum fuzzy confidence |
| EV_ARBITRAGE_THRESH | 1.0   | EV threshold to override confidence check |
| MIN_POOL_BNB        | 0.005 | Minimum pool size before evaluating |
| MIN_POOL_TIME       | 30s   | Minimum seconds into round before evaluating |

---

## Layer 6: Kelly Criterion (Bet Sizing)

### Theory

The Kelly Criterion calculates the mathematically optimal fraction of
your bankroll to bet in order to maximize long-term exponential growth
while avoiding ruin.

### Mathematics

**Full Kelly:**
  f* = (P × Multiplier - (1 - P)) / Multiplier

**Quarter-Kelly (used here for safety):**
  bet_fraction = 0.25 × f*

The Quarter-Kelly is used because:
  1. P is an estimate, not a certainty
  2. Reduces variance and protects against model errors
  3. Prevents large drawdowns from consecutive losses

**Example:**
  P_down = 0.45, Multiplier_DOWN = 7.462
  f* = (0.45 × 7.462 - 0.55) / 7.462
     = (3.358 - 0.55) / 7.462
     = 2.808 / 7.462
     = 0.376  (37.6% full Kelly)

  Quarter-Kelly = 0.25 × 0.376 = 0.094 (9.4% of bankroll)

  If bankroll = 1.0 BNB → bet 0.094 BNB this round

### Parameters

| Parameter      | Value | Meaning |
|----------------|-------|---------|
| KELLY_FRACTION | 0.25  | Safety multiplier (Quarter-Kelly) |

---

## Decision Timeline Per Round

```
Round N (0s → 300s):
  t=0s    New epoch detected
          → next_round_decision promoted to current_round_decision
          → Banner locked: shows BET UP/DOWN/SKIP for this round

  t=0-270s  Next Round Signal updates every 10s
            → DWT, Hurst, Fuzzy, EV all recalculated
            → Shows what you'll bet on Round N+1

  t=270s  DECISION WINDOW opens (30s before lock)
          → Banner shows: ⚡ DECISION WINDOW — Xs left
          → Signal log written with full pool data

  t=300s  Round N LOCKS
          → lock_price recorded on-chain by Chainlink oracle
          → next_round_decision frozen as bet for Round N+1

Round N+1 (300s → 600s):
  t=300s  New epoch N+1 detected
          → next_round_decision (from Round N) → current_round_decision
          → Banner: BET DOWN (locked, cannot change)

  t=600s  Round N+1 CLOSES
          → close_price recorded on-chain
          → Result: UP if close > lock, DOWN if close < lock
          → WIN if bet matches result, LOSS otherwise
          → Banner updates: ✓ WIN or ✗ LOSS
```

---

## Data Sources

| Data | Source | Update Frequency |
|------|--------|-----------------|
| BNB/USD price | Chainlink oracle on BSC | ~30 seconds |
| Round pool sizes | PancakeSwap contract storage | Real-time |
| Lock/close prices | PancakeSwap contract storage | At round boundaries |
| Round epoch | PancakeSwap contract storage | Every 5 minutes |

---

## System Configuration Parameters (server.py)

| Parameter       | Value  | Description |
|-----------------|--------|-------------|
| WAVELET         | db4    | Daubechies 4 wavelet family |
| LEVEL           | 2      | DWT decomposition depth |
| HISTORY         | 400    | Chainlink rounds kept in memory (~200 min) |
| CHUNK           | 20     | Batch size for RPC calls |
| SLOPE_THRESH    | 0.05   | DWT slope threshold for UP/DOWN signal |
| LOOKBACK        | 3      | Rounds to average slope over |
| PANCAKE_SEC     | 300    | Round duration in seconds |
| HURST_WINDOW    | 50     | Rounds used for Hurst calculation |
| MARKOV_HISTORY  | 30     | Past rounds for Markov matrix |
| KELLY_FRACTION  | 0.25   | Quarter-Kelly safety factor |
| MIN_EV          | 0.0    | Minimum EV to bet |
| MIN_CONFIDENCE  | 0.52   | Minimum fuzzy confidence to bet |
| MIN_POOL_BNB    | 0.005  | Minimum pool BNB before evaluating |
| MIN_POOL_TIME   | 30     | Seconds into round before evaluating |
| REFRESH_SEC     | 10     | Dashboard refresh interval |
| DECISION_WINDOW | 30     | Seconds before lock to snapshot decision |

---

## Win Rate Interpretation

The win rate shown in the dashboard is calculated from completed rounds only:
  win_rate = wins / (wins + losses) × 100%

SKIP rounds are excluded from win rate calculation.

**Break-even win rate** (accounting for 3% house fee):
  At equal multipliers (2x): need >51.5% to profit
  At 3x multiplier: need >37.5% to profit
  At 5x multiplier: need >25.0% to profit

This is why EV arbitrage is powerful — a high multiplier means you can
profit even with a below-50% win rate.

---

## Backtest Methodology

The 5-minute backtest uses real Chainlink price history:

1. Divide history into non-overlapping 5-minute buckets
2. For each bucket, take the DWT signal at the LAST Chainlink tick
3. Locked price = price at that last tick
4. Close price = first Chainlink tick in the NEXT bucket
5. Compare prediction vs actual direction → WIN / LOSS / SKIP

One bet per 5-minute round (mirrors real PancakeSwap behavior).
NEUTRAL signals (slope within threshold) are counted as SKIP.

---

## Known Limitations

1. **Oracle latency**: Chainlink commits prices every ~30s. The exact
   price at lock/close time may differ slightly from the last tick.

2. **Late-pool changes**: Pool ratios can shift dramatically in the
   last 10 seconds as other bots submit transactions. The system
   reads pool data at ~30s before lock, which may not reflect the
   final state.

3. **Small sample size**: Win rate is only meaningful after 50+
   completed rounds. Early sessions show high variance.

4. **Model uncertainty**: P values from the fuzzy system are estimates.
   The Quarter-Kelly safety factor accounts for this uncertainty.

5. **Market regime changes**: Hurst exponent is calculated on recent
   history. Sudden regime changes (news events, large trades) may
   not be detected immediately.

---

## File Structure

```
dwt_denoising/
├── server.py          — FastAPI backend, all strategy layers
├── wavelet_denoise.py — DWT de-noising function
├── index.html         — Web dashboard frontend
├── study_bnb_dwt.py   — Standalone matplotlib study tool
├── pancake_study.py   — Standalone matplotlib full strategy tool
├── requirements.txt   — Python dependencies
├── logs/              — Round log files (JSONL format)
│   └── rounds_YYYYMMDD_HHMMSS.jsonl
└── STRATEGY_THEORY.md — This document
```

---

*Generated: 2026-05-28*
*System: PancakeSwap Prediction Strategy Dashboard*
