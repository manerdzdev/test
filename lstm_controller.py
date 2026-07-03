"""
Step 2 & 3: LSTM Controller — Training Script
===============================================
Trains the DynamicParamController LSTM on the audit log.
Designed to run on Lightning.ai free GPU (A10G / T4).

Architecture:
  Input:  sequence of K=12 rounds, each with (X_t, θ_t, C_t) features
  LSTM:   2-layer, hidden_dim=128
  Heads:
    - wavelet_level_head:   3-class softmax  (levels 1,2,3)
    - hurst_window_head:    4-class softmax  (50,100,200,400)
    - fuzzy_threshold_head: sigmoid → [0,1]  (continuous)
    - slope_thresh_head:    sigmoid → [0,1]  (continuous)
    - lookback_head:        2-class softmax  (2,4)

Loss:
  L_total = λ1 * CE(wavelet) + λ2 * CE(hurst_window) + λ3 * CE(lookback)
          + λ4 * MSE(fuzzy_threshold) + λ5 * MSE(slope_thresh)

Usage on Lightning.ai:
  1. Upload dwt_denoising/audit_log.jsonl to Lightning.ai storage
  2. pip install torch numpy
  3. python lstm_controller.py --train
  4. Download lstm_controller.pt back to dwt_denoising/

Local inference (no GPU needed):
  python lstm_controller.py --infer  (uses CPU)
"""

import json
import numpy as np
import argparse
import os
import sys
from datetime import datetime

# ── Parameter space (must match audit_log_generator.py) ───────────────────────
WAVELET_LEVELS    = [1, 2, 3]
HURST_WINDOWS     = [400]              # fixed — short windows always give H=0.5 (broken)
FUZZY_THRESHOLDS  = [0.30, 0.45, 0.60]   # continuous — normalized to [0,1]
SLOPE_THRESHOLDS  = [0.03, 0.07, 0.12]   # continuous — normalized to [0,1]
LOOKBACKS         = [2, 4]

# Normalization helpers
FUZZY_MIN, FUZZY_MAX   = 0.20, 0.80
SLOPE_MIN, SLOPE_MAX   = 0.01, 0.20

def normalize_fuzzy(v):  return (v - FUZZY_MIN) / (FUZZY_MAX - FUZZY_MIN)
def normalize_slope(v):  return (v - SLOPE_MIN) / (SLOPE_MAX - SLOPE_MIN)
def denormalize_fuzzy(v): return v * (FUZZY_MAX - FUZZY_MIN) + FUZZY_MIN
def denormalize_slope(v): return v * (SLOPE_MAX - SLOPE_MIN) + SLOPE_MIN

# ── Feature dimensions ─────────────────────────────────────────────────────────
# X features (market state): 14
X_FEATURES = [
    "vol_20", "vol_50", "trend", "momentum", "skewness", "kurtosis",
    "pool_up_pct", "pool_imbalance", "mult_up", "mult_down",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
# θ features (parameters): 5 (normalized)
THETA_FEATURES = [
    "wavelet_level_norm",   # (level-1)/2
    "hurst_window_norm",    # (window-50)/350
    "fuzzy_threshold_norm", # normalized
    "slope_thresh_norm",    # normalized
    "lookback_norm",        # (lookback-2)/2
]
# C features (performance): 4
C_FEATURES = [
    "pnl",          # realized PnL (-1 to +max_mult)
    "price_error",  # |smoothed - close|
    "dir_correct",  # 1/0/-1 (correct/wrong/no bet)
    "won",          # 1/0/-1
]

INPUT_DIM = len(X_FEATURES) + len(THETA_FEATURES) + len(C_FEATURES)  # 23
SEQ_LEN   = 12   # K = 12 rounds lookback (~1 hour)

print(f"Input dim: {INPUT_DIM}  Seq len: {SEQ_LEN}")


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_dataset(audit_log_path, max_records=None):
    """
    Load compact audit log (one record per round) and build (sequence, target) pairs.

    Each record has:
      - X: market state dict
      - best_theta: optimal parameter combo for that round (training target)
      - combo_pnls: PnL for all 216 combos (for analysis)

    For each round t, input = last K rounds' (X, θ_used, C_observed) vectors.
    Target = best_theta for round t.

    Returns:
      sequences: np.ndarray (N, K, INPUT_DIM)
      targets:   dict of np.ndarray per head
    """
    print(f"Loading {audit_log_path}...")

    records = []
    count = 0
    with open(audit_log_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except:
                pass
            count += 1
            if max_records and count >= max_records:
                break

    print(f"Loaded {len(records):,} round records")

    def make_x_vec(rec):
        X = rec.get("X", {})
        return np.array([X.get(f, 0.0) for f in X_FEATURES], dtype=np.float32)

    def make_theta_vec(rec):
        # Use best_theta as the "params used" for sequence context
        th = rec.get("best_theta", {})
        return np.array([
            (th.get("wavelet_level", 1) - 1) / 2.0,
            (th.get("hurst_window", 400) - 50) / 350.0,  # default 400 → normalizes to 1.0
            normalize_fuzzy(th.get("fuzzy_threshold", 0.45)),
            normalize_slope(th.get("slope_thresh", 0.07)),
            (th.get("lookback", 3) - 2) / 2.0,
        ], dtype=np.float32)

    def make_c_vec(rec):
        pnl       = float(rec.get("best_pnl", 0.0))
        price_err = float(rec.get("price_error", 0.0))
        # Directional: did best combo predict correctly?
        actual = rec.get("actual_outcome")
        best_th = rec.get("best_theta", {})
        # We don't store signal directly, use pnl sign as proxy
        dir_c = 1.0 if pnl > 0 else (-1.0 if pnl < 0 else 0.0)
        won   = 1.0 if pnl > 0 else (-1.0 if pnl < 0 else 0.0)
        return np.array([
            np.clip(pnl / 5.0, -1.0, 1.0),
            np.clip(price_err / 10.0, 0.0, 1.0),
            dir_c, won,
        ], dtype=np.float32)

    def make_target(rec):
        th = rec.get("best_theta", {})
        return {
            "wavelet_level":    WAVELET_LEVELS.index(th.get("wavelet_level", 1)),
            "hurst_window":     HURST_WINDOWS.index(th.get("hurst_window", 400)) if th.get("hurst_window", 400) in HURST_WINDOWS else 0,
            "lookback":         LOOKBACKS.index(th.get("lookback", 2)),
            "fuzzy_threshold":  normalize_fuzzy(th.get("fuzzy_threshold", 0.45)),
            "slope_thresh":     normalize_slope(th.get("slope_thresh", 0.07)),
        }

    # Build sequences
    sequences = []
    targets   = {"wavelet_level": [], "hurst_window": [], "lookback": [],
                 "fuzzy_threshold": [], "slope_thresh": []}

    for i in range(SEQ_LEN, len(records)):
        seq = []
        for j in range(i - SEQ_LEN, i):
            rec = records[j]
            x_vec     = make_x_vec(rec)
            theta_vec = make_theta_vec(rec)
            c_vec     = make_c_vec(rec)
            seq.append(np.concatenate([x_vec, theta_vec, c_vec]))
        sequences.append(np.stack(seq))

        tgt = make_target(records[i])
        for k, v in tgt.items():
            targets[k].append(v)

    sequences = np.stack(sequences).astype(np.float32)
    for k in targets:
        targets[k] = np.array(targets[k])

    print(f"Dataset: {sequences.shape[0]:,} sequences  shape={sequences.shape}")
    return sequences, targets


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    class DynamicParamController(nn.Module):
        def __init__(self, input_dim=INPUT_DIM, hidden_dim=128, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
            )
            self.norm = nn.LayerNorm(hidden_dim)

            # Discrete heads (classification)
            self.wavelet_level_head = nn.Sequential(
                nn.Linear(hidden_dim, 64), nn.ReLU(),
                nn.Linear(64, len(WAVELET_LEVELS))   # 3 classes
            )
            self.hurst_window_head = nn.Sequential(
                nn.Linear(hidden_dim, 64), nn.ReLU(),
                nn.Linear(64, len(HURST_WINDOWS))    # 1 class (fixed at 400)
            )
            self.lookback_head = nn.Sequential(
                nn.Linear(hidden_dim, 32), nn.ReLU(),
                nn.Linear(32, len(LOOKBACKS))         # 2 classes
            )

            # Continuous heads (regression → sigmoid → [0,1])
            self.fuzzy_threshold_head = nn.Sequential(
                nn.Linear(hidden_dim, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Sigmoid()
            )
            self.slope_thresh_head = nn.Sequential(
                nn.Linear(hidden_dim, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Sigmoid()
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            h = self.norm(lstm_out[:, -1, :])   # last timestep
            return {
                "wavelet_level":    self.wavelet_level_head(h),
                "hurst_window":     self.hurst_window_head(h),
                "lookback":         self.lookback_head(h),
                "fuzzy_threshold":  self.fuzzy_threshold_head(h).squeeze(-1),
                "slope_thresh":     self.slope_thresh_head(h).squeeze(-1),
            }

    return DynamicParamController()


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    audit_log_path="dwt_denoising/audit_log.jsonl",
    model_path="dwt_denoising/lstm_controller.pt",
    epochs=50,
    batch_size=256,
    lr=1e-3,
    val_split=0.1,
    max_records=None,
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    sequences, targets = load_dataset(audit_log_path, max_records=max_records)

    X = torch.tensor(sequences)
    y_wl  = torch.tensor(targets["wavelet_level"],   dtype=torch.long)
    y_hw  = torch.tensor(targets["hurst_window"],    dtype=torch.long)
    y_lb  = torch.tensor(targets["lookback"],        dtype=torch.long)
    y_ft  = torch.tensor(targets["fuzzy_threshold"], dtype=torch.float32)
    y_st  = torch.tensor(targets["slope_thresh"],    dtype=torch.float32)

    dataset = TensorDataset(X, y_wl, y_hw, y_lb, y_ft, y_st)
    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ce_loss  = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    # Loss weights
    λ = {"wavelet": 1.0, "hurst": 1.0, "lookback": 0.5, "fuzzy": 2.0, "slope": 2.0}

    best_val_loss = float("inf")
    print(f"\nTraining {epochs} epochs, batch={batch_size}, lr={lr}")
    print(f"Train: {n_train:,}  Val: {n_val:,}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            xb, ywl, yhw, ylb, yft, yst = [b.to(device) for b in batch]
            optimizer.zero_grad()
            out = model(xb)
            loss = (
                λ["wavelet"] * ce_loss(out["wavelet_level"], ywl) +
                λ["hurst"]   * ce_loss(out["hurst_window"],  yhw) +
                λ["lookback"]* ce_loss(out["lookback"],      ylb) +
                λ["fuzzy"]   * mse_loss(out["fuzzy_threshold"], yft) +
                λ["slope"]   * mse_loss(out["slope_thresh"],    yst)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        wl_acc = hw_acc = lb_acc = 0
        with torch.no_grad():
            for batch in val_loader:
                xb, ywl, yhw, ylb, yft, yst = [b.to(device) for b in batch]
                out = model(xb)
                loss = (
                    λ["wavelet"] * ce_loss(out["wavelet_level"], ywl) +
                    λ["hurst"]   * ce_loss(out["hurst_window"],  yhw) +
                    λ["lookback"]* ce_loss(out["lookback"],      ylb) +
                    λ["fuzzy"]   * mse_loss(out["fuzzy_threshold"], yft) +
                    λ["slope"]   * mse_loss(out["slope_thresh"],    yst)
                )
                val_loss += loss.item() * len(xb)
                wl_acc += (out["wavelet_level"].argmax(1) == ywl).sum().item()
                hw_acc += (out["hurst_window"].argmax(1)  == yhw).sum().item()
                lb_acc += (out["lookback"].argmax(1)      == ylb).sum().item()
        val_loss /= n_val
        wl_acc   /= n_val
        hw_acc   /= n_val
        lb_acc   /= n_val

        scheduler.step()

        marker = " ← best" if val_loss < best_val_loss else ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "epoch":       epoch,
                "val_loss":    val_loss,
                "input_dim":   INPUT_DIM,
                "seq_len":     SEQ_LEN,
            }, model_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"wl_acc={wl_acc:.2f}  hw_acc={hw_acc:.2f}  lb_acc={lb_acc:.2f}"
                  f"{marker}")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved: {model_path}")


# ── Inference (used by server.py at runtime) ───────────────────────────────────

class LSTMInference:
    """
    Lightweight inference wrapper — no GPU needed.
    Used by server.py to get optimized parameters every round.
    Falls back to default parameters if torch is not installed or model not found.
    """
    def __init__(self, model_path="dwt_denoising/lstm_controller.pt"):
        self.device = None
        self.torch  = None
        self.model  = None
        self.model_path = model_path
        self.history = []
        self._load()

    def _load(self):
        try:
            import torch
            self.torch  = torch
            self.device = torch.device("cpu")
        except ImportError:
            print(f"[LSTM] PyTorch not installed — using default params")
            return

        if not os.path.exists(self.model_path):
            print(f"[LSTM] Model not found at {self.model_path} — using defaults")
            return
        try:
            ckpt = self.torch.load(self.model_path, map_location=self.device)
            self.model = build_model().to(self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            print(f"[LSTM] Loaded model from {self.model_path} "
                  f"(epoch={ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss',0):.4f})")
        except Exception as e:
            print(f"[LSTM] Failed to load model: {e}")
            self.model = None

    def push(self, X_dict, theta_dict, C_dict):
        """
        Add a completed round to the rolling history buffer.
        Call this after each round closes.
        """
        x_vec = np.array([X_dict.get(f, 0.0) for f in X_FEATURES], dtype=np.float32)
        theta_vec = np.array([
            (theta_dict.get("wavelet_level", 1) - 1) / 2.0,
            1.0,  # hurst_window fixed at 400 — normalize: (400-50)/350 = 1.0
            normalize_fuzzy(theta_dict.get("fuzzy_threshold", 0.45)),
            normalize_slope(theta_dict.get("slope_thresh", 0.07)),
            (theta_dict.get("lookback", 3) - 2) / 2.0,
        ], dtype=np.float32)
        pnl       = float(C_dict.get("pnl", 0.0))
        price_err = float(C_dict.get("price_error", 0.0))
        dir_c     = 1.0 if C_dict.get("dir_correct") == True else (-1.0 if C_dict.get("dir_correct") == False else 0.0)
        won       = 1.0 if C_dict.get("won") == True else (-1.0 if C_dict.get("won") == False else 0.0)
        c_vec = np.array([
            np.clip(pnl / 5.0, -1.0, 1.0),
            np.clip(price_err / 10.0, 0.0, 1.0),
            dir_c, won,
        ], dtype=np.float32)

        self.history.append(np.concatenate([x_vec, theta_vec, c_vec]))
        if len(self.history) > SEQ_LEN * 2:
            self.history = self.history[-SEQ_LEN * 2:]

    def predict(self):
        """
        Predict optimal parameters for the next round.
        Returns dict with all parameter values (or defaults if model not ready).
        """
        defaults = {
            "wavelet_level":   1,
            "hurst_window":    400,
            "fuzzy_threshold": 0.45,
            "slope_thresh":    0.07,
            "lookback":        3,
            "source":          "default",
        }

        if self.model is None or self.torch is None or len(self.history) < SEQ_LEN:
            defaults["source"] = f"default (need {SEQ_LEN} rounds, have {len(self.history)})"
            return defaults

        try:
            seq = np.stack(self.history[-SEQ_LEN:])
            x   = self.torch.tensor(seq).unsqueeze(0)

            with self.torch.no_grad():
                out = self.model(x)

            wl_idx = int(out["wavelet_level"].argmax(1).item())
            hw_idx = int(out["hurst_window"].argmax(1).item())
            lb_idx = int(out["lookback"].argmax(1).item())
            ft_raw = float(out["fuzzy_threshold"].item())
            st_raw = float(out["slope_thresh"].item())

            return {
                "wavelet_level":   WAVELET_LEVELS[wl_idx],
                "hurst_window":    400,  # always 400
                "fuzzy_threshold": round(denormalize_fuzzy(ft_raw), 3),
                "slope_thresh":    round(denormalize_slope(st_raw), 3),
                "lookback":        LOOKBACKS[lb_idx],
                "source":          "lstm",
                "raw": {
                    "wl_probs": out["wavelet_level"].softmax(1).squeeze().tolist(),
                    "hw_probs": out["hurst_window"].softmax(1).squeeze().tolist(),
                    "lb_probs": out["lookback"].softmax(1).squeeze().tolist(),
                    "ft_raw":   round(ft_raw, 4),
                    "st_raw":   round(st_raw, 4),
                }
            }
        except Exception as e:
            defaults["source"] = f"error: {e}"
            return defaults


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",       action="store_true", help="Train the model")
    parser.add_argument("--generate",    action="store_true", help="Generate audit log first")
    parser.add_argument("--infer",       action="store_true", help="Test inference")
    parser.add_argument("--audit-log",   default="dwt_denoising/audit_log.jsonl")
    parser.add_argument("--model-path",  default="dwt_denoising/lstm_controller.pt")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch-size",  type=int, default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    if args.generate:
        sys.path.insert(0, ".")
        from dwt_denoising.audit_log_generator import generate_audit_log
        generate_audit_log(out_file=args.audit_log)

    if args.train:
        train(
            audit_log_path=args.audit_log,
            model_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_records=args.max_records,
        )

    if args.infer:
        inf = LSTMInference(args.model_path)
        # Push some dummy history
        for _ in range(SEQ_LEN):
            inf.push(
                X_dict={"vol_20": 0.3, "pool_up_pct": 0.55, "pool_imbalance": 0.1},
                theta_dict={"wavelet_level": 1, "hurst_window": 200,
                            "fuzzy_threshold": 0.45, "slope_thresh": 0.07, "lookback": 3},
                C_dict={"pnl": -1.0, "price_error": 0.5, "dir_correct": False, "won": False},
            )
        params = inf.predict()
        print(f"\nPredicted params: {json.dumps(params, indent=2)}")
