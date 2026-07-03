"""
Lightning.ai Training Script
==============================
Run this file on Lightning.ai Studio (free GPU: A10G or T4).

SETUP STEPS ON LIGHTNING.AI:
─────────────────────────────
1. Go to https://lightning.ai  → Create free account
2. New Studio → "Blank" → Select GPU (A10G free tier)
3. In the terminal:
     pip install torch numpy
4. Upload files:
     - dwt_denoising/audit_log.jsonl      (generated locally first)
     - dwt_denoising/lstm_controller.py
     - dwt_denoising/train_on_lightning.py
5. Run:
     python train_on_lightning.py
6. Download:
     - lstm_controller.pt  → put in dwt_denoising/

GENERATE AUDIT LOG LOCALLY FIRST:
──────────────────────────────────
  python dwt_denoising/audit_log_generator.py
  # Takes ~15 min, produces ~500 MB audit_log.jsonl

EXPECTED TRAINING TIME ON A10G:
─────────────────────────────────
  ~105,000 rounds × 48 combos = ~5M records
  50 epochs, batch=512 → ~20 min on A10G
"""

import os
import sys

# ── Paths (adjust if needed on Lightning.ai) ──────────────────────────────────
AUDIT_LOG  = "audit_log.jsonl"       # upload this to Lightning.ai
MODEL_OUT  = "lstm_controller.pt"    # download this after training

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS     = 60
BATCH_SIZE = 512
LR         = 1e-3
VAL_SPLIT  = 0.1
HIDDEN_DIM = 128

# ── Install deps if needed ────────────────────────────────────────────────────
try:
    import torch
    print(f"PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    os.system("pip install torch --quiet")
    import torch

import numpy as np
import json
from datetime import datetime

# ── Import the controller (same file, just call train()) ─────────────────────
sys.path.insert(0, ".")

# Inline the key pieces so this file is self-contained on Lightning.ai
from lstm_controller import (
    load_dataset, build_model, train,
    INPUT_DIM, SEQ_LEN,
    WAVELET_LEVELS, HURST_WINDOWS, LOOKBACKS,
    normalize_fuzzy, normalize_slope,
)

if __name__ == "__main__":
    print("=" * 60)
    print("  PancakeSwap LSTM Controller — Training")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not os.path.exists(AUDIT_LOG):
        print(f"ERROR: {AUDIT_LOG} not found.")
        print("Generate it locally with: python audit_log_generator.py")
        sys.exit(1)

    train(
        audit_log_path=AUDIT_LOG,
        model_path=MODEL_OUT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        val_split=VAL_SPLIT,
    )

    print(f"\n✓ Training complete. Download {MODEL_OUT} and put it in dwt_denoising/")
