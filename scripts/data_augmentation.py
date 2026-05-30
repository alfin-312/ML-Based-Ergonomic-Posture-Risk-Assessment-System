"""
data_augmentation.py — Augment posture_features.csv by adding Gaussian noise.

Produces posture_features_augmented.csv with ~5× the original samples,
while preserving the original rows unchanged (noise std is small).

Usage
-----
    python scripts/data_augmentation.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import CSV_PATH, CSV_AUG_PATH, FEATURE_COLUMNS, LABEL_NAMES

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
AUGMENTATION_FACTOR = 5          # number of noisy copies per original row
NOISE_SCALE = {                  # std-dev of Gaussian noise per feature
    "shoulder_angle":    0.6,    # degrees
    "torso_angle":       0.8,    # degrees
    "shoulder_distance": 2.5,    # pixels
    "vertical_offset":   2.0,    # pixels
}

np.random.seed(42)

# ─────────────────────────────────────────────
#  Load
# ─────────────────────────────────────────────
if not os.path.exists(CSV_PATH):
    print(f"[ERROR] Source CSV not found: {CSV_PATH}")
    print("  Run feature_extraction.py first to collect data.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print(f"[INFO] Loaded {len(df)} samples from {CSV_PATH}")
print(f"       Class distribution: {df['label'].value_counts().to_dict()}\n")

# ─────────────────────────────────────────────
#  Augment
# ─────────────────────────────────────────────
augmented_rows = [df]   # start with original

for copy_idx in range(1, AUGMENTATION_FACTOR + 1):
    noisy = df[FEATURE_COLUMNS].copy().astype(float)
    for col in FEATURE_COLUMNS:
        noise = np.random.normal(0, NOISE_SCALE[col], size=len(noisy))
        noisy[col] += noise

    noisy["label"] = df["label"].values
    augmented_rows.append(noisy)
    print(f"  [+] Generated augmented copy {copy_idx}/{AUGMENTATION_FACTOR}")

augmented_df = pd.concat(augmented_rows, ignore_index=True)

# ─────────────────────────────────────────────
#  Save
# ─────────────────────────────────────────────
os.makedirs(os.path.dirname(CSV_AUG_PATH), exist_ok=True)
augmented_df.to_csv(CSV_AUG_PATH, index=False)

print(f"\n[OK] Augmented dataset saved → {CSV_AUG_PATH}")
print(f"     Total samples : {len(augmented_df)}")
print(f"     Class distribution: {augmented_df['label'].value_counts().to_dict()}")
