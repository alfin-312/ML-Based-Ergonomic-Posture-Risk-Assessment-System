"""
config.py — Central configuration for the Posture Risk Assessment System.
Edit HSV ranges here if your red markers look different under your lighting.
"""

import os

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CSV_PATH          = os.path.join(DATA_DIR,   "posture_features.csv")
CSV_AUG_PATH      = os.path.join(DATA_DIR,   "posture_features_augmented.csv")
MODEL_SVM_PATH    = os.path.join(MODELS_DIR, "posture_svm.pkl")
MODEL_RF_PATH     = os.path.join(MODELS_DIR, "posture_rf.pkl")
TRAINING_REPORT   = os.path.join(OUTPUT_DIR, "training_report.txt")

# ─────────────────────────────────────────────
#  Red Marker — HSV Ranges
#  Red wraps around 0° in hue, so we need two bands.
# ─────────────────────────────────────────────
import numpy as np

LOWER_RED_1 = np.array([0,   120, 100])
UPPER_RED_1 = np.array([10,  255, 255])

LOWER_RED_2 = np.array([165, 120, 100])
UPPER_RED_2 = np.array([180, 255, 255])

# ─────────────────────────────────────────────
#  Detection Parameters
# ─────────────────────────────────────────────
MIN_CONTOUR_AREA  = 400    # px² — ignore noise smaller than this
MORPH_KERNEL_SIZE = 5      # morphological kernel size
MIN_CIRCULARITY   = 0.35   # filter blobs that are too elongated (0–1)

# ─────────────────────────────────────────────
#  Classification Labels
# ─────────────────────────────────────────────
LABEL_NAMES = {
    0: "Good Posture",
    1: "Medium Risk",
    2: "High Risk",
}

# BGR colors for each label (used in OpenCV overlays)
LABEL_COLORS = {
    0: (50,  200, 50),    # green
    1: (0,   180, 255),   # orange
    2: (0,   50,  240),   # red
}

# ─────────────────────────────────────────────
#  Feature Column Names
# ─────────────────────────────────────────────
FEATURE_COLUMNS = [
    "shoulder_angle",
    "torso_angle",
    "shoulder_distance",
    "vertical_offset",
]
