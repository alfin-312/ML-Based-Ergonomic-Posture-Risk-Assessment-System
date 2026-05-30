"""
real_time_inference.py — Live posture risk assessment using trained SVM model.

Prerequisites
-------------
1. Run  feature_extraction.py   to collect training data
2. Run  data_augmentation.py    to expand the dataset
3. Run  train_classifier.py     to build and save the model
4. Then run this script.

Controls
--------
  S  — save a snapshot of the current frame to output/snapshots/
  Q  — quit
"""

import cv2
import numpy as np
import joblib
import os
import sys
import time
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_utils import (
    detect_red_markers, extract_features,
    draw_skeleton, draw_risk_badge, draw_session_stats,
)
from utils.config import (
    MODEL_SVM_PATH, FEATURE_COLUMNS, LABEL_NAMES, LABEL_COLORS,
    OUTPUT_DIR,
)

# ─────────────────────────────────────────────
#  Load Model
# ─────────────────────────────────────────────
if not os.path.exists(MODEL_SVM_PATH):
    print(f"[ERROR] No trained model found at: {MODEL_SVM_PATH}")
    print("  Run train_classifier.py first.")
    sys.exit(1)

model = joblib.load(MODEL_SVM_PATH)
print(f"[OK] Loaded model from {MODEL_SVM_PATH}\n")

# ─────────────────────────────────────────────
#  Snapshot folder
# ─────────────────────────────────────────────
SNAP_DIR = os.path.join(OUTPUT_DIR, "snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  Webcam
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    sys.exit(1)

# ─────────────────────────────────────────────
#  State
# ─────────────────────────────────────────────
session_counts   = {0: 0, 1: 0, 2: 0}
SMOOTHING_WINDOW = 8                         # vote over last N frames
recent_labels    = deque(maxlen=SMOOTHING_WINDOW)
snap_count       = 0
start_time       = time.time()

print("=== Real-time Posture Inference ===")
print("Place 3 red markers (Left shoulder / Right shoulder / Torso).")
print("Press  S  to save snapshot  |  Q  to quit\n")

# ─────────────────────────────────────────────
#  Helper — smoothed prediction
# ─────────────────────────────────────────────
def smoothed_label(label_id):
    """Return the majority vote from the recent_labels buffer."""
    recent_labels.append(label_id)
    return max(set(recent_labels), key=recent_labels.count)


# ─────────────────────────────────────────────
#  Main Loop
# ─────────────────────────────────────────────
last_label   = None
last_conf    = None
no_marker_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    points, mask = detect_red_markers(frame)
    feats        = extract_features(points) if len(points) >= 3 else None

    if feats:
        no_marker_frames = 0

        # Prepare feature vector
        feat_vec = np.array([[feats[col] for col in FEATURE_COLUMNS]])

        raw_label = int(model.predict(feat_vec)[0])

        # Confidence (probability if SVC has probability=True)
        try:
            proba = model.predict_proba(feat_vec)[0]
            confidence = float(proba[raw_label])
        except AttributeError:
            confidence = None

        # Temporal smoothing
        label    = smoothed_label(raw_label)
        last_label  = label
        last_conf   = confidence

        session_counts[label] += 1

        # Skeleton overlay (color = risk level)
        draw_skeleton(frame, feats, label_id=label)

        # Risk badge at top
        draw_risk_badge(frame, label, confidence)

    else:
        no_marker_frames += 1

        # Show last known label for a few frames, then "searching..."
        if last_label is not None and no_marker_frames < 30:
            draw_risk_badge(frame, last_label, last_conf)
        else:
            # Searching banner
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (60, 60, 60), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            txt = f"Searching markers ({len(points)}/3 found)..."
            cv2.putText(frame, txt, (12, 42),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (200, 200, 200), 2,
                        cv2.LINE_AA)

    # Session stats (bottom-right stacked bar)
    draw_session_stats(frame, session_counts)

    # Elapsed time (bottom-left)
    elapsed = int(time.time() - start_time)
    m, s = divmod(elapsed, 60)
    cv2.putText(frame, f"Session: {m:02d}:{s:02d}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    # Controls reminder
    cv2.putText(frame, "S=Snapshot  Q=Quit",
                (frame.shape[1] - 200, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    cv2.imshow("Ergonomic Posture Monitor", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_count += 1
        snap_path = os.path.join(SNAP_DIR, f"snapshot_{ts}_{snap_count:03d}.png")
        cv2.imwrite(snap_path, frame)
        print(f"[SNAP] Saved → {snap_path}")

    elif key == ord('q'):
        break

# ─────────────────────────────────────────────
#  Session Summary
# ─────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

total = sum(session_counts.values()) or 1
elapsed = int(time.time() - start_time)
m, s = divmod(elapsed, 60)

print("\n=== Session Summary ===")
print(f"Duration : {m:02d}:{s:02d}")
for lbl, cnt in session_counts.items():
    pct = 100 * cnt / total
    print(f"  {LABEL_NAMES[lbl]:<16}: {cnt:>5} frames  ({pct:.1f}%)")
print(f"  Total frames classified: {total}")
if snap_count:
    print(f"  Snapshots saved: {snap_count} → {SNAP_DIR}")