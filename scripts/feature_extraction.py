"""
feature_extraction.py — Collect labelled posture data for ML training.

Instructions
------------
1. Place 3 red markers on your body:
     • Left shoulder
     • Right shoulder
     • Torso / lower-back
2. Adopt a posture and press the corresponding key to label it:
     1 → Good posture
     2 → Medium risk
     3 → High risk
     Q → Save CSV and quit

Data is APPENDED to data/posture_features.csv so you can run multiple sessions.
"""

import cv2
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_utils import detect_red_markers, extract_features, draw_skeleton
from utils.config import (
    CSV_PATH, FEATURE_COLUMNS, LABEL_NAMES, LABEL_COLORS,
)

# ─────────────────────────────────────────────
#  Setup
# ─────────────────────────────────────────────
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    sys.exit(1)

data = []
class_counts = {0: 0, 1: 0, 2: 0}

print("\n=== Posture Data Collection ===")
print("Press  1  → Good posture  |  2  → Medium risk  |  3  → High risk")
print("Press  Q  → Save CSV and quit\n")

# ─────────────────────────────────────────────
#  HUD helpers
# ─────────────────────────────────────────────
def draw_hud(frame, n_markers):
    h, w = frame.shape[:2]

    # Instructions bar (top)
    bar = np.zeros((50, w, 3), dtype=np.uint8)
    texts = [
        ("1=Good", LABEL_COLORS[0]),
        ("2=Medium", LABEL_COLORS[1]),
        ("3=High", LABEL_COLORS[2]),
        ("Q=Save+Quit", (180, 180, 180)),
    ]
    x = 10
    for t, c in texts:
        cv2.putText(bar, t, (x, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2)
        x += cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] + 20

    frame[:50, :] = cv2.addWeighted(
        frame[:50, :], 0.3, bar, 0.7, 0
    )

    # Marker count indicator
    mc = f"Markers: {n_markers}/3"
    col = (0, 220, 0) if n_markers == 3 else (0, 120, 255)
    cv2.putText(frame, mc, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

    # Sample count per class (bottom right)
    for i, (lbl, cnt) in enumerate(class_counts.items()):
        txt = f"{LABEL_NAMES[lbl]}: {cnt} samples"
        cv2.putText(frame, txt, (w - 260, h - 60 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, LABEL_COLORS[lbl], 1)

    # Total
    total = sum(class_counts.values())
    cv2.putText(frame, f"Total: {total}", (w - 110, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)


# ─────────────────────────────────────────────
#  Last-sample flash feedback
# ─────────────────────────────────────────────
flash_frames = 0
flash_label  = None


# ─────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    points, mask = detect_red_markers(frame)
    feats        = extract_features(points) if len(points) >= 3 else None

    if feats:
        draw_skeleton(frame, feats)

    draw_hud(frame, len(points))

    # Flash feedback on sample save
    if flash_frames > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0),
                      (frame.shape[1], frame.shape[0]),
                      LABEL_COLORS.get(flash_label, (255, 255, 255)), -1)
        alpha = flash_frames / 8
        cv2.addWeighted(overlay, alpha * 0.35, frame, 1 - alpha * 0.35, 0, frame)
        flash_frames -= 1

    cv2.imshow("Feature Extraction | Posture Collection", frame)
    cv2.imshow("Mask (debug)", mask)

    key = cv2.waitKey(1) & 0xFF

    if key in [ord('1'), ord('2'), ord('3')]:
        if feats is None:
            print("[WARN] Need exactly 3 markers to record a sample.")
            continue

        label = int(chr(key)) - 1   # '1'→0, '2'→1, '3'→2
        row = [feats[col] for col in FEATURE_COLUMNS] + [label]
        data.append(row)
        class_counts[label] += 1

        flash_frames = 8
        flash_label  = label
        print(f"  [+] Saved — {LABEL_NAMES[label]} "
              f"| slope={feats['shoulder_angle']:+.1f}° "
              f"| lean={feats['torso_angle']:+.1f}°  "
              f"(total {class_counts[label]} for this class)")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
#  Save / Append CSV
# ─────────────────────────────────────────────
if data:
    new_df = pd.DataFrame(data, columns=FEATURE_COLUMNS + ["label"])

    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(CSV_PATH, index=False)
    print(f"\n[OK] Saved {len(data)} new samples → {CSV_PATH}")
    print(f"     Total samples in file: {len(combined)}")
    print(f"     Class breakdown: {combined['label'].value_counts().to_dict()}")
else:
    print("\n[INFO] No new samples collected. CSV unchanged.")
