"""
marker_detection.py — Visual debugging / validation of red marker detection.

Run this script first to verify that your 3 red markers are being detected
reliably before proceeding to data collection.

Controls
--------
  Q  — quit
"""

import cv2
import sys
import os
import math
import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_utils import detect_red_markers, extract_features, draw_skeleton
from utils.config import LABEL_COLORS


# ─────────────────────────────────────────────
#  Webcam
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam. Check that it is connected.")
    sys.exit(1)

print("\n=== Marker Detection Test ===")
print("Hold 3 red markers in front of the webcam:")
print("  • Left shoulder (LS)")
print("  • Right shoulder (RS)")
print("  • Torso / lower-back (T)\n")
print("Press  Q  to quit.\n")

MARKER_LABELS = ["LS", "RS", "T"]

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Failed to grab frame.")
        break

    points, mask = detect_red_markers(frame)
    n = len(points)

    # ── Status overlay ──────────────────────────────────────────────────────
    status_color = (0, 220, 0) if n == 3 else (0, 100, 240)
    status_text  = f"Detected: {n} marker(s)"
    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, status_color, 2, cv2.LINE_AA)

    # ── Draw each detected marker ───────────────────────────────────────────
    for i, (cx, cy) in enumerate(points[:3]):
        color = list(LABEL_COLORS.values())[i % 3]
        cv2.circle(frame, (cx, cy), 10, color, -1)
        cv2.circle(frame, (cx, cy), 11, (255, 255, 255), 1)
        tag = MARKER_LABELS[i] if i < 3 else str(i)
        cv2.putText(frame, tag, (cx + 14, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"({cx},{cy})", (cx + 14, cy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # ── Draw skeleton when all 3 are found ─────────────────────────────────
    if n >= 3:
        feats = extract_features(points)
        if feats:
            draw_skeleton(frame, feats)

            # Feature readout
            lines = [
                f"Shoulder slope : {feats['shoulder_angle']:+.1f} deg",
                f"Torso lean     : {feats['torso_angle']:+.1f} deg",
                f"Shoulder dist  : {feats['shoulder_distance']:.1f} px",
                f"Vertical offset: {feats['vertical_offset']:.1f} px",
            ]
            for j, txt in enumerate(lines):
                cv2.putText(frame, txt, (10, 60 + j * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230),
                            1, cv2.LINE_AA)
    else:
        hint = "  Waiting for 3 markers..." if n < 3 else ""
        cv2.putText(frame, hint, (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 180, 255), 1)

    # ── Instruction footer ──────────────────────────────────────────────────
    h = frame.shape[0]
    cv2.putText(frame, "Q = quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    cv2.imshow("Marker Detection", frame)
    cv2.imshow("Mask (debug)",     mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Closed.")
