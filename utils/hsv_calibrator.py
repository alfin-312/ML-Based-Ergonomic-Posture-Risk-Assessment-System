"""
hsv_calibrator.py — Interactive HSV trackbar tuner for red marker detection.

Run this script standalone while holding a red marker in front of the webcam.
Adjust the six sliders until the white blob in the "Mask" window matches your
marker and nothing else. Press  S  to print the final values (copy them into
config.py). Press  Q  to quit.
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import LOWER_RED_1, UPPER_RED_1, LOWER_RED_2, UPPER_RED_2


# ─────────────────────────────────────────────
#  Trackbar callback (no-op)
# ─────────────────────────────────────────────
def nothing(x):
    pass


# ─────────────────────────────────────────────
#  Setup windows + trackbars
# ─────────────────────────────────────────────
cv2.namedWindow("Calibrator", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Calibrator", 700, 60)

# Band 1 (hue 0–10)
cv2.createTrackbar("H_low_1",  "Calibrator", int(LOWER_RED_1[0]), 180, nothing)
cv2.createTrackbar("S_low_1",  "Calibrator", int(LOWER_RED_1[1]), 255, nothing)
cv2.createTrackbar("V_low_1",  "Calibrator", int(LOWER_RED_1[2]), 255, nothing)
cv2.createTrackbar("H_high_1", "Calibrator", int(UPPER_RED_1[0]), 180, nothing)
cv2.createTrackbar("S_high_1", "Calibrator", int(UPPER_RED_1[1]), 255, nothing)
cv2.createTrackbar("V_high_1", "Calibrator", int(UPPER_RED_1[2]), 255, nothing)

# Band 2 (hue 165–180)
cv2.createTrackbar("H_low_2",  "Calibrator", int(LOWER_RED_2[0]), 180, nothing)
cv2.createTrackbar("S_low_2",  "Calibrator", int(LOWER_RED_2[1]), 255, nothing)
cv2.createTrackbar("V_low_2",  "Calibrator", int(LOWER_RED_2[2]), 255, nothing)
cv2.createTrackbar("H_high_2", "Calibrator", int(UPPER_RED_2[0]), 180, nothing)
cv2.createTrackbar("S_high_2", "Calibrator", int(UPPER_RED_2[1]), 255, nothing)
cv2.createTrackbar("V_high_2", "Calibrator", int(UPPER_RED_2[2]), 255, nothing)


# ─────────────────────────────────────────────
#  Webcam
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit(1)

print("\n=== HSV Calibrator ===")
print("Adjust sliders so the mask highlights ONLY your red markers.")
print("Press  S  to save values  |  Q  to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Read trackbar positions
    def tb(name): return cv2.getTrackbarPos(name, "Calibrator")

    lower1 = np.array([tb("H_low_1"),  tb("S_low_1"),  tb("V_low_1")])
    upper1 = np.array([tb("H_high_1"), tb("S_high_1"), tb("V_high_1")])
    lower2 = np.array([tb("H_low_2"),  tb("S_low_2"),  tb("V_low_2")])
    upper2 = np.array([tb("H_high_2"), tb("S_high_2"), tb("V_high_2")])

    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask   = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2),
    )

    # Clean mask
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Overlay blobs on frame
    result = frame.copy()
    result[mask > 0] = [0, 0, 255]   # highlight detected region in red

    # Info overlay
    info = f"Band1: H[{lower1[0]}-{upper1[0]}] S[{lower1[1]}-{upper1[1]}] V[{lower1[2]}-{upper1[2]}]"
    cv2.putText(result, info, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (255, 255, 0), 1, cv2.LINE_AA)
    info2 = f"Band2: H[{lower2[0]}-{upper2[0]}] S[{lower2[1]}-{upper2[1]}] V[{lower2[2]}-{upper2[2]}]"
    cv2.putText(result, info2, (8, 46), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(result, "S = save values | Q = quit", (8, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    cv2.imshow("Result (red = detected)", result)
    cv2.imshow("Mask",  mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        print("\n=== Copy these values into utils/config.py ===")
        print(f"LOWER_RED_1 = np.array([{lower1[0]}, {lower1[1]}, {lower1[2]}])")
        print(f"UPPER_RED_1 = np.array([{upper1[0]}, {upper1[1]}, {upper1[2]}])")
        print(f"LOWER_RED_2 = np.array([{lower2[0]}, {lower2[1]}, {lower2[2]}])")
        print(f"UPPER_RED_2 = np.array([{upper2[0]}, {upper2[1]}, {upper2[2]}])")
        print()
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
