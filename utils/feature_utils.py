"""
feature_utils.py — Shared helpers for marker detection, feature extraction,
                   and skeleton drawing. Used by feature_extraction.py and
                   real_time_inference.py.
"""

import cv2
import numpy as np
import math
import sys
import os

# Allow importing config from the utils package itself
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    LOWER_RED_1, UPPER_RED_1,
    LOWER_RED_2, UPPER_RED_2,
    MIN_CONTOUR_AREA, MORPH_KERNEL_SIZE, MIN_CIRCULARITY,
    LABEL_NAMES, LABEL_COLORS, FEATURE_COLUMNS,
)


# ─────────────────────────────────────────────
#  Red Mask
# ─────────────────────────────────────────────

def build_red_mask(hsv_frame):
    """Return a binary mask of red-coloured pixels."""
    m1 = cv2.inRange(hsv_frame, LOWER_RED_1, UPPER_RED_1)
    m2 = cv2.inRange(hsv_frame, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.bitwise_or(m1, m2)

    # Morphological clean-up
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)   # remove salt noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # fill small holes
    mask = cv2.dilate(mask, kernel, iterations=1)             # grow blobs slightly
    return mask


# ─────────────────────────────────────────────
#  Marker Detection
# ─────────────────────────────────────────────

def detect_red_markers(frame):
    """
    Detect up to N red circular markers in *frame*.

    Returns
    -------
    points  : list of (cx, cy) tuples, sorted top-to-bottom
    mask    : the processed binary mask (useful for debugging)
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = build_red_mask(hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    points = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        # Circularity filter  (4π·A / P²)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4 * math.pi * area) / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        points.append((cx, cy))

    # Sort top → bottom (lowest y = highest on screen)
    points.sort(key=lambda p: p[1])
    return points, mask


# ─────────────────────────────────────────────
#  Feature Extraction
# ─────────────────────────────────────────────

def extract_features(points):
    """
    Given exactly 3 sorted (top→bottom) points, compute posture features.

    Marker assignment (by vertical position after sorting):
        points[0], points[1] — the two shoulders (topmost)
        points[2]            — torso / lower-back marker

    Returns a dict with keys matching FEATURE_COLUMNS.
    Returns None if fewer than 3 points are provided.
    """
    if len(points) < 3:
        return None

    # The two topmost points are the shoulders
    top_two = sorted(points[:2], key=lambda p: p[0])  # left → right
    LS, RS  = top_two[0], top_two[1]
    T       = points[2]   # torso marker (lowest)

    # 1. Shoulder slope angle (degrees from horizontal)
    shoulder_angle = math.degrees(
        math.atan2(RS[1] - LS[1], RS[0] - LS[0])
    )

    # 2. Torso lean angle (degrees from vertical)
    mid_x = (LS[0] + RS[0]) / 2
    mid_y = (LS[1] + RS[1]) / 2
    dx = T[0] - mid_x
    dy = T[1] - mid_y
    torso_angle = math.degrees(math.atan2(dx, dy))   # swap for vertical ref

    # 3. Shoulder distance (px)
    shoulder_distance = math.dist(LS, RS)

    # 4. Vertical offset between torso marker and shoulder midpoint (px)
    vertical_offset = abs(T[1] - mid_y)

    return {
        "shoulder_angle":    shoulder_angle,
        "torso_angle":       torso_angle,
        "shoulder_distance": shoulder_distance,
        "vertical_offset":   vertical_offset,
        # internal — used for drawing
        "_LS":  LS,
        "_RS":  RS,
        "_T":   T,
        "_mid": (int(mid_x), int(mid_y)),
    }


# ─────────────────────────────────────────────
#  Skeleton Drawing
# ─────────────────────────────────────────────

def draw_skeleton(frame, features, label_id=None):
    """
    Draw marker circles + skeleton lines + feature text onto *frame* in-place.
    If *label_id* is given (0/1/2), the overlay color is taken from LABEL_COLORS.
    """
    if features is None:
        return

    LS  = features["_LS"]
    RS  = features["_RS"]
    T   = features["_T"]
    mid = features["_mid"]

    color = LABEL_COLORS.get(label_id, (0, 220, 220))

    # Skeleton lines
    cv2.line(frame, LS,  RS,  color, 2)          # shoulder bar
    cv2.line(frame, mid, T,   color, 2)          # torso spine

    # Marker circles
    for pt, tag in [(LS, "LS"), (RS, "RS"), (T, "T")]:
        cv2.circle(frame, pt, 9, color, -1)
        cv2.circle(frame, pt, 10, (255, 255, 255), 1)
        cv2.putText(frame, tag, (pt[0] + 12, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Feature text (bottom-left panel)
    lines = [
        f"Shoulder slope : {features['shoulder_angle']:+.1f} deg",
        f"Torso lean     : {features['torso_angle']:+.1f} deg",
        f"Shoulder dist  : {features['shoulder_distance']:.1f} px",
        f"Vertical offset: {features['vertical_offset']:.1f} px",
    ]
    y0 = frame.shape[0] - 10 - len(lines) * 22
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (10, y0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1,
                    cv2.LINE_AA)


# ─────────────────────────────────────────────
#  Risk Badge Overlay
# ─────────────────────────────────────────────

def draw_risk_badge(frame, label_id, confidence=None):
    """
    Render a coloured risk badge at the top of *frame*.
    """
    label_text = LABEL_NAMES.get(label_id, "Unknown")
    color      = LABEL_COLORS.get(label_id, (150, 150, 150))

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), color, -1)
    cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)

    # Risk label
    emoji_map = {0: "[ GOOD ]", 1: "[ MEDIUM RISK ]", 2: "[ HIGH RISK ]"}
    badge_txt = emoji_map.get(label_id, label_text)
    cv2.putText(frame, badge_txt, (12, 42),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    if confidence is not None:
        conf_txt = f"Conf: {confidence * 100:.0f}%"
        tw = cv2.getTextSize(conf_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
        cv2.putText(frame, conf_txt, (frame.shape[1] - tw - 12, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)


# ─────────────────────────────────────────────
#  Session Stats Bar
# ─────────────────────────────────────────────

def draw_session_stats(frame, counts):
    """
    Draw a small stacked bar at the bottom-right corner.
    counts = {0: n_good, 1: n_medium, 2: n_high}
    """
    total = sum(counts.values()) or 1
    bar_w, bar_h = 140, 14
    x0 = frame.shape[1] - bar_w - 10
    y0 = frame.shape[0] - 60

    cv2.putText(frame, "Session Stats", (x0, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    x = x0
    for lbl in [0, 1, 2]:
        w = int(bar_w * counts.get(lbl, 0) / total)
        if w == 0:
            continue
        cv2.rectangle(frame, (x, y0), (x + w, y0 + bar_h),
                      LABEL_COLORS[lbl], -1)
        x += w

    # Border
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (200, 200, 200), 1)

    # Legend
    for i, lbl in enumerate([0, 1, 2]):
        pct = 100 * counts.get(lbl, 0) / total
        leg = f"{LABEL_NAMES[lbl][0]}: {pct:.0f}%"
        cv2.putText(frame, leg, (x0, y0 + bar_h + 16 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, LABEL_COLORS[lbl], 1)
