import cv2
import numpy as np
import math
import pandas as pd

cap = cv2.VideoCapture(0)

data = []

print("\nControls:")
print("1 - Good posture")
print("2 - Medium risk")
print("3 - High risk")
print("Q - Quit\n")


# =========================
# BALANCED RED HSV RANGE
# =========================
lower_red_1 = np.array([0, 70, 60])
upper_red_1 = np.array([10, 255, 255])

lower_red_2 = np.array([170, 70, 60])
upper_red_2 = np.array([180, 255, 255])


# =========================
# HELPER
# =========================
def distance(p1, p2):
    return math.dist(p1, p2)


# =========================
# MAIN LOOP
# =========================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_red_1, upper_red_1) + \
           cv2.inRange(hsv, lower_red_2, upper_red_2)

    # clean noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    # =========================
    # DETECT ALL RED BLOBS
    # =========================
    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 100:  # ignore tiny noise

            M = cv2.moments(cnt)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                candidates.append((area, (cx, cy)))

    # =========================
    # KEEP ONLY LARGEST 3 MARKERS
    # =========================
    candidates = sorted(candidates, reverse=True)[:3]

    points = [pt for (_, pt) in candidates]

    for p in points:
        cv2.circle(frame, p, 8, (0, 255, 0), -1)

    print("Points detected:", len(points))

    # =========================
    # FEATURE EXTRACTION
    # =========================
    if len(points) == 3:

        # sort top to bottom
        points = sorted(points, key=lambda x: x[1])

        LS, RS, T = points

        shoulder_angle = math.degrees(
            math.atan2(RS[1] - LS[1], RS[0] - LS[0])
        )

        mid = ((LS[0] + RS[0]) // 2, (LS[1] + RS[1]) // 2)

        torso_angle = math.degrees(
            math.atan2(T[1] - mid[1], T[0] - mid[0])
        )

        shoulder_dist = distance(LS, RS)
        vertical_offset = abs(T[1] - mid[1])

        # =========================
        # DISPLAY LIVE FEATURES
        # =========================
        cv2.putText(frame, f"Shoulder Angle : {shoulder_angle:.1f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.putText(frame, f"Torso Angle : {torso_angle:.1f}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.putText(frame, f"Distance : {shoulder_dist:.1f}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.putText(frame, f"Offset : {vertical_offset:.1f}",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # =========================
        # LABEL & SAVE SAMPLE
        # =========================
        if key in [ord('1'), ord('2'), ord('3')]:

            label = int(chr(key)) - 1

            print(
                f"Saved -> "
                f"S:{shoulder_angle:.2f}, "
                f"T:{torso_angle:.2f}, "
                f"D:{shoulder_dist:.2f}, "
                f"O:{vertical_offset:.2f}, "
                f"L:{label}"
            )

            data.append([
                shoulder_angle,
                torso_angle,
                shoulder_dist,
                vertical_offset,
                label
            ])

    if key == ord('q'):
        break

    cv2.imshow("Feature Extraction", frame)
    cv2.imshow("Mask", mask)


# =========================
# SAVE CSV
# =========================
cap.release()
cv2.destroyAllWindows()

if len(data) > 0:
    df = pd.DataFrame(data, columns=[
        "shoulder_angle",
        "torso_angle",
        "shoulder_distance",
        "vertical_offset",
        "label"
    ])

    df.to_csv("data/posture_features.csv", index=False)
    print("\nDataset saved -> data/posture_features.csv")
else:
    print("\nNo samples saved.")