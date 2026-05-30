# 🧠 Machine Learning Based Ergonomic Posture Risk Assessment System

A **classical Computer Vision + Machine Learning** system that monitors posture in real time using
three red body markers and a standard webcam — no deep learning, no special hardware.

---

## 🎯 Project Objectives

| Goal | Approach |
|---|---|
| Low-cost monitoring | Webcam + red sticker markers |
| No deep learning | SVM + Random Forest (scikit-learn) |
| Geometric features | Shoulder slope, torso lean, distance, offset |
| Real-time feedback | Color-coded badge: 🟢 Good / 🟡 Medium / 🔴 High |
| Lightweight execution | Classical CV only — runs on any laptop |

---

## 🏗️ System Architecture

```
Webcam Frame
    │
    ▼
HSV Color Segmentation (Red Mask)
    │
    ▼
Morphological Filtering (Open → Close → Dilate)
    │
    ▼
Contour Detection + Circularity Filter
    │
    ▼
3 Marker Centroids (Left Shoulder, Right Shoulder, Torso)
    │
    ▼
Geometric Feature Extraction
    ├── Shoulder Slope Angle (°)
    ├── Torso Lean Angle (°)
    ├── Shoulder Distance (px)
    └── Vertical Offset (px)
    │
    ▼
SVM Classifier (RBF kernel, StandardScaler)
    │
    ▼
Posture Risk Label: Good (0) / Medium Risk (1) / High Risk (2)
    │
    ▼
Real-time HUD Overlay + Session Statistics
```

---

## 📁 Project Structure

```
Mini_Project/
├── scripts/
│   ├── camera_test.py          # Webcam sanity check
│   ├── marker_detection.py     # Step 1 — visualize marker detection
│   ├── feature_extraction.py   # Step 2 — collect labelled training data
│   ├── data_augmentation.py    # Step 3 — expand dataset with Gaussian noise
│   ├── train_classifier.py     # Step 4 — train SVM + Random Forest
│   └── real_time_inference.py  # Step 5 — live posture assessment
├── utils/
│   ├── config.py               # Central HSV ranges, paths, labels
│   ├── feature_utils.py        # Shared CV + drawing helpers
│   └── hsv_calibrator.py       # Interactive HSV trackbar tuner
├── data/
│   ├── posture_features.csv             # Raw labelled samples
│   └── posture_features_augmented.csv   # Augmented training set
├── models/
│   ├── posture_svm.pkl         # Trained SVM model
│   └── posture_rf.pkl          # Trained Random Forest model
├── output/
│   ├── training_report.txt     # Full CV metrics
│   ├── confusion_matrices.png  # Side-by-side confusion matrices
│   ├── feature_importance.png  # RF feature importances
│   └── snapshots/              # Press S during inference to save
└── requirements.txt
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

**Requirements:** `opencv-python`, `scikit-learn`, `numpy`, `pandas`, `joblib`, `matplotlib`

---

## 🚀 Usage — Step by Step

### Step 0 — Test your webcam
```bash
python scripts/camera_test.py
```

### Step 1 — Verify marker detection
Attach **3 red stickers / tape** to:
- Left shoulder
- Right shoulder
- Torso / lower-back

```bash
python scripts/marker_detection.py
```
✅ You should see 3 green circles on the markers and a skeleton connecting them.

> 💡 If detection is poor, run the HSV calibrator first:
> ```bash
> python utils/hsv_calibrator.py
> ```
> Adjust sliders → press **S** → copy values into `utils/config.py`.

### Step 2 — Collect training data
```bash
python scripts/feature_extraction.py
```

| Key | Label |
|---|---|
| `1` | Good posture (sit/stand straight) |
| `2` | Medium risk (slight lean/slouch) |
| `3` | High risk (heavy forward lean / hunch) |
| `Q` | Save CSV and quit |

Aim for **≥ 15–20 samples per class** across different postures.

### Step 3 — Augment dataset
```bash
python scripts/data_augmentation.py
```
Creates `data/posture_features_augmented.csv` with 5× more samples using Gaussian noise.

### Step 4 — Train the classifiers
```bash
python scripts/train_classifier.py
```
- Trains **SVM (RBF)** and **Random Forest** with 5-fold stratified cross-validation
- Saves `models/posture_svm.pkl` and `models/posture_rf.pkl`
- Writes `output/training_report.txt`, `output/confusion_matrices.png`, `output/feature_importance.png`

### Step 5 — Run real-time assessment
```bash
python scripts/real_time_inference.py
```

| Key | Action |
|---|---|
| `S` | Save snapshot to `output/snapshots/` |
| `Q` | Quit and show session summary |

---

## 📊 Feature Description

| Feature | Description | Unit |
|---|---|---|
| `shoulder_angle` | Slope angle between left and right shoulder markers | degrees |
| `torso_angle` | Lean angle of torso relative to vertical | degrees |
| `shoulder_distance` | Pixel distance between shoulder markers | px |
| `vertical_offset` | Vertical gap between shoulder midpoint and torso marker | px |

---

## 🤖 ML Models

| Model | Algorithm | Notes |
|---|---|---|
| `posture_svm.pkl` | SVM with RBF kernel + StandardScaler | Primary inference model |
| `posture_rf.pkl` | Random Forest (200 trees) + StandardScaler | Alternative; provides feature importances |

Both are evaluated with **5-fold stratified cross-validation**.

---

## 📋 Risk Classification

| Label | Class | Description |
|---|---|---|
| 0 | 🟢 **Good Posture** | Shoulders level, torso upright |
| 1 | 🟡 **Medium Risk** | Slight shoulder tilt or torso lean |
| 2 | 🔴 **High Risk** | Heavy slouch, forward lean, or asymmetric posture |

---

## 🔧 Customisation

Edit `utils/config.py` to adjust:
- `LOWER_RED_1 / UPPER_RED_1 / LOWER_RED_2 / UPPER_RED_2` — HSV thresholds for your markers
- `MIN_CONTOUR_AREA` — filter noise smaller than this pixel area
- `MIN_CIRCULARITY` — strictness of blob shape filter (0 = any shape, 1 = perfect circle)

---

## 🧪 Technique Summary

| Stage | Technique |
|---|---|
| Color detection | HSV color segmentation (two red hue bands) |
| Noise removal | Morphological Open + Close + Dilate |
| Blob detection | Contour finding + circularity filter |
| Centroid finding | Image moments |
| Feature computation | Trigonometry (atan2, dist) |
| Classification | SVM (RBF) / Random Forest |
| Data augmentation | Gaussian noise injection |
| Evaluation | Stratified K-Fold CV + confusion matrix |
