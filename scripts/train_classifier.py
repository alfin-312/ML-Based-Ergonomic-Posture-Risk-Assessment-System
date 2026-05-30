import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import make_pipeline
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    CSV_PATH, CSV_AUG_PATH,
    MODEL_SVM_PATH, MODEL_RF_PATH,
    TRAINING_REPORT, OUTPUT_DIR,
    FEATURE_COLUMNS, LABEL_NAMES,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SVM_PATH), exist_ok=True)


# ─────────────────────────────────────────────
#  Load Dataset
# ─────────────────────────────────────────────
if os.path.exists(CSV_AUG_PATH):
    csv_used = CSV_AUG_PATH
    print(f"[INFO] Using augmented dataset: {CSV_AUG_PATH}")
elif os.path.exists(CSV_PATH):
    csv_used = CSV_PATH
    print(f"[INFO] Using base dataset:      {CSV_PATH}")
else:
    print("[ERROR] No dataset found. Run feature_extraction.py first.")
    sys.exit(1)

df = pd.read_csv(csv_used)
print(f"       Samples: {len(df)}")
print(f"       Class distribution: {df['label'].value_counts().sort_index().to_dict()}\n")

X = df[FEATURE_COLUMNS].values
y = df["label"].values

# ─────────────────────────────────────────────
#  Train / Test Split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
#  Model Definitions
# ─────────────────────────────────────────────
models = {
    "SVM (RBF kernel)": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, gamma="scale", probability=True),
    ),
    "Random Forest": make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
        ),
    ),
}

# ─────────────────────────────────────────────
#  Training + Evaluation
# ─────────────────────────────────────────────
cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
report_lines = []

def header(title):
    line = "=" * 60
    return f"\n{line}\n  {title}\n{line}"

report_lines.append("ML-Based Ergonomic Posture Risk Assessment — Training Report")
report_lines.append(f"Dataset : {csv_used}")
report_lines.append(f"Samples : {len(df)}  |  Features: {FEATURE_COLUMNS}")

trained = {}

for name, pipe in models.items():
    print(f"--- Training: {name} ---")
    report_lines.append(header(name))

    # Cross-validation on training split
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    report_lines.append(f"5-Fold CV Accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    report_lines.append(f"CV Fold Scores     : {[round(s,4) for s in cv_scores]}")

    # Final train on full training set, evaluate on hold-out
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    print(f"  Hold-out Accuracy: {acc:.4f}\n")

    report_lines.append(f"Hold-out Accuracy  : {acc:.4f}")
    report_lines.append("\nClassification Report:")
    target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    report_lines.append(
        classification_report(y_test, y_pred, target_names=target_names)
    )

    cm = confusion_matrix(y_test, y_pred)
    report_lines.append("Confusion Matrix (rows=actual, cols=predicted):")
    report_lines.append(str(cm))

    trained[name] = (pipe, y_pred, cm)

# ─────────────────────────────────────────────
#  Save Models
# ─────────────────────────────────────────────
svm_pipe, _, _ = trained["SVM (RBF kernel)"]
rf_pipe,  _, _ = trained["Random Forest"]

joblib.dump(svm_pipe, MODEL_SVM_PATH)
joblib.dump(rf_pipe,  MODEL_RF_PATH)
print(f"[OK] SVM model saved → {MODEL_SVM_PATH}")
print(f"[OK] RF  model saved → {MODEL_RF_PATH}")

# ─────────────────────────────────────────────
#  Confusion Matrix Plots
# ─────────────────────────────────────────────
target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Confusion Matrices — Posture Risk Classifier", fontsize=14, fontweight="bold")

for ax, (name, (pipe, y_pred, cm)) in zip(axes, trained.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=target_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=12)
    ax.tick_params(axis='x', labelrotation=15)

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
plt.savefig(cm_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"[OK] Confusion matrix plot → {cm_path}")

# ─────────────────────────────────────────────
#  Feature Importance (Random Forest)
# ─────────────────────────────────────────────
rf_raw = rf_pipe.named_steps["randomforestclassifier"]
importances = rf_raw.feature_importances_

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.barh(FEATURE_COLUMNS, importances, color="#4f8ef7")
ax2.set_xlabel("Feature Importance (Gini)")
ax2.set_title("Random Forest — Feature Importances")
ax2.invert_yaxis()
plt.tight_layout()
fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
plt.savefig(fi_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"[OK] Feature importance plot → {fi_path}")

# ─────────────────────────────────────────────
#  Save Text Report
# ─────────────────────────────────────────────
report_lines.append(f"\nModels saved:")
report_lines.append(f"  SVM : {MODEL_SVM_PATH}")
report_lines.append(f"  RF  : {MODEL_RF_PATH}")

with open(TRAINING_REPORT, "w") as f:
    f.write("\n".join(report_lines))
print(f"[OK] Training report → {TRAINING_REPORT}")
print("\n✓ Training complete.")
