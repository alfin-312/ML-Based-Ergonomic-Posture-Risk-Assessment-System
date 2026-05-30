import joblib, os, pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
svm_path = os.path.join(BASE, '..', 'models', 'posture_svm.pkl')
rf_path  = os.path.join(BASE, '..', 'models', 'posture_rf.pkl')
aug_csv  = os.path.join(BASE, '..', 'data', 'posture_features_augmented.csv')

print("SVM exists:", os.path.exists(svm_path))
print("RF  exists:", os.path.exists(rf_path))
print("CSV exists:", os.path.exists(aug_csv))

df = pd.read_csv(aug_csv)
print(f"Augmented CSV rows: {len(df)}")
print(f"Class distribution: {df['label'].value_counts().sort_index().to_dict()}")

cols = ['shoulder_angle','torso_angle','shoulder_distance','vertical_offset']
X = df[cols].values
y = df['label'].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

svm = joblib.load(svm_path)
rf  = joblib.load(rf_path)

for name, m in [('SVM', svm), ('RF', rf)]:
    preds = m.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    print(f"\n{name} hold-out accuracy: {acc:.4f}")
    print(classification_report(y_te, preds, target_names=['Good','Medium','High']))

out_dir = os.path.join(BASE, '..', 'output')
print("=== Output files ===")
for f in sorted(os.listdir(out_dir)):
    fp = os.path.join(out_dir, f)
    if os.path.isfile(fp):
        print(f"  {f}  ({os.path.getsize(fp):,} bytes)")
print("DONE")
