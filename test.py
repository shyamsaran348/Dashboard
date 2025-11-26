import joblib, pandas as pd, os

MODEL_FILE = "artifacts/rf_baseline.joblib"   # or rf_baseline_fast.joblib
FEATURES_FILE = "artifacts/features_table.csv"  # your loaded CSV

print("Paths:", MODEL_FILE, FEATURES_FILE)
model = joblib.load(MODEL_FILE)
print("model.n_features_in_:", getattr(model, "n_features_in_", None))
print("model.feature_names_in_:", getattr(model, "feature_names_in_", None))

df = pd.read_csv(FEATURES_FILE)
print("df.shape:", df.shape)
print("df.columns (first 50):", list(df.columns)[:50])

# show exactly what the app would send as features
# use same detection logic as app:
feature_cols = [c for c in df.columns if c.startswith('ch')]
if not feature_cols:
    numeric_cols = df.select_dtypes(include=[float,int]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ('semantic_relevance','label_enc')]
print("Detected feature_cols (count):", len(feature_cols))
print(feature_cols[:100])
