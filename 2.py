"""
Expanded starter notebook: EEG semantic relevance - full EDA, feature extraction, baseline model,
and visualization helpers.

This expanded file contains everything from the previous starter pipeline plus a
comprehensive EDA section covering:
 - participant/topic summaries
 - PSD comparisons per label
 - ERP / average waveform plots
 - PCA / UMAP embeddings (UMAP optional)
 - feature correlation heatmap
 - outlier detection (length and amplitude)
 - spectrogram/time-frequency plots
 - interactive sample plotting helpers
 - convenience functions to save EDA plots to ./artifacts/eda

How to use:
- Paste into a Jupyter notebook cell or save as a .py and run stepwise.
- Set SAMPLE_LIMIT for quicker runs when prototyping.
- Install optional packages if needed: umap-learn, seaborn

"""

# === Imports ===
import os
from pprint import pprint
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram

# optional
try:
    import seaborn as sns
except Exception:
    sns = None

try:
    import umap
except Exception:
    umap = None

from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

# === Settings ===
HF_DATASET_ID = "Quoron/EEG-semantic-text-relevance"
SAMPLE_LIMIT = None   # set small for quick tests (e.g., 500)
SF = 250              # sampling frequency (Hz). Change if dataset meta says otherwise.
N_PERSEG = 256
RANDOM_STATE = 42

ARTIFACT_DIR = "artifacts"
EDA_DIR = os.path.join(ARTIFACT_DIR, "eda")
os.makedirs(EDA_DIR, exist_ok=True)

# === Utility functions ===

def safe_pprint_row(row):
    r = dict(row)
    eeg = r.get("eeg", None)
    if eeg is not None:
        try:
            arr = np.asarray(eeg)
            r["eeg_summary"] = f"type={type(eeg).__name__}, shape={arr.shape}, dtype={arr.dtype}"
        except Exception:
            r["eeg_summary"] = f"type={type(eeg).__name__} (unreadable)"
        r.pop("eeg", None)
    pprint(r)


def bandpower(x, sf=250, band=(8,13), nperseg=256):
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    f, Pxx = welch(x, fs=sf, nperseg=nperseg)
    idx_band = np.logical_and(f >= band[0], f <= band[1])
    return np.trapz(Pxx[idx_band], f[idx_band]) if np.any(idx_band) else 0.0


def extract_features_from_signal(sig, sf=250, nperseg=256):
    x = np.asarray(sig)
    if x.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "bp_delta": 0.0, "bp_theta": 0.0, "bp_alpha": 0.0, "bp_beta": 0.0, "len": 0}

    feats = {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "len": int(x.shape[0])
    }
    bands = {"delta": (1,4), "theta": (4,8), "alpha": (8,13), "beta": (13,30)}
    for name, band in bands.items():
        feats[f"bp_{name}"] = float(bandpower(x, sf=sf, band=band, nperseg=nperseg))
    return feats

# === 1. Load dataset ===
print("Loading dataset from Hugging Face:", HF_DATASET_ID)
dset = load_dataset(HF_DATASET_ID)
print(dset)

# === 2. Safe inspection ===
print("Features:")
print(dset["train"].features)

print("Example row (safe):")
safe_pprint_row(dset["train"][0])

print("Sample table (excluding EEG column):")
small_pd = dset["train"].to_pandas()[[c for c in dset["train"].column_names if c != "eeg"]].head(10)
print(small_pd.to_string())

# Inspect EEG shape statistics for first N rows
N_CHECK = 50
lengths = []
shapes = Counter()
for i in range(min(len(dset["train"]), N_CHECK)):
    e = dset["train"][i]["eeg"]
    arr = np.asarray(e)
    shapes[arr.shape] += 1
    lengths.append(arr.shape)
print("EEG shapes observed (first", N_CHECK, "rows):")
for s, cnt in shapes.items():
    print(s, "->", cnt)

# === 3. Build feature matrix (with SAMPLE_LIMIT) ===
print("Building feature matrix... (this may take some time)")
rows = []
limit = SAMPLE_LIMIT or len(dset["train"])
for i in range(limit):
    r = dset["train"][i]
    eeg = np.asarray(r["eeg"])
    # If EEG is multi-channel (2D), flatten by averaging channels -> 1D
    if eeg.ndim == 2:
        proc_sig = eeg.mean(axis=0)
    else:
        proc_sig = eeg
    feats = extract_features_from_signal(proc_sig, sf=SF, nperseg=N_PERSEG)
    feats.update({
        "semantic_relevance": r.get("semantic_relevance"),
        "interestingness": r.get("interestingness"),
        "participant": r.get("participant"),
        "topic": r.get("topic"),
        "word": r.get("word")
    })
    rows.append(feats)

df = pd.DataFrame(rows)
print("Feature dataframe shape:", df.shape)
print(df.head())

# Quick label exploration
print("Label distribution (semantic_relevance):")
print(df["semantic_relevance"].value_counts(dropna=False))

# === 4. Prepare X, y, groups for baseline model ===
df = df.dropna(subset=["semantic_relevance"]).reset_index(drop=True)
if df["semantic_relevance"].dtype == object:
    label_map = {v:i for i,v in enumerate(sorted(df["semantic_relevance"].unique()))}
    df["label_enc"] = df["semantic_relevance"].map(label_map)
else:
    df["label_enc"] = df["semantic_relevance"].astype(int)

feature_cols = [c for c in df.columns if c.startswith(("mean","std","min","max","len","bp_"))]
X = df[feature_cols].values
y = df["label_enc"].values
groups = df["participant"].values
print("Features used:", feature_cols)
print("X shape:", X.shape, "y shape:", y.shape)

# === 5. Group-aware cross validation baseline ===
print("Running GroupKFold cross-validation (participant-wise)...")
gkf = GroupKFold(n_splits=5)
clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
scoring = "f1_macro" if len(np.unique(y)) > 2 else "f1"
scores = cross_val_score(clf, X, y, groups=groups, cv=gkf, scoring=scoring)
print(f"CV {scoring} scores (by fold): {scores}")
print(f"Mean {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
clf.fit(X, y)


# === MODEL EVALUATION AFTER TRAINING ===
print("\n================ MODEL EVALUATION ================")

# 1. Predictions on the training set (baseline)
y_pred = clf.predict(X)

# 2. Accuracy
acc = accuracy_score(y, y_pred)
print(f"Training Accuracy: {acc:.4f}")

# 3. F1 Score
f1 = f1_score(y, y_pred)
print(f"Training F1 Score: {f1:.4f}")

# 4. Confusion Matrix
cm = confusion_matrix(y, y_pred)
print("\nConfusion Matrix:")
print(cm)

# 5. Detailed Classification Report
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y, y_pred))

# 6. Feature importance
importances = clf.feature_importances_
print("\nFeature Importances:")
for fname, imp in zip(feature_cols, importances):
    print(f"{fname:12} : {imp:.6f}")

# 7. Save evaluation output to a text file (optional)
eval_path = os.path.join(ARTIFACT_DIR, "model_evaluation.txt")
with open(eval_path, "w") as f:
    f.write("=== Random Forest Model Evaluation ===\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y, y_pred))
    f.write("\nFeature Importances:\n")
    for fname, imp in zip(feature_cols, importances):
        f.write(f"{fname:12} : {imp:.6f}\n")

print(f"\nModel evaluation saved to {eval_path}")

# === 6. EDA: participant-level summary ===
print("Participant counts (top 20):")
pc = df.groupby('participant').size().sort_values(ascending=False)
print(pc.head(20))

print("Class balance per participant (first 10 rows):")
balance = df.pivot_table(index='participant', columns='label_enc', values='word', aggfunc='count', fill_value=0)
print(balance.head(10))

# save
pc.head(50).to_csv(os.path.join(EDA_DIR, 'participant_counts.csv'))

# === 7. EDA: topic distribution ===
print("Topic distribution (top 20):")
print(df['topic'].value_counts().head(20))
print("Selected topic distribution (top 20):")
print(df.get('selected_topic', pd.Series()).value_counts().head(20))

# === 8. EDA: PSD comparison per label ===

def avg_psd_for_indices(indices, maxlen=5000, sf=SF):
    psd_list=[]
    freqs=None
    for i in indices:
        sig = np.asarray(dset['train'][i]['eeg'])
        if sig.ndim==2:
            sig = sig.mean(axis=0)
        sig = sig[:maxlen]
        f,P = welch(sig, fs=sf, nperseg=N_PERSEG)
        psd_list.append(P)
        freqs = f
    return freqs, np.mean(psd_list, axis=0)

labels = sorted(df['label_enc'].unique())
plt.figure(figsize=(8,4))
for lab in labels:
    idxs = df.index[df['label_enc']==lab].tolist()[:200]
    if len(idxs)==0: continue
    f, Pavg = avg_psd_for_indices(idxs)
    plt.semilogy(f, Pavg, label=f'label_{lab}')
plt.xlim(0,60)
plt.legend(); plt.title('Average PSD by label'); plt.xlabel('Freq (Hz)'); plt.ylabel('PSD')
plt.tight_layout(); plt.savefig(os.path.join(EDA_DIR,'avg_psd_by_label.png'))
plt.show()

# === 9. EDA: ERP / Average waveform per label ===

def average_waveform(indices, maxlen=2000):
    sigs=[]
    for i in indices:
        raw = np.asarray(dset['train'][i]['eeg'])
        if raw.ndim==2: raw = raw.mean(axis=0)
        sigs.append(raw[:maxlen])
    minlen = min(len(s) for s in sigs)
    return np.mean([s[:minlen] for s in sigs], axis=0)

plt.figure(figsize=(10,4))
for lab in labels:
    idxs = df.index[df['label_enc']==lab].tolist()[:200]
    if len(idxs)==0: continue
    avg = average_waveform(idxs, maxlen=2000)
    plt.plot(avg, label=f'label_{lab}')
plt.legend(); plt.title('Average waveform by label (first 200 samples)')
plt.tight_layout(); plt.savefig(os.path.join(EDA_DIR,'avg_waveform_by_label.png'))
plt.show()

# === 10. EDA: PCA / UMAP embeddings of features ===

pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
plt.figure(figsize=(6,5));
plt.scatter(pca[:,0], pca[:,1], c=df['label_enc'], cmap='tab10', s=8)
plt.title('PCA of features'); plt.xlabel('PC1'); plt.ylabel('PC2'); plt.colorbar();
plt.tight_layout(); plt.savefig(os.path.join(EDA_DIR,'pca_features.png'))
plt.show()

if umap is not None:
    try:
        emb = umap.UMAP(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
        plt.figure(figsize=(6,5)); plt.scatter(emb[:,0], emb[:,1], c=df['label_enc'], s=6)
        plt.title('UMAP of features'); plt.tight_layout(); plt.savefig(os.path.join(EDA_DIR,'umap_features.png'))
        plt.show()
    except Exception as e:
        print('UMAP embedding failed:', e)
else:
    print('UMAP not installed; skip UMAP step. To install: pip install umap-learn')

# === 11. EDA: Correlation / feature importance heatmap ===
if sns is not None:
    corr = df[feature_cols].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=False, cmap='vlag', center=0)
    plt.title('Feature correlation'); plt.tight_layout(); plt.savefig(os.path.join(EDA_DIR,'feature_correlation.png'))
    plt.show()
else:
    print('seaborn not installed; install it for nicer heatmaps (pip install seaborn)')

# === 12. EDA: Outlier detection ===
lengths = df['len']
q1 = lengths.quantile(0.25); q3 = lengths.quantile(0.75); iqr = q3 - q1
upper = q3 + 1.5*iqr
outliers_len = df[lengths > upper]
print('Length outliers count:', len(outliers_len))
outliers_len.head().to_csv(os.path.join(EDA_DIR,'length_outliers_sample.csv'), index=False)

amp_out = df[(df['max'] > df['max'].quantile(0.99)) | (df['min'] < df['min'].quantile(0.01))]
print('Amplitude outliers count:', len(amp_out))
amp_out.head().to_csv(os.path.join(EDA_DIR,'amplitude_outliers_sample.csv'), index=False)

# === 13. EDA: Spectrogram example ===
example_idx = 0
sig = np.asarray(dset['train'][example_idx]['eeg'])
if sig.ndim==2: sig = sig.mean(axis=0)
f, t, Sxx = spectrogram(sig, fs=SF, nperseg=N_PERSEG, noverlap=N_PERSEG//2)
plt.figure(figsize=(8,4))
plt.pcolormesh(t, f, 10*np.log10(Sxx+1e-12))
plt.ylabel('Freq (Hz)'); plt.xlabel('Time (s)'); plt.title(f'Spectrogram sample {example_idx}'); plt.colorbar(label='dB')
plt.ylim(0,60); plt.tight_layout(); plt.savefig(os.path.join(EDA_DIR,f'spectrogram_{example_idx}.png'))
plt.show()

# === 14. Interactive / helper plot functions ===

def plot_sample(i, maxlen=None, show_psd=True):
    sig = np.asarray(dset['train'][i]['eeg'])
    if sig.ndim==2: sig = sig.mean(axis=0)
    if maxlen:
        sig = sig[:maxlen]
    plt.figure(figsize=(12,3)); plt.plot(sig); plt.title(f'sample {i} (len={len(sig)})')
    plt.show()
    if show_psd:
        f,P = welch(sig, fs=SF, nperseg=N_PERSEG)
        plt.figure(figsize=(8,3)); plt.semilogy(f, P); plt.xlim(0,60)
        plt.title(f'PSD sample {i}'); plt.xlabel('Freq (Hz)'); plt.ylabel('PSD'); plt.show()

# Save a few sample plots
for idx in [0, 10, 50]:
    try:
        plot_sample(idx, maxlen=2000)
        # Save PSD plots as files
        sig = np.asarray(dset['train'][idx]['eeg'])
        if sig.ndim==2: sig = sig.mean(axis=0)
        f,P = welch(sig, fs=SF, nperseg=N_PERSEG)
        plt.figure(); plt.semilogy(f,P); plt.xlim(0,60); plt.title(f'PSD sample {idx}')
        plt.tight_layout(); plt.savefig(os.path.join(EDA_DIR,f'psd_sample_{idx}.png'))
        plt.close()
    except Exception as e:
        print('Failed to plot sample', idx, e)

# === 15. Save model and feature table ===
import joblib
os.makedirs(ARTIFACT_DIR, exist_ok=True)
joblib.dump(clf, os.path.join(ARTIFACT_DIR, 'rf_baseline.joblib'))
df.to_csv(os.path.join(ARTIFACT_DIR, 'features_table.csv'), index=False)
print('Saved artifacts to', ARTIFACT_DIR)

# === 16. Streamlit skeleton (optional) ===

print('If you want a Streamlit app skeleton, I can generate a separate file streamlit_app.py with controls to:')
print(' - select participant/topic/sample')
print(' - plot raw signal, PSD, spectrogram, and show feature values')
print(' - show model metrics and confusion matrix')

# === End ===
print("Expanded EDA pipeline completed. All EDA figures and CSVs are in:", EDA_DIR)
