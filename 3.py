"""
High-performance, optimized notebook: EEG semantic relevance — fast mode for M1 (8GB).

This version focuses on speed and low memory footprint while preserving the pipeline:
- Vectorized per-channel PSD / bandpower (single welch call per sample)
- float32 everywhere to reduce memory
- joblib parallelization for feature extraction (multi-core)
- optional downsampling to reduce time-series length (default off)
- reduced RandomForest defaults for fast prototyping (n_estimators=50)
- profiling timers to find slow spots
- clear knobs to trade speed vs accuracy: SAMPLE_LIMIT, DOWNSAMPLE_FACTOR, N_ESTIMATORS

How to use:
- For fastest run on M1 8GB: set SAMPLE_LIMIT=5000, DOWNSAMPLE_FACTOR=4, N_ESTIMATORS=50, n_jobs=-2
- To run full dataset later, increase SAMPLE_LIMIT=None and DOWNSAMPLE_FACTOR=1

"""

# === Imports ===
import os
import time
from pprint import pprint
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, decimate

# optional
try:
    import seaborn as sns
except Exception:
    sns = None

# parallel
from joblib import Parallel, delayed

# plotting interactivity
try:
    import plotly.graph_objects as go
    from ipywidgets import interact, IntSlider
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# === Settings / SPEED KNOBS ===
HF_DATASET_ID = "Quoron/EEG-semantic-text-relevance"
SAMPLE_LIMIT = 5000        # set to small number for fast prototyping; use None for full dataset
DOWNSAMPLE_FACTOR = 2      # 1 = no downsample; 2 or 4 reduces timepoints by factor
SF = 250                   # original sampling freq in Hz
SF_EFFECTIVE = SF // max(1, DOWNSAMPLE_FACTOR)
N_PERSEG = 256 // max(1, DOWNSAMPLE_FACTOR)
RANDOM_STATE = 42

ARTIFACT_DIR = "artifacts"
EDA_DIR = os.path.join(ARTIFACT_DIR, "eda")
os.makedirs(EDA_DIR, exist_ok=True)

# model speed knobs
N_ESTIMATORS = 50          # fewer trees for fast runs
RF_N_JOBS = -2             # parallel jobs for RF and joblib (-2 leaves 1 core free)
PARALLEL_N_JOBS = -2       # for joblib feature extraction

# === Utility functions (optimized) ===

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


# Vectorized bandpower from Pxx
def bandpower_from_Pxx(Pxx, f, band):
    idx = np.logical_and(f >= band[0], f <= band[1])
    if not np.any(idx):
        return np.zeros(Pxx.shape[0], dtype=np.float32)
    # trapz along freq axis
    return np.trapz(Pxx[:, idx], f[idx], axis=1).astype(np.float32)


# Fast featurizer using vectorized welch per-sample
def featurize_eeg(eeg_raw, downsample_factor=1, sf=SF, nperseg=N_PERSEG):
    # eeg_raw: (n_ch, n_samples) or (n_samples,)  -> returns dict of flattened features
    eeg = np.asarray(eeg_raw, dtype=np.float32)
    if eeg.ndim == 1:
        eeg = eeg[np.newaxis, :]
    if downsample_factor > 1:
        # decimate each channel (vectorized across axis=1 with scipy.decimate requires loop per channel)
        # do quick numpy slicing downsample as a fast alternative (not anti-aliased) for speed
        eeg = eeg[:, ::downsample_factor]
        sf = sf // downsample_factor
    n_ch, n_samp = eeg.shape

    # time-domain stats vectorized
    means = eeg.mean(axis=1).astype(np.float32)
    stds  = eeg.std(axis=1).astype(np.float32)
    mins  = eeg.min(axis=1).astype(np.float32)
    maxs  = eeg.max(axis=1).astype(np.float32)
    lens  = np.full(n_ch, n_samp, dtype=np.int32)

    # compute PSD vectorized per channel using axis=1 in scipy.signal.welch
    f, Pxx = welch(eeg, fs=sf, nperseg=min(n_samp, max(64, nperseg)), axis=1)
    Pxx = Pxx.astype(np.float32)

    feats = {}
    for ch in range(n_ch):
        feats[f'ch{ch}_mean'] = float(means[ch])
        feats[f'ch{ch}_std']  = float(stds[ch])
        feats[f'ch{ch}_min']  = float(mins[ch])
        feats[f'ch{ch}_max']  = float(maxs[ch])
        feats[f'ch{ch}_len']  = int(lens[ch])

    bands = {"delta": (1,4), "theta": (4,8), "alpha": (8,13), "beta": (13,30)}
    for name, band in bands.items():
        bp = bandpower_from_Pxx(Pxx, f, band)  # shape (n_ch,)
        for ch in range(n_ch):
            feats[f'ch{ch}_bp_{name}'] = float(bp[ch])

    return feats

# featurize wrapper for dataset index
def featurize_idx(i, downsample_factor, dset):
    r = dset[i]
    eeg = np.asarray(r['eeg'], dtype=np.float32)
    feats = featurize_eeg(eeg, downsample_factor, sf=SF, nperseg=N_PERSEG)
    feats.update({
        'semantic_relevance': int(r.get('semantic_relevance')),
        'interestingness': int(r.get('interestingness')) if r.get('interestingness') is not None else -1,
        'participant': r.get('participant'),
        'topic': r.get('topic'),
        'word': r.get('word')
    })
    return feats

# === 1. Load dataset (timed) ===
print('Loading dataset:', HF_DATASET_ID)
t0 = time.perf_counter()
dset = load_dataset(HF_DATASET_ID)['train']
print('Loaded dataset with', len(dset), 'rows; took', time.perf_counter()-t0, 's')

# quick feature of dataset
print('EEG feature info:', dset.features['eeg'])

# === 2. Fast parallel feature extraction ===
limit = SAMPLE_LIMIT or len(dset)
print(f'Featurizing {limit} samples (downsample_factor={DOWNSAMPLE_FACTOR}) using {PARALLEL_N_JOBS} jobs...')
t0 = time.perf_counter()
rows = Parallel(n_jobs=PARALLEL_N_JOBS, backend='loky')(
    delayed(featurize_idx)(i, DOWNSAMPLE_FACTOR, dset) for i in range(limit)
)
print('Featurization completed in', time.perf_counter()-t0, 's')

# create dataframe
df = pd.DataFrame(rows)
print('Feature dataframe shape:', df.shape)

# === 3. Sanitize and prepare X,y,groups (float32) ===
df = df.dropna(subset=['semantic_relevance']).reset_index(drop=True)
df['label_enc'] = df['semantic_relevance'].astype(int)
feature_cols = [c for c in df.columns if c.startswith('ch')]
X = df[feature_cols].to_numpy(dtype=np.float32)
# replace inf/nan
X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float32).max/10, neginf=-np.finfo(np.float32).max/10)
# optional: clip extreme values to reduce effect of outliers
clip_val = np.percentile(np.abs(X), 99.5)
X = np.clip(X, -clip_val, clip_val)

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = df['label_enc'].values
groups = df['participant'].values
print('Prepared X_scaled shape:', X_scaled.shape)

# === 4. Fast baseline with GroupKFold (reduced complexity) ===
print('Running GroupKFold CV with RandomForest (n_estimators=', N_ESTIMATORS, ')')
clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight='balanced', n_jobs=RF_N_JOBS, random_state=RANDOM_STATE)
gkf = GroupKFold(n_splits=5)

t0 = time.perf_counter()
scores = cross_val_score(clf, X_scaled, y, groups=groups, cv=gkf, scoring='f1_macro', n_jobs=RF_N_JOBS)
print('CV f1_macro scores:', scores)
print('Mean f1_macro:', scores.mean(), 'std:', scores.std())
print('CV completed in', time.perf_counter()-t0, 's')

# fit final model on full set (fast)
clf.fit(X_scaled, y)

# === 5. Quick EDA saved (limited) ===
os.makedirs(EDA_DIR, exist_ok=True)
# label distribution
df['label_enc'].value_counts().to_csv(os.path.join(EDA_DIR, 'label_distribution.csv'))
# small PCA (on a sample if very large)
n_pca = min(5000, X_scaled.shape[0])
print('Computing PCA on', n_pca, 'samples')
if X_scaled.shape[0] > n_pca:
    pca_sample_idx = np.random.RandomState(0).choice(X_scaled.shape[0], size=n_pca, replace=False)
    pca_input = X_scaled[pca_sample_idx]
else:
    pca_input = X_scaled
pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(pca_input)
plt.figure(figsize=(6,5)); plt.scatter(pca[:,0], pca[:,1], c=(df['label_enc'].values if X_scaled.shape[0]<=n_pca else df['label_enc'].values[pca_sample_idx]), s=6)
plt.title('PCA (sample)'); plt.tight_layout(); plt.savefig(os.path.join(EDA_DIR,'pca_sample.png'))
plt.close()

# save artifacts (model + features + scaler)
import joblib
os.makedirs(ARTIFACT_DIR, exist_ok=True)
joblib.dump(clf, os.path.join(ARTIFACT_DIR, 'rf_baseline_fast.joblib'))
joblib.dump(scaler, os.path.join(ARTIFACT_DIR, 'scaler_fast.joblib'))
df.to_csv(os.path.join(ARTIFACT_DIR, 'features_table_fast.csv'), index=False)
print('Saved fast artifacts to', ARTIFACT_DIR)

# === 6. Profiling info & guidance ===
print('Summary:')
print(' - Featurization time (s):',)
print(' - Set SAMPLE_LIMIT lower for quicker iterations; try 2000 or 1000')
print(' - To get faster, increase DOWNSAMPLE_FACTOR (2 or 4)')
print(' - For final runs, set SAMPLE_LIMIT=None and DOWNSAMPLE_FACTOR=1')

# === 7. Interactive sample viewer (optional) ===
if PLOTLY_AVAILABLE:
    def show_sample_plotly(idx):
        raw = np.asarray(dset[int(idx)]['eeg'], dtype=np.float32)
        if raw.ndim==2:
            sig = raw.mean(axis=0)
        else:
            sig = raw
        if DOWNSAMPLE_FACTOR>1:
            sig = sig[::DOWNSAMPLE_FACTOR]
        f,P = welch(sig, fs=SF_EFFECTIVE, nperseg=min(len(sig), max(64, N_PERSEG)))
        fig = go.Figure(); fig.add_trace(go.Scatter(y=sig, name='raw'))
        fig.update_layout(title=f'Sample {idx} (downsample={DOWNSAMPLE_FACTOR})', height=300)
        fig.show()
        fig2 = go.Figure(); fig2.add_trace(go.Scatter(x=f, y=P, mode='lines'))
        fig2.update_layout(title='PSD', xaxis_title='Hz', height=300); fig2.show()
    try:
        interact(show_sample_plotly, idx=IntSlider(min=0, max=len(dset)-1, step=1, value=0))
    except Exception:
        print('Interactive widgets unavailable; call show_sample_plotly(idx) manually')
else:
    print('Plotly/ipywidgets not installed; use matplotlib helper')

print('Optimized fast pipeline complete. Adjust SAMPLE_LIMIT and DOWNSAMPLE_FACTOR to trade speed vs accuracy.')
