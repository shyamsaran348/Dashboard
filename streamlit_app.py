# streamlit_app.py
"""
Full Streamlit dashboard for EEG Semantic Relevance with LIVE EDA.

Features:
- Robust artifact auto-detection (fast/slow filenames)
- Safe model + scaler loading with graceful fallbacks
- Automatic feature-alignment to model.n_features_in_ (trims 'interestingness' if needed)
- Static EDA explorer: lists files under artifacts/ and displays images in artifacts/eda/
- LIVE EDA computations on demand: PSD by label, ERP (avg waveform), correlation heatmap,
  outlier detection, spectrogram viewer, optional UMAP
- Replaces deprecated use_container_width with width='stretch'
- Run with: streamlit run streamlit_app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.signal import welch, spectrogram
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# optional imports (installed if available)
try:
    import seaborn as sns
except Exception:
    sns = None

try:
    import umap
except Exception:
    umap = None

# ---- Config ----
ARTIFACT_DIR = "artifacts"
FAST_FEATURES = os.path.join(ARTIFACT_DIR, "features_table_fast.csv")
SLOW_FEATURES = os.path.join(ARTIFACT_DIR, "features_table.csv")

FAST_MODEL = os.path.join(ARTIFACT_DIR, "rf_baseline_fast.joblib")
SLOW_MODEL = os.path.join(ARTIFACT_DIR, "rf_baseline.joblib")

FAST_SCALER = os.path.join(ARTIFACT_DIR, "scaler_fast.joblib")
SLOW_SCALER = os.path.join(ARTIFACT_DIR, "scaler.joblib")

HF_DATASET_ID = "Quoron/EEG-semantic-text-relevance"  # optional raw viewer

# sampling params (used when viewing raw signals)
SF = 250
DOWNSAMPLE_FACTOR = 1
N_PERSEG = 256 // max(1, DOWNSAMPLE_FACTOR)

# optional screenshot path from your session (displayed if present)
UPLOADED_SCREENSHOT_PATH = "/mnt/data/Screenshot 2025-11-24 at 11.01.29 PM.png"

st.set_page_config(layout="wide", page_title="EEG Semantic Relevance Dashboard")
st.title("EEG Semantic Relevance — Dashboard (robust loader + live EDA)")

# ---- helpers to auto-detect files ----
def find_features_file():
    if os.path.exists(FAST_FEATURES):
        return FAST_FEATURES
    if os.path.exists(SLOW_FEATURES):
        return SLOW_FEATURES
    return None

def find_model_file():
    if os.path.exists(FAST_MODEL):
        return FAST_MODEL
    if os.path.exists(SLOW_MODEL):
        return SLOW_MODEL
    return None

def find_scaler_file():
    if os.path.exists(FAST_SCALER):
        return FAST_SCALER
    if os.path.exists(SLOW_SCALER):
        return SLOW_SCALER
    return None

# ---- Load features (or stop with guidance) ----
FEATURES_FILE = find_features_file()
if FEATURES_FILE is None:
    st.error(f"No features CSV found. Put one of these in `{ARTIFACT_DIR}/`: \n"
             "- features_table_fast.csv\n- features_table.csv")
    st.stop()

@st.cache_data(ttl=3600)
def load_features(path):
    return pd.read_csv(path)

df = load_features(FEATURES_FILE)
st.sidebar.markdown(f"**Loaded features file:** `{os.path.basename(FEATURES_FILE)}`")

# ---- Load model & scaler with graceful fallback ----
MODEL_FILE = find_model_file()
SCALER_FILE = find_scaler_file()

@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    model = None
    scaler = None
    if model_path is not None and os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.warning(f"Failed to load model `{os.path.basename(model_path)}`: {e}")
            model = None
    if scaler_path is not None and os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.warning(f"Failed to load scaler `{os.path.basename(scaler_path)}`: {e}")
            scaler = None
    return model, scaler

model, scaler = load_model_and_scaler(MODEL_FILE, SCALER_FILE)
st.sidebar.write("Model:", os.path.basename(MODEL_FILE) if MODEL_FILE else "None")
st.sidebar.write("Scaler:", os.path.basename(SCALER_FILE) if SCALER_FILE else "None")

# ---- Sidebar filters ----
st.sidebar.header("Filters / Controls")
participants = df['participant'].dropna().unique().tolist() if 'participant' in df.columns else []
participants = sorted(participants)
selected_participant = st.sidebar.selectbox("Participant", ["All"] + participants) if participants else "All"

topics = ["All"]
if 'topic' in df.columns:
    topics = ["All"] + sorted(df['topic'].fillna("NA").unique().tolist())
selected_topic = st.sidebar.selectbox("Topic", topics)

# ---- Filtered view ----
filtered = df.copy()
if selected_participant != "All":
    filtered = filtered[filtered['participant'] == selected_participant]
if selected_topic != "All":
    filtered = filtered[filtered['topic'] == selected_topic]

st.subheader("Dataset overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Samples (filtered)", int(len(filtered)))
col2.metric("Total participants", int(df['participant'].nunique()) if 'participant' in df.columns else 0)
col3.metric("Label 0", int((filtered['label_enc']==0).sum()) if 'label_enc' in filtered.columns else 0)
col4.metric("Label 1", int((filtered['label_enc']==1).sum()) if 'label_enc' in filtered.columns else 0)

st.markdown("### Sample table (filtered)")
show_cols = [c for c in ['word', 'topic', 'participant', 'interestingness', 'label_enc'] if c in df.columns]
st.dataframe(filtered[show_cols].reset_index().rename(columns={'label_enc':'semantic_relevance'}).head(200))

# ---- Feature detection (robust) ----
def detect_feature_columns(df):
    # prefer channel style 'ch' columns
    ch_cols = [c for c in df.columns if c.startswith('ch')]
    if len(ch_cols) > 0:
        return sorted(ch_cols)
    # fallback: numeric columns excluding label/meta
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop = {'semantic_relevance', 'label_enc'}
    numeric_cols = [c for c in numeric_cols if c not in drop]
    return sorted(numeric_cols)

candidate_feature_cols = detect_feature_columns(df)
st.sidebar.write(f"Detected candidate feature columns: {len(candidate_feature_cols)}")

# If model is available and reports expected feature count, align to that.
feature_cols = candidate_feature_cols.copy()
if model is not None:
    expected_n = getattr(model, "n_features_in_", None)
    if expected_n is not None:
        canonical_order = ['mean','std','min','max','len','bp_delta','bp_theta','bp_alpha','bp_beta','interestingness']
        ordered = [c for c in canonical_order if c in candidate_feature_cols]
        remaining = [c for c in candidate_feature_cols if c not in ordered]
        ordered.extend(sorted(remaining))
        if len(ordered) >= expected_n:
            feature_cols = ordered[:expected_n]
            st.sidebar.write(f"Auto-aligned features to model's expected count: {expected_n} features.")
        else:
            feature_cols = ordered
            st.sidebar.warning(f"Model expects {expected_n} features but only found {len(ordered)} in the CSV. Missing features will be zero-filled at prediction time.")
else:
    st.sidebar.info("No model loaded — using detected feature columns from CSV.")

st.sidebar.write(f"Final feature columns used (count={len(feature_cols)}):")
st.sidebar.write(feature_cols)

if len(feature_cols) == 0:
    st.error("No feature columns detected in the features file. Expected columns starting with 'ch' or numeric feature columns.")
    st.stop()

# ---- Sample inspector + prediction (robust selection) ----
st.markdown("### Inspect a sample & predict")
min_index, max_index = int(df.index.min()), int(df.index.max())
sample_idx = st.number_input("Select sample index (from features CSV index)", min_value=min_index, max_value=max_index, value=min_index, step=1)

# robust retrieval: try loc, then iloc
sample = None
try:
    sample = df.loc[sample_idx]
except Exception:
    try:
        sample = df.iloc[int(sample_idx)]
    except Exception:
        sample = None

if sample is None:
    st.warning("Couldn't locate the selected sample index in the features table.")
else:
    st.write("Metadata:", {k: sample.get(k, None) for k in ['word','topic','participant','interestingness','semantic_relevance'] if k in df.columns})

    # optional raw viewer (requires HF dataset) — lazy loaded
    if st.checkbox("Show raw waveform & PSD (requires HF dataset)"):
        try:
            from datasets import load_dataset
            dset = load_dataset(HF_DATASET_ID)['train']
            raw = np.asarray(dset[int(sample_idx)]['eeg'])
            if raw.ndim == 2:
                ch_count = raw.shape[0]
                ch_choice = st.selectbox("Channel", ["Average"] + [f"ch{c}" for c in range(ch_count)])
                if ch_choice == "Average":
                    sig = raw.mean(axis=0)
                else:
                    sig = raw[int(ch_choice.replace("ch",""))]
            else:
                sig = raw
            if DOWNSAMPLE_FACTOR > 1:
                sig = sig[::DOWNSAMPLE_FACTOR]
            t = np.arange(len(sig)) / (SF // max(1, DOWNSAMPLE_FACTOR))
            fig = go.Figure(); fig.add_trace(go.Scatter(x=t, y=sig)); fig.update_layout(title='Raw EEG', height=300)
            st.plotly_chart(fig, width='stretch')
            f, Pxx = welch(sig, fs=SF//max(1,DOWNSAMPLE_FACTOR), nperseg=min(len(sig), N_PERSEG))
            fig2 = go.Figure(); fig2.add_trace(go.Scatter(x=f, y=Pxx)); fig2.update_layout(title='PSD', xaxis=dict(range=[0,60]), height=300)
            st.plotly_chart(fig2, width='stretch')
        except Exception as e:
            st.warning(f"Could not load HF dataset raw signals: {e}")

    # display subset of features
    st.markdown("#### Feature preview (first 40 columns)")
    preview_cols = feature_cols[:40]
    if len(preview_cols) > 0:
        st.table(pd.DataFrame(sample[preview_cols]).T.rename(columns={0:'value'}))
    else:
        st.info("No features to show.")

    # prepare feature vector for prediction
    candidate_map = {c: float(sample[c]) for c in candidate_feature_cols if c in sample.index}
    aligned_vals = []
    missing_cols = []
    for c in feature_cols:
        if c in candidate_map:
            aligned_vals.append(candidate_map[c])
        else:
            aligned_vals.append(0.0)
            missing_cols.append(c)

    extras = [c for c in candidate_feature_cols if c not in feature_cols]
    if missing_cols:
        st.warning(f"Filling missing {len(missing_cols)} feature(s) with zeros: {missing_cols}")
    if extras:
        st.info(f"Ignoring {len(extras)} extra column(s) from CSV not used by model: {extras}")

    feat_vals = np.asarray(aligned_vals, dtype=np.float32).reshape(1, -1) if len(aligned_vals) > 0 else None

    # scaling if scaler available
    if feat_vals is not None and scaler is not None:
        try:
            feat_vals_scaled = scaler.transform(feat_vals)
        except Exception as e:
            st.warning(f"Scaler transform failed: {e}. Using unscaled features.")
            feat_vals_scaled = feat_vals
    else:
        feat_vals_scaled = feat_vals

    # prediction (guarded)
    if model is None:
        st.info("Model not loaded — prediction is disabled. Load a compatible model or retrain locally.")
    else:
        if feat_vals_scaled is None or feat_vals_scaled.size == 0:
            st.error("Feature vector is empty. Check feature detection and sample index.")
        else:
            try:
                expected_n = getattr(model, "n_features_in_", feat_vals_scaled.shape[1])
                if feat_vals_scaled.shape[1] != expected_n:
                    st.error(f"Aligned vector has {feat_vals_scaled.shape[1]} features but model expects {expected_n}. Prediction aborted.")
                else:
                    pred = model.predict(feat_vals_scaled)[0]
                    proba = model.predict_proba(feat_vals_scaled).tolist()[0] if hasattr(model, "predict_proba") else None
                    st.success(f"Predicted label: {int(pred)}")
                    if proba:
                        st.write("Probabilities:", proba)
            except Exception as e:
                st.error(f"Model prediction failed: {e}")

# ---- PCA scatter (sampled) ----
st.markdown("### PCA of features (sample)")
n_samples_pca = min(3000, len(df))
sample_idx_pca = np.random.RandomState(0).choice(len(df), size=n_samples_pca, replace=False)
X_sample = df.iloc[sample_idx_pca][feature_cols].to_numpy(dtype=np.float32)
if scaler is not None:
    try:
        Xs = scaler.transform(X_sample)
    except Exception:
        Xs = X_sample
else:
    Xs = X_sample

from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=0).fit_transform(Xs)
pca_df = pd.DataFrame(pca, columns=['PC1','PC2'])
if 'label_enc' in df.columns:
    pca_df['label'] = df.iloc[sample_idx_pca]['label_enc'].values
else:
    pca_df['label'] = 0
fig = px.scatter(pca_df, x='PC1', y='PC2', color='label', title='PCA (sampled)')
st.plotly_chart(fig, width='stretch')

# ---- LIVE EDA COMPUTATION PANEL ----
st.markdown("## Live EDA tools (compute on demand)")
eda_cols = st.columns([1,1,1])
# need HF dataset only for waveform/PSD across samples
hf_loaded = False
hf_dataset = None

with eda_cols[0]:
    if st.button("Compute PSD by label"):
        st.info("Computing average PSD per label (may take a while, requires HF dataset)...")
        try:
            from datasets import load_dataset
            hf_dataset = load_dataset(HF_DATASET_ID)['train']
            hf_loaded = True
        except Exception as e:
            st.warning(f"Could not load HF dataset: {e}")
            hf_loaded = False

        labels = sorted(df['label_enc'].unique()) if 'label_enc' in df.columns else [0]
        plt.figure(figsize=(8,4))
        any_plotted = False
        for lab in labels:
            idxs = df.index[df['label_enc']==lab].tolist()[:200]
            psd_list = []
            for i in idxs:
                try:
                    raw = np.asarray(hf_dataset[int(i)]['eeg'])
                    sig = raw.mean(axis=0) if raw.ndim == 2 else raw
                    f,P = welch(sig, fs=SF, nperseg=min(len(sig), N_PERSEG))
                    psd_list.append(P)
                except Exception:
                    continue
            if len(psd_list) > 0:
                Pavg = np.mean(psd_list, axis=0)
                plt.semilogy(f, Pavg, label=f'label_{lab}')
                any_plotted = True
        if any_plotted:
            plt.xlim(0,60); plt.legend(); plt.title("Average PSD by label"); plt.xlabel("Hz")
            st.pyplot(plt.gcf())
            plt.clf()
        else:
            st.info("No PSDs computed (HF dataset may not be available or indices failed).")

with eda_cols[1]:
    if st.button("Compute ERP (avg waveform)"):
        st.info("Computing ERP / average waveform per label (requires HF dataset)...")
        try:
            if not hf_loaded:
                from datasets import load_dataset
                hf_dataset = load_dataset(HF_DATASET_ID)['train']
                hf_loaded = True
        except Exception as e:
            st.warning(f"Could not load HF dataset: {e}")
            hf_loaded = False

        plt.figure(figsize=(10,4))
        any_plotted = False
        for lab in sorted(df['label_enc'].unique()):
            idxs = df.index[df['label_enc']==lab].tolist()[:200]
            sigs=[]
            for i in idxs:
                try:
                    raw = np.asarray(hf_dataset[int(i)]['eeg'])
                    sig = raw.mean(axis=0) if raw.ndim==2 else raw
                    sigs.append(sig[:2000])
                except Exception:
                    continue
            if len(sigs) > 0:
                minlen = min(len(s) for s in sigs)
                avg = np.mean([s[:minlen] for s in sigs], axis=0)
                plt.plot(avg, label=f"label_{lab}")
                any_plotted = True
        if any_plotted:
            plt.legend(); plt.title("Average waveform by label"); st.pyplot(plt.gcf()); plt.clf()
        else:
            st.info("No waveforms computed (HF dataset missing or indices failed).")

with eda_cols[2]:
    if st.button("Feature correlation heatmap"):
        st.info("Computing correlation heatmap...")
        if sns is None:
            try:
                import seaborn as sns  # attempt import
            except Exception:
                sns = None
        if sns is None:
            st.warning("seaborn not available. Install with `pip install seaborn` to use heatmap.")
        else:
            corr = df[feature_cols].corr()
            plt.figure(figsize=(8,6))
            sns.heatmap(corr, cmap='vlag', center=0)
            plt.title("Feature correlation")
            st.pyplot(plt.gcf())
            plt.clf()

# Outlier detection
if st.button("Show outlier summary"):
    lengths = df['len'] if 'len' in df.columns else None
    if lengths is not None:
        q1,q3 = lengths.quantile(0.25), lengths.quantile(0.75)
        iqr = q3-q1
        upper = q3 + 1.5*iqr
        out_len = df[lengths > upper]
        st.write("Length outliers:", len(out_len))
        st.dataframe(out_len.head(50))
    amp_out = df[
        ((df['max'] > df['max'].quantile(0.99)) if 'max' in df.columns else False) |
        ((df['min'] < df['min'].quantile(0.01)) if 'min' in df.columns else False)
    ]
    st.write("Amplitude outliers:", len(amp_out))
    st.dataframe(amp_out.head(50))

# Spectrogram viewer for arbitrary sample
st.markdown("#### Spectrogram viewer")
spec_idx = st.number_input("Spectrogram sample index", min_value=min_index, max_value=max_index, value=min_index, step=1, key="spec_idx")
if st.button("Compute spectrogram for sample", key="spec_btn"):
    try:
        from datasets import load_dataset
        hf_dataset = load_dataset(HF_DATASET_ID)['train']
        raw = np.asarray(hf_dataset[int(spec_idx)]['eeg'])
        sig = raw.mean(axis=0) if raw.ndim==2 else raw
        f,t,Sxx = spectrogram(sig, fs=SF, nperseg=min(N_PERSEG, len(sig)), noverlap=min(N_PERSEG//2, len(sig)//2))
        plt.figure(figsize=(8,4))
        plt.pcolormesh(t, f, 10*np.log10(Sxx+1e-12))
        plt.ylabel("Freq (Hz)"); plt.xlabel("Time (s)")
        plt.ylim(0,60)
        st.pyplot(plt.gcf()); plt.clf()
    except Exception as e:
        st.warning(f"Could not compute spectrogram: {e}")

# UMAP (optional)
if umap is None:
    try:
        import umap  # try import if available now
        umap = umap
    except Exception:
        umap = None

if umap is not None:
    if st.button("Compute UMAP embedding"):
        st.info("Computing UMAP — this may take time for large sets.")
        try:
            sample_n = min(3000, len(df))
            idxs = np.random.RandomState(0).choice(len(df), size=sample_n, replace=False)
            emb = umap.UMAP(n_components=2, random_state=0).fit_transform(df.iloc[idxs][feature_cols].values)
            emb_df = pd.DataFrame(emb, columns=['u1','u2'])
            if 'label_enc' in df.columns:
                emb_df['label'] = df.iloc[idxs]['label_enc'].values
            fig = px.scatter(emb_df, x='u1', y='u2', color='label', title='UMAP (sampled)')
            st.plotly_chart(fig, width='stretch')
        except Exception as e:
            st.warning(f"UMAP failed: {e}")
else:
    st.info("UMAP not installed. Install with `pip install umap-learn` to enable UMAP.")

# ---- PCA done earlier, now EDA artifact explorer ----
st.markdown("### Artifacts & EDA figures")

def list_artifacts(root="artifacts", max_items=1000):
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            try:
                size_kb = round(os.path.getsize(full)/1024, 1)
            except Exception:
                size_kb = None
            rel = os.path.relpath(full, start=root)
            rows.append({"path": full, "relpath": rel, "size_kb": size_kb})
            if len(rows) >= max_items:
                return rows
    return rows

art_rows = list_artifacts(ARTIFACT_DIR)
st.write(f"Found {len(art_rows)} artifact files under `{ARTIFACT_DIR}/`")
if len(art_rows) > 0:
    art_df = pd.DataFrame(art_rows)
    st.dataframe(art_df.head(400))

# display images in artifacts/eda
EDA_DIR = os.path.join(ARTIFACT_DIR, "eda")
st.markdown("#### Saved EDA figures (artifacts/eda)")

if os.path.exists(EDA_DIR):
    imgs = [os.path.join(EDA_DIR, f) for f in sorted(os.listdir(EDA_DIR)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if len(imgs) == 0:
        st.info("No image files found in artifacts/eda/.")
    else:
        st.write(f"Found {len(imgs)} image(s) in `{EDA_DIR}`.")
        for img_path in imgs:
            fn = os.path.basename(img_path)
            st.markdown(f"**{fn}**")
            try:
                st.image(img_path, caption=fn, width=800)
            except Exception as e:
                st.warning(f"Could not display `{fn}`: {e}")
            with open(img_path, "rb") as f:
                st.download_button(label=f"Download {fn}", data=f, file_name=fn, key=f"dl_{fn}")
            st.markdown("---")
else:
    st.info("No `artifacts/eda` directory found. Place EDA images under `artifacts/eda/` to see them here.")

# Show uploaded screenshot if present
if os.path.exists(UPLOADED_SCREENSHOT_PATH):
    st.markdown("#### Uploaded screenshot (session)")
    try:
        st.image(UPLOADED_SCREENSHOT_PATH, caption=os.path.basename(UPLOADED_SCREENSHOT_PATH), width=800)
    except Exception as e:
        st.warning(f"Could not display uploaded screenshot: {e}")
else:
    st.info(f"No session screenshot found at `{UPLOADED_SCREENSHOT_PATH}` (optional)")

# ---- Downloads ----
st.sidebar.markdown("### Downloads")
if st.sidebar.button("Download features CSV"):
    try:
        with open(FEATURES_FILE, "rb") as f:
            st.sidebar.download_button("Download features CSV", f, file_name=os.path.basename(FEATURES_FILE))
    except Exception as e:
        st.sidebar.warning(f"Failed to provide features download: {e}")

if MODEL_FILE and st.sidebar.button("Download model file"):
    try:
        with open(MODEL_FILE, "rb") as f:
            st.sidebar.download_button("Download model", f, file_name=os.path.basename(MODEL_FILE))
    except Exception as e:
        st.sidebar.warning(f"Failed to provide model download: {e}")

st.sidebar.info("App loaded. If predictions or live EDA fail, check model/scaler sklearn versions or install optional packages (seaborn, umap-learn).")
