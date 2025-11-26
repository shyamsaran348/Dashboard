# streamlit_app.py
"""
Robust Streamlit dashboard for EEG Semantic Relevance.
Place your artifacts inside ./artifacts/ (features CSV and joblib model/scaler).
This app will try fast filenames first and fall back to the slower filenames.
It now auto-aligns feature columns to the model's expected feature count (if available).
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.signal import welch
import plotly.express as px
import plotly.graph_objects as go
    
# ---- Config ----
ARTIFACT_DIR = "artifacts"
# prefer fast artifacts if present, fall back to normal names
FAST_FEATURES = os.path.join(ARTIFACT_DIR, "features_table_fast.csv")
SLOW_FEATURES = os.path.join(ARTIFACT_DIR, "features_table.csv")

FAST_MODEL = os.path.join(ARTIFACT_DIR, "rf_baseline_fast.joblib")
SLOW_MODEL = os.path.join(ARTIFACT_DIR, "rf_baseline.joblib")

FAST_SCALER = os.path.join(ARTIFACT_DIR, "scaler_fast.joblib")
SLOW_SCALER = os.path.join(ARTIFACT_DIR, "scaler.joblib")  # try this name if used

HF_DATASET_ID = "Quoron/EEG-semantic-text-relevance"  # optional raw viewer

# sampling params (not used for features but used for raw viewer)
SF = 250
DOWNSAMPLE_FACTOR = 1
N_PERSEG = 256 // max(1, DOWNSAMPLE_FACTOR)

st.set_page_config(layout="wide", page_title="EEG Semantic Relevance Dashboard")
st.title("EEG Semantic Relevance — Dashboard (robust loader)")

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
    if model_path is not None:
        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.warning(f"Failed to load model `{os.path.basename(model_path)}`: {e}")
            model = None
    if scaler_path is not None:
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
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
    # allow NA topics
    filtered = filtered[filtered['topic'] == selected_topic]

st.subheader("Dataset overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Samples (filtered)", int(len(filtered)))
col2.metric("Total participants", int(df['participant'].nunique()) if 'participant' in df.columns else 0)
col3.metric("Label 0", int((filtered['label_enc']==0).sum()) if 'label_enc' in filtered.columns else 0)
col4.metric("Label 1", int((filtered['label_enc']==1).sum()) if 'label_enc' in filtered.columns else 0)

st.markdown("### Sample table (filtered)")
show_cols = [c for c in ['word', 'topic', 'participant', 'interestingness', 'label_enc'] if c in df.columns]
st.dataframe(filtered[show_cols].reset_index().rename(columns={'label_enc':'semantic_relevance'}) .head(200))

# ---- Feature detection (robust) ----
def detect_feature_columns(df):
    # prefer channel style 'ch0_' columns
    ch_cols = [c for c in df.columns if c.startswith('ch')]
    if len(ch_cols) > 0:
        return sorted(ch_cols)
    # fallback: any numeric columns excluding label/meta
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop = {'semantic_relevance', 'label_enc'}
    numeric_cols = [c for c in numeric_cols if c not in drop]
    return sorted(numeric_cols)

# initial detect
candidate_feature_cols = detect_feature_columns(df)
st.sidebar.write(f"Detected candidate feature columns: {len(candidate_feature_cols)}")

# If model is available and reports expected feature count, align to that.
feature_cols = candidate_feature_cols.copy()
if model is not None:
    expected_n = getattr(model, "n_features_in_", None)
    if expected_n is not None:
        # Use a canonical order where possible so we can remove meta columns like 'interestingness'
        canonical_order = ['mean','std','min','max','len','bp_delta','bp_theta','bp_alpha','bp_beta','interestingness']
        # build ordered list of candidate features following canonical order
        ordered = [c for c in canonical_order if c in candidate_feature_cols]
        # if canonical didn't include all, append remaining candidates
        remaining = [c for c in candidate_feature_cols if c not in ordered]
        ordered.extend(sorted(remaining))
        # Now trim/pick the first expected_n features
        if len(ordered) >= expected_n:
            feature_cols = ordered[:expected_n]
            st.sidebar.write(f"Auto-aligned features to model's expected count: {expected_n} features.")
        else:
            # if too few features present, keep ordered and we'll fill missing during prediction
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
        # if sample_idx is a position
        sample = df.iloc[int(sample_idx)]
    except Exception:
        sample = None

if sample is None:
    st.warning("Couldn't locate the selected sample index in the features table.")
else:
    st.write("Metadata:", {k: sample.get(k, None) for k in ['word','topic','participant','interestingness','semantic_relevance'] if k in df.columns})

    # optional raw viewer (requires HF dataset, lazy load)
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
            st.plotly_chart(fig, use_container_width=True)
            f, Pxx = welch(sig, fs=SF//max(1,DOWNSAMPLE_FACTOR), nperseg=min(len(sig), N_PERSEG))
            fig2 = go.Figure(); fig2.add_trace(go.Scatter(x=f, y=Pxx)); fig2.update_layout(title='PSD', xaxis=dict(range=[0,60]), height=300)
            st.plotly_chart(fig2, use_container_width=True)
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
    # Build a map of available candidate values
    candidate_map = {c: float(sample[c]) for c in candidate_feature_cols if c in sample.index}
    # Build aligned feature vector according to feature_cols (fill missing with 0.0)
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
                # final sanity check vs model
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
# try to scale
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
st.plotly_chart(fig, use_container_width=True)

# ---- Downloads ----
st.sidebar.markdown("### Downloads")
if st.sidebar.button("Download features CSV"):
    with open(FEATURES_FILE, "rb") as f:
        st.sidebar.download_button("Download features CSV", f, file_name=os.path.basename(FEATURES_FILE))
if MODEL_FILE and st.sidebar.button("Download model file"):
    try:
        with open(MODEL_FILE, "rb") as f:
            st.sidebar.download_button("Download model", f, file_name=os.path.basename(MODEL_FILE))
    except Exception as e:
        st.sidebar.warning(f"Failed to provide model download: {e}")

st.sidebar.info("App loaded. If predictions fail, check model/scaler sklearn versions or retrain locally.")
