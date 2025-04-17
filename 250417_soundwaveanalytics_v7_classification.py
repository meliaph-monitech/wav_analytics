import streamlit as st
import zipfile
import os
import io
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.io import wavfile
from scipy.signal import welch, find_peaks
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory

st.set_page_config(layout="wide")
st.title("🔊 Welding Sound Classification App")

# --- Sidebar ---
st.sidebar.header("Upload ZIP Files")
train_zip = st.sidebar.file_uploader("📁 Training ZIP (folders as labels)", type="zip")
test_zip = st.sidebar.file_uploader("🧪 Test ZIP (WAV files only)", type="zip")
# normalize = st.sidebar.checkbox("☑️ Normalize Amplitude and Features", value=False)

# --- Constants ---
BANDS = [(0, 5000), (5000, 10000), (10000, 15000), (15000, 20000)]
BAND_LABELS = [f"{b[0]//1000}-{b[1]//1000}kHz" for b in BANDS]
UNIFORM_FREQS = np.linspace(0, 20000, 200)

def extract_features(wav_data, samplerate):
    if wav_data.ndim > 1:
        wav_data = wav_data.mean(axis=1)
    # if normalize:
    #     wav_data = wav_data / np.max(np.abs(wav_data))

    freqs, psd = welch(wav_data, fs=samplerate, nperseg=2048)
    db = 10 * np.log10(psd + 1e-12)

    # Interpolated FFT profile
    interp_fft = np.interp(UNIFORM_FREQS, freqs, db, left=db[0], right=db[-1])

    # Band energy
    band_energies = []
    for start, end in BANDS:
        mask = (freqs >= start) & (freqs < end)
        energy = np.mean(psd[mask]) if np.any(mask) else 0
        band_energies.append(energy)

    return band_energies + interp_fft.tolist()

def process_zip_file(zip_file, is_training=True):
    features = []
    labels = []
    names = []
    with zipfile.ZipFile(zip_file, 'r') as z:
        if is_training:
            folders = [f.filename for f in z.filelist if f.is_dir()]
            for folder in folders:
                for f in z.namelist():
                    if f.startswith(folder) and f.endswith(".wav"):
                        label = os.path.basename(folder.strip("/"))
                        with z.open(f) as wav_file:
                            sr, data = wavfile.read(io.BytesIO(wav_file.read()))
                            feat = extract_features(data, sr)
                            features.append(feat)
                            labels.append(label)
                            names.append(os.path.basename(f))
        else:
            for f in z.namelist():
                if f.endswith(".wav"):
                    with z.open(f) as wav_file:
                        sr, data = wavfile.read(io.BytesIO(wav_file.read()))
                        feat = extract_features(data, sr)
                        features.append(feat)
                        names.append(os.path.basename(f))
    return features, labels if is_training else None, names

# --- MAIN WORKFLOW ---
if train_zip:
    st.subheader("📊 Training Data Summary & Visualization")
    train_X, train_y, train_names = process_zip_file(train_zip, is_training=True)
    st.write(f"✅ {len(train_X)} training samples loaded from ZIP.")

    # if normalize:
    #     scaler = StandardScaler()
        # train_X = scaler.fit_transform(train_X)

    # PCA for visualization
    pca_vis = PCA(n_components=2)
    train_2d = pca_vis.fit_transform(train_X)
    df_vis = pd.DataFrame(train_2d, columns=["PC1", "PC2"])
    df_vis["label"] = train_y
    fig_train = go.Figure()
    for label in sorted(set(train_y)):
        subset = df_vis[df_vis["label"] == label]
        fig_train.add_trace(go.Scatter(x=subset["PC1"], y=subset["PC2"], mode="markers", name=label))
    fig_train.update_layout(title="PCA Visualization of Training Data", xaxis_title="PC1", yaxis_title="PC2")
    st.plotly_chart(fig_train, use_container_width=True)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(train_X, train_y)
    st.success("🎯 Classifier trained successfully!")

    if test_zip:
        st.subheader("🧪 Test Data & Predictions")
        test_X, _, test_names = process_zip_file(test_zip, is_training=False)
        if normalize:
            test_X = scaler.transform(test_X)

        preds = clf.predict(test_X)

        result_df = pd.DataFrame({
            "Filename": test_names,
            "Predicted Label": preds
        })
        st.dataframe(result_df)

        # Prediction count bar chart
        bar_fig = go.Figure()
        label_counts = result_df["Predicted Label"].value_counts()
        bar_fig.add_trace(go.Bar(x=label_counts.index, y=label_counts.values))
        bar_fig.update_layout(title="Prediction Label Distribution", xaxis_title="Label", yaxis_title="Count")
        st.plotly_chart(bar_fig, use_container_width=True)

        # Download results
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Prediction CSV", csv, "predictions.csv", mime="text/csv")

else:
    st.info("📁 Please upload a training ZIP file to begin.")
