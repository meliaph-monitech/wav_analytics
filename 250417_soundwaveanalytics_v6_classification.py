import streamlit as st
import zipfile
import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.io import wavfile
from scipy.signal import welch, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tempfile import TemporaryDirectory

# Page settings
st.set_page_config(layout="wide")
st.title("🎧 Welding Sound Classification App")

# --- Global Settings ---
label_colors = {
    "OK": "green",
    "ALU_GAP": "orange",
    "ALU_POWER": "red",
    "CU_GAP": "blue",
    "CU_POWER": "purple"
}

bands = [(0, 5000), (5000, 10000), (10000, 15000), (15000, 20000)]

# --- Helper Functions ---

def extract_label_from_train_path(path):
    parts = path.strip("/").split("/")
    if len(parts) > 1:
        return parts[-2].replace("ALL_", "OK").upper()
    return "UNKNOWN"

def downsample_waveform(data, samplerate):
    window_size = int(samplerate * 0.01)
    trimmed_len = len(data) - len(data) % window_size
    reshaped = data[:trimmed_len].reshape(-1, window_size)
    return reshaped.mean(axis=1)

def extract_band_energy(freqs, psd):
    band_energies = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        energy = np.mean(psd[mask]) if mask.any() else 0
        band_energies.append(energy)
    return band_energies

def process_zip_file(zip_file, is_training=True):
    waveform_fig = go.Figure()
    freq_fig = go.Figure()
    bar_fig = go.Figure()
    radar_fig = go.Figure()
    fft_peak_hist = []

    features, labels = [], []
    test_file_features = []

    with TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

            wav_paths = []
            if is_training:
                # Traverse subfolders
                for root, _, files in os.walk(tmpdir):
                    for file in files:
                        if file.endswith(".wav"):
                            wav_paths.append(os.path.join(root, file))
            else:
                # Flat structure
                wav_paths = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".wav")]

            for path in wav_paths:
                samplerate, data = wavfile.read(path)
                if data.ndim > 1:
                    data = data.mean(axis=1)

                label = extract_label_from_train_path(path) if is_training else None
                color = label_colors.get(label, "black") if label else "black"
                line_dash = "solid" if is_training else "dash"

                # --- Time-Domain Plot ---
                waveform = downsample_waveform(data, samplerate)
                time_axis = np.arange(len(waveform)) * 10  # ms
                waveform_fig.add_trace(go.Scatter(
                    x=time_axis, y=waveform,
                    mode="lines", name=os.path.basename(path),
                    line=dict(color=color, dash=line_dash)
                ))

                # --- Frequency-Domain ---
                freqs, psd = welch(data, fs=samplerate, nperseg=2048)
                db = 10 * np.log10(psd + 1e-12)
                mask = (freqs >= 0) & (freqs <= 20000)
                freqs, db = freqs[mask], db[mask]

                freq_fig.add_trace(go.Scatter(
                    x=freqs, y=db,
                    mode="lines", name=os.path.basename(path),
                    line=dict(color=color, dash=line_dash),
                    fill='tozeroy'
                ))

                # --- Band Energy ---
                energy = extract_band_energy(freqs, psd)
                band_labels = [f"{low//1000}-{high//1000}kHz" for low, high in bands]

                bar_fig.add_trace(go.Bar(
                    x=band_labels, y=energy,
                    name=os.path.basename(path),
                    marker_color=color
                ))

                radar_fig.add_trace(go.Scatterpolar(
                    r=energy + [energy[0]],
                    theta=band_labels + [band_labels[0]],
                    fill='toself',
                    name=os.path.basename(path),
                    line=dict(color=color, dash=line_dash)
                ))

                # --- FFT Peak Histogram ---
                peaks, _ = find_peaks(db, height=np.max(db) - 10)
                fft_peak_hist.extend(freqs[peaks])

                if is_training:
                    features.append(energy)
                    labels.append(label)
                else:
                    test_file_features.append((os.path.basename(path), energy))

    return {
        "waveform_fig": waveform_fig,
        "freq_fig": freq_fig,
        "bar_fig": bar_fig,
        "radar_fig": radar_fig,
        "fft_peak_freqs": fft_peak_hist,
        "features": features,
        "labels": labels,
        "test_features": test_file_features
    }

# --- Streamlit Layout ---

st.sidebar.header("📥 Upload Files")
train_zip = st.sidebar.file_uploader("Upload Training ZIP", type="zip", key="train_zip")
test_zip = st.sidebar.file_uploader("Upload Test ZIP", type="zip", key="test_zip")

# --- Training Phase ---
if train_zip:
    st.header("🔧 Training Data Visualizations")
    train_data = process_zip_file(train_zip, is_training=True)

    st.subheader("📈 Waveform")
    st.plotly_chart(train_data["waveform_fig"], use_container_width=True)

    st.subheader("🔊 Spectrum")
    st.plotly_chart(train_data["freq_fig"], use_container_width=True)

    st.subheader("📊 Band Energy - Bar")
    st.plotly_chart(train_data["bar_fig"], use_container_width=True)

    st.subheader("🧭 Band Energy - Radar")
    st.plotly_chart(train_data["radar_fig"], use_container_width=True)

    st.subheader("📉 Histogram of FFT Peak Frequencies")
    st.plotly_chart(go.Figure([go.Histogram(x=train_data["fft_peak_freqs"], nbinsx=50)]), use_container_width=True)

    st.subheader("🤖 Training Classifier")
    X_train, X_test, y_train, y_test = train_test_split(train_data["features"], train_data["labels"], test_size=0.2, stratify=train_data["labels"], random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.text(classification_report(y_test, y_pred))

    # --- Test Phase ---
    if test_zip:
        st.header("🧪 Test Data & Predictions")
        test_data = process_zip_file(test_zip, is_training=False)

        for filename, feature in test_data["test_features"]:
            prediction = model.predict([feature])[0]
            st.success(f"✅ **{filename}** → Predicted as **{prediction}**")

        # Add test plots
        for trace in test_data["waveform_fig"].data:
            train_data["waveform_fig"].add_trace(trace)
        st.subheader("📈 Waveform (with Test)")
        st.plotly_chart(train_data["waveform_fig"], use_container_width=True)

        for trace in test_data["freq_fig"].data:
            train_data["freq_fig"].add_trace(trace)
        st.subheader("🔊 Spectrum (with Test)")
        st.plotly_chart(train_data["freq_fig"], use_container_width=True)

        for trace in test_data["radar_fig"].data:
            train_data["radar_fig"].add_trace(trace)
        st.subheader("🧭 Radar Energy (with Test)")
        st.plotly_chart(train_data["radar_fig"], use_container_width=True)
