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
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from tempfile import TemporaryDirectory

# Set page config
st.set_page_config(layout="wide")
st.title("🎧 Welding Sound Classifier")

# Color map for labels
label_colors = {
    "OK": "green",
    "ALU_GAP": "orange",
    "ALU_POWER": "red",
    "CU_GAP": "blue",
    "CU_POWER": "purple"
}

# Extract label from folder name
def extract_label_from_path(path):
    folder = os.path.dirname(path).split("/")[-1].upper()
    return folder.replace("ALL_", "OK")

# Extract WAV data from ZIP
def process_zip(zip_file, is_training=True):
    features = []
    labels = []
    waveforms = []
    fft_peaks = []
    band_energies = []

    waveform_fig = go.Figure()
    freq_fig = go.Figure()
    radar_fig = go.Figure()
    bar_fig = go.Figure()

    peak_freq_all = []

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        wav_files = [f for f in zip_ref.namelist() if f.endswith(".wav")]
        for file in wav_files:
            with zip_ref.open(file) as wav_file:
                samplerate, data = wavfile.read(io.BytesIO(wav_file.read()))
                if data.ndim > 1:
                    data = data.mean(axis=1)
                
                label = extract_label_from_path(file) if is_training else None
                color = label_colors.get(label, "gray") if label else "black"

                # --- Time-Domain Downsample ---
                window_size = int(samplerate * 0.01)
                trimmed = len(data) - len(data) % window_size
                reshaped = data[:trimmed].reshape(-1, window_size)
                avg = reshaped.mean(axis=1)
                time_axis = np.arange(len(avg)) * 10  # ms

                waveform_fig.add_trace(go.Scatter(
                    x=time_axis, y=avg,
                    mode="lines", name=f"{file} ({label if label else 'Test'})",
                    line=dict(color=color, dash="solid" if is_training else "dash")
                ))

                # --- Frequency-Domain ---
                freqs, psd = welch(data, fs=samplerate, nperseg=2048)
                db = 10 * np.log10(psd + 1e-12)
                mask = (freqs >= 0) & (freqs <= 20000)
                freqs_masked = freqs[mask]
                db_masked = db[mask]

                freq_fig.add_trace(go.Scatter(
                    x=freqs_masked,
                    y=db_masked,
                    mode="lines",
                    fill='tozeroy',
                    name=f"{file} ({label if label else 'Test'})",
                    line=dict(color=color, dash="solid" if is_training else "dash")
                ))

                # --- Band Energy ---
                bands = [(0, 5000), (5000, 10000), (10000, 15000), (15000, 20000)]
                energy_per_band = []
                for b_start, b_end in bands:
                    band_mask = (freqs >= b_start) & (freqs < b_end)
                    energy = np.mean(psd[band_mask]) if band_mask.any() else 0
                    energy_per_band.append(energy)

                bar_fig.add_trace(go.Bar(
                    x=[f"{s//1000}-{e//1000}kHz" for s, e in bands],
                    y=energy_per_band,
                    name=f"{file} ({label if label else 'Test'})",
                    marker_color=color
                ))

                radar_fig.add_trace(go.Scatterpolar(
                    r=energy_per_band + [energy_per_band[0]],
                    theta=[f"{s//1000}-{e//1000}kHz" for s, e in bands] + [f"{bands[0][0]//1000}-{bands[0][1]//1000}kHz"],
                    fill='toself',
                    name=f"{file} ({label if label else 'Test'})",
                    line=dict(color=color, dash="solid" if is_training else "dash")
                ))

                # --- FFT Peak Frequencies ---
                peaks, _ = find_peaks(db, height=np.max(db) - 10)
                peak_freqs = freqs[peaks]
                peak_freq_all.extend(peak_freqs)

                if is_training:
                    features.append(energy_per_band)
                    labels.append(label)
                else:
                    band_energies.append((file, energy_per_band))

    return {
        "waveform_fig": waveform_fig,
        "freq_fig": freq_fig,
        "radar_fig": radar_fig,
        "bar_fig": bar_fig,
        "features": features,
        "labels": labels,
        "peak_freqs": peak_freq_all,
        "test_features": band_energies
    }

# --- Upload Training Data ---
st.sidebar.header("🔧 Training Data")
train_zip = st.sidebar.file_uploader("Upload Training ZIP", type="zip")

if train_zip:
    st.header("📊 Training Data Visualizations")
    result = process_zip(train_zip, is_training=True)

    st.subheader("Waveform (Time-Domain)")
    st.plotly_chart(result["waveform_fig"], use_container_width=True)

    st.subheader("Spectrum (Frequency-Domain)")
    st.plotly_chart(result["freq_fig"], use_container_width=True)

    st.subheader("Band Energy - Bar Plot")
    st.plotly_chart(result["bar_fig"], use_container_width=True)

    st.subheader("Band Energy - Radar Plot")
    st.plotly_chart(result["radar_fig"], use_container_width=True)

    st.subheader("Histogram of FFT Peak Frequencies")
    st.plotly_chart(go.Figure([go.Histogram(x=result["peak_freqs"], nbinsx=50)]), use_container_width=True)

    # --- Train Classifier ---
    st.subheader("🤖 Classifier Performance")
    X_train, X_test, y_train, y_test = train_test_split(result["features"], result["labels"], test_size=0.2, stratify=result["labels"])
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.text(classification_report(y_test, y_pred))

    # --- Upload Test Data ---
    st.sidebar.header("🧪 Test Data")
    test_zip = st.sidebar.file_uploader("Upload Test ZIP", type="zip")

    if test_zip:
        st.header("🧪 Test Data Predictions & Visualizations")
        test_result = process_zip(test_zip, is_training=False)

        # Predict
        test_file_labels = []
        for file, feature in test_result["test_features"]:
            predicted = model.predict([feature])[0]
            test_file_labels.append((file, predicted))
            st.write(f"✅ **{file}** → Predicted as **{predicted}**")

        # Merge plots
        st.subheader("Waveform (Including Test Data)")
        merged_waveform = result["waveform_fig"]
        for trace in test_result["waveform_fig"].data:
            merged_waveform.add_trace(trace)
        st.plotly_chart(merged_waveform, use_container_width=True)

        st.subheader("Spectrum (Including Test Data)")
        merged_freq = result["freq_fig"]
        for trace in test_result["freq_fig"].data:
            merged_freq.add_trace
