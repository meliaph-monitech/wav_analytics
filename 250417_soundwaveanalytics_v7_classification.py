import streamlit as st
import zipfile
import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.io import wavfile
from scipy.signal import welch, find_peaks
from sklearn.decomposition import PCA
from tempfile import TemporaryDirectory
from scipy.interpolate import interp1d

st.set_page_config(layout="wide")
st.title("🎧 Welding Sound Classification Explorer")

# -- Sidebar Configs --
normalize = st.sidebar.checkbox("☑️ Normalize Amplitude and Features", value=True)

train_zip = st.sidebar.file_uploader("Upload TRAINING ZIP (Folders = Labels)", type="zip")
test_zip = st.sidebar.file_uploader("Upload TEST ZIP (WAV files only)", type="zip")

interval_ms = st.sidebar.slider("Downsampling Interval (ms)", 1, 100, 10)
min_freq = st.sidebar.number_input("Min Frequency (Hz)", value=0)
max_freq = st.sidebar.number_input("Max Frequency (Hz)", value=20000)
min_db = st.sidebar.number_input("Min dB", value=-100)
max_db = st.sidebar.number_input("Max dB", value=0)

bands = [(0, 5000), (5000, 10000), (10000, 15000), (15000, 20000)]
band_labels = [f"{lo//1000}-{hi//1000}kHz" for lo, hi in bands]
uniform_freqs = np.linspace(min_freq, max_freq, 200)

label_colors = {
    "All_OK": "green",
    "Alu_GAP": "orange",
    "Alu_POWER": "red",
    "Cu_GAP": "blue",
    "Cu_POWER": "purple",
    "UNKNOWN": "gray"
}


def extract_band_energy(freqs, psd):
    min_len = min(len(freqs), len(psd))
    freqs = freqs[:min_len]
    psd = psd[:min_len]
    band_energies = []
    for low, high in bands:
        band_mask = (freqs >= low) & (freqs < high)
        if not np.any(band_mask):
            band_energies.append(0)
        else:
            psd_band = psd[band_mask]
            band_energies.append(np.mean(psd_band))
    band_energies = np.array(band_energies)
    if normalize and np.sum(band_energies) > 0:
        band_energies = band_energies / np.sum(band_energies)
    return band_energies.tolist()


def process_zip_file(zip_file, is_training):
    features = []
    wave_fig = go.Figure()
    fft_fig = go.Figure()
    radar_data = []
    hist_peaks = []
    pca_vectors = []
    pca_labels = []
    file_names = []

    with zipfile.ZipFile(zip_file, "r") as z:
        wav_paths = [f for f in z.namelist() if f.endswith(".wav")]

        for path in wav_paths:
            label = path.split("/")[0] if is_training else "UNKNOWN"
            filename = os.path.basename(path)
            color = label_colors.get(label, "gray")

            with z.open(path) as file:
                samplerate, data = wavfile.read(io.BytesIO(file.read()))
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if normalize:
                    data = data / (np.max(np.abs(data)) + 1e-8)

                # --- Time-Domain ---
                window = int(samplerate * interval_ms / 1000)
                trim = len(data) - len(data) % window
                avg = data[:trim].reshape(-1, window).mean(axis=1)
                time_axis = np.arange(len(avg)) * interval_ms

                wave_fig.add_trace(go.Scatter(x=time_axis, y=avg, mode="lines",
                                              name=f"{filename} ({label})", line=dict(color=color)))

                # --- FFT / Welch ---
                freqs, psd = welch(data, fs=samplerate, nperseg=2048)
                db = 10 * np.log10(psd + 1e-12)
                mask = (freqs >= min_freq) & (freqs <= max_freq)
                freqs_masked, db_masked = freqs[mask], db[mask]

                fft_fig.add_trace(go.Scatter(
                    x=freqs_masked, y=np.clip(db_masked, min_db, max_db),
                    mode="lines", fill="tozeroy", name=f"{filename} ({label})",
                    line=dict(color=color)
                ))

                # --- Band Energy ---
                band_energy = extract_band_energy(freqs, psd)
                radar_data.append((filename, label, band_energy))

                # --- FFT Peaks ---
                peaks, _ = find_peaks(db, height=np.max(db) - 10)
                peak_freqs = freqs[peaks]
                hist_peaks.extend(peak_freqs)

                # --- PCA Vector ---
                if len(freqs_masked) >= 2:
                    interp = interp1d(freqs_masked, db_masked, bounds_error=False, fill_value="extrapolate")
                    interpolated = interp(uniform_freqs)
                    if normalize:
                        interpolated = (interpolated - interpolated.mean()) / (interpolated.std() + 1e-8)
                    pca_vectors.append(interpolated)
                    pca_labels.append(label)
                    file_names.append(filename)

    return {
        "waveform": wave_fig,
        "fft": fft_fig,
        "radar": radar_data,
        "hist": hist_peaks,
        "pca": (pca_vectors, pca_labels, file_names)
    }


# --- TRAINING SECTION ---
if train_zip:
    st.subheader("📘 Training Data Visualizations")
    train_data = process_zip_file(train_zip, is_training=True)

    st.markdown("### 📈 Time-Domain Signal")
    st.plotly_chart(train_data["waveform"], use_container_width=True)

    st.markdown("### 🔊 Frequency-Domain Spectrum")
    train_data["fft"].update_layout(xaxis_title="Frequency (Hz)", yaxis_title="dB")
    st.plotly_chart(train_data["fft"], use_container_width=True)

    st.markdown("### 📊 Band Energy (Radar & Bar)")
    radar_fig = go.Figure()
    bar_fig = go.Figure()
    for filename, label, energies in train_data["radar"]:
        radar_fig.add_trace(go.Scatterpolar(
            r=energies + [energies[0]],
            theta=band_labels + [band_labels[0]],
            fill="toself", name=f"{filename} ({label})",
            line=dict(color=label_colors.get(label, "gray"))
        ))
        bar_fig.add_trace(go.Bar(
            x=band_labels, y=energies,
            name=f"{filename} ({label})", marker_color=label_colors.get(label, "gray")
        ))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    bar_fig.update_layout(barmode="group", xaxis_title="Band", yaxis_title="Energy")
    st.plotly_chart(bar_fig, use_container_width=True)
    st.plotly_chart(radar_fig, use_container_width=True)

    st.markdown("### 📌 Histogram of FFT Peak Frequencies")
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=train_data["hist"], nbinsx=50))
    hist_fig.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Count")
    st.plotly_chart(hist_fig, use_container_width=True)

    st.markdown("### 📉 PCA Projection of FFT Vectors")
    try:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(np.array(train_data["pca"][0]))
        pca_fig = go.Figure()
        for i, label in enumerate(train_data["pca"][1]):
            pca_fig.add_trace(go.Scatter(
                x=[reduced[i, 0]], y=[reduced[i, 1]],
                mode="markers+text", text=[train_data["pca"][2][i]],
                name=label, marker=dict(color=label_colors.get(label, "gray"), size=10),
                textposition="top center"
            ))
        pca_fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
        st.plotly_chart(pca_fig, use_container_width=True)
    except Exception as e:
        st.error(f"PCA error: {e}")

# --- TEST SECTION ---
if test_zip:
    st.subheader("🧪 Test Data Visualizations")
    test_data = process_zip_file(test_zip, is_training=False)

    st.markdown("### 📈 Time-Domain Signal")
    st.plotly_chart(test_data["waveform"], use_container_width=True)

    st.markdown("### 🔊 Frequency-Domain Spectrum")
    test_data["fft"].update_layout(xaxis_title="Frequency (Hz)", yaxis_title="dB")
    st.plotly_chart(test_data["fft"], use_container_width=True)

    st.markdown("### 📊 Band Energy (Radar & Bar)")
    radar_fig = go.Figure()
    bar_fig = go.Figure()
    for filename, label, energies in test_data["radar"]:
        radar_fig.add_trace(go.Scatterpolar(
            r=energies + [energies[0]],
            theta=band_labels + [band_labels[0]],
            fill="toself", name=filename,
            line=dict(color="gray")
        ))
        bar_fig.add_trace(go.Bar(x=band_labels, y=energies, name=filename, marker_color="gray"))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    bar_fig.update_layout(barmode="group", xaxis_title="Band", yaxis_title="Energy")
    st.plotly_chart(bar_fig, use_container_width=True)
    st.plotly_chart(radar_fig, use_container_width=True)

    st.markdown("### 📌 Histogram of FFT Peak Frequencies")
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=test_data["hist"], nbinsx=50))
    hist_fig.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Count")
    st.plotly_chart(hist_fig, use_container_width=True)
