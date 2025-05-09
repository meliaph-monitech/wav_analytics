import streamlit as st
import zipfile
import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.io import wavfile
from scipy.signal import spectrogram, welch, find_peaks
from sklearn.decomposition import PCA
from tempfile import TemporaryDirectory
from scipy.interpolate import interp1d

st.set_page_config(layout="wide")
st.title("🎧 Welding Sound Analyzer V3 with Label Filtering")

uploaded_zip = st.sidebar.file_uploader("Upload a ZIP file containing WAV files", type="zip")

label_colors = {
    "OK": "green",
    "GAP": "orange",
    "POWER": "red"
}

def extract_label(filename):
    return filename.split("_")[-1].replace(".wav", "").upper()

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        wav_files = [f for f in zip_ref.namelist() if f.endswith(".wav")]
        if not wav_files:
            st.warning("No .wav files found.")
        else:
            selected_files = st.sidebar.multiselect("Select WAV files to analyze", wav_files, default=wav_files)
            available_labels = sorted(set(extract_label(f) for f in selected_files))
            selected_labels = st.sidebar.multiselect("Filter by Label", options=available_labels, default=available_labels)

            interval_ms = st.sidebar.slider("Sampling interval (ms)", 1, 100, 10)

            st.sidebar.markdown("### Frequency Range")
            min_freq = st.sidebar.number_input("Min Frequency (Hz)", value=0)
            max_freq = st.sidebar.number_input("Max Frequency (Hz)", value=20000)

            st.sidebar.markdown("### dB Display Range")
            min_db = st.sidebar.number_input("Min dB", value=-100)
            max_db = st.sidebar.number_input("Max dB", value=0)

            waveform_fig = go.Figure()
            freq_fig = go.Figure()
            peak_data = []
            band_energy_data = []
            pca_vectors = []
            labels = []
            file_labels = []
            processed_csvs = []

            uniform_freqs = np.linspace(min_freq, max_freq, 200)  # Standardized frequency bins

            for file in selected_files:
                label = extract_label(file)
                if label not in selected_labels:
                    continue

                color = label_colors.get(label, "gray")

                with zip_ref.open(file) as wav_file:
                    samplerate, data = wavfile.read(io.BytesIO(wav_file.read()))
                    if data.ndim > 1:
                        data = data.mean(axis=1)

                    # Time-domain downsampling
                    window_size = int(samplerate * interval_ms / 1000)
                    trimmed = len(data) - len(data) % window_size
                    reshaped = data[:trimmed].reshape(-1, window_size)
                    avg = reshaped.mean(axis=1)
                    time_axis = np.arange(len(avg)) * interval_ms

                    df = pd.DataFrame({"Time (ms)": time_axis, "Amplitude": avg})
                    processed_csvs.append((file, df))

                    waveform_fig.add_trace(go.Scatter(
                        x=df["Time (ms)"], y=df["Amplitude"],
                        mode="lines", name=f"{file} ({label})",
                        line=dict(color=color)
                    ))

                    # FFT and dB conversion
                    freqs, psd = welch(data, fs=samplerate, nperseg=2048)
                    db = 10 * np.log10(psd + 1e-12)
                    mask = (freqs >= min_freq) & (freqs <= max_freq)
                    freqs_masked = freqs[mask]
                    db_masked = db[mask]

                    # Interpolate FFT profile for PCA
                    if len(freqs_masked) >= 2:
                        interp = interp1d(freqs_masked, db_masked, bounds_error=False, fill_value="extrapolate")
                        interpolated_db = interp(uniform_freqs)
                        pca_vectors.append(interpolated_db)
                        labels.append(label)
                        file_labels.append(file)

                    # Frequency plot
                    freq_fig.add_trace(go.Scatter(
                        x=freqs_masked,
                        y=np.clip(db_masked, min_db, max_db),
                        mode="lines",
                        fill='tozeroy',
                        name=f"{file} ({label})",
                        line=dict(color=color)
                    ))

                    # Band energy
                    bands = [(0, 5000), (5000, 10000), (10000, 15000), (15000, 20000)]
                    energy_per_band = []
                    for b_start, b_end in bands:
                        band_mask = (freqs >= b_start) & (freqs < b_end)
                        energy = np.mean(psd[band_mask]) if band_mask.any() else 0
                        energy_per_band.append(energy)
                    band_energy_data.append((file, label, energy_per_band))

                    # Peak frequencies
                    peaks, _ = find_peaks(db, height=np.max(db) - 10)
                    peak_freqs = freqs[peaks]
                    peak_data.extend(peak_freqs)

            # --- Waveform Plot ---
            st.subheader("📈 Time-Domain Signal")
            st.plotly_chart(waveform_fig, use_container_width=True)

            # --- FFT Area Plot ---
            st.subheader("🔊 Frequency-Domain Spectrum")
            freq_fig.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="dB")
            st.plotly_chart(freq_fig, use_container_width=True)

            # --- Band Energy Bar + Radar ---
            st.subheader("📊 Band Energy (Bar & Radar)")
            band_labels = [f"{start//1000}-{end//1000}kHz" for start, end in bands]

            bar_fig = go.Figure()
            for file, label, energies in band_energy_data:
                if label in selected_labels:
                    color = label_colors.get(label, "gray")
                    bar_fig.add_trace(go.Bar(
                        x=band_labels, y=energies,
                        name=f"{file} ({label})",
                        marker_color=color
                    ))
            bar_fig.update_layout(barmode="group", xaxis_title="Frequency Band", yaxis_title="Avg Energy")
            st.plotly_chart(bar_fig, use_container_width=True)

            radar_fig = go.Figure()
            for file, label, energies in band_energy_data:
                if label in selected_labels:
                    color = label_colors.get(label, "gray")
                    radar_fig.add_trace(go.Scatterpolar(
                        r=energies + [energies[0]],
                        theta=band_labels + [band_labels[0]],
                        fill='toself',
                        name=f"{file} ({label})",
                        line=dict(color=color)
                    ))
            radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
            st.plotly_chart(radar_fig, use_container_width=True)

            # --- FFT Peak Histogram ---
            st.subheader("📌 Histogram of FFT Peak Frequencies")
            hist_fig = go.Figure()
            hist_fig.add_trace(go.Histogram(x=peak_data, nbinsx=50))
            hist_fig.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Count")
            st.plotly_chart(hist_fig, use_container_width=True)

            # --- PCA Scatter ---
            st.subheader("📉 PCA - Frequency Profile Projection")
            if pca_vectors:
                try:
                    pca = PCA(n_components=2)
                    reduced = pca.fit_transform(np.array(pca_vectors))
                    pca_fig = go.Figure()
                    for i, label in enumerate(labels):
                        if label in selected_labels:
                            color = label_colors.get(label, "gray")
                            pca_fig.add_trace(go.Scatter(
                                x=[reduced[i, 0]], y=[reduced[i, 1]],
                                mode="markers+text",
                                text=[file_labels[i]],
                                name=label,
                                textposition="top center",
                                marker=dict(color=color, size=10)
                            ))
                    pca_fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
                    st.plotly_chart(pca_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"PCA Error: {e}")

            # --- CSV Download Buttons ---
            st.subheader("📥 Download Processed CSVs")
            filtered_csvs = [(f, df) for f, df in processed_csvs if extract_label(f) in selected_labels]
            if len(filtered_csvs) == 1:
                file, df = filtered_csvs[0]
                st.download_button(
                    f"Download CSV for {file}",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name=os.path.splitext(os.path.basename(file))[0] + ".csv",
                    mime="text/csv"
                )
            elif len(filtered_csvs) > 1:
                with TemporaryDirectory() as temp_dir:
                    for file, df in filtered_csvs:
                        name = os.path.splitext(os.path.basename(file))[0] + ".csv"
                        df.to_csv(os.path.join(temp_dir, name), index=False)
                    zip_bytes = io.BytesIO()
                    with zipfile.ZipFile(zip_bytes, "w") as zf:
                        for file, df in filtered_csvs:
                            name = os.path.splitext(os.path.basename(file))[0] + ".csv"
                            zf.write(os.path.join(temp_dir, name), arcname=name)
                    zip_bytes.seek(0)
                    st.download_button(
                        "Download All CSVs as ZIP",
                        zip_bytes,
                        file_name="converted_csvs.zip",
                        mime="application/zip"
                    )
