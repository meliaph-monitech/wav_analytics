import streamlit as st
import zipfile
import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.io import wavfile
from scipy.signal import welch
from tempfile import TemporaryDirectory

st.title("📊 WAV Visualizer & Frequency Explorer")

uploaded_zip = st.file_uploader("Upload ZIP with WAV files", type="zip")

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        wav_files = [f for f in zip_ref.namelist() if f.endswith(".wav")]

        if not wav_files:
            st.warning("No .wav files found in the uploaded ZIP.")
        else:
            selected_files = st.multiselect("Select WAV files to visualize", wav_files)

            interval_ms = st.slider("Select interval (ms)", 1, 100, 10)

            st.markdown("### Frequency Range Settings (for Spectral View)")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                min_freq = st.number_input("Min Frequency (Hz)", value=0)
            with col2:
                max_freq = st.number_input("Max Frequency (Hz)", value=8000)
            with col3:
                min_db = st.number_input("Min dB", value=-100)
            with col4:
                max_db = st.number_input("Max dB", value=0)

            waveform_fig = go.Figure()
            freq_fig = go.Figure()
            processed_csvs = {}

            for file in selected_files:
                with zip_ref.open(file) as wav_file:
                    samplerate, data = wavfile.read(io.BytesIO(wav_file.read()))

                    if data.ndim > 1:
                        data = data.mean(axis=1)  # mono

                    # Waveform (Amplitude over Time)
                    window_size = int(samplerate * (interval_ms / 1000.0))
                    trimmed_len = len(data) - len(data) % window_size
                    data_wave = data[:trimmed_len]
                    reshaped = data_wave.reshape(-1, window_size)
                    avg_values = reshaped.mean(axis=1)
                    time_axis = np.arange(len(avg_values)) * interval_ms

                    df = pd.DataFrame({
                        "Time (ms)": time_axis,
                        "Amplitude": avg_values
                    })
                    processed_csvs[file] = df

                    waveform_fig.add_trace(go.Scatter(
                        x=df["Time (ms)"],
                        y=df["Amplitude"],
                        mode="lines",
                        name=file
                    ))

                    # Frequency Domain (dB vs Frequency)
                    freqs, psd = welch(data, fs=samplerate, nperseg=1024)
                    db = 10 * np.log10(psd + 1e-12)  # avoid log(0)
                    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)

                    freq_fig.add_trace(go.Scatter(
                        x=freqs[freq_mask],
                        y=np.clip(db[freq_mask], min_db, max_db),
                        mode="lines",
                        fill='tozeroy',
                        name=file
                    ))

            if selected_files:
                st.subheader("📈 Time-Domain Waveform")
                st.plotly_chart(waveform_fig, use_container_width=True)

                st.subheader("🔊 Frequency-Domain (Power Spectral Density)")
                freq_fig.update_layout(
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="dB",
                    yaxis=dict(range=[min_db, max_db])
                )
                st.plotly_chart(freq_fig, use_container_width=True)

                # Download buttons
                if len(selected_files) == 1:
                    file = selected_files[0]
                    csv = processed_csvs[file].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        f"Download CSV for {file}",
                        csv,
                        file_name=f"{os.path.splitext(os.path.basename(file))[0]}.csv",
                        mime="text/csv"
                    )
                else:
                    with TemporaryDirectory() as temp_dir:
                        for file in selected_files:
                            name = os.path.splitext(os.path.basename(file))[0] + ".csv"
                            processed_csvs[file].to_csv(os.path.join(temp_dir, name), index=False)
                        zip_bytes = io.BytesIO()
                        with zipfile.ZipFile(zip_bytes, "w") as zf:
                            for file in selected_files:
                                name = os.path.splitext(os.path.basename(file))[0] + ".csv"
                                zf.write(os.path.join(temp_dir, name), arcname=name)
                        zip_bytes.seek(0)
                        st.download_button(
                            "Download All CSVs as ZIP",
                            zip_bytes,
                            file_name="converted_csvs.zip",
                            mime="application/zip"
                        )
