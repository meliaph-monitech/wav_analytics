# -*- coding: utf-8 -*-
"""250417_SoundWaveAnalytics_V0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15CSf-8djdEQfY1qc6RUyJBAE6_XNvxre
"""

import streamlit as st
import zipfile
import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.io import wavfile
from tempfile import TemporaryDirectory

st.title("WAV File Visualizer and CSV Exporter (10ms Resolution)")

# Upload ZIP
uploaded_zip = st.file_uploader("Upload a ZIP file containing WAV files", type="zip")

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        wav_files = [f for f in zip_ref.namelist() if f.endswith(".wav")]

        if not wav_files:
            st.warning("No .wav files found in the uploaded ZIP.")
        else:
            selected_files = st.multiselect("Select WAV files to visualize", wav_files)

            # Store processed CSVs in memory
            processed_csvs = {}
            fig = go.Figure()

            for file in selected_files:
                with zip_ref.open(file) as wav_file:
                    samplerate, data = wavfile.read(io.BytesIO(wav_file.read()))

                    if data.ndim > 1:
                        data = data.mean(axis=1)  # Convert to mono by averaging channels

                    # Calculate average absolute amplitude every 10ms
                    window_size = int(samplerate * 0.01)  # 10ms
                    trimmed_len = len(data) - len(data) % window_size
                    data = data[:trimmed_len]
                    reshaped = data.reshape(-1, window_size)
                    avg_values = reshaped.mean(axis=1)
                    time_axis = np.arange(len(avg_values)) * 10  # in ms

                    df = pd.DataFrame({
                        "Time (ms)": time_axis,
                        "Amplitude": avg_values
                    })

                    processed_csvs[file] = df

                    fig.add_trace(go.Scatter(
                        x=df["Time (ms)"],
                        y=df["Amplitude"],
                        mode="lines",
                        name=file
                    ))

            if selected_files:
                st.plotly_chart(fig, use_container_width=True)

                # Single or bulk download options
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