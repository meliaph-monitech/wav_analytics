import streamlit as st
import zipfile
import os
import tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import librosa
import io
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Streamlit UI
st.set_page_config(layout="wide")
st.title("WAV File Unsupervised Analysis")

with st.sidebar:
    zip_file = st.file_uploader("Upload a ZIP of WAV files", type="zip")
    interval_ms = st.slider("Time Interval (ms)", 1, 100, 10)
    freq_min = st.number_input("Min Frequency (Hz)", value=0)
    freq_max = st.number_input("Max Frequency (Hz)", value=22050)
    db_min = st.number_input("Min dB", value=-100)
    db_max = st.number_input("Max dB", value=0)
    cluster_k = st.slider("Number of Clusters (KMeans)", 2, 10, 3)
    category_filter = st.multiselect("Filter by Category (OK, GAP, POWER)", ["OK", "GAP", "POWER"], default=["OK", "GAP", "POWER"])

# Temp directory to extract
if zip_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        wav_files = [f for f in os.listdir(tmpdir) if f.endswith(".wav")]
        wav_files.sort()

        all_time_data = {}
        all_fft_data = {}
        all_features = []
        labels = []
        file_names = []

        for fname in wav_files:
            label = fname.split("_")[-1].replace(".wav", "").upper()
            if label not in category_filter:
                continue
            fpath = os.path.join(tmpdir, fname)
            y, sr = librosa.load(fpath, sr=None)
            interval_samples = int((interval_ms / 1000) * sr)
            samples = y[::interval_samples]
            time = np.arange(len(samples)) * interval_ms / 1000
            all_time_data[fname] = (time, samples)

            # FFT
            fft = np.abs(np.fft.rfft(y))
            fft_db = librosa.amplitude_to_db(fft, ref=np.max)
            freqs = np.fft.rfftfreq(len(y), 1/sr)
            all_fft_data[fname] = (freqs, fft_db)

            # Feature extraction
            band_1 = np.mean(fft_db[(freqs >= 0) & (freqs < 5000)])
            band_2 = np.mean(fft_db[(freqs >= 5000) & (freqs < 10000)])
            band_3 = np.mean(fft_db[(freqs >= 15000) & (freqs < 20000)])
            band_4 = np.mean(fft_db[(freqs >= 20000)])
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
            flatness = librosa.feature.spectral_flatness(y=y)[0].mean()
            rms = librosa.feature.rms(y=y)[0].mean()

            feature_vector = [band_1, band_2, band_3, band_4, centroid, bandwidth, flatness, rms]
            all_features.append(feature_vector)

            labels.append(label)
            file_names.append(fname)

        # Plot time domain
        fig_time = go.Figure()
        for fname, (t, s) in all_time_data.items():
            label = fname.split("_")[-1].replace(".wav", "").upper()
            fig_time.add_trace(go.Scatter(x=t, y=s, mode='lines', name=label, legendgroup=label))
        fig_time.update_layout(title="Time Domain (Waveform)", xaxis_title="Time (s)", yaxis_title="Amplitude")
        st.plotly_chart(fig_time, use_container_width=True)

        # Plot FFT area plot
        fig_fft = go.Figure()
        for fname, (freqs, fft_db) in all_fft_data.items():
            mask = (freqs >= freq_min) & (freqs <= freq_max)
            label = fname.split("_")[-1].replace(".wav", "").upper()
            fig_fft.add_trace(go.Scatter(x=freqs[mask], y=fft_db[mask], fill='tozeroy', mode='lines', name=label, legendgroup=label))
        fig_fft.update_layout(title="Frequency Domain (FFT dB)", xaxis_title="Frequency (Hz)", yaxis_title="dB", yaxis_range=[db_min, db_max])
        st.plotly_chart(fig_fft, use_container_width=True)

        # Radar + Bar chart
        feature_df = pd.DataFrame(all_features, columns=[
            'band_0_5k', 'band_5k_10k', 'band_15k_20k', 'band_20k_up',
            'centroid', 'bandwidth', 'flatness', 'rms'])
        feature_df['label'] = labels
        feature_df['filename'] = file_names

        # Select only numeric feature columns for averaging
        numeric_cols = ['band_0_5k', 'band_5k_10k', 'band_15k_20k', 'band_20k_up',
                        'centroid', 'bandwidth', 'flatness', 'rms']
        
        avg_features = feature_df.groupby('label')[numeric_cols].mean()

        categories = avg_features.columns.tolist()

        fig_radar = go.Figure()
        for label in avg_features.index:
            fig_radar.add_trace(go.Scatterpolar(
                r=avg_features.loc[label].values,
                theta=categories,
                fill='toself',
                name=label
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Radar Plot of Average Features")
        st.plotly_chart(fig_radar, use_container_width=True)

        fig_bar = go.Figure()
        for label in avg_features.index:
            fig_bar.add_trace(go.Bar(x=categories, y=avg_features.loc[label].values, name=label))
        fig_bar.update_layout(barmode='group', title="Band Energy Bar Chart")
        st.plotly_chart(fig_bar, use_container_width=True)

        # PCA + KMeans
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(feature_df.drop(columns=['label', 'filename']))
        kmeans = KMeans(n_clusters=cluster_k, random_state=42)
        clusters = kmeans.fit_predict(features_2d)
        feature_df['PC1'] = features_2d[:, 0]
        feature_df['PC2'] = features_2d[:, 1]
        feature_df['cluster'] = clusters

        fig_pca = px.scatter(
            feature_df,
            x='PC1',
            y='PC2',
            color='label',                    # Color by category (OK, GAP, POWER)
            symbol='cluster',                # Shape by cluster number
            symbol_sequence=["circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down", "star"],
            hover_data=['filename'],
            title="PCA Scatter with Cluster Overlay"
        )
        fig_pca.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig_pca, use_container_width=True)


        # Download processed time data
        st.markdown("---")
        st.subheader("Download Time-Domain CSVs")
        for fname, (t, s) in all_time_data.items():
            df = pd.DataFrame({"time_s": t, "amplitude": s})
            csv = df.to_csv(index=False).encode()
            st.download_button(
                label=f"Download CSV for {fname}",
                data=csv,
                file_name=fname.replace(".wav", ".csv"),
                mime='text/csv'
            )
