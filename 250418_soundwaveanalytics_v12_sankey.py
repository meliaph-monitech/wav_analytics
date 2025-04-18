import streamlit as st
import zipfile
import os
import io
import time
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.graph_objects as go
from collections import Counter
from scipy.io import wavfile
from scipy.signal import welch, find_peaks
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
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
classifier_name = st.sidebar.selectbox(
    "🤖 Select Classifier",
    [
        "RandomForest", "SVM", "KNN", "LogisticRegression",
        "DecisionTree", "GradientBoosting", "AdaBoost",
        "NaiveBayes", "MLP", "ExtraTrees", "QDA", "LDA"
    ]
)


# --- Constants ---
BANDS = [(0, 5000), (5000, 10000), (10000, 15000), (15000, 20000)]
BAND_LABELS = [f"{b[0]//1000}-{b[1]//1000}kHz" for b in BANDS]
UNIFORM_FREQS = np.linspace(0, 20000, 200)

def extract_features(wav_data, samplerate):
    if wav_data.ndim > 1:
        wav_data = wav_data.mean(axis=1)

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

def get_classifier(name):
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif name == "SVM":
        return SVC(probability=True)
    elif name == "KNN":
        return KNeighborsClassifier()
    elif name == "LogisticRegression":
        return LogisticRegression(max_iter=1000)
    elif name == "DecisionTree":
        return DecisionTreeClassifier()
    elif name == "GradientBoosting":
        return GradientBoostingClassifier()
    elif name == "AdaBoost":
        return AdaBoostClassifier()
    elif name == "NaiveBayes":
        return GaussianNB()
    elif name == "MLP":
        return MLPClassifier(max_iter=500)
    elif name == "ExtraTrees":
        return ExtraTreesClassifier()
    elif name == "QDA":
        return QuadraticDiscriminantAnalysis()
    elif name == "LDA":
        return LinearDiscriminantAnalysis()
    return RandomForestClassifier()

def extract_label_from_filename(name):
    parts = name.rsplit("_", 1)
    if len(parts) > 1 and "." in parts[1]:
        return parts[1].split(".")[0]
    return None

# --- MAIN WORKFLOW ---
if train_zip:
    st.subheader("📊 Training Data Summary & Visualization")
    train_X, train_y, train_names = process_zip_file(train_zip, is_training=True)
    st.write(f"✅ {len(train_X)} training samples loaded from ZIP.")

    # PCA for visualization
    pca_vis = PCA(n_components=2)
    train_2d = pca_vis.fit_transform(train_X)
    df_vis = pd.DataFrame(train_2d, columns=["PC1", "PC2"])
    df_vis["label"] = train_y
    # fig_train = go.Figure()
    # for label in sorted(set(train_y)):
    #     subset = df_vis[df_vis["label"] == label]
    #     fig_train.add_trace(go.Scatter(x=subset["PC1"], y=subset["PC2"], mode="markers", name=label))
    df_vis["Filename"] = train_names
    
    fig_train = go.Figure()
    for label in sorted(set(train_y)):
        subset = df_vis[df_vis["label"] == label]
        fig_train.add_trace(go.Scatter(
            x=subset["PC1"],
            y=subset["PC2"],
            mode="markers",
            name=label,
            text=subset["Filename"],
            hoverinfo="text+name+x+y"
        ))

    fig_train.update_layout(title="PCA Visualization of Training Data", xaxis_title="PC1", yaxis_title="PC2")
    st.plotly_chart(fig_train, use_container_width=True)

    # # Train classifier
    # clf = get_classifier(classifier_name)
    # clf.fit(train_X, train_y)
    # st.success(f"🎯 {classifier_name} trained successfully!")

    st.subheader("🛠️ Training Classifier")
    
    # Progress bar and training time
    progress = st.progress(0)
    status_text = st.empty()
    start_time = time.time()
    
    clf = get_classifier(classifier_name)
    status_text.text(f"Training {classifier_name} model...")
    progress.progress(30)
    clf.fit(train_X, train_y)
    progress.progress(100)
    
    end_time = time.time()
    train_duration = end_time - start_time
    status_text.text(f"{classifier_name} trained in {train_duration:.2f} seconds ✅")


    # Evaluate on training data
    train_preds = clf.predict(train_X)
    # st.subheader("📈 Training Evaluation Metrics")
    # st.text("Confusion Matrix:")
    # fig_cm, ax_cm = plt.subplots()
    # ConfusionMatrixDisplay.from_predictions(train_y, train_preds, ax=ax_cm)
    # st.pyplot(fig_cm)
    # st.text("Classification Report:")
    # st.text(classification_report(train_y, train_preds)
    st.subheader("📈 Model Evaluation on Training Data")
    
    col1, col2 = st.columns(2)
    
    # Confusion Matrix
    with col1:
        st.markdown("#### Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(train_y, clf.predict(train_X), ax=ax_cm)
        st.pyplot(fig_cm)
    
    # Classification Report as Table
    with col2:
        st.markdown("#### Classification Report")
        report_dict = classification_report(train_y, clf.predict(train_X), output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format(precision=2))
    # with col2:
    #     st.markdown("#### Classification Report")
    #     report_dict = classification_report(train_y, clf.predict(train_X), output_dict=True)
    #     report_df = pd.DataFrame(report_dict).transpose()
    #     st.dataframe(report_df.style.format(precision=2))
    
    #     st.markdown(f"⏱️ **Training Time:** `{train_duration:.2f} seconds`")

    if test_zip:
        st.subheader("🧪 Test Data & Predictions")
        test_X, _, test_names = process_zip_file(test_zip, is_training=False)
        preds = clf.predict(test_X)

        result_df = pd.DataFrame({
            "Filename": test_names,
            "Predicted Label": preds
        })

        # Extract label from filename
        result_df["True Label"] = result_df["Filename"].apply(extract_label_from_filename)
        result_df["Match"] = result_df["Predicted Label"] == result_df["True Label"]

        st.dataframe(result_df)

        # Real-world accuracy
        real_labels = result_df.dropna(subset=["True Label"])
        if not real_labels.empty:
            accuracy = (real_labels["Match"].sum() / len(real_labels)) * 100
            st.metric("🎯 Real Test Accuracy", f"{accuracy:.2f}%")
        else:
            st.warning("⚠️ No ground truth labels found in filenames for real test evaluation.")

        # # Prediction count bar chart
        # bar_fig = go.Figure()
        # label_counts = result_df["Predicted Label"].value_counts()
        # bar_fig.add_trace(go.Bar(x=label_counts.index, y=label_counts.values))
        # bar_fig.update_layout(title="Prediction Label Distribution", xaxis_title="Label", yaxis_title="Count")
        # st.plotly_chart(bar_fig, use_container_width=True)

        # 🔄 Sankey Diagram to visualize real test label flow        
        true_labels = result_df["True Label"].tolist()
        predicted_labels = result_df["Predicted Label"].tolist()
        
        # Unique label mapping
        label_set = sorted(set(true_labels + predicted_labels))
        label_to_index = {label: i for i, label in enumerate(label_set)}
        
        # Build source → target links
        source = [label_to_index[true] for true, pred in zip(true_labels, predicted_labels)]
        target = [label_to_index[pred] for true, pred in zip(true_labels, predicted_labels)]
        values = [1] * len(source)
        
        # Aggregate duplicate flows
        flow_counter = Counter(zip(source, target))
        source_unique, target_unique, value_unique = zip(*[(s, t, v) for (s, t), v in flow_counter.items()])
        
        # Plot Sankey
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=label_set
            ),
            link=dict(
                source=source_unique,
                target=target_unique,
                value=value_unique
            )
        )])
        
        st.markdown("### 🔄 Real Test Label Flow (True → Predicted)")
        st.plotly_chart(fig_sankey, use_container_width=True)
        
        # Show accuracy
        accuracy = np.mean(result_df["True Label"] == result_df["Predicted Label"])
        st.markdown(f"✅ **Real Test Accuracy:** `{accuracy*100:.2f}%`")

        st.markdown("### ❌ Misclassified Files")
        misclassified = result_df[result_df["True Label"] != result_df["Predicted Label"]]
        st.dataframe(misclassified)

        # Download results
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Prediction CSV", csv, "predictions.csv", mime="text/csv")

else:
    st.info("📁 Please upload a training ZIP file to begin.")
