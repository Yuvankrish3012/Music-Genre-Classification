import streamlit as st
import numpy as np
import librosa
import pickle
from sklearn.preprocessing import StandardScaler

# --- Load model and label encoder ---
MODEL_PATH = "D:/ML PROJECTS/Music Genre Classification/music_genre_model_26.pkl"
ENCODER_PATH = "D:/ML PROJECTS/Music Genre Classification/genre_label_encoder.pkl"

model = pickle.load(open(MODEL_PATH, "rb"))
label_encoder = pickle.load(open(ENCODER_PATH, "rb"))

# --- Streamlit UI setup ---
st.set_page_config(page_title="🎵 Music Genre Classifier", page_icon="🎶")
st.markdown("<h1 style='text-align: center;'>🎧 Music Genre Prediction</h1>", unsafe_allow_html=True)
st.write("Upload a 30-second `.wav` audio file and I'll tell you its genre!")

# --- File uploader ---
uploaded_file = st.file_uploader("Upload your audio file", type=["wav"])

# --- Feature extractor function ---
def extract_features(file):
    y, sr = librosa.load(file, duration=30)
    features = []

    features.append(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.rms(y=y)))
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features.append(np.mean(mfcc[i]))

    return np.array(features).reshape(1, -1)

# --- Prediction & Result Display ---
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.info("⏳ Extracting features and predicting...")

    try:
        features = extract_features(uploaded_file)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        prediction = model.predict(features_scaled)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        st.success(f"🎶 Predicted Genre: **{predicted_label.upper()}**")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
