# Music-Genre-Classification

This project classifies 30-second music clips into genres such as **rock, classical, metal, reggae, jazz, and more** using machine learning on extracted audio features.

---

## 📁 Dataset Used

- Dataset: [GTZAN Genre Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- Files Used:
  - `features_30_sec.csv` (for model training)
  - Audio `.wav` files (for real-time testing via UI)

---

## 🧠 ML Model Details

| Model                  | Random Forest Classifier |
|------------------------|--------------------------|
| Input Features         | 26                       |
| Train/Test Split       | 80/20                    |
| Test Accuracy          | ✅ **0.65**              |

---

## 📈 Evaluation Results

✅ **Accuracy**: `0.65`  
📋 **Classification Report (Summary)**:

| Genre     | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Classical | 0.95      | 0.90   | 0.90     |
| Metal     | 0.90      | 0.84   | 0.84     |
| Blues     | 0.70      | 0.65   | 0.65     |
| Pop       | 0.70      | 0.62   | 0.62     |
| Others    | Moderate scores |

---

## 📊 Visualizations

> These visualizations give insights into the distribution and relevance of features used in genre classification.

### 🎶 Genre Distribution

![image](https://github.com/user-attachments/assets/9721b520-9b51-4772-a636-7ee8527c57a6)


### 📊 Mean Feature Value Per Genre

![image](https://github.com/user-attachments/assets/9613c51c-79a8-4950-b3dc-1037bdc87b07

### 🎼 Tempo Distribution Across Genres

![image](https://github.com/user-attachments/assets/725727ca-62cd-4c61-9279-7a0b1cde6d2e)


### 🔍 Pairplot of Key Features

![image](https://github.com/user-attachments/assets/61451ca2-7483-45a9-8c91-62eab14a57af)


---

## 🌐 Streamlit Web App

We built a live prediction UI using **Streamlit** that takes in a `.wav` file and predicts the genre using the trained model.

### ▶️ How to Launch the UI

```bash
conda activate myenv
streamlit run "D:/ML PROJECTS/Music Genre Classification/music_genre_app.py"
📁 Files Needed
music_genre_model_26.pkl – trained ML model

genre_label_encoder.pkl – encoder to map genre labels

music_genre_app.py – Streamlit frontend

💡 Features Used (26)
Type	Examples
Spectral	spectral_centroid_mean, rolloff_mean
Temporal	zero_crossing_rate_mean, tempo
Harmonic	chroma_stft_mean, rms_mean
MFCCs	mfcc1_mean to mfcc20_mean

🧪 What the UI Does
Accepts .wav file (30 seconds)

Extracts features using librosa

Predicts genre with RandomForest

Shows result + plays audio in-browser

![image](https://github.com/user-attachments/assets/b46d7b48-8077-42a0-8c8d-fcdbbfabdaca)


📦 Optional
If model files (.pkl) are too large for GitHub:

🔗 [Insert: Google Drive Link to Download .pkl Files]

🚀 Future Improvements
📈 Improve model accuracy using CNN on spectrograms

🧠 Try deep learning models like CRNN

🎛️ Add top-3 predictions using predict_proba()

🔉 Add spectrogram visualization in UI

🧪 Use audio augmentation for training

🙌 Credits
Dataset: GTZAN via Kaggle

Feature Extraction: librosa

Modeling: scikit-learn

Frontend: Streamlit

Made with ❤️ by Yuvan Krishnan
