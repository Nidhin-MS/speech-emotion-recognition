import streamlit as st
import numpy as np
import librosa
import tensorflow as tf


model = tf.keras.models.load_model("SER_model.h5")

emotion_labels = ["angry","happy","sad"]

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    max_len = 130

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc

st.title("🎙 Speech Emotion Recognition")
st.write("Upload a speech audio file (.wav)")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    features = extract_features(uploaded_file)

    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)

    emotion = emotion_labels[predicted_class]

    st.success(f"Predicted Emotion: **{emotion.upper()}**")