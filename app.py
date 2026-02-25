import streamlit as st
import numpy as np
import librosa
import tensorflow as tf


model = tf.keras.models.load_model("SER_model.h5")

emotion_labels = ["angry","happy","sad"]

def extract_features(file):
    audio, sr = librosa.load(file, duration=3, offset=0.5)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    if mfcc.shape[1] >= 130:
        mfcc = mfcc[:, :130]
    else:
        pad_width = 130 - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)))

    mfcc = mfcc[..., np.newaxis]

    mfcc = np.expand_dims(mfcc, axis=0)

    return mfcc

st.title("🎙 Speech Emotion Recognition")
st.write("Upload a speech audio file (.wav)")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    features = extract_features(uploaded_file)
    print(features.shape)
    prediction = model.predict(features)
    
    predicted_class = np.argmax(prediction)

    emotion = emotion_labels[predicted_class]

    st.success(f"Predicted Emotion: **{emotion.upper()}**")