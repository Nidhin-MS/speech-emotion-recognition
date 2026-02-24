import streamlit as st
import numpy as np
import librosa
import tensorflow as tf


model = tf.keras.models.load_model("SER_model.h5")

emotion_labels = ["happy", "sad","angry"]

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
