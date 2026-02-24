import streamlit as st
import numpy as np
import librosa
import tensorflow as tf


model = tf.keras.models.load_model("SER_model.h5")


