# Speech Emotion Recognition(SER)

This project is a deep learning based system that identifies human emotions from speech audio files.When a user uploads a .wav audio file,the system extracts important sound features and uses a trained CNN model to predict whether the emotion is happy,sad,or angry.

# Dataset

I used the RAVDESS dataset from kaggle.  
It contains speech recordings from 24 professional actors(audio_speech_actors_01-24).  

For this project,I used only three emotions:
- Happy
- Sad
- Angry

# How It Works

1. The system takes an audio(.wav) file as input.
2. It extracts audio features(MFCC–Mel Frequency Cepstral Coefficients).
3. The features are reshaped into a format suitable for deep learning.
4. A Convolutional Neural Network(CNN) model predicts the emotion.

# Model Training

The complete model training process is available in "SER_training.ipynb".

It includes:
- Loading the RAVDESS dataset(audio_speech_actors_01-24)from google drive
- Feature extraction(MFCC)
- Data preprocessing
- CNN model building
- Model training and evaluation
- Saving the trained model as SER_model.h5

 I trained the CNN model in colab and saved in HDF5(.h5) format as "SER_model.h5" and added this file to VScode.This file contains the learned weights of the model trained on the RAVDESS dataset.Since it is a binary file GitHub cannot display its internal content in the browser.However,it can be downloaded and directly loaded when running the project.

# Model Architecture

The model is built using TensorFlow/Keras and includes:

- Conv2D layers for feature extraction
- MaxPooling2D layers for downsampling
- Flatten layer to convert features into 1D
- Dense layer for final emotion classification

Input shape used:(40,130,1)

# Technologies Used

- Python
- TensorFlow/Keras
- Librosa
- NumPy
- Streamlit(for simple UI)

# How to Run the Project

1. Install dependencies:
pip install -r requirements.txt

2. Run the application:
streamlit run app.py

3. You can either upload your own .wav audio file or use the sample audio files provided in the "sample_audios" folder to test the system.

# Conclusion

This project shows how we can use deep learning to understand human emotions from voice recordings.The system listens to a speech audio file,analyzes important sound patterns,and predicts whether the emotion is happy,sad,or angry.
It is a complete working system where audio is taken as input,processed using a CNN model,and the predicted emotion is shown through a simple and easy to use interface.