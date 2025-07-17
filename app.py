from flask import Flask, render_template, request
import librosa
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import pyttsx3

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
WAVEFORM_PATH = 'static/waveform.png'
STATIC_FOLDER = 'static'
os.makedirs(STATIC_FOLDER, exist_ok=True)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = tf.keras.models.load_model('model/emotion_bilstm_final_model.keras')
labels = ['angry', 'boredom', 'disgust', 'fear', 'happy', 'neutral', 'sad']

# Extract features
def extract_features(file_path, max_len=160):
    y, sr = librosa.load(file_path, duration=4, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2]).T
    if combined.shape[0] < max_len:
        combined = np.pad(combined, ((0, max_len - combined.shape[0]), (0, 0)), mode='constant')
    else:
        combined = combined[:max_len, :]
    return combined.reshape(1, max_len, 60)

# Plot waveform
def plot_waveform(file_path, save_path=WAVEFORM_PATH):
    y, sr = librosa.load(file_path, duration=4)
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, color='#00f29c')
    plt.title("Waveform")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    message = None
    if request.method == 'POST':
        file = request.files['audio']
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            # Predict emotion
            features = extract_features(path)
            pred = model.predict(features)
            emotion = labels[np.argmax(pred)]

            # Voice output
            engine = pyttsx3.init()
            engine.say(f"You are {emotion} now")
            engine.runAndWait()
            message = f"You are {emotion} now"

            # Plot waveform
            plot_waveform(path)

    return render_template('index.html', message=message)
if __name__ == '__main__':
    app.run(debug=True)