import sounddevice as sd
import numpy as np
import librosa
import joblib
import time
import noisereduce as nr
import serial
from collections import deque

model_filename = r"your_trained_model.pkl"
loaded_model = joblib.load(model_filename)

arduino = serial.Serial(port='COM4', baudrate=115200, timeout=1) #make sure u chose the correct port

duration = 3  
sampling_rate = 22050  
n_fft = 2048
WINDOW_SIZE = 3  
THRESHOLD = 2  

sliding_window = deque(maxlen=WINDOW_SIZE)

def extract_features(audio_data, sample_rate):
    audio = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.7, time_mask_smooth_ms=50)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T)
    rms = np.mean(librosa.feature.rms(y=audio).T)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate, n_fft=n_fft).T)
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate, n_fft=n_fft).T)
    spectral_rolloff_50 = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, roll_percent=0.5, n_fft=n_fft).T)
    spectral_rolloff_25 = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, roll_percent=0.25, n_fft=n_fft).T)
    spectral_rolloff_75 = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, roll_percent=0.75, n_fft=n_fft).T)
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio, n_fft=n_fft).T)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_fft=n_fft).T, axis=0)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_fft=n_fft).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=audio, sr=sample_rate).T, axis=0)

    features = [zcr, rms, spectral_centroid, spectral_bandwidth, spectral_rolloff_50, spectral_rolloff_25, spectral_rolloff_75, spectral_flatness]
    features.extend(spectral_contrast)
    features.extend(mfcc)
    features.extend(chroma)
    features.extend(tonnetz)

    print(f"Extracted features: {len(features)} features.")  #debug
    return features

def real_time_detection():
    with sd.InputStream(samplerate=sampling_rate, channels=1, dtype='float32') as stream:
        while True:
            audio_data, overflowed = stream.read(int(sampling_rate * duration))
            if overflowed:
                print("Warning: Input overflow detected!")

            features = extract_features(audio_data.flatten(), sampling_rate)
            features_reshaped = np.array(features).reshape(1, -1)

            print(f"Feature shape for prediction: {features_reshaped.shape[1]} features.")  #debug

            try:
                prediction = loaded_model.predict(features_reshaped)
                print(f"Prediction: {prediction[0]}")

                sliding_window.append(prediction[0])

                if sliding_window.count(1) >= THRESHOLD:
                    print("Drone detected consistently!")
                    arduino.write(b'2')  

                elif sliding_window.count(1) > 0:
                    print("Some detection, but not consistent.")
                    arduino.write(b'1')  

                else:
                    print("No consistent drone detection.")
                    arduino.write(b'0')  

            except ValueError as e:
                print(f"Error with prediction: {e}")

            time.sleep(0.1)

real_time_detection()
