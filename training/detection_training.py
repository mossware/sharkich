import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import os
import joblib
import noisereduce as nr

print("Running...")

def shift_pitch(audio, sample_rate, n_steps):
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)

def augment_audio(audio, sample_rate):
    pitch_shift = np.random.uniform(-2, 2)
    audio = shift_pitch(audio, sample_rate, pitch_shift)
    return audio

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    n_fft = 2048  # Set FFT size to 2048
    audio = nr.reduce_noise(y=audio, sr=sample_rate, prop_decrease=0.7, time_mask_smooth_ms=50)
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
    return features

def prepare_data(drone_folder, non_drone_folder, moving_drone_folder):
    features, labels = [], []
    
    # Process drone samples
    for file_name in os.listdir(drone_folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(drone_folder, file_name)
            feature = extract_features(file_path)
            features.append(feature)
            labels.append(1)  
    
    # Process non-drone samples
    for file_name in os.listdir(non_drone_folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(non_drone_folder, file_name)
            feature = extract_features(file_path)
            features.append(feature)
            labels.append(0)  
    
    # Process moving drone samples
    for file_name in os.listdir(moving_drone_folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(moving_drone_folder, file_name)
            feature = extract_features(file_path)
            features.append(feature)
            labels.append(2)  
    
    return np.array(features), np.array(labels)

drone_folder = "C:/Users/bansh/OneDrive/Рабочий стол/audio detection/new data set/drone_samples"
non_drone_folder = "C:/Users/bansh/OneDrive/Рабочий стол/audio detection/new data set/non_drone_samples"
moving_drone_folder = "C:/Users/bansh/OneDrive/Рабочий стол/audio detection/new data set/moving_drone_samples"

features, labels = prepare_data(drone_folder, non_drone_folder, moving_drone_folder)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 20, None],  
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_
y_pred_best = best_clf.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print("Improved accuracy: ", accuracy_best)

joblib.dump(best_clf, "sharkichAI_updated_three_classes.pkl")
print("Model trained and saved as 'sharkichAI_updated_three_classes.pkl'")
