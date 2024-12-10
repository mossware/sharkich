Acoustic Drone Detection System

This project is an acoustic-based real-time drone detection system using machine learning. The system detects the presence of drones by analyzing audio signals captured in real-time, processes them with a trained model, and then triggers actions (via an Arduino) based on the detection result.

Features:
Real-time audio processing to detect drone sounds.
Uses a pre-trained machine learning model for classification.
Sends signals to an Arduino to activate indicators based on detection (e.g., LED or buzzer).
Noise reduction using the noisereduce library.
Sliding window mechanism for more consistent and reliable detection.
Requirements:
Python 3.6+
Libraries: sounddevice, numpy, librosa, joblib, noisereduce, serial
Hardware: Raspberry Pi / Orange Pi, Arduino for signal output
A trained machine learning model (e.g., sharkichAI_updated_three_classes.pkl)

    Setup: 
        Install Dependencies:

pip install sounddevice numpy librosa joblib noisereduce serial

        Model:
Download sharkichAI_updated_three_classes.pkl in the same directory as the script or specify its path.

        Arduino:
Connect your Arduino via USB and ensure it communicates with the specified port (COM4 in this example).
The system sends '2' for drone detected, '1' for partial detection, and '0' for no detection.

    How It Works:
Audio Input: The system continuously listens for audio input using the sounddevice library.
Noise Reduction: Noise is reduced in the audio signal using the noisereduce library to improve detection accuracy.
Feature Extraction: The audio data is converted into a set of features using librosa (e.g., spectral centroid, MFCC, and tonnetz).
Prediction: The extracted features are input into a pre-trained machine learning model to classify whether a drone is present.
Sliding Window: A sliding window is used to ensure that drone detection is consistent over multiple frames.
Arduino Control: Based on the prediction, the system sends a signal to the Arduino to indicate the detection status (no detection, partial detection, or consistent detection).
Real-time Detection:
The real_time_detection() function continuously reads audio, extracts features, performs predictions, and controls the Arduino in real-time.


Acknowledgments:
Librosa: Used for audio feature extraction.
Noisereduce: Used to reduce background noise in audio recordings.
Joblib: Used for loading the trained model.
Arduino: Used for hardware control.