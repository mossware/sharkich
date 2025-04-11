# Acoustic Drone Detection System 🚁🔊

This project is an **acoustic-based real-time drone detection system** using **machine learning**. The system detects the presence of drones by analyzing audio signals captured in real-time, processes them with a trained **algorithm**, and then triggers actions (via an Arduino) based on the detection result.

---

### Features 🌟

- Real-time audio processing to detect drone sounds.
- Uses a pre-trained **machine learning algorithm** for classification.
- Sends signals to an **Arduino** to activate indicators based on detection (e.g., LED or buzzer).
- **Noise reduction** using the `noisereduce` library.
- **Sliding window mechanism** for more consistent and reliable detection.

---

### Requirements ⚙️

**Python 3.6+**  
Libraries:
- `sounddevice`
- `numpy`
- `librosa`
- `joblib`
- `noisereduce`
- `serial`

**Hardware**:
- **Raspberry Pi** / **Orange Pi**
- **Arduino** for signal output

**Algorithm**:
- A trained machine learning algorithm (e.g., `sharkichAI_updated_three_classes.pkl`)

---

### Setup 🛠️

#### 1. Install Dependencies:

```bash
pip install sounddevice numpy librosa joblib noisereduce serial
```

## 2. Algorithm

Download `sharkichAI_updated_three_classes.pkl` into the same directory as the script,  
or specify its path in the script manually.

## 3. Arduino

Connect your **Arduino** via USB and make sure it communicates with the correct port  
(**COM4** in this example — adjust if needed).

The system sends detection signals as follows:

- `2` — drone detected  
- `1` — partial detection  
- `0` — no detection

---

## System Diagram

Here's a flow of how the system processes audio input:

       ┌────────────┐
       │  Microphone│
       └─────┬──────┘
             │
    [ Audio Signal Stream ]
             │
    ┌────────▼────────┐
    │  Noise Reduction│ ◄─── Using noisereduce
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Feature Extraction│ ◄─── Using librosa (MFCC, spectral features)
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ ML Classification│ ◄─── Trained on 3 classes:
    │                  │        - Non-drone  
    │                  │        - Drone  
    │                  │        - Moving Drone  
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Output Decision │
    │ (Binary: Drone/ │
    │  Non-drone)     │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │   Arduino Signal│
    └─────────────────┘


**Audio Input**  
Continuously listens for sound using the `sounddevice` library.

**Noise Reduction**  
Applies noise filtering via `noisereduce` to improve detection accuracy.

**Feature Extraction**  
Extracts audio features using `librosa` (e.g., spectral centroid, MFCC, tonnetz).

**Prediction**  
Extracted features are passed into a pre-trained machine learning algorithm  
to classify whether a drone is present.

**Sliding Window**  
Uses a sliding window mechanism to make detection more reliable.

**Arduino Control**  
Sends appropriate signals to the Arduino based on detection results.

**Real-time Detection**  
The `real_time_detection()` function continuously handles audio input,  
feature extraction, prediction, and hardware signaling.

---

---

### 📬 Feedback Welcome!

If you end up using or testing this system, I’d love to hear how it worked for you.  
Feel free to drop a message at: `mossware@tuta.io`  

Let’s make drones fear us together 
