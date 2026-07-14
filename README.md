# 🤟 Sign Language Translator (SLT) System

**An AI-powered, real-time assistive technology for the Deaf & Hard-of-Hearing (DHH) community.**

The SLT System captures live hand gestures through a standard webcam, detects 21-point hand landmarks using Google's **MediaPipe**, classifies the gesture using a trained **scikit-learn** model, and converts the recognized sign into audible speech via **pyttsx3** — entirely offline.

---

## 📌 Overview

Sign language is a complete, natural language used by an estimated 500,000–2 million people in the US and Canada alone (ASL), yet most hearing people cannot understand it. This project bridges that communication gap with a real-time, low-cost, webcam-only translator that requires no internet connection.

**Key features:**
- Real-time hand landmark detection (21 points/hand) via MediaPipe's HandLandmarker
- Single-hand and two-hand gesture recognition modes
- 11 ASL-inspired gestures covering common conversational phrases
- Confidence scoring with a color-coded bar (green ≥ 70%, yellow ≥ 40%, gray < 40%)
- 15-frame sliding window (majority vote) for stable predictions
- Sentence building with auto-punctuation
- Offline text-to-speech output
- Timestamped, persistent conversation logging

---

## 🧠 How It Works

1. **Capture** — OpenCV grabs frames from the webcam.
2. **Detect** — MediaPipe's `HandLandmarker` extracts 21 (x, y, z) landmarks per hand.
3. **Classify** — Landmarks are flattened into a feature vector (63 values single-hand, 126 two-hand) and passed to a trained scikit-learn (Random Forest) classifier.
4. **Stabilize** — A `deque`-based sliding window of the last 15 predictions applies majority voting to smooth out noise.
5. **Speak** — The recognized gesture is converted to speech via `pyttsx3`; sentences are built with `SPACE` and finalized with `ENTER`/`e`, auto-punctuated, and logged with a timestamp.

---
## Screenshot Of SLT-System
<img width="1920" height="1080" alt="Screenshot 2026-03-14 180209" src="https://github.com/user-attachments/assets/2ad38629-50ca-4a8d-ade8-a583f85b6971" />
<img width="1920" height="1080" alt="Screenshot 2026-03-14 180245" src="https://github.com/user-attachments/assets/0e126f19-239e-42d8-bd6e-8f3f8053b733" />
<img width="1920" height="1080" alt="Screenshot 2026-03-14 180301" src="https://github.com/user-attachments/assets/6835cc7a-c5b6-4057-b176-944df22b3d91" />
<img width="1920" height="1080" alt="Screenshot 2026-03-14 180316" src="https://github.com/user-attachments/assets/de21570b-2a64-4c37-ba98-dee6f3072e18" />
<img width="790" height="638" alt="Screenshot 2026-03-14 180643" src="https://github.com/user-attachments/assets/fbfe8edc-8775-4957-b2d5-1cd1a9575e4e" />
<img width="609" height="418" alt="Screenshot 2026-03-14 180654" src="https://github.com/user-attachments/assets/576d2ae3-14fc-4ee4-af88-d1e0a0a5d6d4" />
<img width="1919" height="1079" alt="Screenshot 2026-03-14 180036" src="https://github.com/user-attachments/assets/be8bfa9f-cd04-4db3-975d-d7566adeaf58" />
<img width="1920" height="1080" alt="Screenshot 2026-03-14 180058" src="https://github.com/user-attachments/assets/d78cca8c-6e90-4fc5-8dfb-4845ba3ad227" />

---

## 🗂️ Project Structure

```
├── collect_landmarks.py                  # Data collection: capture landmarks + labels → CSV
├── hand_tracker.py                       # Basic MediaPipe landmark visualization (no ML)
├── realtime_translator.py                # Single-hand real-time sign-to-speech translator
├── realtime_translator_twohands.py       # Two-hand translator: confidence bar, sentence
│                                          # building, auto-punctuation, conversation logging
├── hand_landmarker.task                  # Pre-trained MediaPipe hand landmark model
├── gesture_model.pkl                     # Trained single-hand gesture classifier
├── gesture_model_twohands.pkl            # Trained two-hand gesture classifier
├── label_encoder.pkl                     # Label encoder — single-hand mode
├── label_encoder_twohands.pkl            # Label encoder — two-hand mode
├── landmarks_data.csv                    # Training data: single-hand landmarks
├── landmarks_data_twohands.csv           # Training data: two-hand landmarks
├── conversation_log.txt                  # Timestamped log of finalized sentences
└── README.md
```

---

## 🖐️ Supported Gestures

| Key | Gesture |
|-----|---------|
| 1 | HELLO |
| 2 | YES |
| 3 | NO |
| 4 | GOODBYE |
| 5 | THANKYOU |
| 6 | WELCOME |
| 7 | PLEASE |
| 8 | ILOVEYOU |
| 9 | SORRY |
| h | HOWAREYOU |
| g | GOOD |

Each gesture class was trained on a minimum of 50 labeled samples.
You can Delete Preivious samples of Gesture, and Train New Samples by Model.
---

## ⚙️ Installation

Requires **Python 3**. Install dependencies:

```bash
pip install opencv-python mediapipe scikit-learn numpy pyttsx3 pandas joblib
```

| Package | Purpose |
|---|---|
| `opencv-python` | Camera capture, frame processing, UI overlay |
| `mediapipe` | Real-time hand landmark detection |
| `scikit-learn` | Gesture classifier & label encoding |
| `numpy` | Feature vector construction |
| `pyttsx3` | Offline text-to-speech |
| `pandas` | CSV read/write for training data |
| `joblib` | Model serialization |

---

## ▶️ Usage

**Run the two-hand real-time translator (recommended):**
```bash
python realtime_translator_twohands.py
```

**Run the single-hand translator:**
```bash
python realtime_translator.py
```

**Controls:**
- `SPACE` — append the current predicted word to the sentence
- `b` — backspace / remove last word
- `ENTER` / `e` — finalize the sentence (auto-punctuated), speak it, and log it with a timestamp
- Number keys (`1`–`9`) and `h`, `g` — used during data collection to label gestures

**Collect your own training data:**
```bash
python collect_landmarks.py
```

**Preview raw hand tracking (no classification):**
```bash
python hand_tracker.py
```

---

## 📊 Results

- Real-time performance at standard webcam frame rates (25–30 fps)
- High confidence (≥ 70%) for static, clear gestures under good lighting
- Two-hand mode improved accuracy for gestures requiring both hands (e.g., ILOVEYOU, HOWAREYOU)
- Auto-punctuation correctly distinguished questions (e.g., "HOWAREYOU?") from statements
- TTS latency under 0.5s per word, fully offline

### Limitations
- Sensitive to poor lighting / high-contrast backgrounds
- Only 11 static gestures supported; no dynamic/motion-based signs yet
- Model doesn't account for user-specific variability (handedness, hand size, skin tone)
- No visual feedback for the hearing person beyond voice output

---

## 🚀 Future Work

- Expand vocabulary to the full ASL alphabet and more phrases via data augmentation
- Add dynamic/motion-based gesture recognition using LSTM or Transformer models
- Support Indian Sign Language (ISL) for regional relevance
- Build a mobile app (TensorFlow Lite / on-device MediaPipe) for portable use
- Add speech-to-text for bidirectional communication
- Conduct usability studies with DHH community members

---

## 🎓 Acknowledgments

Developed as an Applied Physics project (BSL2012: Semiconductor Physics Lab), Department of Humanities & Applied Sciences, **A.P. Shah Institute of Technology, Thane** (AY 2025–26).

**Contributors:**
- Suhani Hiwale
- Aayush Humne
- Om Eknath Jadhav
- Om Sameer Jadhav
- Himanshu Bhupendra Jain
- Himanshu Paras Jain

---

## 📚 References

- [Python Documentation](https://www.python.org)
- [OpenCV Documentation](https://opencv.org)
- [MediaPipe Documentation](https://mediapipe.dev)
