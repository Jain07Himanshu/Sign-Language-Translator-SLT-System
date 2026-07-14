import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque, Counter
import pyttsx3

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "gesture_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"

def main():
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        num_hands=1,
        running_mode=VisionRunningMode.VIDEO,
    )
    
    history = deque(maxlen=15)   # last 15 predictions

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        current_pred = ""
        sentence = ""
        # engine = pyttsx3.init()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                h, w, _ = frame.shape
                hand = result.hand_landmarks[0]

                xs = [lm.x for lm in hand]
                ys = [lm.y for lm in hand]
                zs = [lm.z for lm in hand]

                feat = np.array(xs + ys + zs).reshape(1, -1)
                pred_idx = clf.predict(feat)[0]
                pred_label = le.inverse_transform([pred_idx])[0]

                history.append(pred_label)
                # choose most common in history
                most_common, count = Counter(history).most_common(1)[0]
                current_pred = most_common


                for lm in hand:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            cv2.putText(frame, f"Prediction: {current_pred}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(frame, f"Sentence: {sentence}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Sign Translator", frame)
            
            key = cv2.waitKey(1) & 0xFF

            # press SPACE to add current word to sentence and speak it
            if key == ord(' '):
                if current_pred:
                    if sentence == "":
                        sentence = current_pred
                    else:
                        sentence += " " + current_pred
                    print("Sentence:", sentence)

                    # speak using a fresh engine
                    tts = pyttsx3.init()
                    tts.say(current_pred)
                    tts.runAndWait()
                    del tts

            # press 'c' to clear sentence
            if key == ord('c'):
                sentence = ""
                print("Sentence cleared")

            # press 'q' to quit
            if key == ord('q'):
                break




        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

