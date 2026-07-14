import cv2
import mediapipe as mp
import joblib
import numpy as np
import pyttsx3
from collections import deque, Counter
from datetime import datetime

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

LOG_PATH = "conversation_log.txt"
HAND_MODEL_PATH = "hand_landmarker.task"
MODEL_PATH = "gesture_model_twohands.pkl"
ENCODER_PATH = "label_encoder_twohands.pkl"

def speak_word(word):
    tts = pyttsx3.init()
    tts.say(word)
    tts.runAndWait()
    del tts

def main():
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        num_hands=2,
        running_mode=VisionRunningMode.VIDEO,
    )

    history = deque(maxlen=15)
    current_pred = ""
    sentence = ""
    last_word = ""


    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            h, w, _ = frame.shape

            # ---- build two-hand feature vector (same as in collector) ----
            confidence = 0.0            
            if result.hand_landmarks:
                # draw both hands
                for hand in result.hand_landmarks:
                    for lm in hand:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                # zero placeholders
                left_x = [0.0] * 21
                left_y = [0.0] * 21
                left_z = [0.0] * 21
                right_x = [0.0] * 21
                right_y = [0.0] * 21
                right_z = [0.0] * 21

                hands_list = result.hand_landmarks

                if len(hands_list) >= 1:
                    h0 = hands_list[0]
                    left_x = [lm.x for lm in h0]
                    left_y = [lm.y for lm in h0]
                    left_z = [lm.z for lm in h0]

                if len(hands_list) >= 2:
                    h1 = hands_list[1]
                    right_x = [lm.x for lm in h1]
                    right_y = [lm.y for lm in h1]
                    right_z = [lm.z for lm in h1]

                feat = np.array(
                    left_x + left_y + left_z + right_x + right_y + right_z
                ).reshape(1, -1)

                pred_idx = clf.predict(feat)[0]
                pred_label = le.inverse_transform([pred_idx])[0]

                history.append(pred_label)
                most_common, count = Counter(history).most_common(1)[0]
                current_pred = most_common

                confidence = count / len(history)  # 0.0–1.0
            else:
                confidence = 0.0


            if confidence >= 0.7:
                color = (0, 255, 0)      # green
            elif confidence >= 0.4:
                color = (0, 255, 255)    # yellow
            else:
                color = (128, 128, 128)  # gray

            cv2.putText(frame, f"Prediction: {current_pred}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.putText(frame, f"Sentence: {sentence}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # --- confidence bar ---
            bar_x, bar_y = 10, frame.shape[0] - 20
            bar_width, bar_height = 200, 10

            # background bar (gray)
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + bar_width, bar_y + bar_height),
                          (80, 80, 80), 1)

            # filled part
            filled = int(bar_width * confidence)
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + filled, bar_y + bar_height),
                          color, -1)

            cv2.putText(frame, f"{int(confidence * 100)}%",
                        (bar_x + bar_width + 10, bar_y + bar_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # --- small help text ---
            help_lines = [
                "Space: add word  |  Enter/e: speak sentence",
                "b: backspace  |  c: clear  |  q: quit"
            ]
            y0 = 110
            for i, line in enumerate(help_lines):
                y = y0 + i * 20
                cv2.putText(frame, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Two-Hand Sign Translator", frame)
            key = cv2.waitKey(1) & 0xFF

            # SPACE: accept current word, add to sentence, speak that word
            if key == ord(' '):
                if current_pred:
                    if sentence == "":
                        sentence = current_pred
                    else:
                        sentence += " " + current_pred
                    last_word = current_pred
                    print("Sentence:", sentence)
                    speak_word(current_pred)

            # ENTER or 'e': finalize sentence, add punctuation, speak full sentence, log it
            if key in [13, ord('e')]:   # 13 = Enter key code in waitKey
                if sentence.strip():
                    lower_last = last_word.lower()
                    if any(q in lower_last for q in ["howareyou", "how", "what", "why", "where", "when"]):
                        punct = "?"
                    else:
                        punct = "."

                    final_sentence = sentence.strip()
                    if not final_sentence.endswith(('.', '?', '!')):
                        final_sentence += punct

                    print("Final sentence:", final_sentence)
                    speak_word(final_sentence)

                    # ---- append to log file with timestamp ----
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(f"[{now}] {final_sentence}\n")

                    # reset for next sentence
                    sentence = ""
                    last_word = ""


            # 'b': backspace last word
            if key == ord('b'):
                if sentence.strip():
                    parts = sentence.strip().split()
                    parts = parts[:-1]
                    sentence = " ".join(parts)
                    print("Sentence (backspace):", sentence)

            # 'c': clear sentence
            if key == ord('c'):
                sentence = ""
                print("Sentence cleared")
                

            # 'q': quit
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
