import cv2
import mediapipe as mp
import joblib
import numpy as np
import pyttsx3
from collections import deque, Counter
from datetime import datetime

# Paths (adjust if needed)
HAND_MODEL_PATH = "hand_landmarker.task"
MODEL_PATH = "gesture_model_twohands.pkl"
ENCODER_PATH = "label_encoder_twohands.pkl"
LOG_PATH = "conversation_log.txt"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class SignTranslator:
    def __init__(self):
        # load classifier + encoder
        self.clf = joblib.load(MODEL_PATH)
        self.le = joblib.load(ENCODER_PATH)

        # mediapipe options
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
            num_hands=2,
            running_mode=VisionRunningMode.VIDEO,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        # state
        self.history = deque(maxlen=15)
        self.current_pred = ""
        self.confidence = 0.0
        self.sentence = ""
        self.last_word = ""

    def _speak(self, text):
        tts = pyttsx3.init()
        tts.say(text)
        tts.runAndWait()
        del tts

    def _twohand_features(self, result):
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

        return feat

    def process_frame(self, frame):
        """Process one frame: update prediction, draw overlays, return new frame."""
        self.confidence = 0.0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = 0  # webcam stream, we don't need real time here

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        h, w, _ = frame.shape

        if result.hand_landmarks:
            # draw hands
            for hand in result.hand_landmarks:
                for lm in hand:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # features + prediction
            feat = self._twohand_features(result)
            pred_idx = self.clf.predict(feat)[0]
            pred_label = self.le.inverse_transform([pred_idx])[0]

            self.history.append(pred_label)
            most_common, count = Counter(self.history).most_common(1)[0]
            self.current_pred = most_common
            self.confidence = count / len(self.history)
        else:
            self.current_pred = ""
            self.confidence = 0.0

        # choose color by confidence
        if self.confidence >= 0.7:
            color = (0, 255, 0)
        elif self.confidence >= 0.4:
            color = (0, 255, 255)
        else:
            color = (128, 128, 128)

        # overlays: prediction + sentence
        cv2.putText(frame, f"Prediction: {self.current_pred}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, f"Sentence: {self.sentence}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # confidence bar
        bar_x, bar_y = 10, frame.shape[0] - 20
        bar_width, bar_height = 200, 10
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_width, bar_y + bar_height),
                      (80, 80, 80), 1)
        filled = int(bar_width * self.confidence)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + filled, bar_y + bar_height),
                      color, -1)
        cv2.putText(frame, f"{int(self.confidence * 100)}%",
                    (bar_x + bar_width + 10, bar_y + bar_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # help text
        help_lines = [
            "Space:add word  Enter/e:speak sentence",
            "b:backspace  c:clear  q:quit"
        ]
        y0 = 110
        for i, line in enumerate(help_lines):
            y = y0 + i * 20
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def handle_key(self, key):
        """
        Handle keyboard controls.
        Returns True if caller should exit (q pressed), else False.
        """
        # SPACE: accept current word
        if key == ord(' '):
            if self.current_pred:
                if self.sentence == "":
                    self.sentence = self.current_pred
                else:
                    self.sentence += " " + self.current_pred
                self.last_word = self.current_pred
                print("Sentence:", self.sentence)
                self._speak(self.current_pred)

        # ENTER/e: finalize sentence
        if key in [13, ord('e')]:
            if self.sentence.strip():
                lower_last = self.last_word.lower()
                if any(q in lower_last for q in ["howareyou", "how", "what", "why", "where", "when"]):
                    punct = "?"
                else:
                    punct = "."

                final_sentence = self.sentence.strip()
                if not final_sentence.endswith(('.', '?', '!')):
                    final_sentence += punct

                print("Final sentence:", final_sentence)
                self._speak(final_sentence)

                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(f"[{now}] {final_sentence}\n")

                self.sentence = ""
                self.last_word = ""

        # backspace
        if key == ord('b'):
            if self.sentence.strip():
                parts = self.sentence.strip().split()
                if parts:
                    parts = parts[:-1]
                    self.sentence = " ".join(parts)
                    self.last_word = parts[-1] if parts else ""
                print("Sentence (backspace):", self.sentence)

        # clear
        if key == ord('c'):
            self.sentence = ""
            self.last_word = ""
            print("Sentence cleared")

        # quit
        if key == ord('q'):
            return True

        return False

    def close(self):
        self.landmarker.close()
