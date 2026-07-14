import cv2
import mediapipe as mp
import pandas as pd
import os

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"
CSV_PATH = "landmarks_data_twohands.csv"   # two-hand data

def main():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2,
        running_mode=VisionRunningMode.VIDEO,
    )

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        columns = (
            [f"lx{i}" for i in range(21)] + [f"ly{i}" for i in range(21)] + [f"lz{i}" for i in range(21)] +
            [f"rx{i}" for i in range(21)] + [f"ry{i}" for i in range(21)] + [f"rz{i}" for i in range(21)] +
            ["label"]
        )
        df = pd.DataFrame(columns=columns)

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)

        print("Start collecting TWO-HAND gestures.")
        print("Show gesture and press:")
        print("  '1' = HELLO")
        print("  '2' = YES")
        print("  '3' = NO")
        print("  '4' = GOODBYE")
        print("  '5' = THANKYOU")
        print("  '6' = WELCOME")
        print("  '7' = PLEASE")
        print("  '8' = ILOVEYOU")
        print("  '9' = SORRY")
        print("  'h' = HOWAREYOU")
        print("  'g' = GOOD")
        print("Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # draw hands
            if result.hand_landmarks:
                h, w, _ = frame.shape
                for hand in result.hand_landmarks:
                    for lm in hand:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            cv2.imshow("Collect Two-Hand Landmarks", frame)
            key = cv2.waitKey(1) & 0xFF

            valid_keys = [ord('1'), ord('2'), ord('3'),
                          ord('4'), ord('5'), ord('6'),
                          ord('7'), ord('8'), ord('9'),
                          ord('h'), ord('g')]

            if key in valid_keys and result.hand_landmarks:
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

                # map key -> label
                if key == ord('1'):
                    label = "HELLO"
                elif key == ord('2'):
                    label = "YES"
                elif key == ord('3'):
                    label = "NO"
                elif key == ord('4'):
                    label = "GOODBYE"
                elif key == ord('5'):
                    label = "THANKYOU"
                elif key == ord('6'):
                    label = "WELCOME"
                elif key == ord('7'):
                    label = "PLEASE"
                elif key == ord('8'):
                    label = "ILOVEYOU"
                elif key == ord('9'):
                    label = "SORRY"
                elif key == ord('h'):
                    label = "HOWAREYOU"
                elif key == ord('g'):
                    label = "GOOD"
                else:
                    continue

                row = left_x + left_y + left_z + right_x + right_y + right_z + [label]
                df.loc[len(df)] = row
                print(f"Saved sample for {label}. Total samples: {len(df)}")

            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    df.to_csv(CSV_PATH, index=False)
    print(f"Saved all data to {CSV_PATH}")

if __name__ == "__main__":
    main()
