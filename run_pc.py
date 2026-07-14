import cv2
from sign_core import SignTranslator

def main():
    translator = SignTranslator()
    cap = cv2.VideoCapture(0)

    # optional: lower resolution for speed / Pi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = translator.process_frame(frame)
        cv2.imshow("Two-Hand Sign Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if translator.handle_key(key):
            break

    cap.release()
    cv2.destroyAllWindows()
    translator.close()

if __name__ == "__main__":
    main()
