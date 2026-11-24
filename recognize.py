"""Real-time sign language recognition from the webcam.

Hold a sign steady for ~1 second to add it to the sentence.

Controls: SPACE = add space | BACKSPACE = delete | C = clear | S = speak | Q = quit
Usage:    python recognize.py [--camera 0] [--threshold 0.7] [--hold 25]
"""
import argparse
from collections import Counter, deque

import cv2

from classifier import SignClassifier
from features import HandDetector


def speak(text):
    try:
        import pyttsx3
        engine = pyttsx3.init(); engine.say(text); engine.runAndWait()
    except Exception:
        print("Text-to-speech unavailable (pip install pyttsx3).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default="models/sign_model.joblib")
    ap.add_argument("--threshold", type=float, default=0.7, help="min confidence to accept a prediction")
    ap.add_argument("--hold", type=int, default=25, help="frames a sign must be stable to be typed")
    args = ap.parse_args()

    clf = SignClassifier(args.model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")
    detector = HandDetector()

    recent = deque(maxlen=10)       # smoothing window
    stable_label, stable_count, locked = None, 0, False
    sentence = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        det = detector.process(frame)
        label, conf, display = None, 0.0, "-"

        if det:
            detector.draw(frame, det["landmarks"])
            pred, conf = clf.predict(det["features"])
            recent.append(pred if conf >= args.threshold else None)
            votes = Counter(p for p in recent if p)
            if votes:
                label, n = votes.most_common(1)[0]
                if n < len(recent) * 0.6:
                    label = None
            display = label or "..."
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{display} {conf*100:.0f}%", (x1, max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            recent.clear()
            locked = False                      # lowering the hand allows repeating a letter
            stable_label, stable_count = None, 0

        # hold-to-type logic: a sign must stay stable for `--hold` frames to be typed once
        if label and label == stable_label:
            stable_count += 1
        elif label:                                  # a different sign appeared
            stable_label, stable_count, locked = label, 1, False
        else:                                        # unsure frame: keep lock, restart count
            stable_count = 0
        if label and stable_count >= args.hold and not locked:
            sentence += label
            locked = True
        progress = min(stable_count / args.hold, 1.0) if label and not locked else 0

        # HUD
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 90), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, sentence[-40:] + "_", (15, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        cv2.rectangle(frame, (15, h - 25), (15 + int(200 * progress), h - 12), (0, 255, 255), -1)
        cv2.putText(frame, "SPACE add space | BKSP delete | C clear | S speak | Q quit", (230, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.imshow("Sign Language Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            sentence += " "
        elif key in (8, 127):
            sentence = sentence[:-1]
        elif key == ord("c"):
            sentence = ""
        elif key == ord("s"):
            speak(sentence)

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
