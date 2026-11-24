"""Record your own training data with the webcam.

Usage:
    python collect_data.py                      # letters A-Z, 200 samples each
    python collect_data.py --labels A B C HELLO # custom signs
    python collect_data.py --samples 300 --camera 1

Controls:  SPACE = start recording the current sign | N = skip sign | Q = quit
Data is appended to data/landmarks.csv, so you can stop and continue later.
"""
import argparse
import csv
import string
from pathlib import Path

import cv2

from features import NUM_FEATURES, HandDetector

CSV_PATH = Path("data/landmarks.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=list(string.ascii_uppercase))
    ap.add_argument("--samples", type=int, default=200, help="samples per sign")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default=str(CSV_PATH))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")
    detector = HandDetector()

    with open(out, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["label"] + [f"f{i}" for i in range(NUM_FEATURES)])

        for label in args.labels:
            recording, count = False, 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)  # selfie view
                det = detector.process(frame)
                if det:
                    detector.draw(frame, det["landmarks"])
                    if recording:
                        writer.writerow([label] + det["features"].tolist())
                        count += 1

                status = f"Recording {count}/{args.samples}" if recording else "SPACE=record  N=skip  Q=quit"
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (30, 30, 30), -1)
                cv2.putText(frame, f"Sign: {label}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(frame, status, (260, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                if not det:
                    cv2.putText(frame, "No hand detected", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Collect data", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    cap.release(); cv2.destroyAllWindows(); return
                if key == ord("n"):
                    break
                if key == ord(" "):
                    recording = True
                if recording and count >= args.samples:
                    f.flush()
                    print(f"Saved {count} samples for '{label}'")
                    break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Data saved to {out}")


if __name__ == "__main__":
    main()
