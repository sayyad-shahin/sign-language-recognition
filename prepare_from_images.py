"""Build data/landmarks.csv from an image dataset instead of the webcam.

Expected layout (e.g. the Kaggle "ASL Alphabet" dataset):
    dataset/A/*.jpg
    dataset/B/*.jpg   ...

Usage:
    python prepare_from_images.py --dataset dataset --max-per-class 300

Images are mirrored before detection so they match the selfie-view webcam frames
used by collect_data.py / recognize.py. Use --no-flip if your photos are already mirrored.
"""
import argparse
import csv
from pathlib import Path

import cv2

from features import NUM_FEATURES, HandDetector

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="data/landmarks.csv")
    ap.add_argument("--max-per-class", type=int, default=300)
    ap.add_argument("--no-flip", action="store_true")
    args = ap.parse_args()

    root, out = Path(args.dataset), Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    detector = HandDetector(static_image_mode=True, detection_conf=0.5)
    total = 0
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [f"f{i}" for i in range(NUM_FEATURES)])
        for cls in sorted(p for p in root.iterdir() if p.is_dir()):
            kept = tried = 0
            for img_path in sorted(cls.iterdir()):
                if img_path.suffix.lower() not in EXTS:
                    continue
                if kept >= args.max_per_class:
                    break
                tried += 1
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                if not args.no_flip:
                    img = cv2.flip(img, 1)
                det = detector.process(img)
                if det:
                    writer.writerow([cls.name] + det["features"].tolist())
                    kept += 1
            total += kept
            print(f"{cls.name}: {kept}/{tried} images had a detectable hand")
    print(f"Done: {total} samples -> {out}")


if __name__ == "__main__":
    main()
