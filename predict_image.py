"""Predict the sign in a single image.  Usage: python predict_image.py photo.jpg"""
import sys

import cv2

from classifier import SignClassifier
from features import HandDetector

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python predict_image.py <image>")
    img = cv2.imread(sys.argv[1])
    if img is None:
        raise SystemExit("Could not read image.")
    det = HandDetector(static_image_mode=True, detection_conf=0.5).process(cv2.flip(img, 1))
    if not det:
        raise SystemExit("No hand detected.")
    label, conf = SignClassifier().predict(det["features"])
    print(f"Prediction: {label} ({conf*100:.1f}% confidence)")
