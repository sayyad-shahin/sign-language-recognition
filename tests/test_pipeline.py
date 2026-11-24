import numpy as np

from classifier import SignClassifier
from features import NUM_FEATURES, normalize_landmarks
from tests.synthetic import make_dataset
from train import train

import joblib


def _hand(seed=0):
    return np.random.default_rng(seed).normal(size=(21, 3))


def test_feature_length_and_wrist_origin():
    f = normalize_landmarks(_hand())
    assert f.shape == (NUM_FEATURES,)
    assert np.allclose(f[:3], 0)


def test_translation_and_scale_invariant():
    pts = _hand(1)
    a = normalize_landmarks(pts)
    b = normalize_landmarks(pts * 3.7 + np.array([5.0, -2.0, 1.0]))
    assert np.allclose(a, b, atol=1e-5)


def test_left_hand_is_mirrored():
    pts = _hand(2)
    mirrored = pts.copy(); mirrored[:, 0] *= -1
    assert np.allclose(normalize_landmarks(pts, is_left=True), normalize_landmarks(mirrored), atol=1e-5)


def test_train_and_predict_roundtrip(tmp_path):
    df = make_dataset()
    name, model, scores, y_te, pred = train(df)
    assert max(scores.values()) > 0.9
    path = tmp_path / "m.joblib"
    joblib.dump({"model": model, "classes": [str(c) for c in model.classes_]}, path)
    clf = SignClassifier(path)
    row = df[df.label == "C"].iloc[0, 1:].to_numpy(dtype=np.float32)
    label, conf = clf.predict(row)
    assert label == "C" and 0 < conf <= 1


def test_detector_returns_none_on_blank_frame():
    from features import HandDetector
    det = HandDetector(static_image_mode=True)
    assert det.process(np.zeros((480, 640, 3), np.uint8)) is None
