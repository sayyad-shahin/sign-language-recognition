"""Synthetic hand-landmark generator used only for testing the ML pipeline."""
import numpy as np
import pandas as pd

from features import normalize_landmarks


def make_dataset(n_classes=6, per_class=60, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(21, 3)) * 0.2
    rows = []
    for c in range(n_classes):
        template = base + np.random.default_rng(100 + c).normal(size=(21, 3)) * 0.25
        for _ in range(per_class):
            pts = template + rng.normal(size=(21, 3)) * 0.02
            pts = pts * rng.uniform(0.5, 2.0) + rng.uniform(-1, 1, size=3)   # random scale + shift
            rows.append([chr(65 + c)] + normalize_landmarks(pts).tolist())
    return pd.DataFrame(rows, columns=["label"] + [f"f{i}" for i in range(63)])
