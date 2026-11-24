"""Train and compare classifiers on data/landmarks.csv, save the best to models/.

Usage: python train.py [--data data/landmarks.csv]
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = Path("models")


def train(df: pd.DataFrame, seed: int = 42):
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    y = df["label"].astype(str).to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=2000)),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=600, random_state=seed)),
    }
    scores, best_name, best_model, best_acc = {}, None, None, -1.0
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te))
        scores[name] = round(float(acc), 4)
        print(f"{name:18s} test accuracy = {acc:.4f}")
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc
    pred = best_model.predict(X_te)
    return best_name, best_model, scores, y_te, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/landmarks.csv")
    ap.add_argument("--out", default=str(MODELS / "sign_model.joblib"))
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"{args.data} not found. Run collect_data.py or prepare_from_images.py first.")
    df = pd.read_csv(args.data)
    counts = df["label"].value_counts()
    print(f"Loaded {len(df)} samples, {counts.size} signs\n")
    if counts.size < 2 or counts.min() < 10:
        raise SystemExit("Need at least 2 signs with 10+ samples each.")

    name, model, scores, y_te, pred = train(df)
    print(f"\nBest model: {name}\n")
    print(classification_report(y_te, pred, zero_division=0))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": [str(c) for c in model.classes_]}, out)
    (out.parent / "metrics.json").write_text(json.dumps(
        {"best_model": name, "test_accuracy": scores, "samples": int(len(df)), "signs": sorted(map(str, counts.index))}, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = sorted(set(y_te))
        cm = confusion_matrix(y_te, pred, labels=labels)
        fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.4), max(5, len(labels) * 0.4)))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, fontsize=7); ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix - {name}")
        fig.tight_layout(); fig.savefig(out.parent / "confusion_matrix.png", dpi=150)
    except Exception as e:
        print("Skipped plot:", e)
    print(f"Saved model -> {out}")


if __name__ == "__main__":
    main()
