"""
Confusion matrix for the chosen model (HistGradientBoosting) on student-test.

Loads models/model.pkl (the {model, feature_cols} bundle) and
results/features_studenttest.parquet, scores every candidate row, thresholds at
0.5, and saves a 2x2 confusion matrix (candidate is the true arc vs not) plus a
classification report to results/figures/.

Run from the repo root:  python models/confusion_matrix.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

MODEL = Path("models/model.pkl")
STEST = Path("results/features_studenttest.parquet")
OUTDIR = Path("results/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)
THRESH = 0.5


def main():
    bundle = joblib.load(MODEL)
    model, feat_cols = bundle["model"], bundle["feature_cols"]

    df = pd.read_parquet(STEST)
    y = df["label"].to_numpy()
    proba = model.predict_proba(df[feat_cols])[:, 1]
    pred = (proba >= THRESH).astype(int)

    cm = confusion_matrix(y, pred, labels=[0, 1])
    print("Confusion matrix (rows=true, cols=pred):\n", cm)
    print("\n", classification_report(y, pred, target_names=["not true arc", "true arc"],
                                      digits=4, zero_division=0))

    # Plot
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["not true arc", "true arc"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("HistGradientBoosting — confusion matrix (student-test)")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100 if total else 0
            ax.text(j, i, f"{cm[i, j]:,}\n{pct:.1f}%", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = OUTDIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
