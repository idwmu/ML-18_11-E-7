
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

COMPARISON = Path("models_comparison_all.csv")
if not COMPARISON.exists():
    COMPARISON = Path("models_comparison.csv")
TRIALS = Path("results/optuna_trials.csv")
TUNED  = Path("results/tuned_vs_untuned.csv")


def _barh(df, col, title, xlabel, fname, ascending):
    d = df.sort_values(col, ascending=ascending)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(d["model"], d[col], color="#4C72B0")
    ax.set_xlabel(xlabel); ax.set_title(title)
    for y, v in enumerate(d[col]):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3); fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150); plt.close(fig)
    print("wrote", OUT / fname)


def comparison_plots():
    if not COMPARISON.exists():
        print("skip comparison plots:", COMPARISON, "not found"); return
    df = pd.read_csv(COMPARISON)
    _barh(df, "cv_macro_f1", "Cross-validation macro-F1 by model",
          "CV macro-F1 (higher is better)", "cv_macro_f1.png", ascending=True)
    _barh(df, "top1_acc", "Observation-level Top-1 accuracy by model",
          "Top-1 (higher is better)", "top1.png", ascending=True)
    _barh(df, "log_loss", "Log loss by model",
          "Log loss (lower is better)", "log_loss.png", ascending=False)


def optuna_history():
    if not TRIALS.exists():
        print("skip optuna history:", TRIALS, "not found"); return
    df = pd.read_csv(TRIALS)
    vcol = "value" if "value" in df.columns else [c for c in df.columns if "value" in c][0]
    ncol = "number" if "number" in df.columns else df.index
    x = df[ncol] if isinstance(ncol, str) else df.index
    best = df[vcol].cummax()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, df[vcol], "o", alpha=0.5, label="trial value", color="#999999")
    ax.plot(x, best, "-", color="#C44E52", linewidth=2, label="best so far")
    ax.set_xlabel("Trial"); ax.set_ylabel("CV macro-F1")
    ax.set_title("Optuna optimisation history (HistGB)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(OUT / "optuna_history.png", dpi=150); plt.close(fig)
    print("wrote", OUT / "optuna_history.png")


def tuned_vs_untuned():
    if not TUNED.exists():
        print("skip tuned vs untuned:", TUNED, "not found"); return
    df = pd.read_csv(TUNED)
    metrics = [c for c in ["top1", "macro_f1", "log_loss"] if c in df.columns]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(metrics)); w = 0.38
    for i, (_, row) in enumerate(df.iterrows()):
        ax.bar([xx + i * w for xx in x], [row[m] for m in metrics],
               width=w, label=str(row["model"]))
    ax.set_xticks([xx + w / 2 for xx in x]); ax.set_xticklabels(metrics)
    ax.set_title("Untuned vs tuned (student-test)"); ax.legend()
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(OUT / "tuned_vs_untuned.png", dpi=150); plt.close(fig)
    print("wrote", OUT / "tuned_vs_untuned.png")


if __name__ == "__main__":
    comparison_plots()
    optuna_history()
    tuned_vs_untuned()
    print("\nFigures in", OUT.resolve())
