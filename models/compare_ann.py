
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, log_loss

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")  # silence MLP convergence chatter

FEATURES_TRAIN = Path("results/features_train.parquet")
EXISTING_CSV   = Path("models_comparison.csv")
OUT_CSV        = Path("models_comparison_all.csv")
GROUP_COL = "trip_id"
OBS_COLS  = ["trip_id", "point_idx"]
LABEL_COL = "label"
DROP_COLS = ["trip_id", "point_idx", "candidate_rank", "arc_id", "label", "candidate_miss"]
N_SPLITS  = 5
SEED      = 42


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def top1_accuracy(df, proba):
    t = df.copy(); t["_p"] = proba
    idx = t.groupby(OBS_COLS)["_p"].idxmax()
    return t.loc[idx, LABEL_COL].mean()


def _mock(n_trips, seed):
    rng = np.random.default_rng(seed); rows = []
    for tr in range(n_trips):
        for p in range(120):
            true = rng.integers(6)
            for c in range(6):
                it = int(c == true)
                rows.append(dict(trip_id=str(tr), point_idx=p, candidate_rank=c,
                    arc_id=int(rng.integers(5000)),
                    dist_m=rng.gamma(2, 4) + (0 if it else rng.gamma(2, 20)),
                    bearing_diff_deg=min(abs(rng.normal(0, 10 if it else 60)), 180),
                    arc_length_m=rng.gamma(3, 30), speed_mps=abs(rng.normal(8, 3)),
                    heading_deg=rng.random() * 360, arc_degree=int(rng.integers(2, 6)),
                    proj_t=rng.random(), dist_start_node_m=rng.gamma(2, 15),
                    dist_end_node_m=rng.gamma(2, 15), label=it, candidate_miss=0))
    return pd.DataFrame(rows)


def load():
    if FEATURES_TRAIN.exists():
        log(f"Loading {FEATURES_TRAIN}"); return pd.read_parquet(FEATURES_TRAIN).reset_index(drop=True)
    log("MOCK"); return _mock(20, SEED)


def build_ann():
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),  
            activation="relu",
            solver="adam",
            alpha=1e-4,                    
            batch_size=512,
            learning_rate_init=1e-3,
            max_iter=100,              
            early_stopping=True,           
            n_iter_no_change=8,
            random_state=SEED,
        )),
    ])


def evaluate_one(df, name, model):
    feat = [c for c in df.columns if c not in DROP_COLS]
    X = df[feat].to_numpy(); y = df[LABEL_COL].to_numpy(); groups = df[GROUP_COL].to_numpy()
    folds = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))
    f1s, top1s, lls, fit_t, inf_t = [], [], [], [], []
    log(f"=== {name} ===")
    for k, (tr, te) in enumerate(folds, 1):
        t0 = time.perf_counter(); model.fit(X[tr], y[tr]); ft = time.perf_counter() - t0
        t0 = time.perf_counter(); proba = model.predict_proba(X[te])[:, 1]; it = time.perf_counter() - t0
        fit_t.append(ft); inf_t.append(it)
        f1s.append(f1_score(y[te], (proba >= 0.5).astype(int), average="macro", zero_division=0))
        lls.append(log_loss(y[te], proba, labels=[0, 1]))
        top1s.append(top1_accuracy(df.iloc[te], proba))
        log(f"  fold {k}/{N_SPLITS}: fit {ft:5.1f}s  infer {it:4.1f}s  F1 {f1s[-1]:.3f}  top1 {top1s[-1]:.3f}")
    return dict(model=name, cv_macro_f1=float(np.mean(f1s)), top1_acc=float(np.mean(top1s)),
                log_loss=float(np.mean(lls)), fit_s=float(np.mean(fit_t)), infer_s=float(np.mean(inf_t)))


if __name__ == "__main__":
    df = load()
    log(f"{len(df):,} rows, {df[GROUP_COL].nunique()} trips")
    ann_row = evaluate_one(df, "ANN (MLP 64-32)", build_ann())

    if EXISTING_CSV.exists():
        prev = pd.read_csv(EXISTING_CSV)
        combined = pd.concat([prev, pd.DataFrame([ann_row])], ignore_index=True)
    else:
        log(f"{EXISTING_CSV} not found - writing ANN row only")
        combined = pd.DataFrame([ann_row])

    combined = combined.sort_values("top1_acc", ascending=False)
    pd.set_option("display.float_format", lambda v: f"{v:0.4f}")
    print("\n=== Combined comparison (incl. ANN) ===")
    print(combined.to_string(index=False), flush=True)
    combined.to_csv(OUT_CSV, index=False)
    log(f"Wrote {OUT_CSV}")
