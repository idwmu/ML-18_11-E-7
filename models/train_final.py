
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, log_loss

warnings.filterwarnings("ignore", category=RuntimeWarning)

TRAIN = Path("results/features_train.parquet")
STEST = Path("results/features_studenttest.parquet")
OUT   = Path("models/model.pkl")
OBS_COLS  = ["trip_id", "point_idx"]
LABEL_COL = "label"
DROP_COLS = ["trip_id", "point_idx", "candidate_rank", "arc_id", "label", "candidate_miss"]
SEED = 42


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


def load(path, mock_trips):
    if path.exists():
        log(f"Loading {path}")
        return pd.read_parquet(path)
    log(f"{path} not found - MOCK")
    return _mock(mock_trips, SEED)


if __name__ == "__main__":
    train = load(TRAIN, 20)
    stest = load(STEST, 10)
    feat = [c for c in train.columns if c not in DROP_COLS]
    log(f"Features ({len(feat)}): {feat}")

    Xtr, ytr = train[feat].to_numpy(), train[LABEL_COL].to_numpy()
    model = HistGradientBoostingClassifier(random_state=SEED)
    t0 = time.perf_counter(); model.fit(Xtr, ytr)
    log(f"Fit on {len(train):,} rows / {train['trip_id'].nunique()} trips "
        f"in {time.perf_counter()-t0:.1f}s")

    # student-test evaluation
    Xte, yte = stest[feat].to_numpy(), stest[LABEL_COL].to_numpy()
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    log("STUDENT-TEST:")
    log(f"  Top-1 accuracy : {top1_accuracy(stest, proba):.4f}")
    log(f"  macro-F1       : {f1_score(yte, pred, average='macro', zero_division=0):.4f}")
    log(f"  log loss       : {log_loss(yte, proba, labels=[0,1]):.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feat}, OUT)
    log(f"Saved bundle -> {OUT}")
    log("C loads it with: b = joblib.load('models/model.pkl'); "
        "p = b['model'].predict_proba(df[b['feature_cols']])[:,1]")