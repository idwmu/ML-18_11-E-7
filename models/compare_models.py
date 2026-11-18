"""
PathSnap - Model comparison harness (System B).

Runs on A's real feature parquet (results/features_train.parquet); falls back to
a built-in mock if that file is absent. Group-aware 5-fold CV (group = trip).
Reports CV macro-F1, observation-level Top-1 accuracy, log loss, train/infer time,
and PRINTS PROGRESS as it goes.

Syllabus models: Logistic Regression, Gaussian Naive Bayes, KNN, Random Forest,
HistGradientBoosting (the fast histogram-based gradient boosting; same family).
"""
from __future__ import annotations
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score, log_loss

FEATURES_TRAIN = Path("results/features_train.parquet")
GROUP_COL = "trip_id"
OBS_COLS  = ["trip_id", "point_idx"]
LABEL_COL = "label"
DROP_COLS = ["trip_id", "point_idx", "candidate_rank", "arc_id", "label", "candidate_miss"]
N_SPLITS  = 5
SEED      = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_mock(n_trips=20, pts_per_trip=120, k_cand=6, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_trips):
        for p in range(pts_per_trip):
            true = rng.integers(k_cand)
            for c in range(k_cand):
                is_true = int(c == true)
                rows.append(dict(
                    trip_id=str(t), point_idx=p, candidate_rank=c, arc_id=int(rng.integers(5000)),
                    dist_m=rng.gamma(2, 4) + (0 if is_true else rng.gamma(2, 20)),
                    bearing_diff_deg=min(abs(rng.normal(0, 10 if is_true else 60)), 180),
                    arc_length_m=rng.gamma(3, 30), speed_mps=abs(rng.normal(8, 3)),
                    heading_deg=rng.random() * 360, arc_degree=int(rng.integers(2, 6)),
                    proj_t=rng.random(), dist_start_node_m=rng.gamma(2, 15),
                    dist_end_node_m=rng.gamma(2, 15), label=is_true, candidate_miss=0,
                ))
    return pd.DataFrame(rows)


def load_data() -> pd.DataFrame:
    if FEATURES_TRAIN.exists():
        log(f"Loading real features from {FEATURES_TRAIN}")
        df = pd.read_parquet(FEATURES_TRAIN)
    else:
        log("Real features not found - using MOCK table.")
        df = make_mock()
    return df.reset_index(drop=True)


def build_models() -> dict:
    scale = lambda est: Pipeline([("sc", StandardScaler()), ("clf", est)])
    return {
        "LogisticRegression": scale(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "GaussianNB":         GaussianNB(),
        "KNN":                scale(KNeighborsClassifier(n_neighbors=15, n_jobs=-1)),
        "RandomForest":       RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                                     random_state=SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
    }


def top1_accuracy(df_obs: pd.DataFrame, proba_pos: np.ndarray) -> float:
    tmp = df_obs.copy()
    tmp["_p"] = proba_pos
    idx = tmp.groupby(OBS_COLS)["_p"].idxmax()
    return tmp.loc[idx, LABEL_COL].mean()


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    log(f"Features ({len(feat_cols)}): {feat_cols}")
    X = df[feat_cols].to_numpy()
    y = df[LABEL_COL].to_numpy()
    groups = df[GROUP_COL].to_numpy()
    gkf = GroupKFold(n_splits=N_SPLITS)
    folds = list(gkf.split(X, y, groups))

    results = []
    for name, model in build_models().items():
        log(f"=== {name} ===")
        m0 = time.perf_counter()
        f1s, top1s, lls, fit_t, inf_t = [], [], [], [], []
        for k, (tr, te) in enumerate(folds, 1):
            t0 = time.perf_counter(); model.fit(X[tr], y[tr]); ft = time.perf_counter() - t0
            t0 = time.perf_counter(); proba = model.predict_proba(X[te])[:, 1]; it = time.perf_counter() - t0
            pred = (proba >= 0.5).astype(int)
            f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
            lls.append(log_loss(y[te], proba, labels=[0, 1]))
            top1s.append(top1_accuracy(df.iloc[te], proba))
            log(f"  fold {k}/{N_SPLITS}: fit {ft:5.1f}s  infer {it:5.1f}s  "
                f"F1 {f1s[-1]:.3f}  top1 {top1s[-1]:.3f}")
        log(f"  {name} done in {time.perf_counter()-m0:.1f}s  "
            f"(CV F1 {np.mean(f1s):.4f}, top1 {np.mean(top1s):.4f})")
        results.append(dict(
            model=name, cv_macro_f1=float(np.mean(f1s)), top1_acc=float(np.mean(top1s)),
            log_loss=float(np.mean(lls)), fit_s=float(np.mean(fit_t)), infer_s=float(np.mean(inf_t)),
        ))
    return pd.DataFrame(results).sort_values("top1_acc", ascending=False)


if __name__ == "__main__":
    df = load_data()
    log(f"{len(df):,} candidate rows, "
        f"{df[OBS_COLS].drop_duplicates().shape[0]:,} observations, "
        f"{df[GROUP_COL].nunique()} trips")
    table = evaluate(df)
    pd.set_option("display.float_format", lambda v: f"{v:0.4f}")
    print("\n" + table.to_string(index=False), flush=True)
    table.to_csv("models_comparison.csv", index=False)
    log("Wrote models_comparison.csv")