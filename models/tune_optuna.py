
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss

warnings.filterwarnings("ignore", category=RuntimeWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAIN = Path("results/features_train.parquet")
STEST = Path("results/features_studenttest.parquet")
OUT   = Path("models/model_tuned.pkl")
TRIALS_CSV = Path("results/optuna_trials.csv")
OBS_COLS  = ["trip_id", "point_idx"]
LABEL_COL = "label"
DROP_COLS = ["trip_id", "point_idx", "candidate_rank", "arc_id", "label", "candidate_miss"]
N_TRIALS  = 30
CV_SPLITS = 3           # 3 folds during search for speed; final report can use 5
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


def load(path, mock_trips):
    if path.exists():
        log(f"Loading {path}"); return pd.read_parquet(path)
    log(f"{path} not found - MOCK"); return _mock(mock_trips, SEED)


def main():
    train = load(TRAIN, 20)
    stest = load(STEST, 10)
    feat = [c for c in train.columns if c not in DROP_COLS]
    X = train[feat].to_numpy(); y = train[LABEL_COL].to_numpy()
    groups = train["trip_id"].to_numpy()
    folds = list(GroupKFold(n_splits=CV_SPLITS).split(X, y, groups))
    log(f"{len(train):,} rows, {train['trip_id'].nunique()} trips, {len(feat)} features")

    def objective(trial):
        params = dict(
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_iter          = trial.suggest_int("max_iter", 100, 600),
            max_leaf_nodes    = trial.suggest_int("max_leaf_nodes", 15, 255),
            max_depth         = trial.suggest_int("max_depth", 3, 12),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 20, 200),
            l2_regularization = trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        )
        f1s = []
        for tr, te in folds:
            m = HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **params)
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            f1s.append(f1_score(y[te], (p >= 0.5).astype(int), average="macro", zero_division=0))
        score = float(np.mean(f1s))
        log(f"  trial {trial.number:2d}: CV macroF1 {score:.4f}  "
            f"(lr={params['learning_rate']:.3f}, it={params['max_iter']}, "
            f"leaves={params['max_leaf_nodes']}, depth={params['max_depth']})")
        return score

    log(f"Starting Optuna: {N_TRIALS} trials, {CV_SPLITS}-fold group CV, objective = macro-F1")
    t0 = time.perf_counter()
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)
    log(f"Search done in {time.perf_counter()-t0:.1f}s")
    log(f"Best CV macro-F1: {study.best_value:.4f}")
    log(f"Best params: {study.best_params}")

    TRIALS_CSV.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(TRIALS_CSV, index=False)
    log(f"Wrote {TRIALS_CSV}")

    # --- untuned vs tuned on student-test ---------------------------------
    Xte, yte = stest[feat].to_numpy(), stest[LABEL_COL].to_numpy()

    def eval_on_stest(model, name):
        model.fit(X, y)
        p = model.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        return dict(model=name, top1=top1_accuracy(stest, p),
                    macro_f1=f1_score(yte, pred, average="macro", zero_division=0),
                    log_loss=log_loss(yte, p, labels=[0, 1]))

    untuned = eval_on_stest(HistGradientBoostingClassifier(random_state=SEED), "HistGB untuned")
    tuned_model = HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **study.best_params)
    tuned = eval_on_stest(tuned_model, "HistGB tuned")

    cmp = pd.DataFrame([untuned, tuned])
    pd.set_option("display.float_format", lambda v: f"{v:0.4f}")
    print("\n=== Untuned vs Tuned (student-test) ===")
    print(cmp.to_string(index=False), flush=True)
    cmp.to_csv("results/tuned_vs_untuned.csv", index=False)

    joblib.dump({"model": tuned_model, "feature_cols": feat, "params": study.best_params}, OUT)
    log(f"Saved tuned bundle -> {OUT}")


if __name__ == "__main__":
    main()