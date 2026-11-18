"""
Emits features_train.parquet + features_studenttest.parquet
Split: by trip_id with GroupKFold downstream (group = trip_id). No trip spans
partitions - train and student_test are already separate folders, so this is
satisfied by construction; just don't mix the two output files.
"""
import time
import pandas as pd
from loaders import list_trip_ids
from features import build_trip_features

OUT_DIR = "results"


def build_partition(partition: str) -> pd.DataFrame:
    frames = []
    trip_ids = list_trip_ids(partition)
    t0 = time.time()
    for i, tid in enumerate(trip_ids):
        df = build_trip_features(tid, partition)
        frames.append(df)
        if (i + 1) % 10 == 0 or (i + 1) == len(trip_ids):
            print(f"  [{partition}] {i+1}/{len(trip_ids)} trips done ({time.time()-t0:.1f}s)")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    train_df = build_partition("train")
    train_path = f"{OUT_DIR}/features_train.parquet"
    train_df.to_parquet(train_path, index=False)
    print("TRAIN:", train_df.shape, "->", train_path)
    print("  overall label rate:", train_df["label"].mean())
    print("  per-point positive-candidate recall:",
          train_df.groupby(["trip_id", "point_idx"])["label"].max().mean())
    print("  unique trips:", train_df["trip_id"].nunique())

    test_df = build_partition("student_test")
    test_path = f"{OUT_DIR}/features_studenttest.parquet"
    test_df.to_parquet(test_path, index=False)
    print("STUDENT_TEST:", test_df.shape, "->", test_path)
    print("  unique trips:", test_df["trip_id"].nunique())
