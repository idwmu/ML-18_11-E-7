import pandas as pd
import numpy as np

def report_miss_rates(train_path, test_path):
    for name, path in [("TRAIN", train_path), ("STUDENT_TEST", test_path)]:
        df = pd.read_parquet(path)
        if "candidate_miss" not in df.columns:
            print(f"Error: candidate_miss column missing in {name}")
            continue
        
        # Miss rate per point
        point_misses = df.groupby("point_idx")["candidate_miss"].max()
        # But actually, candidate_miss is constant per point already.
        # Let's just get the mean of the first candidate for each point.
        
        # Correct way: since candidate_miss is a property of the point, 
        # we can just average it across the unique points.
        trip_groups = df.groupby("trip_id")
        
        print(f"\n--- {name} Miss Rate Report ---")
        overall_miss_rate = df.groupby(["trip_id", "point_idx"])["candidate_miss"].first().mean()
        print(f"Overall candidate-miss rate: {overall_miss_rate:.2%}")
        
        trip_miss_rates = df.groupby(["trip_id", "point_idx"])["candidate_miss"].first().groupby("trip_id").mean()
        
        high_miss_trips = trip_miss_rates[trip_miss_rates > 0.20]
        if not high_miss_trips.empty:
            print("Trips with >20% miss rate:")
            print(high_miss_trips)
        else:
            print("No trips exceeded 20% miss rate.")

if __name__ == "__main__":
    report_miss_rates("results/features_train.parquet", "results/features_studenttest.parquet")
