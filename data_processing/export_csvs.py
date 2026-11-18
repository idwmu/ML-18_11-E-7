import pandas as pd

def export_parquet_to_csv(parquet_path, csv_path):
    df = pd.read_parquet(parquet_path)
    # Ensure column order matches SCHEMA.md
    # Schema: trip_id, point_idx, candidate_rank, arc_id, dist_m, bearing_diff_deg, 
    # arc_length_m, speed_mps, heading_deg, arc_degree, proj_t, dist_start_node_m, 
    # dist_end_node_m, label, candidate_miss
    
    # We only export columns that actually exist in the df (e.g. label/miss might be missing in some tests)
    cols = [
        "trip_id", "point_idx", "candidate_rank", "arc_id", "dist_m", "bearing_diff_deg", 
        "arc_length_m", "speed_mps", "heading_deg", "arc_degree", "proj_t", 
        "dist_start_node_m", "dist_end_node_m", "label", "candidate_miss"
    ]
    existing_cols = [c for c in cols if c in df.columns]
    df[existing_cols].to_csv(csv_path, index=False)
    print(f"Exported {parquet_path} -> {csv_path}")

if __name__ == "__main__":
    export_parquet_to_csv("results/features_train.parquet", "results/features_train.csv")
    export_parquet_to_csv("results/features_studenttest.parquet", "results/features_studenttest.csv")
