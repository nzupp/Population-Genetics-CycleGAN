import pandas as pd
import glob
import os
import sys


def combine_stats(suffix, stats_dir):
    output_dir = os.path.join(stats_dir, suffix)
    csv_files = glob.glob(os.path.join(output_dir, "*_stats.csv"))
    csv_files = [f for f in csv_files if "combined_stats.csv" not in f]

    if not csv_files:
        print(f"No CSV files found for {suffix}")
        return

    dfs = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(["chrom", "window_start"])

    output_file = os.path.join(stats_dir, f"{suffix}.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined.to_csv(output_file, index=False)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python combine_stats.py <suffix> <stats_dir>")
        sys.exit(1)

    suffix = sys.argv[1]
    stats_dir = sys.argv[2]
    combine_stats(suffix, stats_dir)
