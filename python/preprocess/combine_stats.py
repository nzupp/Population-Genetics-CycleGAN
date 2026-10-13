import pandas as pd
import glob
import os
import sys


def combine_stats(pop, stats_dir):
    output_dir = os.path.join(stats_dir, pop)
    csv_files = glob.glob(os.path.join(output_dir, "*_stats.csv"))
    csv_files = [f for f in csv_files if "combined_stats.csv" not in f]

    if not csv_files:
        return

    dfs = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(["chrom", "window_start"])

    output_file = os.path.join(stats_dir, f"{pop}.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined.to_csv(output_file, index=False)


if __name__ == "__main__":
    pop = sys.argv[1]
    stats_dir = sys.argv[2]
    combine_stats(pop, stats_dir)
