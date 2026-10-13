import os
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

# HPC paths
LOO_OUTPUT_DIR = ("/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/Population_Genetics_CycleGAN/LOO/output")
SUBSAMPLE_BASE = ("/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/preprocess/allel_stats")
FULL_DIR = ("/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/preprocess/allel_stats/full_all")

STATS = ["pi", "tajima_d", "maf", "seg_sites", "h_mean", "h_std"]

full = {}
for fname in os.listdir(FULL_DIR):
    if not fname.endswith(".csv"):
        continue
    pop = fname.replace(".csv", "")
    full[pop] = pd.read_csv(os.path.join(FULL_DIR, fname))

records = []
for dirname in sorted(os.listdir(LOO_OUTPUT_DIR)):
    dirpath = os.path.join(LOO_OUTPUT_DIR, dirname)
    if not os.path.isdir(dirpath):
        continue

    parts = dirname.split("_")
    if len(parts) < 5 or parts[3] != "LOO":
        continue

    ref = parts[0]
    tgt = parts[1]
    size = int(parts[2])
    dropped = "_".join(parts[4:])

    eval_stats = [s for s in STATS if s != dropped]

    full_tgt = full[tgt]

    for fname in os.listdir(dirpath):
        if not fname.endswith("_transformed.csv") or "_replicate" not in fname:
            continue

        after = fname.split("_replicate", 1)[1]
        rep_str = after.split("_to_", 1)[0]
        if not rep_str.isdigit():
            continue
        rep = int(rep_str)

        transformed = pd.read_csv(os.path.join(dirpath, fname))
        subsample_path = os.path.join(SUBSAMPLE_BASE, "subsamples_all", f"{tgt}_subsample{size}_replicate{rep}.csv")
        subsample = pd.read_csv(subsample_path)

        for stat in eval_stats:
            full_vals = full_tgt[stat].dropna()
            trans_vals = transformed[stat].dropna()
            sub_vals = subsample[stat].dropna()

            full_mean = full_vals.mean()
            norm = abs(full_mean) if abs(full_mean) > 1e-12 else 1.0

            wd_transformed = wasserstein_distance(full_vals, trans_vals) / norm
            wd_subsample = wasserstein_distance(full_vals, sub_vals) / norm

            records.append({
                "ref_pop": ref,
                "tgt_pop": tgt,
                "subsample_n": size,
                "dropped_feature": dropped,
                "rep": rep,
                "stat": stat,
                "norm_wd_full_vs_transformed": wd_transformed,
                "norm_wd_full_vs_subsample": wd_subsample,
            })

df = pd.DataFrame(records)

if df.empty:
    print("No records found")
else:
    df["norm_wd_ratio"] = df["norm_wd_full_vs_subsample"] / df["norm_wd_full_vs_transformed"]
    df["pair"] = df["ref_pop"] + "-->" + df["tgt_pop"]
    df.to_csv("loo_feature_wd_merged.csv", index=False)
    print(f"Saved: loo_feature_wd_merged.csv ({len(df)} rows)")