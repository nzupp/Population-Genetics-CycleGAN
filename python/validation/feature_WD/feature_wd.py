import os
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

# HPC paths
TRANSFORMED_DIR = "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/Population_Genetics_CycleGAN/output/transformed"
SUBSAMPLE_BASE = "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/preprocess/allel_stats"
FULL_DIR = "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/preprocess/allel_stats/full_all"

STATS = ["pi", "tajima_d", "maf", "seg_sites", "h_mean", "h_std"]

full = {}
for fname in os.listdir(FULL_DIR):
    if not fname.endswith(".csv"):
        continue
    pop = fname.replace(".csv", "")
    full[pop] = pd.read_csv(os.path.join(FULL_DIR, fname))

records = []
for folder in os.listdir(TRANSFORMED_DIR):
    if not folder.endswith("_transformed"):
        continue
    parts = folder[:-len("_transformed")].split("_")
    if len(parts) != 3 or not parts[0].isalpha() or not parts[0].isupper() or not parts[1].isalpha() or not parts[1].isupper() or not parts[2].isdigit():
        continue
    ref, tgt, size = parts[0], parts[1], int(parts[2])
    folder_path = os.path.join(TRANSFORMED_DIR, folder)
    full_tgt = full[tgt]

    for fname in os.listdir(folder_path):
        if not fname.endswith("_transformed.csv") or "_replicate" not in fname or "_to_" not in fname:
            continue
        after = fname.split("_replicate", 1)[1]
        rep_str = after.split("_to_", 1)[0]
        if not rep_str.isdigit():
            continue
        rep = int(rep_str)

        transformed = pd.read_csv(os.path.join(folder_path, fname))
        subsample_path = os.path.join(SUBSAMPLE_BASE, "subsamples_all", f"{tgt}_subsample{size}_replicate{rep}.csv")
        subsample = pd.read_csv(subsample_path)

        for stat in STATS:
            wd_full_vs_transformed = wasserstein_distance(full_tgt[stat].dropna(), transformed[stat].dropna())
            wd_full_vs_subsample = wasserstein_distance(full_tgt[stat].dropna(), subsample[stat].dropna())
            records.append({"ref_pop": ref, "tgt_pop": tgt, "subsample_n": size, "rep": rep, "stat": stat, "full_vs_transformed_WD": wd_full_vs_transformed, "full_vs_subsample_WD": wd_full_vs_subsample})

df = pd.DataFrame(records)
df["wd_ratio"] = df["full_vs_subsample_WD"] / df["full_vs_transformed_WD"]
df.to_csv("feature_wd.csv", index=False)