import os
import numpy as np
import pandas as pd

TRANSFORMED_DIR = "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/Population_Genetics_CycleGAN/output/transformed"
SUBSAMPLE_BASE = "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/preprocess/allel_stats"
FULL_DIR = "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/preprocess/allel_stats/full_all"

STATS = ["pi", "tajima_d", "maf", "seg_sites", "h_mean", "h_std"]


def geomean(x):
    return float(np.exp(np.mean(np.log(x))))


full = {}
for fname in os.listdir(FULL_DIR):
    if not fname.endswith(".csv"):
        continue
    pop = fname.replace(".csv", "")
    full[pop] = pd.read_csv(os.path.join(FULL_DIR, fname))

records = []
for folder in os.listdir(TRANSFORMED_DIR):
    folder_name = folder.removesuffix("_transformed")
    parts = folder_name.split("_")
    ref, tgt, size_str = parts
    size = int(size_str)
    folder_path = os.path.join(TRANSFORMED_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    full_tgt = full[tgt]

    for fname in os.listdir(folder_path):
        if not fname.endswith("_transformed.csv"):
            continue
        if "_replicate" not in fname or "_to_" not in fname:
            continue

        after = fname.split("_replicate", 1)[1]
        rep_str = after.split("_to_", 1)[0]
        if not rep_str.isdigit():
            continue
        rep = int(rep_str)

        transformed = pd.read_csv(os.path.join(folder_path, fname))
        subsample_path = os.path.join(SUBSAMPLE_BASE, "subsamples_all", f"{tgt}_subsample{size}_replicate{rep}.csv")
        subsample = pd.read_csv(subsample_path)

        corr_full = full_tgt[STATS].dropna().corr().values
        corr_trans = transformed[STATS].dropna().corr().values
        corr_sub = subsample[STATS].dropna().corr().values

        frob_full_vs_transformed = np.linalg.norm(corr_full - corr_trans, "fro")
        frob_full_vs_subsample = np.linalg.norm(corr_full - corr_sub, "fro")

        records.append({"ref": ref, "tgt": tgt, "sample_size": size, "replicate": rep, "frob_full_vs_transformed": frob_full_vs_transformed, "frob_full_vs_subsample": frob_full_vs_subsample})

df = pd.DataFrame(records)
df["frob_ratio"] = df["frob_full_vs_subsample"] / df["frob_full_vs_transformed"]
df.to_csv("frobenius_corr_by_replicate.csv", index=False)

avg = df.groupby(["ref", "tgt", "sample_size"]).agg(frob_full_vs_transformed=("frob_full_vs_transformed", geomean), frob_full_vs_subsample=("frob_full_vs_subsample", geomean), frob_ratio=("frob_ratio", geomean)).reset_index()
avg.to_csv("frobenius_corr_by_group.csv", index=False)