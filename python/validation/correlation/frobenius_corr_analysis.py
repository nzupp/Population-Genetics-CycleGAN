import os
import numpy as np
import pandas as pd

# Point to QN for comparison as needed
#TODO take directories as an argument

TRANSFORMED_DIR = (
    "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/"
    "Population_Genetics_CycleGAN/output/transformed"
)

SUBSAMPLE_BASE = (
    "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/preprocess/allel_stats"
)

FULL_DIR = (
    "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/preprocess/allel_stats/full_all"
)

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
    if len(parts) != 3 or not parts[0].isalpha() or not parts[0].isupper() \
       or not parts[1].isalpha() or not parts[1].isupper() or not parts[2].isdigit():
        continue
    ref, tgt, size = parts[0], parts[1], int(parts[2])
    folder_path = os.path.join(TRANSFORMED_DIR, folder)

    for fname in os.listdir(folder_path):
        if not fname.endswith("_transformed.csv") or "_replicate" not in fname or "_to_" not in fname:
            continue
        after = fname.split("_replicate", 1)[1]
        rep_str = after.split("_to_", 1)[0]
        if not rep_str.isdigit():
            continue
        rep = int(rep_str)

        transformed = pd.read_csv(os.path.join(folder_path, fname))
        subsample_path = os.path.join(
            SUBSAMPLE_BASE,
            f"subsamples_{tgt}",
            f"{tgt}_subsample{size}_replicate{rep}.csv",
        )
        subsample = pd.read_csv(subsample_path)

        full_tgt = full[tgt]
        corr_full = full_tgt[STATS].dropna().corr().values
        corr_trans = transformed[STATS].dropna().corr().values
        corr_sub = subsample[STATS].dropna().corr().values

        frob_full_vs_transformed = np.linalg.norm(
            corr_full - corr_trans, "fro"
        )
        frob_full_vs_subsample = np.linalg.norm(
            corr_full - corr_sub, "fro"
        )

        records.append(
            {
                "ref": ref,
                "tgt": tgt,
                "sample_size": size,
                "replicate": rep,
                "frob_full_vs_transformed": frob_full_vs_transformed,
                "frob_full_vs_subsample": frob_full_vs_subsample,
            }
        )

df = pd.DataFrame(records)
df["frob_ratio"] = df["frob_full_vs_subsample"] / df["frob_full_vs_transformed"]
df.to_csv("frobenius_corr_by_replicate.csv", index=False)

avg = (
    df.groupby(["ref", "tgt", "sample_size"])[
        ["frob_full_vs_transformed", "frob_full_vs_subsample", "frob_ratio"]
    ]
    .mean()
    .reset_index()
)
