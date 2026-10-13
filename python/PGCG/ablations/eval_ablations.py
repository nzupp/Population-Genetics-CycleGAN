import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wasserstein_distance

FEATURES = ["pi", "tajima_d", "maf", "seg_sites", "h_mean", "h_std"]

ABLATIONS = {
    "null": "no_abl",
    "cov": "cov_abl",
    "demo": "demo_abl",
}

# HPC paths
FULL_DIR = Path("/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/preprocess/allel_stats/full_all")
SUBSAMPLE_DIR = Path("/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/preprocess/allel_stats/subsamples_all")

full = {}
for fpath in FULL_DIR.glob("*.csv"):
    pop = fpath.stem
    full[pop] = pd.read_csv(fpath).dropna(subset=FEATURES)

rows = []
for ablation, base_dir in ABLATIONS.items():
    base_dir = Path(base_dir)
    for subdir in base_dir.iterdir():
        if not subdir.is_dir():
            continue
        parts = subdir.name.split("_")
        if len(parts) != 3:
            continue
        ref_pop, tgt_pop, subsample_n = parts
        if not subsample_n.isdigit():
            continue
        subsample_n = int(subsample_n)

        if tgt_pop not in full:
            continue
        full_tgt = full[tgt_pop]

        for fpath in subdir.glob("*_transformed.csv"):
            fname = fpath.name
            if "_replicate" not in fname or "_to_" not in fname:
                continue
            after = fname.split("_replicate", 1)[1]
            rep_str = after.split("_to_", 1)[0]
            if not rep_str.isdigit():
                continue
            rep = int(rep_str)

            transformed = pd.read_csv(fpath).dropna(subset=FEATURES)

            sub_path = SUBSAMPLE_DIR / f"{tgt_pop}_subsample{subsample_n}_replicate{rep}.csv"
            if not sub_path.exists():
                continue
            subsample = pd.read_csv(sub_path).dropna(subset=FEATURES)

            for stat in FEATURES:
                wd_full_vs_transformed = wasserstein_distance(full_tgt[stat].dropna(), transformed[stat].dropna())
                wd_full_vs_subsample = wasserstein_distance(full_tgt[stat].dropna(), subsample[stat].dropna())
                rows.append({
                    "ablation": ablation,
                    "metric_type": "feature",
                    "stat": stat,
                    "ref_pop": ref_pop,
                    "tgt_pop": tgt_pop,
                    "subsample_n": subsample_n,
                    "replicate": rep,
                    "ratio": wd_full_vs_subsample / wd_full_vs_transformed,
                })

            corr_full = full_tgt[FEATURES].dropna().corr().values
            corr_trans = transformed[FEATURES].dropna().corr().values
            corr_sub = subsample[FEATURES].dropna().corr().values

            frob_full_vs_transformed = np.linalg.norm(corr_full - corr_trans, "fro")
            frob_full_vs_subsample = np.linalg.norm(corr_full - corr_sub, "fro")

            rows.append({
                "ablation": ablation,
                "metric_type": "covariance",
                "stat": "corr",
                "ref_pop": ref_pop,
                "tgt_pop": tgt_pop,
                "subsample_n": subsample_n,
                "replicate": rep,
                "ratio": frob_full_vs_subsample / frob_full_vs_transformed,
            })

all_df = pd.DataFrame(rows)
all_df["ratio"] = all_df["ratio"].replace([np.inf, -np.inf], np.nan)

null_df = all_df[all_df["ablation"] == "null"].rename(columns={"ratio": "ratio_null"})

all_df = all_df.merge(
    null_df[["metric_type", "stat", "ref_pop", "tgt_pop", "subsample_n", "replicate", "ratio_null"]],
    on=["metric_type", "stat", "ref_pop", "tgt_pop", "subsample_n", "replicate"],
    how="left",
)

all_df["delta_from_null"] = all_df["ratio"] - all_df["ratio_null"]
all_df["log_ratio"] = np.log(all_df["ratio"])
all_df["log_delta_from_null"] = np.log(all_df["ratio"]) - np.log(all_df["ratio_null"])

all_df.to_csv("ablation_all_unified.csv", index=False)

summary = all_df.groupby(["ablation", "metric_type"])["log_delta_from_null"].mean().reset_index()
print(summary)
print(all_df.groupby("ablation").size())
print(all_df.groupby(["ablation", "ref_pop", "tgt_pop", "metric_type"])["log_delta_from_null"].mean().reset_index().to_string())