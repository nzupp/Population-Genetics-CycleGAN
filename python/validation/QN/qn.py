import os
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

SEED = 42
rng = np.random.default_rng(SEED)

FEATURES = ["pi", "tajima_d", "maf", "seg_sites", "h_mean", "h_std"]
POPS = ["CEU", "CHB", "YRI"]
NS = [5, 10, 15, 20]
REPS = list(range(1, 11))
EPS = 1e-12

PGCG_TRANSFORMED_DIR = "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/Population_Genetics_CycleGAN/output/transformed"
GAN_FOLDER = "{ref}_{tgt}_{size}_transformed"
SUBSAMPLE_BASE = "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/preprocess/allel_stats"
FULL_DIR = "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/preprocess/allel_stats/full_all"


def quantile_normalize_1d(source, reference):
    n = len(source)
    order = np.lexsort((rng.random(n), source))
    q = np.empty(n)
    q[order] = np.arange(n) / max(n - 1, 1)
    ref_sorted = np.sort(reference)
    return np.interp(q, np.linspace(0, 1, len(ref_sorted)), ref_sorted)


def quantile_normalize_df(tgt_df, ref_df):
    out = tgt_df.copy()
    for feat in FEATURES:
        out[feat] = quantile_normalize_1d(tgt_df[feat].values, ref_df[feat].values)
    return out


def frobenius_corr_distance(df1, df2):
    corr1 = df1[FEATURES].dropna().corr().values
    corr2 = df2[FEATURES].dropna().corr().values
    return np.linalg.norm(corr1 - corr2, "fro")


def geomean(x):
    return float(np.exp(np.mean(np.log(x))))


def compare(g):
    return {"ratio_gan": geomean(g["dist_sub"] / (g["dist_gan"] + EPS)), "ratio_qn": geomean(g["dist_sub"] / (g["dist_qn"] + EPS))}


full = {}
for fname in os.listdir(FULL_DIR):
    if fname.endswith(".csv"):
        pop = fname.replace(".csv", "")
        full[pop] = pd.read_csv(os.path.join(FULL_DIR, fname)).dropna(subset=FEATURES)

records = []
for tgt in POPS:
    full_tgt = full[tgt]
    for ref in POPS:
        ref_df = full[ref]
        for size in NS:
            for rep in REPS:
                subsample_path = os.path.join(SUBSAMPLE_BASE, "subsamples_all", f"{tgt}_subsample{size}_replicate{rep}.csv")
                if not os.path.exists(subsample_path):
                    continue
                sub_df = pd.read_csv(subsample_path).dropna(subset=FEATURES)
                qn_df = quantile_normalize_df(sub_df, ref_df)

                gan_df = None
                gan_folder = os.path.join(PGCG_TRANSFORMED_DIR, GAN_FOLDER.format(ref=ref, tgt=tgt, size=size))
                if os.path.isdir(gan_folder):
                    for fname in os.listdir(gan_folder):
                        if f"_replicate{rep}_to_" in fname and fname.endswith("_transformed.csv"):
                            gan_df = pd.read_csv(os.path.join(gan_folder, fname)).dropna(subset=FEATURES)
                            break
                if gan_df is None:
                    continue

                pair = f"{ref}-->{tgt}"
                ids = {"ref": ref, "tgt": tgt, "pair": pair, "sample_size": size, "replicate": rep}

                for stat in FEATURES:
                    f = full_tgt[stat]
                    records.append({**ids, "metric": "wd", "stat": stat, "dist_sub": wasserstein_distance(f, sub_df[stat]), "dist_qn": wasserstein_distance(f, qn_df[stat]), "dist_gan": wasserstein_distance(f, gan_df[stat])})

                records.append({**ids, "metric": "frobenius", "stat": "all", "dist_sub": frobenius_corr_distance(full_tgt, sub_df), "dist_qn": frobenius_corr_distance(full_tgt, qn_df), "dist_gan": frobenius_corr_distance(full_tgt, gan_df)})

df = pd.DataFrame(records)

wd_rep = df[df.metric == "wd"].groupby(["ref", "tgt", "pair", "sample_size", "replicate"])[["dist_sub", "dist_qn", "dist_gan"]].mean().reset_index()
frob = df[df.metric == "frobenius"]

rows = []
for metric, data in [("Marginal", wd_rep), ("Joint", frob)]:
    for (pair, size), g in data.groupby(["pair", "sample_size"]):
        rows.append({"metric": metric, "pair": pair, "sample_size": size, **compare(g)})

res = pd.DataFrame(rows)

pair_order = [f"{r}-->{t}" for r in POPS for t in POPS]
res["_pair"] = res["pair"].map({p: i for i, p in enumerate(pair_order)})
res = res.sort_values(["metric", "_pair", "sample_size"]).drop(columns=["_pair"])

output = []
for metric in ["Marginal", "Joint"]:
    sub = res[res["metric"] == metric]
    table = pd.DataFrame({"Pair": sub["pair"].values, "n": sub["sample_size"].values, "PGCG ratio": [f"{v:.3g}" for v in sub["ratio_gan"]], "QN ratio": [f"{v:.3g}" for v in sub["ratio_qn"]]})
    output.append(f"\n{metric}")
    output.append(table.to_string(index=False))

text = "\n".join(output)
with open("QN_results.txt", "w") as fh:
    fh.write(text + "\n")
print(text)