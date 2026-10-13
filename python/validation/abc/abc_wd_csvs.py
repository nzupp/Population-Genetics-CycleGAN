import os
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit

# HPC paths
BASE_RAW_FULL = "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/preprocess/allel_stats/full_all"
BASE_RAW_SUB = "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/preprocess/allel_stats/subsamples_all"
BASE_RAW_TRANS = "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/Population_Genetics_CycleGAN/output/transformed"
BASE_SIM = "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/validation/abc/simulated_stats"
OUTPUT_DIR = "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/validation/abc/abc_csvs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

pop = sys.argv[1]

POPULATIONS = ("CEU", "YRI", "CHB")
SIZES = (5, 10, 15, 20)
STAT_COLS = ("pi", "tajima_d", "maf", "seg_sites", "h_mean", "h_std")
MAF_COL = STAT_COLS.index("maf")
QUARTILE_PROBS = np.array([0.25, 0.50, 0.75])

N_STATS = len(STAT_COLS)
N_PER_STAT = 1 + len(QUARTILE_PROBS)
N_MARGINAL = N_STATS * N_PER_STAT
N_PAIRS = N_STATS * (N_STATS - 1) // 2
N_FEATURES = N_MARGINAL + N_PAIRS

SEED = 42
rng = np.random.default_rng(SEED)

col_headers = []
for s in STAT_COLS:
    col_headers.append(f"{s}_mean")
    col_headers.append(f"{s}_q25")
    col_headers.append(f"{s}_q50")
    col_headers.append(f"{s}_q75")

for i in range(N_STATS):
    for j in range(i + 1, N_STATS):
        col_headers.append(f"corr_{STAT_COLS[i]}__{STAT_COLS[j]}")


# Naming adjustments
def get_time_param(p):
    return "T_AMH" if p == "YRI" else "T_OOA"


def get_out_time_param(p):
    return "T_B" if p == "YRI" else "T_EU_AS"


def clean_windows(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(STAT_COLS):
        raise ValueError(f"expected (n, {len(STAT_COLS)}) windows, got {arr.shape}")
    return arr[~np.isnan(arr).any(axis=1)]


def load_observed_windows(filepath):
    df = pd.read_csv(filepath)
    return clean_windows(df[list(STAT_COLS)].values)


def maf_floor(reference):
    if reference not in _maf_lo:
        maf = pd.read_csv(os.path.join(BASE_RAW_FULL, f"{reference}.csv"))["maf"].dropna()
        _maf_lo[reference] = float(maf[maf > 0].min())
    return _maf_lo[reference]


_maf_lo = {}


def frac_below_maf_floor(windows, reference):
    return float(np.mean(windows[:, MAF_COL] < maf_floor(reference)))


# specifically used for speed
@njit
def _compute_marginals(windows, quartile_probs):
    nrows, ncols = windows.shape
    n_q = quartile_probs.shape[0]
    per_stat = 1 + n_q
    out = np.empty(ncols * per_stat)

    for i in range(ncols):
        col = windows[:, i].copy()

        m = 0.0
        for k in range(nrows):
            m += col[k]
        m /= nrows

        base = i * per_stat
        out[base] = m

        col.sort()
        for p in range(n_q):
            frac = quartile_probs[p] * (nrows - 1)
            lo = int(frac)
            hi = lo + 1
            if hi >= nrows:
                hi = nrows - 1
            w = frac - lo
            out[base + 1 + p] = col[lo] * (1.0 - w) + col[hi] * w

    return out


def summarize(windows):
    marginal = _compute_marginals(windows, QUARTILE_PROBS)

    corr = np.corrcoef(windows, rowvar=False)
    corr_vec = np.empty(N_PAIRS)
    idx = 0
    for i in range(N_STATS):
        for j in range(i + 1, N_STATS):
            corr_vec[idx] = corr[i, j]
            idx += 1

    out = np.concatenate([marginal, corr_vec])
    return out


def preload_sims(sim_dir, size_label, sim_ids):
    cache = {}
    n_dropped_rows = 0
    n_empty = 0
    for sid in sim_ids:
        sim_path = os.path.join(sim_dir, f"{size_label}_{sid}.npy")
        if not os.path.exists(sim_path):
            continue
        raw = np.load(sim_path)
        arr = clean_windows(raw)
        n_dropped_rows += raw.shape[0] - arr.shape[0]
        if arr.shape[0] == 0:
            n_empty += 1
            continue
        cache[sid] = arr
    return cache


def precompute_sim_summaries(sim_cache, sim_ids):
    valid_ids = []
    summaries = []
    for sid in sim_ids:
        if sid not in sim_cache:
            continue
        summaries.append(summarize(sim_cache[sid]))
        valid_ids.append(sid)
    arr = np.array(summaries)
    return valid_ids, arr


# another speed measure, fast file reading
def write_parquet(out_path, valid_ids, params_df, p, sim_summary_arr, obs_vec, extra_cols=None):
    t_param_in = get_time_param(p)
    t_param_out = get_out_time_param(p)
    param_lookup = params_df.set_index("sim_id")
    assert param_lookup.index.is_unique

    n = len(valid_ids)
    sim_id_arr = np.array(valid_ids, dtype=np.int64)
    t_arr = param_lookup.loc[valid_ids, t_param_in].to_numpy()
    n0_arr = param_lookup.loc[valid_ids, "N0"].to_numpy()

    arrays = [pa.array(sim_id_arr), pa.array(t_arr), pa.array(n0_arr)]
    names = ["sim_id", t_param_out, "N0"]

    for j, h in enumerate(col_headers):
        arrays.append(pa.array(np.full(n, obs_vec[j])))
        names.append(f"obs_{h}")

    for j, h in enumerate(col_headers):
        arrays.append(pa.array(sim_summary_arr[:, j]))
        names.append(f"sim_{h}")

    if extra_cols:
        for k, v in extra_cols.items():
            arrays.append(pa.array(np.full(n, v)))
            names.append(k)

    pq.write_table(pa.table(arrays, names=names), out_path)
    return n


sim_dir = os.path.join(BASE_SIM, pop, "merged")
params_df = pd.read_csv(os.path.join(sim_dir, "parameters.csv"))
params_df["sim_id"] = params_df["sim_id"].astype(int)
assert get_time_param(pop) in params_df.columns, f"{get_time_param(pop)} not in parameters.csv"
sim_ids = sorted(params_df["sim_id"].tolist())

obs_full = load_observed_windows(os.path.join(BASE_RAW_FULL, f"{pop}.csv"))

full_cache = preload_sims(sim_dir, "full", sim_ids)

full_valid_ids, full_sim_summaries = precompute_sim_summaries(full_cache, sim_ids)
print(f"Computed {len(full_valid_ids)} full summaries", flush=True)
del full_cache

sub_valid = {}
sub_summaries = {}
for size in SIZES:
    label = f"n{size}"
    sub_cache = preload_sims(sim_dir, label, sim_ids)
    sub_valid[size], sub_summaries[size] = precompute_sim_summaries(sub_cache, sim_ids)
    print(f"Computed {len(sub_valid[size])} {label} summaries", flush=True)
    del sub_cache

out_path = os.path.join(OUTPUT_DIR, f"{pop}_full_v_sims.parquet")
if os.path.exists(out_path):
    print(f"SKIP {out_path}", flush=True)
else:
    obs_vec = summarize(obs_full)
    n = write_parquet(out_path, full_valid_ids, params_df, pop, full_sim_summaries, obs_vec)

floor_log = []
for size in SIZES:
    for rep in range(1, 11):
        out_path = os.path.join(OUTPUT_DIR, f"{pop}_n{size}_rep{rep}_subsample_v_sims.parquet")
        if os.path.exists(out_path):
            continue

        obs_sub_path = os.path.join(BASE_RAW_SUB, f"{pop}_subsample{size}_replicate{rep}.csv")
        obs_vec = summarize(load_observed_windows(obs_sub_path))

        n = write_parquet(out_path, sub_valid[size], params_df, pop, sub_summaries[size], obs_vec, {"replicate": rep})

    for reference in POPULATIONS:
        folder = os.path.join(BASE_RAW_TRANS, f"{reference}_{pop}_{size}_transformed")
        for rep in range(1, 11):
            out_path = os.path.join(OUTPUT_DIR, f"{pop}_from_{reference}_n{size}_rep{rep}_transform_v_sims.parquet")
            if os.path.exists(out_path):
                continue

            filename = f"{pop}_subsample{size}_replicate{rep}_to_{reference}_transformed.csv"
            windows = load_observed_windows(os.path.join(folder, filename))
            frac_low = frac_below_maf_floor(windows, reference)
            floor_log.append({"target": pop, "reference": reference, "n": size, "replicate": rep, "maf_floor": maf_floor(reference), "frac_below_floor": frac_low})
            obs_vec = summarize(windows)

            n = write_parquet(out_path, full_valid_ids, params_df, pop, full_sim_summaries, obs_vec, {"replicate": rep})

if floor_log:
    log_path = os.path.join(OUTPUT_DIR, f"{pop}_maf_floor_check.csv")
    pd.DataFrame(floor_log).to_csv(log_path, index=False)