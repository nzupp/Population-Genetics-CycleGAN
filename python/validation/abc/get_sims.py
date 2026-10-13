import os
import math
import msprime
import numpy as np
import pandas as pd
import stdpopsim
import sys
from abc_stat_calc import calculate_stats_from_ts

# Parameters derived primarily from OutOfAfrica_3G09 (Gutenkunst et al. 2009)
# Note: prior naming conventions changed to match literature post running code
GENERATION_TIME = 25
T_EU_AS_TRUE = 21.2e3 / GENERATION_TIME
T_B_TRUE = 140e3 / GENERATION_TIME
T_AF = 220e3 / GENERATION_TIME
R_EU = 0.004
R_AS = 0.0055
N_B = 2100
N_AF = 12300
N_A = 7300
TRUE_N_CEU = 1000 / math.exp(-R_EU * T_EU_AS_TRUE)
TRUE_N_CHB = 510 / math.exp(-R_AS * T_EU_AS_TRUE)
M_EU_AS = 9.6e-5
M_AF_AS = 1.9e-5
M_AF_EU = 3e-5
M_AF_B = 25e-5
MU_MODEL = 2.35e-8
MU_SCALE = 0.3
GENETIC_MAP = "HapMapII_GRCh38"
FLUSH_EVERY = 10

POP_PARAMS = {
    "CEU": {"TRUE_N0": TRUE_N_CEU, "TRUE_T": T_EU_AS_TRUE, "PARAM_NAME": "T_EU_AS", "N_SAMPLES": 99},
    "CHB": {"TRUE_N0": TRUE_N_CHB, "TRUE_T": T_EU_AS_TRUE, "PARAM_NAME": "T_EU_AS", "N_SAMPLES": 103},
    "YRI": {"TRUE_N0": N_AF, "TRUE_T": T_B_TRUE, "PARAM_NAME": "T_B", "N_SAMPLES": 108},
}

STATS = ['pi', 'tajima_d', 'maf', 'seg_sites', 'h_mean', 'h_std']
N_SMALL_LIST = [5, 10, 15, 20]

POP = sys.argv[1]

TRUE_N0 = POP_PARAMS[POP]["TRUE_N0"]
TRUE_T = POP_PARAMS[POP]["TRUE_T"]
PARAM_NAME = POP_PARAMS[POP]["PARAM_NAME"]
N_SAMPLES = POP_PARAMS[POP]["N_SAMPLES"]

species = stdpopsim.get_species("HomSap")
CONTIGS = {}
for chrom in [f"chr{i}" for i in range(1, 23)]:
    c = species.get_contig(chrom, length_multiplier=0.01)
    CONTIGS[chrom] = {'length': float(c.length), 'recombination_map': species.get_contig(chrom, genetic_map=GENETIC_MAP).recombination_map}

def segment_recombination_map(chrom, rng):
    full_map = CONTIGS[chrom]['recombination_map']
    seg = CONTIGS[chrom]['length']
    for _ in range(1000):
        left = float(rng.integers(0, int(full_map.sequence_length - seg)))
        try:
            rmap = full_map.slice(left=left, right=left + seg, trim=True)
        except ValueError:  # seg missing
            continue
        if not np.isnan(rmap.rate).any():
            return rmap

def sample_prior(scale=4):
    if POP == "YRI":
        T = np.random.uniform(TRUE_T / scale, T_AF - 1)
    else:
        T = np.random.uniform(TRUE_T / scale, TRUE_T * scale)
    N0 = np.exp(np.random.uniform(np.log(TRUE_N0 / scale), np.log(TRUE_N0 * scale)))
    mu = np.random.uniform(MU_MODEL * (1 - MU_SCALE), MU_MODEL * (1 + MU_SCALE))
    return {PARAM_NAME: T, "N0": N0, "mu": mu}

def build_demography(params):
    N0 = params["N0"]

    if POP == "YRI":
        t_eu_as = T_EU_AS_TRUE
        t_b = params["T_B"]
        r_ceu = R_EU
        r_chb = R_AS
        n_ceu = 1000 / math.exp(-r_ceu * t_eu_as)
        n_chb = 510 / math.exp(-r_chb * t_eu_as)
        n_yri = N0
    elif POP == "CEU":
        t_eu_as = params["T_EU_AS"]
        t_b = T_B_TRUE
        n_ceu = N0
        r_ceu = math.log(N0 / 1000) / t_eu_as
        r_chb = R_AS
        n_chb = 510 / math.exp(-r_chb * t_eu_as)
        n_yri = N_AF
    else:
        t_eu_as = params["T_EU_AS"]
        t_b = T_B_TRUE
        n_chb = N0
        r_chb = math.log(N0 / 510) / t_eu_as
        r_ceu = R_EU
        n_ceu = 1000 / math.exp(-r_ceu * t_eu_as)
        n_yri = N_AF

    demography = msprime.Demography()
    demography.add_population(name="YRI", initial_size=n_yri, initially_active=True)
    demography.add_population(name="CEU", initial_size=n_ceu, growth_rate=r_ceu)
    demography.add_population(name="CHB", initial_size=n_chb, growth_rate=r_chb)
    demography.add_population(name="OOA", initial_size=N_B)

    demography.set_symmetric_migration_rate(["CEU", "CHB"], M_EU_AS)
    demography.set_symmetric_migration_rate(["YRI", "CHB"], M_AF_AS)
    demography.set_symmetric_migration_rate(["YRI", "CEU"], M_AF_EU)

    demography.add_population_split(time=t_eu_as, derived=["CEU", "CHB"], ancestral="OOA")
    demography.add_symmetric_migration_rate_change(time=t_eu_as, populations=["YRI", "OOA"], rate=M_AF_B)
    demography.add_population_split(time=t_b, derived=["OOA"], ancestral="YRI")
    demography.add_population_parameters_change(time=T_AF, population="YRI", initial_size=N_A)

    demography.sort_events()
    return demography

def subsample_ts(ts, n_individuals):
    all_individuals = list(ts.individuals())
    chosen = np.random.choice(len(all_individuals), size=n_individuals, replace=False)
    nodes = []
    for i in chosen:
        nodes.extend(all_individuals[i].nodes)
    return ts.simplify(samples=nodes)

def get_sample_dict(pop, n_samples):
    d = {"YRI": 0, "CEU": 0, "CHB": 0}
    d[pop] = n_samples
    return d

def run_one_sim(sim_id):
    np.random.seed(sim_id)
    params = sample_prior()
    demog = build_demography(params)
    seg_rng = np.random.default_rng(sim_id)

    full_stats_list = []
    small_stats_by_n = {n: [] for n in N_SMALL_LIST}

    for chrom in [f"chr{i}" for i in range(1, 23)]:
        ts = msprime.sim_ancestry(
            samples=get_sample_dict(POP, N_SAMPLES),
            demography=demog,
            recombination_rate=segment_recombination_map(chrom, seg_rng),
        )
        ts = msprime.sim_mutations(ts, rate=params["mu"])

        full_stats_list.append(calculate_stats_from_ts(ts))

        for n_small in N_SMALL_LIST:
            ts_small = subsample_ts(ts, n_small)
            small_stats_by_n[n_small].append(calculate_stats_from_ts(ts_small))

        del ts

    full_df = pd.concat(full_stats_list, ignore_index=True)
    full_array = full_df[STATS].dropna().values

    small_arrays = {}
    for n_small in N_SMALL_LIST:
        df = pd.concat(small_stats_by_n[n_small], ignore_index=True)
        small_arrays[n_small] = df[STATS].dropna().values

    return full_array, small_arrays, params

# safe handling, temp and flush help but may revisit
def sim_files(output_dir, sim_id):
    return [os.path.join(output_dir, f"full_{sim_id}.npy")] + \
           [os.path.join(output_dir, f"n{n}_{sim_id}.npy") for n in N_SMALL_LIST]

def write_params(params_path, df):
    tmp = params_path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, params_path)

def load_done(output_dir):
    params_path = os.path.join(output_dir, "parameters.csv")
    if not os.path.exists(params_path):
        return set()
    p = pd.read_csv(params_path).drop_duplicates("sim_id", keep="last")
    complete = p["sim_id"].apply(lambda s: all(os.path.exists(f) for f in sim_files(output_dir, int(s))))
    p = p[complete]
    write_params(params_path, p)
    return set(p["sim_id"].astype(int))

def flush(output_dir, full_arrays, small_arrays, all_params):
    for i, arr in enumerate(full_arrays):
        sim_id = all_params[i]["sim_id"]
        np.save(os.path.join(output_dir, f"full_{sim_id}.npy"), arr.astype(np.float32))

    for n in N_SMALL_LIST:
        for i, arr in enumerate(small_arrays[n]):
            sim_id = all_params[i]["sim_id"]
            np.save(os.path.join(output_dir, f"n{n}_{sim_id}.npy"), arr.astype(np.float32))

    params_path = os.path.join(output_dir, "parameters.csv")
    new_params = pd.DataFrame(all_params)
    if os.path.exists(params_path):
        new_params = pd.concat([pd.read_csv(params_path), new_params], ignore_index=True)
    write_params(params_path, new_params.drop_duplicates("sim_id", keep="last"))

if __name__ == "__main__":
    OUTPUT_DIR = sys.argv[2]
    START_ID = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    END_ID = int(sys.argv[4]) if len(sys.argv) > 4 else 10000

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    done_ids = load_done(OUTPUT_DIR)

    missing_ids = [i for i in range(START_ID, END_ID) if i not in done_ids]
    print(f"{len(done_ids)} complete, {len(missing_ids)} to run.")

    if not missing_ids:
        sys.exit(0)

    full_arrays = []
    small_arrays = {n: [] for n in N_SMALL_LIST}
    all_params = []

    for idx, sim_id in enumerate(missing_ids):
        if idx % 100 == 0:
            print(f"Running sim {idx}/{len(missing_ids)}...")

        full_array, small_arrs, params = run_one_sim(sim_id)
        full_arrays.append(full_array)
        for n in N_SMALL_LIST:
            small_arrays[n].append(small_arrs[n])
        params["sim_id"] = sim_id
        all_params.append(params)

        if (idx + 1) % FLUSH_EVERY == 0 or idx == len(missing_ids) - 1:
            flush(OUTPUT_DIR, full_arrays, small_arrays, all_params)
            full_arrays = []
            small_arrays = {n: [] for n in N_SMALL_LIST}
            all_params = []
            print(f"Flushed to disk at sim {idx + 1}.", flush=True)

    print("Done.")



