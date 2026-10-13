import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import wasserstein_distance, gaussian_kde, spearmanr, f_oneway

ALLEL_DIR = "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/preprocess/allel_stats"
FULL_DIR = os.path.join(ALLEL_DIR, "full_all")
OUTPUT_DIR = "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/Population_Genetics_CycleGAN/output"
LOO_DIR = "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/Population_Genetics_CycleGAN/LOO"
VALIDATION_DIR = "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/validation"
ABC_BASE = os.path.join(VALIDATION_DIR, "abc")
ABC_COMBINED_DIR = os.path.join(ABC_BASE, "abc_results_combined")
FEAT_WD_DIR = os.path.join(VALIDATION_DIR, "feature_WD")
CORR_CSV = os.path.join(VALIDATION_DIR, "correlation", "frobenius_corr_by_replicate.csv")
FEAT_WD_CSV = os.path.join(FEAT_WD_DIR, "feature_wd.csv")
LOO_CSV = os.path.join(LOO_DIR, "loo_feature_wd_merged.csv")
TRAIN_CSV = os.path.join(FEAT_WD_DIR, "all_training_losses.csv")

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Constant plot params
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["axes.titlesize"] = 12
matplotlib.rcParams["axes.labelsize"] = 11
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10

# Set plt
_VIRIDIS_FULL = plt.cm.viridis
VIRIDIS = mcolors.LinearSegmentedColormap.from_list("viridis_trunc", _VIRIDIS_FULL(np.linspace(0, 0.85, 256)))

POPULATIONS = ["CEU", "CHB", "YRI"]
SUBSAMPLE_SIZES = [5, 10, 15, 20]
N_REPLICATES = 10
STATS_COLS = ["pi", "tajima_d", "maf", "seg_sites", "h_mean", "h_std"]
STAT_LABELS = ["Nucleotide Diversity", "Tajima's D", "MAF", "Segregating Sites", "Mean Heterozygosity", "SD Heterozygosity"]
STAT_LABELS_MULTI = ["Nucleotide\nDiversity", "Tajima's D", "Minor Allele\nFrequency", "Segregating\nSites", "Mean\nHeterozygosity", "SD\nHeterozygosity"]
STAT_LABEL_MAP = dict(zip(STATS_COLS, STAT_LABELS))
STAT_LABEL_MULTI_MAP = dict(zip(STATS_COLS, STAT_LABELS_MULTI))

BAR_COLORS = [VIRIDIS(x) for x in np.linspace(0.2, 0.8, len(STATS_COLS))]
STAT_COLORS = dict(zip(STATS_COLS, BAR_COLORS))

POP_COLORS = {"CEU": VIRIDIS(0.15), "CHB": VIRIDIS(0.50), "YRI": VIRIDIS(0.85)}

METRICS = ["gen_g", "gen_f", "disc_x", "disc_y", "cycle", "cov"]

# Helpers

def _remove_spines(ax, which=("top", "right")):
    for s in which:
        ax.spines[s].set_visible(False)


def _split_pop_pairs(pairs):
    same = sorted(p for p in pairs if p.split("→")[0] == p.split("→")[1])
    cross = sorted(p for p in pairs if p.split("→")[0] != p.split("→")[1])
    return same, cross


def _method_color(method):
    if method == "Full data":
        return "dimgray"
    if method == "Subsample":
        return "#d9d9d9"
    for pop, color in POP_COLORS.items():
        if pop in method:
            return color
    return "silver"

# Fig 2- How bad is subsampling
def plot_figure2():
    sub_dir = os.path.join(ALLEL_DIR, "subsamples_all")
    results = np.zeros((len(POPULATIONS), len(SUBSAMPLE_SIZES)))

    for i, pop in enumerate(POPULATIONS):
        full_path = os.path.join(FULL_DIR, f"{pop}.csv")
        full_df = pd.read_csv(full_path)

        for j, size in enumerate(SUBSAMPLE_SIZES):
            rep_distances = []
            for rep in range(1, N_REPLICATES + 1):
                sub_path = os.path.join(sub_dir, f"{pop}_subsample{size}_replicate{rep}.csv")
                if not os.path.exists(sub_path):
                    continue
                sub_df = pd.read_csv(sub_path)
                distances = []
                for col in STATS_COLS:
                    fv = full_df[col].dropna().values
                    sv = sub_df[col].dropna().values
                    if len(fv) == 0 or len(sv) == 0:
                        continue
                    mu, sigma = fv.mean(), fv.std()
                    if sigma == 0:
                        continue
                    distances.append(wasserstein_distance((fv - mu) / sigma, (sv - mu) / sigma))
                if distances:
                    rep_distances.append(np.mean(distances))
            results[i, j] = np.nanmean(rep_distances) if rep_distances else np.nan

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    results_df = pd.DataFrame(results, index=POPULATIONS, columns=SUBSAMPLE_SIZES)
    vmin, vmax = np.nanmin(results), np.nanmax(results)
    sns.heatmap(results_df, annot=True, fmt=".3f", cmap=VIRIDIS.reversed(), ax=ax, square=True, linewidths=1, linecolor="white", annot_kws={"size": 10, "fontweight": "bold"}, cbar_kws={"shrink": 0.8, "label": "Wasserstein Distance", "pad": 0.02, "ticks": [vmin, vmax], "format": "%.2f"})
    ax.set_xlabel("Subsample Size")
    ax.set_ylabel("Population")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "figure2.png"), dpi=300, bbox_inches="tight")
    plt.close()

# Fig 3- validation results
def plot_figure3():
    df_feat = pd.read_csv(FEAT_WD_CSV)
    df_corr = pd.read_csv(CORR_CSV)
    df_corr = df_corr.rename(columns={"ref": "ref_pop", "tgt": "tgt_pop", "sample_size": "subsample_n", "frob_full_vs_transformed": "full_vs_transformed_corr", "frob_full_vs_subsample": "full_vs_subsample_corr", "frob_ratio": "corr_ratio"})

    df_feat["wd_ratio"] = df_feat["full_vs_subsample_WD"] / df_feat["full_vs_transformed_WD"]
    df_feat["wd_ratio"] = df_feat["wd_ratio"].replace([np.inf, -np.inf], np.nan)
    df_feat["wd_ratio_log"] = np.log10(df_feat["wd_ratio"])
    df_feat["pop_pair"] = df_feat["ref_pop"] + "→" + df_feat["tgt_pop"]

    df_forest = df_feat[df_feat["subsample_n"].isin([5, 10, 15, 20])].dropna(subset=["wd_ratio_log"])
    forest = df_forest.groupby(["stat", "pop_pair"])["wd_ratio_log"].agg(["mean", "std", "count"]).reset_index()

    per_rep_gmean = df_forest.groupby(["subsample_n", "ref_pop", "tgt_pop", "pop_pair", "rep"])["wd_ratio_log"].mean().reset_index()
    forest_gmean = per_rep_gmean.groupby("pop_pair")["wd_ratio_log"].agg(["mean", "std", "count"]).reset_index()
    forest_gmean["stat"] = "geomean"
    forest = pd.concat([forest, forest_gmean], ignore_index=True)

    stat_order = ["geomean", "pi", "tajima_d", "maf", "seg_sites", "h_mean", "h_std"]
    stat_labels = ["Geometric Mean", "Nucleotide Diversity", "Tajima's D", "Minor Allele Frequency", "Segregating Sites", "Mean Heterozygosity", "SD Heterozygosity"]

    same, cross = _split_pop_pairs(forest["pop_pair"].unique())
    pop_pairs_ordered = same + cross
    n_pairs = len(pop_pairs_ordered)
    y_pos = {p: n_pairs - i - 1 for i, p in enumerate(pop_pairs_ordered)}

    pair_colors = {p: VIRIDIS(v) for p, v in zip(pop_pairs_ordered, np.linspace(0.85, 0.05, len(pop_pairs_ordered)))}

    xabs = np.nanmax(np.abs(forest["mean"].values)) * 1.3
    xabs = max(xabs, 0.3)
    ncols = 4
    nrows = 2

    df_corr["corr_ratio"] = df_corr["full_vs_subsample_corr"] / df_corr["full_vs_transformed_corr"]
    df_corr["corr_ratio"] = df_corr["corr_ratio"].replace([np.inf, -np.inf], np.nan)
    avg_corr = df_corr.groupby(["subsample_n", "ref_pop", "tgt_pop"])["corr_ratio"].apply(lambda x: np.exp(np.mean(np.log(x)))).reset_index(name="corr_ratio")

    subsample_order = sorted(df_corr["subsample_n"].unique())
    corr_log_vals = np.log10(avg_corr["corr_ratio"].dropna())
    corr_vabs = max(abs(corr_log_vals.min()), abs(corr_log_vals.max()))
    corr_norm = mcolors.TwoSlopeNorm(vmin=-corr_vabs, vcenter=0, vmax=corr_vabs)

    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1], hspace=0.22, left=0.12, right=0.95, top=0.95, bottom=0.06)

    gs_top = gs[0].subgridspec(nrows, ncols, wspace=0.25, hspace=0.35)
    forest_axes = []
    for i, (stat, label) in enumerate(zip(stat_order, stat_labels)):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs_top[r, c])
        forest_axes.append(ax)

        if stat == "geomean":
            raw = per_rep_gmean.copy()
        else:
            raw = df_forest[df_forest["stat"] == stat].copy()

        for pp in pop_pairs_ordered:
            vals = raw[raw["pop_pair"] == pp]["wd_ratio_log"].dropna().values
            if len(vals) == 0:
                vals = np.array([0.0])
            y = y_pos[pp]
            bp = ax.boxplot(vals, positions=[y], vert=False, widths=0.6, patch_artist=True, boxprops=dict(linewidth=0.8), medianprops=dict(color="black", linewidth=1.2), whiskerprops=dict(linewidth=0.7), capprops=dict(linewidth=0.7), flierprops=dict(marker=".", markersize=3, alpha=0.5))
            bp["boxes"][0].set_facecolor(pair_colors[pp])
            bp["boxes"][0].set_alpha(0.7)
            bp["boxes"][0].set_edgecolor("black")

        ax.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.6)
        ax.axhline(n_pairs - len(same) - 0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlim(-xabs, xabs)
        ax.set_ylim(-0.5, n_pairs - 0.5)
        ax.set_yticks(range(n_pairs))
        ax.set_yticklabels(list(reversed(pop_pairs_ordered)) if c == 0 else [])

        def _fmt_tick(v):
            val = 10 ** v
            if abs(val - round(val)) < 0.01 and val < 1000:
                return str(int(round(val)))
            txt = f"{val:.2g}"
            if "e+" in txt or "e-" in txt:
                base, exp = txt.split("e")
                exp = int(exp)
                if exp == 2:
                    return str(int(float(base) * 100))
                return f"{base}×10$^{{{exp}}}$"
            return txt

        xt = ax.get_xticks()
        xt = xt[(xt >= -xabs) & (xt <= xabs)]
        ax.set_xticks(xt)
        ax.set_xticklabels([_fmt_tick(v) for v in xt], fontsize=8)
        ax.set_title(label, fontsize=10, fontweight="bold" if stat == "geomean" else "normal")
        if r == nrows - 1:
            ax.set_xlabel("WD Ratio", fontsize=9)
        ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.3, color="gray")
        _remove_spines(ax)
        ax.tick_params(left=False)

    gs_bottom = fig.add_gridspec(1, len(subsample_order) + 1, wspace=0.15, width_ratios=[1] * len(subsample_order) + [0.12], left=0.08, right=0.82, top=0.28, bottom=0.06)
    corr_axes = []

    for j, sub_n in enumerate(subsample_order):
        ax = fig.add_subplot(gs_bottom[0, j])
        corr_axes.append(ax)
        subset = avg_corr[avg_corr["subsample_n"] == sub_n]
        pivot = subset.pivot(index="ref_pop", columns="tgt_pop", values="corr_ratio")
        pivot_log = np.log10(pivot)
        sns.heatmap(pivot_log, annot=pivot.values, fmt=".2f", cmap=VIRIDIS, norm=corr_norm, ax=ax, cbar=False, square=True, annot_kws={"size": 10}, linewidths=1, linecolor="white")
        ax.set_title(f"n={sub_n}")
        ax.set_ylabel("Reference Pop." if j == 0 else "")
        ax.set_xlabel("Target Pop.")
        ax.tick_params(left=False, bottom=False)
        if j > 0:
            ax.set_yticklabels([])

    cax2_parent = fig.add_subplot(gs_bottom[0, len(subsample_order)])
    cax2_parent.axis("off")
    cax2 = cax2_parent.inset_axes([0.0, 0.2, 0.8, 0.6])
    sm2 = plt.cm.ScalarMappable(cmap=VIRIDIS, norm=corr_norm)
    sm2.set_array([])
    cbar2 = fig.colorbar(sm2, cax=cax2)
    cbar2.set_label("Correlation Ratio")
    cbar2.set_ticks([np.log10(0.25), 0, np.log10(4)])
    cbar2.set_ticklabels(["0.25x", "1.0x", "4x"])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    bbs_a = [a.get_tightbbox(renderer).transformed(inv) for a in forest_axes]
    bbs_b = [a.get_tightbbox(renderer).transformed(inv) for a in corr_axes]
    label_x = min(min(b.x0 for b in bbs_a), min(b.x0 for b in bbs_b))
    fig.text(label_x, max(b.y1 for b in bbs_a) + 0.01, "A)", fontsize=14, fontweight="bold", ha="left", va="bottom")
    fig.text(label_x, max(b.y1 for b in bbs_b) + 0.02, "B)", fontsize=14, fontweight="bold", ha="left", va="bottom")

    plt.savefig(os.path.join(PLOT_DIR, "figure3.png"), dpi=300, bbox_inches="tight")
    plt.close()

# Fig 4- ABC posterior box plots
def plot_figure4():
    sizes = SUBSAMPLE_SIZES
    n_lab_order = ["Full"] + [str(s) for s in sizes]

    summ_path = os.path.join(ABC_COMBINED_DIR, "posterior_summary.csv")
    if not os.path.exists(summ_path):
        print(f"SKIP figure4: {summ_path} not found")
        return

    summ = pd.read_csv(summ_path)

    params = ["Divergence time (T)", "Population size (N0)"]
    param_labels = ["Divergence Time (T)", "Population Size (N)"]

    all_methods = sorted(summ["method"].unique(), key=lambda m: (0 if m == "Full data" else 1 if m == "Subsample" else 2 if m.startswith("Transform") else 3))

    summ["method"] = pd.Categorical(summ["method"], categories=all_methods, ordered=True)
    summ["n_lab"] = pd.Categorical(summ["n_lab"].astype(str), categories=n_lab_order, ordered=True)

    fig, axes = plt.subplots(len(POPULATIONS), len(params), figsize=(10, 11), sharey=False)

    for col, (param, param_label) in enumerate(zip(params, param_labels)):
        for row, target_pop in enumerate(POPULATIONS):
            ax = axes[row, col]
            sub = summ[(summ["param"] == param) & (summ["target"] == target_pop)].copy()
            sub = sub.sort_values(["n_lab", "method"])

            full_row = sub[sub["method"] == "Full data"]
            if not full_row.empty:
                ax.axhline(full_row["q50"].values[0], color="black", linestyle="--", linewidth=0.8)

            groups = [nl for nl in n_lab_order if nl in sub["n_lab"].values]
            n_groups = len(groups)
            methods_present = [m for m in all_methods if m in sub["method"].values]
            n_methods = len(methods_present)
            box_width = 0.7 / max(n_methods, 1)

            for gi, grp in enumerate(groups):
                grp_sub = sub[sub["n_lab"] == grp]
                for mi, method in enumerate(methods_present):
                    row_data = grp_sub[grp_sub["method"] == method]
                    if row_data.empty:
                        continue
                    r = row_data.iloc[0]
                    x = gi + (mi - (n_methods - 1) / 2) * box_width
                    color = _method_color(method)

                    ax.bar(x, r["q75"] - r["q25"], bottom=r["q25"], width=box_width * 0.9, color=color, edgecolor="black", linewidth=0.3)
                    ax.plot([x - box_width * 0.35, x + box_width * 0.35], [r["q50"], r["q50"]], color="black", linewidth=1.0)
                    ax.plot([x, x], [r["q025"], r["q25"]], color="black", linewidth=0.5)
                    ax.plot([x, x], [r["q75"], r["q975"]], color="black", linewidth=0.5)
                    ax.plot([x - box_width * 0.2, x + box_width * 0.2], [r["q025"], r["q025"]], color="black", linewidth=0.5)
                    ax.plot([x - box_width * 0.2, x + box_width * 0.2], [r["q975"], r["q975"]], color="black", linewidth=0.5)

            ax.set_xticks(range(n_groups))
            ax.set_xticklabels(groups, fontsize=8)
            if row == 0:
                ax.set_title(param_label, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(target_pop, fontsize=11, fontweight="bold")
            if row == len(POPULATIONS) - 1:
                ax.set_xlabel("Subsample size (n)", fontsize=9)
            _remove_spines(ax)

    n_methods = len(all_methods)
    legend_ncol = n_methods if n_methods <= 5 else (n_methods + 1) // 2
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=_method_color(m), edgecolor="black", linewidth=0.3) for m in all_methods]
    fig.legend(handles, all_methods, loc="lower center", ncol=legend_ncol, fontsize=7, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1.0])
    fig.savefig(os.path.join(PLOT_DIR, "figure4.png"), dpi=300, bbox_inches="tight")
    plt.close()


# S.Fig 1- LOO + AMOVA stuff
def plot_supp1():
    wd_all = pd.read_csv(LOO_CSV)

    stat_medians = wd_all.groupby("stat")["norm_wd_full_vs_transformed"].median()
    wd_all["wd_normalized"] = wd_all.apply(lambda r: r["norm_wd_full_vs_transformed"] / stat_medians[r["stat"]], axis=1)

    wd_all["pair"] = wd_all["ref_pop"] + "→" + wd_all["tgt_pop"]
    wd_per_obs = wd_all.groupby(["pair", "dropped_feature", "rep"])["wd_normalized"].mean().reset_index()

    groups = [grp["wd_normalized"].values for _, grp in wd_per_obs.groupby("dropped_feature")]
    f_val, p_val = f_oneway(*groups)
    wd_by_feat = wd_per_obs.groupby("dropped_feature")["wd_normalized"].agg(["mean", "std"]).reindex(STATS_COLS)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(range(len(STATS_COLS)), wd_by_feat["mean"], yerr=wd_by_feat["std"], capsize=4, color=BAR_COLORS, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(STATS_COLS)))
    ax.set_xticklabels(STAT_LABELS_MULTI, fontsize=8)
    ax.set_ylabel("Normalized WD (Transform to Full)")
    ymax = (wd_by_feat["mean"] + wd_by_feat["std"]).max()
    ax.set_ylim(0, ymax * 1.25)
    ax.text(0.50, 0.97, f"ANOVA F={f_val:.2f}, p={p_val:.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=10)
    _remove_spines(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "supp_figure1.png"), dpi=300, bbox_inches="tight")
    plt.close()

# S.Fig 2- Training results
def plot_supp2():
    wd = pd.read_csv(FEAT_WD_CSV)
    wd = wd.rename(columns={"ref_pop": "ref", "tgt_pop": "tgt", "subsample_n": "sample_size", "rep": "replicate"})
    train = pd.read_csv(TRAIN_CSV)

    ep_cols = [c for c in train.columns if c.startswith("gen_g_ep")]
    max_ep_label = sorted(ep_cols, key=lambda c: int(c.split("ep")[1]))[-1]
    ep_suffix = max_ep_label.split("gen_g_")[1]

    rows = []
    for _, r in train.iterrows():
        finals = {m: r[f"{m}_{ep_suffix}"] for m in METRICS}
        rows.append({"ref": r["ref"], "tgt": r["tgt"], "sample_size": r["sample_size"], "replicate": r["replicate"], "max_gen_disc_gap": max(abs(finals["gen_g"] - finals["disc_x"]), abs(finals["gen_g"] - finals["disc_y"]), abs(finals["gen_f"] - finals["disc_x"]), abs(finals["gen_f"] - finals["disc_y"]))})

    train_summary = pd.DataFrame(rows)
    merged = wd.merge(train_summary, on=["ref", "tgt", "sample_size", "replicate"])
    stat_list = merged["stat"].unique()

    fig, axes = plt.subplots(2, 3, figsize=(8.5, 6))
    for ax, stat in zip(axes.flat, stat_list):
        sub = merged[merged["stat"] == stat]
        ax.scatter(sub["max_gen_disc_gap"], sub["wd_ratio"], alpha=0.5, s=15, color=STAT_COLORS.get(stat, VIRIDIS(0.5)), edgecolors="none")
        rho, _ = spearmanr(sub["max_gen_disc_gap"], sub["wd_ratio"])
        ax.set_xlabel("Generator-Discriminator Gap")
        ax.set_ylabel("WD Ratio")
        ax.set_title(f"{STAT_LABEL_MULTI_MAP.get(stat, stat)}\nρ={rho:+.3f}")
        _remove_spines(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "supp_figure2.png"), dpi=300, bbox_inches="tight")
    plt.close()

# S.Fig 3- KDEs of per stat
def plot_supp3():
    n_plot = 10
    rep_plot = 1
    plot_pct = (2.5, 97.5)

    full_data = {}
    for fname in os.listdir(FULL_DIR):
        if fname.endswith(".csv"):
            p = fname.replace(".csv", "")
            full_data[p] = pd.read_csv(os.path.join(FULL_DIR, fname))

    transformed_dir = os.path.join(OUTPUT_DIR, "transformed")
    sub_dir = os.path.join(ALLEL_DIR, "subsamples_all")

    fig, axes = plt.subplots(len(STATS_COLS), len(POPULATIONS), figsize=(5 * len(POPULATIONS), 4 * len(STATS_COLS)), squeeze=False)

    for col_idx, tgt in enumerate(POPULATIONS):
        axes[0][col_idx].set_title(tgt, fontsize=12, fontweight="bold")
        for row_idx, stat in enumerate(STATS_COLS):
            ax = axes[row_idx][col_idx]

            full_vals = full_data[tgt][stat].dropna().values
            sub_path = os.path.join(sub_dir, f"{tgt}_subsample{n_plot}_replicate{rep_plot}.csv")
            sub_vals = pd.read_csv(sub_path)[stat].dropna().values

            transformed = {}
            for ref in POPULATIONS:
                tf_folder = os.path.join(transformed_dir, f"{ref}_{tgt}_{n_plot}_transformed")
                if not os.path.isdir(tf_folder):
                    continue
                for fname in os.listdir(tf_folder):
                    if f"_replicate{rep_plot}_to_" in fname and fname.endswith("_transformed.csv"):
                        transformed[ref] = pd.read_csv(os.path.join(tf_folder, fname))[stat].dropna().values
                        break

            all_arrays = [full_vals, sub_vals, *transformed.values()]
            lows, highs = zip(*(np.percentile(v, plot_pct) for v in all_arrays))
            xmin, xmax = min(lows), max(highs)
            pad = (xmax - xmin) * 0.05
            xs = np.linspace(xmin - pad, xmax + pad, 500)

            kde_full = gaussian_kde(full_vals)
            ax.plot(xs, kde_full(xs), color="dimgray", linewidth=2, label="Full")

            kde_sub = gaussian_kde(sub_vals)
            ax.plot(xs, kde_sub(xs), color="silver", linewidth=2, linestyle="--", label="Subsample")

            for ref in POPULATIONS:
                if ref in transformed:
                    kde_tf = gaussian_kde(transformed[ref])
                    ax.plot(xs, kde_tf(xs), color=POP_COLORS[ref], linewidth=1.5, label=f"Transform (ref={ref})")

            ax.set_xlim(xs[0], xs[-1])
            if col_idx == 0:
                ax.set_ylabel(STAT_LABEL_MAP[stat], fontsize=10)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            _remove_spines(ax)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7)

    plt.tight_layout(rect=(0, 0, 1, 1.0))
    plt.savefig(os.path.join(PLOT_DIR, "supp_figure3.png"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # plot_figure2()
    plot_figure3()
    # plot_figure4()
    # plot_supp1()
    # plot_supp2()
    # plot_supp3()