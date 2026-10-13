import allel
import numpy as np
import pandas as pd


def calculate_pi_windowed(gt, pos, windows):
    ac = gt.count_alleles()
    pi_values = []

    for i in range(len(windows) - 1):
        start, end = windows[i], windows[i + 1]
        mask = (pos >= start) & (pos < end)

        if not np.any(mask):
            pi_values.append(np.nan)
            continue

        window_ac = ac[mask]
        window_pos = pos[mask]

        pi = allel.sequence_diversity(window_pos, window_ac)
        pi_values.append(pi)

    return np.array(pi_values)


def calculate_tajima_d_windowed(gt, pos, windows):
    ac = gt.count_alleles()
    tajima_d_values = []

    for i in range(len(windows) - 1):
        start, end = windows[i], windows[i + 1]
        mask = (pos >= start) & (pos < end)

        if not np.any(mask):
            tajima_d_values.append(np.nan)
            continue

        window_ac = ac[mask]
        window_pos = pos[mask]

        try:
            td = allel.tajima_d(window_ac, window_pos)
            tajima_d_values.append(td)
        except Exception as e:
            tajima_d_values.append(np.nan)

    return np.array(tajima_d_values)


def calculate_maf_windowed(gt, pos, windows):
    np.random.seed(42)
    window_mafs = []

    for i in range(len(windows) - 1):
        start, end = windows[i], windows[i + 1]
        mask = (pos >= start) & (pos < end)

        if not np.any(mask):
            window_mafs.append(np.nan)
            continue

        window_gt = gt[mask]
        ac = window_gt.count_alleles()

        site_mafs = []
        for j in range(len(ac)):
            allele_counts = ac[j]
            total = allele_counts.sum()
            if total > 0:
                freqs = allele_counts / total
                sorted_freqs = np.sort(freqs)[::-1]
                if len(sorted_freqs) > 1:
                    site_mafs.append(sorted_freqs[1])

        if site_mafs:
            window_mafs.append(np.random.choice(site_mafs))
        else:
            window_mafs.append(np.nan)

    return np.array(window_mafs)


def calculate_seg_sites_windowed(gt, pos, windows):
    window_counts = []

    for i in range(len(windows) - 1):
        start, end = windows[i], windows[i + 1]
        mask = (pos >= start) & (pos < end)

        if not np.any(mask):
            window_counts.append(0)
            continue

        window_gt = gt[mask]
        ac = window_gt.count_alleles()

        seg_count = np.sum(ac.is_segregating())
        window_counts.append(seg_count)

    return np.array(window_counts)


def calculate_heterozygosity_stats_windowed(gt, pos, windows):
    h_means = []
    h_stds = []

    for i in range(len(windows) - 1):
        start, end = windows[i], windows[i + 1]
        mask = (pos >= start) & (pos < end)

        if not np.any(mask):
            h_means.append(np.nan)
            h_stds.append(np.nan)
            continue

        window_gt = gt[mask]
        ac = window_gt.count_alleles()
        af = ac.to_frequencies()

        het_values = []
        for j in range(len(af)):
            freqs = af[j]
            freqs = freqs[~np.isnan(freqs)]
            if len(freqs) > 0:
                het = 1 - np.sum(freqs**2)
                het_values.append(het)

        if len(het_values) > 0:
            h_means.append(np.mean(het_values))
            h_stds.append(np.std(het_values))
        else:
            h_means.append(np.nan)
            h_stds.append(np.nan)

    return np.array(h_means), np.array(h_stds)


def calculate_stats_from_ts(ts, window_size=30000):
    windows = np.arange(0, ts.sequence_length, window_size)
    if windows[-1] < ts.sequence_length:
        windows = np.append(windows, ts.sequence_length)

    G = ts.genotype_matrix()

    if G.shape[0] == 0:
        return pd.DataFrame({
            "window_start": windows[:-1],
            "window_end": windows[1:],
            "pi": np.nan,
            "tajima_d": np.nan,
            "maf": np.nan,
            "seg_sites": 0,
            "h_mean": np.nan,
            "h_std": np.nan,
        })

    positions = np.array([s.position for s in ts.sites()]).astype(int)

    n_haps = G.shape[1]
    n_diploid = n_haps // 2
    G_diploid = G.reshape(G.shape[0], n_diploid, 2)
    gt = allel.GenotypeArray(G_diploid)

    ac = gt.count_alleles()
    is_seg = ac.is_segregating()
    gt = gt[is_seg]
    positions = positions[is_seg]

    pi = calculate_pi_windowed(gt, positions, windows)
    tajima_d = calculate_tajima_d_windowed(gt, positions, windows)
    maf = calculate_maf_windowed(gt, positions, windows)
    seg_sites = calculate_seg_sites_windowed(gt, positions, windows)
    h_mean, h_std = calculate_heterozygosity_stats_windowed(gt, positions, windows)

    return pd.DataFrame({
        "window_start": windows[:-1],
        "window_end": windows[1:],
        "pi": pi,
        "tajima_d": tajima_d,
        "maf": maf,
        "seg_sites": seg_sites,
        "h_mean": h_mean,
        "h_std": h_std,
    }).dropna()

