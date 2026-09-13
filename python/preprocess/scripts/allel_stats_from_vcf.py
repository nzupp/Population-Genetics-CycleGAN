import allel
import numpy as np
import pandas as pd
import os
import sys
import re


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

        # Collect MAFs for all segregating sites in window
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
            # Randomly sample one site per window
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


def process_vcf_file(vcf_path, chrom, window_size=30000, output_dir="results"):
    callset = allel.read_vcf(vcf_path, fields=["samples", "variants/POS", "calldata/GT", "variants/CHROM"])

    gt = allel.GenotypeArray(callset["calldata/GT"])
    positions = callset["variants/POS"]

    unique_positions, unique_indices = np.unique(positions, return_index=True)
    positions = unique_positions
    gt = gt[unique_indices]

    ac = gt.count_alleles()
    is_seg = ac.is_segregating()
    gt = gt[is_seg]
    positions = positions[is_seg]

    max_pos = positions.max()
    windows = np.arange(0, max_pos + window_size, window_size)
    if windows[-1] < max_pos:
        windows = np.append(windows, max_pos)

    pi = calculate_pi_windowed(gt, positions, windows)
    tajima_d = calculate_tajima_d_windowed(gt, positions, windows)
    maf = calculate_maf_windowed(gt, positions, windows)
    seg_sites = calculate_seg_sites_windowed(gt, positions, windows)
    h_mean, h_std = calculate_heterozygosity_stats_windowed(gt, positions, windows)

    results = pd.DataFrame(
        {
            "chrom": chrom,
            "window_start": windows[:-1],
            "window_end": windows[1:],
            "pi": pi,
            "tajima_d": tajima_d,
            "maf": maf,
            "seg_sites": seg_sites,
            "h_mean": h_mean,
            "h_std": h_std,
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(vcf_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_stats.csv")

    # Drop windows where any statistic could not be calculated
    results = results.dropna()
    results.to_csv(output_file, index=False)

    print(f"Total windows with data: {len(results)}")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python vcf_stats_allel.py <vcf_file> <suffix>")
        sys.exit(1)

    vcf_file = sys.argv[1]
    suffix = sys.argv[2]

    chrom_match = re.search(r"chr(\d+)", os.path.basename(vcf_file))
    if chrom_match:
        chrom = int(chrom_match.group(1))
    else:
        print("using 'unknown' for chrom number")
        chrom = "unknown"

    output_dir = f"/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/preprocess/allel_stats/{suffix}/"

    results = process_vcf_file(vcf_file, chrom, window_size=30000, output_dir=output_dir)
