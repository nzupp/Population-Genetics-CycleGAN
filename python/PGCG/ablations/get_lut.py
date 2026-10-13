import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import os
from donni_utils import load_donni_models, demographic_loss_fn
from PGCG import (
    build_models,
    load_data,
    calc_cycle_loss,
    generator_loss,
    discriminator_loss,
    identity_loss,
    covariance_loss,
    FEATURES,
    SEED,
)

tf.random.set_seed(SEED)
np.random.seed(SEED)

# HPC pointer to stats calced from VCFs
POPS = {
    "CEU": "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/preprocess/allel_stats/full_all/CEU.csv",
    "CHB": "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/preprocess/allel_stats/full_all/CHB.csv",
    "YRI": "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/preprocess/allel_stats/full_all/YRI.csv",
}


def calibrate(ref_path, tgt_path, donni_models, n_batches=50):
    X_tgt_scaled, scaler, train_ref, train_tgt, _ = load_data(ref_path, tgt_path, FEATURES)
    input_dim = len(FEATURES)
    generator_g, generator_f, discriminator_x, discriminator_y, gen_g_opt, gen_f_opt, disc_x_opt, disc_y_opt = build_models(input_dim)

    maf_idx = FEATURES.index("maf")
    maf_mean = tf.constant(scaler.mean_[maf_idx], dtype=tf.float32)
    maf_std = tf.constant(np.sqrt(scaler.var_[maf_idx]), dtype=tf.float32)

    loss_accum = {
        "gen_g_loss": [],
        "gen_f_loss": [],
        "cycle_loss": [],
        "cov_loss": [],
        "id_loss_x": [],
        "id_loss_y": [],
        "disc_x_loss": [],
        "disc_y_loss": [],
        "demo_loss": [],
    }

    for n, (real_x, real_y) in enumerate(tf.data.Dataset.zip((train_ref, train_tgt))):
        if n >= n_batches:
            break

        fake_y = generator_g(real_x, training=False)
        cycled_x = generator_f(fake_y, training=False)
        fake_x = generator_f(real_y, training=False)
        cycled_y = generator_g(fake_x, training=False)
        same_x = generator_f(real_x, training=False)
        same_y = generator_g(real_y, training=False)

        disc_real_x = discriminator_x(real_x, training=False)
        disc_real_y = discriminator_y(real_y, training=False)
        disc_fake_x = discriminator_x(fake_x, training=False)
        disc_fake_y = discriminator_y(fake_y, training=False)

        maf_x = real_x[:, maf_idx] * maf_std + maf_mean
        maf_cycled_x = cycled_x[:, maf_idx] * maf_std + maf_mean
        maf_y = real_y[:, maf_idx] * maf_std + maf_mean
        maf_cycled_y = cycled_y[:, maf_idx] * maf_std + maf_mean

        demo_loss_x = demographic_loss_fn(maf_x, maf_cycled_x, donni_models, sample_size=20)
        demo_loss_y = demographic_loss_fn(maf_y, maf_cycled_y, donni_models, sample_size=20)

        demo_loss = demo_loss_x + demo_loss_y
        demo_loss_val = float(demo_loss.numpy()) if tf.math.is_finite(demo_loss) else np.nan

        if n < 3:
            print(f"[DEBUG] batch {n} demo_loss: {demo_loss_val}")

        loss_accum["gen_g_loss"].append(float(generator_loss(disc_fake_y)))
        loss_accum["gen_f_loss"].append(float(generator_loss(disc_fake_x)))
        loss_accum["cycle_loss"].append(float(calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)))
        loss_accum["cov_loss"].append(float(covariance_loss(fake_x, real_x)))
        loss_accum["id_loss_x"].append(float(identity_loss(real_x, same_x)))
        loss_accum["id_loss_y"].append(float(identity_loss(real_y, same_y)))
        loss_accum["disc_x_loss"].append(float(discriminator_loss(disc_real_x, disc_fake_x)))
        loss_accum["disc_y_loss"].append(float(discriminator_loss(disc_real_y, disc_fake_y)))
        loss_accum["demo_loss"].append(demo_loss_val)

    means = {k: np.nanmean(v) for k, v in loss_accum.items()}
    scales = {k: 1.0 / v if v > 0 else 1.0 for k, v in means.items()}
    return scales


def main(n_batches, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    donni_models, _ = load_donni_models("two_epoch", sample_size=20)

    rows = []
    for ref, ref_path in POPS.items():
        for tgt, tgt_path in POPS.items():
            print(f"\n=== Calibrating ref={ref} tgt={tgt} ===")
            scales = calibrate(ref_path, tgt_path, donni_models, n_batches)
            row = {"ref": ref, "tgt": tgt}
            row.update(scales)
            rows.append(row)

    full_df = pd.DataFrame(rows)
    full_df = full_df.round(1)
    full_df.to_csv(os.path.join(output_dir, "loss_scaling_lut.csv"), index=False)

    cov_abl_df = full_df.copy()
    cov_abl_df["cov_loss"] = 0.0
    cov_abl_df.to_csv(os.path.join(output_dir, "cov_abl_lut.csv"), index=False)

    demo_abl_df = full_df.copy()
    demo_abl_df["demo_loss"] = 0.0
    demo_abl_df.to_csv(os.path.join(output_dir, "demo_abl_lut.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()
    main(args.n_batches, args.output_dir)
