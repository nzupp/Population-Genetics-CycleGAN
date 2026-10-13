import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
import argparse
import os
from donni_utils import load_donni_models, demographic_loss_fn
import random

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE
FEATURES = ['pi', 'tajima_d', 'maf', 'seg_sites', 'h_mean', 'h_std']
MAF_IDX = FEATURES.index('maf')


def maf_bounds(ref_maf):
    # pad
    floor = float(ref_maf[ref_maf > 0].min())
    return floor, floor / 2, 0.5 + floor / 2


def maf_to_unbounded(maf, lo, hi):
    u = (maf - lo) / (hi - lo)
    return np.log(u / (1 - u))


def maf_from_unbounded(x, lo, hi):
    return lo + (hi - lo) / (1 + np.exp(-x))


def maf_from_unbounded_tf(x, lo, hi):
    return lo + (hi - lo) * tf.sigmoid(x)


class Generator(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim=512):
        super(Generator, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(hidden_dim)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.hidden2 = tf.keras.layers.Dense(hidden_dim // 2)
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.hidden3 = tf.keras.layers.Dense(hidden_dim // 4)
        self.ln3 = tf.keras.layers.LayerNormalization()
        self.output_layer = tf.keras.layers.Dense(input_dim)

    def call(self, x, training=False):
        h = tf.nn.relu(self.ln1(self.hidden1(x), training=training))
        h = tf.nn.relu(self.ln2(self.hidden2(h), training=training))
        h = tf.nn.relu(self.ln3(self.hidden3(h), training=training))
        return self.output_layer(h)

class Discriminator(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(hidden_dim)
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.hidden2 = tf.keras.layers.Dense(hidden_dim // 2)
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.hidden3 = tf.keras.layers.Dense(hidden_dim // 4)
        self.dropout3 = tf.keras.layers.Dropout(0.3)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        h = self.dropout1(tf.nn.relu(self.hidden1(x)), training=training)
        h = self.dropout2(tf.nn.relu(self.hidden2(h)), training=training)
        h = self.dropout3(tf.nn.relu(self.hidden3(h)), training=training)
        return self.output_layer(h)

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
    return (loss_obj(tf.ones_like(real), real) + loss_obj(tf.zeros_like(generated), generated)) * 0.5

def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_x, cycled_x):
    return tf.reduce_mean(tf.abs(real_x - cycled_x))

def identity_loss(real, same):
    return tf.reduce_mean(tf.abs(real - same))

def covariance_loss(fake, real):
    fake_centered = fake - tf.reduce_mean(fake, axis=0, keepdims=True)
    real_centered = real - tf.reduce_mean(real, axis=0, keepdims=True)

    fake_cov = tf.matmul(fake_centered, fake_centered, transpose_a=True) / tf.cast(tf.shape(fake)[0], tf.float32)
    real_cov = tf.matmul(real_centered, real_centered, transpose_a=True) / tf.cast(tf.shape(real)[0], tf.float32)

    return tf.reduce_mean(tf.square(fake_cov - real_cov))

def load_data(ref_path, tgt_path, features, batch_size=256, buffer_size=50000):
    ref_df = pd.read_csv(ref_path).dropna(subset=features)
    tgt_df = pd.read_csv(tgt_path).dropna(subset=features)

    X_ref = ref_df[features].values.astype(np.float64)
    X_tgt = tgt_df[features].values.astype(np.float64)

    maf_floor, maf_lo, maf_hi = maf_bounds(X_ref[:, MAF_IDX])
    X_ref[:, MAF_IDX] = maf_to_unbounded(X_ref[:, MAF_IDX], maf_lo, maf_hi)
    X_tgt[:, MAF_IDX] = maf_to_unbounded(np.clip(X_tgt[:, MAF_IDX], maf_floor, 0.5), maf_lo, maf_hi)
    X_ref, X_tgt = X_ref.astype(np.float32), X_tgt.astype(np.float32)

    scaler = StandardScaler().fit(X_ref)
    X_ref_scaled = scaler.transform(X_ref)
    X_tgt_scaled = scaler.transform(X_tgt)

    train_ref = (tf.data.Dataset.from_tensor_slices(X_ref_scaled).cache().shuffle(buffer_size).batch(batch_size).prefetch(AUTOTUNE))
    train_tgt = (tf.data.Dataset.from_tensor_slices(X_tgt_scaled).cache().shuffle(buffer_size).batch(batch_size).prefetch(AUTOTUNE))
    return X_tgt_scaled, scaler, train_ref, train_tgt, (maf_floor, maf_lo, maf_hi)

def build_models(input_dim, lr=1e-5):
    generator_g = Generator(input_dim)
    generator_f = Generator(input_dim)

    discriminator_x = Discriminator(input_dim)
    discriminator_y = Discriminator(input_dim)

    generator_g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)

    discriminator_x_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)

    return (generator_g, generator_f, discriminator_x, discriminator_y,
            generator_g_optimizer, generator_f_optimizer,
            discriminator_x_optimizer, discriminator_y_optimizer)

def setup_checkpoint(models, optimizers, checkpoint_path):
    generator_g, generator_f, discriminator_x, discriminator_y = models
    gen_g_opt, gen_f_opt, disc_x_opt, disc_y_opt = optimizers

    ckpt = tf.train.Checkpoint(
        generator_g=generator_g, generator_f=generator_f,
        discriminator_x=discriminator_x, discriminator_y=discriminator_y,
        generator_g_optimizer=gen_g_opt, generator_f_optimizer=gen_f_opt,
        discriminator_x_optimizer=disc_x_opt, discriminator_y_optimizer=disc_y_opt
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        return ckpt, ckpt_manager, True

    return ckpt, ckpt_manager, False

def create_train_step(generator_g, generator_f, discriminator_x, discriminator_y,
                      generator_g_optimizer, generator_f_optimizer,
                      discriminator_x_optimizer, discriminator_y_optimizer,
                      donni_models, scales, scaler, maf_params):
    
    # donni needs MAF as a frequency: undo the scaler then the logit
    _, maf_lo, maf_hi = maf_params
    maf_mean = tf.constant(scaler.mean_[MAF_IDX], dtype=tf.float32)
    maf_std = tf.constant(scaler.scale_[MAF_IDX], dtype=tf.float32)

    def to_maf(x_scaled):
        return maf_from_unbounded_tf(x_scaled[:, MAF_IDX] * maf_std + maf_mean, maf_lo, maf_hi)

    @tf.function
    def train_step(real_x, real_y):
        with tf.GradientTape(persistent=True) as tape:
            fake_y = generator_g(real_x, training=True)
            cycled_x = generator_f(fake_y, training=True)

            fake_x = generator_f(real_y, training=True)
            cycled_y = generator_g(fake_x, training=True)

            same_x = generator_f(real_x, training=True)
            same_y = generator_g(real_y, training=True)

            disc_real_x = discriminator_x(real_x, training=True)
            disc_real_y = discriminator_y(real_y, training=True)

            disc_fake_x = discriminator_x(fake_x, training=True)
            disc_fake_y = discriminator_y(fake_y, training=True)

            gen_g_loss = generator_loss(disc_fake_y)
            gen_f_loss = generator_loss(disc_fake_x)

            cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
            cov_loss = covariance_loss(fake_x, real_x)

            id_loss_x = identity_loss(real_x, same_x)
            id_loss_y = identity_loss(real_y, same_y)

            demo_loss_x = demographic_loss_fn(to_maf(real_x), to_maf(cycled_x), donni_models, sample_size=20)
            demo_loss_y = demographic_loss_fn(to_maf(real_y), to_maf(cycled_y), donni_models, sample_size=20)
            demo_loss = demo_loss_x + demo_loss_y
            demo_loss = tf.where(tf.math.is_finite(demo_loss), demo_loss, tf.zeros_like(demo_loss))

            total_gen_g_loss = scales["gen_g_loss"] * gen_g_loss + scales["cycle_loss"] * cycle_loss + scales["demo_loss"] * demo_loss + scales["id_loss_y"] * id_loss_y
            total_gen_f_loss = scales["gen_f_loss"] * gen_f_loss + scales["cycle_loss"] * cycle_loss + scales["demo_loss"] * demo_loss + scales["cov_loss"] * cov_loss + scales["id_loss_x"] * id_loss_x

            disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

        generator_g_optimizer.apply_gradients(zip(tape.gradient(total_gen_g_loss, generator_g.trainable_variables), generator_g.trainable_variables))
        generator_f_optimizer.apply_gradients(zip(tape.gradient(total_gen_f_loss, generator_f.trainable_variables), generator_f.trainable_variables))
        discriminator_x_optimizer.apply_gradients(zip(tape.gradient(disc_x_loss, discriminator_x.trainable_variables), discriminator_x.trainable_variables))
        discriminator_y_optimizer.apply_gradients(zip(tape.gradient(disc_y_loss, discriminator_y.trainable_variables), discriminator_y.trainable_variables))
        return {
            'gen_g_loss': gen_g_loss, 'gen_f_loss': gen_f_loss,
            'total_gen_g_loss': total_gen_g_loss, 'total_gen_f_loss': total_gen_f_loss,
            'disc_x_loss': disc_x_loss, 'disc_y_loss': disc_y_loss,
            'cycle_loss': cycle_loss, 'demo_loss': demo_loss, 'cov_loss': cov_loss,
            'id_loss_x': id_loss_x, 'id_loss_y': id_loss_y,
        }
    return train_step

def train(train_ref, train_tgt, train_step, epochs, log_file):
    loss_history = {
        'gen_g_loss': [], 'gen_f_loss': [], 'total_gen_g_loss': [], 'total_gen_f_loss': [],
        'disc_x_loss': [], 'disc_y_loss': [], 'cycle_loss': [], 'demo_loss': [], 'cov_loss': [],
        'id_loss_x': [], 'id_loss_y': []
    }

    for epoch in range(epochs):
        start = time.time()
        epoch_losses = {key: [] for key in loss_history}
        for n, (real_x, real_y) in enumerate(tf.data.Dataset.zip((train_ref, train_tgt))):
            losses = train_step(real_x, real_y)
            for key in loss_history:
                epoch_losses[key].append(float(losses[key].numpy()))
            if n % 10 == 0:
                log_file.write('.')
        for key in loss_history:
            loss_history[key].append(np.mean(epoch_losses[key]))
        epoch_summary = f'\nEpoch {epoch+1}/{epochs} - Time: {time.time()-start:.2f}s\n'
        for key in loss_history:
            epoch_summary += f'  {key}: {loss_history[key][-1]:.6f}\n'
        log_file.write(epoch_summary)
    return loss_history

def transform_data(generator_f, X_tgt_scaled, scaler, features, maf_params):
    maf_floor, maf_lo, maf_hi = maf_params
    X_tgt_tensor = tf.convert_to_tensor(X_tgt_scaled, dtype=tf.float32)
    X_tgt_to_ref_scaled = generator_f(X_tgt_tensor, training=False).numpy()
    X_tgt_to_ref = scaler.inverse_transform(X_tgt_to_ref_scaled).astype(np.float64)
    X_tgt_to_ref[:, MAF_IDX] = np.clip(maf_from_unbounded(X_tgt_to_ref[:, MAF_IDX], maf_lo, maf_hi), maf_floor, 0.5)
    return pd.DataFrame(X_tgt_to_ref, columns=features)

def load_loss_lut(path="loss_scaling_lut.csv"):
    df = pd.read_csv(path)
    df.set_index(["ref", "tgt"], inplace=True)
    return df

def main(ref_path, tgt_path, output_dir, epochs, loss_lut_pth):
    os.makedirs(output_dir, exist_ok=True)
    ref_name = os.path.splitext(os.path.basename(ref_path))[0]
    tgt_name = os.path.splitext(os.path.basename(tgt_path))[0]

    lut_tgt_name = tgt_name.split("_")[0]
    loss_lut = load_loss_lut(loss_lut_pth)
    scales = loss_lut.loc[(ref_name, lut_tgt_name)].to_dict()

    checkpoint_path = os.path.join(output_dir, f"checkpoints/{ref_name}_{tgt_name}")
    log_path = os.path.join(output_dir, f"{ref_name}_{tgt_name}_training.txt")
    transformed_path = os.path.join(output_dir, f"{tgt_name}_to_{ref_name}_transformed.csv")

    with open(log_path, 'w') as log_file:
        log_file.write(f"=== CycleGAN Training Log ===\n")
        log_file.write(f"REF: {ref_path}\nTGT: {tgt_path}\nEpochs: {epochs}\n\n")
        X_tgt_scaled, scaler, train_ref, train_tgt, maf_params = load_data(ref_path, tgt_path, FEATURES)
        log_file.write(f"MAF bounds: floor={maf_params[0]:.6f}, logit range=({maf_params[1]:.6f}, {maf_params[2]:.6f})\n")
        donni_models, _ = load_donni_models('two_epoch', sample_size=20)
        input_dim = len(FEATURES)

        generator_g, generator_f, discriminator_x, discriminator_y, gen_g_opt, gen_f_opt, disc_x_opt, disc_y_opt = build_models(input_dim)

        models = generator_g, generator_f, discriminator_x, discriminator_y
        optimizers = gen_g_opt, gen_f_opt, disc_x_opt, disc_y_opt
        ckpt, ckpt_manager, restored = setup_checkpoint(models, optimizers, checkpoint_path)

        if restored:
            log_file.write("Latest checkpoint restored!!\n")
        train_step = create_train_step(
            generator_g, generator_f, discriminator_x, discriminator_y,
            gen_g_opt, gen_f_opt, disc_x_opt, disc_y_opt,
            donni_models, scales, scaler, maf_params
        )

        loss_history = train(train_ref, train_tgt, train_step, epochs, log_file)
        tgt_to_ref_df = transform_data(generator_f, X_tgt_scaled, scaler, FEATURES, maf_params)
        tgt_to_ref_df.to_csv(transformed_path, index=False)
        log_file.write(f"\nTransformed data saved to {transformed_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--tgt", required=True)
    parser.add_argument("--output_dir", default="./output_test")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--loss_lut", default="loss_scaling_lut.csv")
    args = parser.parse_args()
    main(args.ref, args.tgt, args.output_dir, args.epochs, args.loss_lut)
