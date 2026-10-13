import os
import shutil
import requests
import tensorflow as tf
from pathlib import Path


def frequency_to_sfs_tf(alt_freq_tensor, sample_size=20, sigma=0.05):
    bins = tf.linspace(0.0, 1.0, sample_size + 1)
    freqs = tf.expand_dims(alt_freq_tensor, 1)
    bins = tf.expand_dims(bins, 0)

    weights = tf.exp(-0.5 * ((freqs - bins) / sigma) ** 2)
    sfs = tf.reduce_sum(weights, axis=0)
    sfs = sfs / tf.reduce_sum(sfs)
    return sfs


def download_donni_model(model_name, param_idx, sample_size=20, folded=False, version="0.9.0"):
    folding = "folded" if folded else "unfolded"
    home_dir = str(Path.home())
    cache_dir = os.path.join(home_dir, ".cache", "donni", version, f"{model_name}_{folding}_ns_{sample_size}")
    os.makedirs(cache_dir, exist_ok=True)
    model_filename = f"param_{param_idx:02d}_predictor.keras"
    model_path = os.path.join(cache_dir, model_filename)
    if os.path.exists(model_path):
        return model_path
    urls_to_try = [
        f"https://data.cyverse.org/dav-anon/iplant/home/shared/donni/{model_name}/{folding}/ss_{sample_size}/v{version}/tuned_models/{model_filename}",
        f"https://de.cyverse.org/api/download?path=/iplant/home/shared/donni/{model_name}/{folding}/ss_{sample_size}/v{version}/tuned_models/{model_filename}",
    ]
    for url in urls_to_try:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(model_path, "wb") as f:
                    f.write(response.content)
                return model_path
        except Exception:
            pass
    return None


def load_model_for_inference(model_path_or_dir):
    try:
        if os.path.isfile(model_path_or_dir) and model_path_or_dir.endswith(".keras"):
            h5_path = model_path_or_dir.replace(".keras", ".h5")
            shutil.copy(model_path_or_dir, h5_path)
            model = tf.keras.models.load_model(h5_path, compile=False)
            return model
    except Exception:
        pass
    return None


def get_model_param_count(model_name):
    return {"two_epoch": 3, "three_epoch": 5, "growth": 4}.get(model_name, 3)


def load_donni_models(model_name, sample_size=20, folded=False, version="0.9.0"):
    num_params = get_model_param_count(model_name)
    donni_models = {}
    for param_idx in range(num_params):
        model_path = download_donni_model(model_name, param_idx, sample_size, folded, version)
        if model_path and os.path.exists(model_path):
            model = load_model_for_inference(model_path)
            if model is not None:
                donni_models[param_idx] = model
    param_names_map = {
        "two_epoch": ["N_anc", "N_A", "T_split"],
        "three_epoch": ["N_anc", "N_A_ancient", "N_A", "T_growth", "T_split"],
        "growth": ["N_anc", "N_A", "growth_rate", "T_split"],
    }
    param_names = param_names_map.get(model_name, [f"param_{i}" for i in range(num_params)])
    return donni_models, param_names


def demographic_loss_fn(original_freqs, reconstructed_freqs, donni_models, sample_size=20):
    source_sfs = frequency_to_sfs_tf(original_freqs, sample_size)
    recov_sfs = frequency_to_sfs_tf(reconstructed_freqs, sample_size)

    param_diffs = []
    for param_idx, model in donni_models.items():
        source_tensor = tf.reshape(tf.convert_to_tensor(source_sfs, dtype=tf.float32), (1, -1))
        recov_tensor = tf.reshape(tf.convert_to_tensor(recov_sfs, dtype=tf.float32), (1, -1))
        source_param = model(source_tensor)[0]
        recov_param = model(recov_tensor)[0]
        diff = tf.abs(source_param - recov_param)
        param_diffs.append(tf.reduce_mean(diff))

    total_loss = tf.reduce_mean(param_diffs)
    return total_loss
