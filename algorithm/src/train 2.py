"""
train.py — XGBoost training pipeline for voice disorder detection.

Loads WAV files from a directory, extracts Mel-Spectrogram features,
applies 3× augmentation, runs 5-fold Stratified K-Fold CV, then trains
a final model on the full train split and evaluates on held-out test set.

Usage:
    python algorithm/src/train.py --audio_dir path/to/renamed_audio_files/

Saves:
    outputs/models/voice_disorder_model.json
    outputs/models/mean_std_values.npy
"""

import os
import argparse
import numpy as np
import librosa
import cv2
import warnings
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

OUT_MODELS = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "models")
OUT_FIGURES = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "figures")


# ── Augmentation ──────────────────────────────────────────────────────────────

def add_noise(y, noise_factor=0.02):
    """
    Add Gaussian noise to an audio signal.

    Args:
        y (np.ndarray): Audio time-series.
        noise_factor (float): Scale of noise relative to signal.

    Returns:
        np.ndarray: Noisy audio signal.

    Example:
        >>> y_noisy = add_noise(y, noise_factor=0.02)
    """
    return y + noise_factor * np.random.randn(len(y))


def shift_pitch(y, sr, n_steps=2):
    """
    Shift the pitch of an audio signal by n semitones.

    Args:
        y (np.ndarray): Audio time-series.
        sr (int): Sample rate.
        n_steps (int): Number of semitones to shift (positive = up).

    Returns:
        np.ndarray: Pitch-shifted audio signal.
    """
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_mel_spectrogram(y, sr=16000, n_mels=128, fixed_shape=(128, 100)):
    """
    Extract a fixed-size Mel-Spectrogram from an audio signal.

    Converts the raw waveform to a log-scaled Mel-Spectrogram, then resizes
    it to a fixed shape so every sample produces the same feature vector size
    regardless of recording length.

    Args:
        y (np.ndarray): Audio time-series.
        sr (int): Sample rate (default 16000 Hz).
        n_mels (int): Number of Mel frequency bands.
        fixed_shape (tuple): (width, height) for the resized spectrogram.

    Returns:
        np.ndarray: Flattened 1D feature vector of length n_mels * fixed_shape[1].

    Example:
        >>> features = extract_mel_spectrogram(y, sr=16000)
        >>> features.shape
        (12800,)
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_resized = cv2.resize(mel_db, fixed_shape, interpolation=cv2.INTER_AREA)
    return mel_resized.flatten()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_and_augment(audio_dir, n_per_class=None):
    """
    Load WAV files, balance classes, and apply 3× augmentation.

    File naming convention: {LABEL}_{GENDER}_{INDEX}.wav
    Normal files start with 'N_', disorder files start with 'D_'.

    Augmentation triples the dataset:
        original + add_noise + shift_pitch  (×3 per sample)

    Args:
        audio_dir (str): Path to directory containing renamed WAV files.
        n_per_class (int, optional): Max samples per class. Uses all available if not set.

    Returns:
        tuple: (X, y) — feature matrix and label array.
    """
    normal_files   = [f for f in os.listdir(audio_dir) if f.endswith(".wav") and f.startswith("N_")]
    disorder_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav") and f.startswith("D_")]

    if n_per_class:
        normal_files   = normal_files[:n_per_class]
        disorder_files = disorder_files[:n_per_class]

    selected = normal_files + disorder_files

    audio_data, labels = [], []
    for fname in selected:
        y, sr = librosa.load(os.path.join(audio_dir, fname), sr=16000)
        label = 0 if fname.startswith("N_") else 1

        for y_aug in [y, add_noise(y), shift_pitch(y, sr, n_steps=2)]:
            audio_data.append(extract_mel_spectrogram(y_aug))
            labels.append(label)

    X = np.array(audio_data)
    y_arr = np.array(labels)

    normal_count   = np.sum(y_arr == 0)
    disorder_count = np.sum(y_arr == 1)
    print(f"After augmentation — Normal: {normal_count}, Disorder: {disorder_count}, Total: {len(y_arr)}")
    return X, y_arr


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, title, save_path=None):
    """
    Plot and optionally save a heatmap confusion matrix.

    Args:
        cm (np.ndarray): 2×2 confusion matrix.
        title (str): Plot title.
        save_path (str, optional): File path to save the figure.
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Disorder"],
                yticklabels=["Normal", "Disorder"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ── Training ──────────────────────────────────────────────────────────────────

def train_fold(X_train, y_train, X_val, y_val, params, num_rounds=100):
    """
    Train XGBoost on one CV fold.

    Normalizes using train-fold statistics only — fitting on the full dataset
    before splitting is a leakage mistake this explicitly avoids.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
        params (dict): XGBoost hyperparameters.
        num_rounds (int): Maximum boosting rounds.

    Returns:
        tuple: (model, mean, std) for this fold.
    """
    mean = np.mean(X_train, axis=0)
    std  = np.std(X_train, axis=0) + 1e-8   # avoid division by zero

    dtrain = xgb.DMatrix((X_train - mean) / std, label=y_train)
    dval   = xgb.DMatrix((X_val   - mean) / std, label=y_val)

    model = xgb.train(
        params, dtrain, num_rounds,
        evals=[(dtrain, "train"), (dval, "eval")],
        early_stopping_rounds=10,
        verbose_eval=False,
    )
    return model, mean, std


# ── Main ──────────────────────────────────────────────────────────────────────

def main(audio_dir, n_splits=5, test_size=0.2, random_state=42, num_rounds=100):
    """
    Full training pipeline: load → split → CV → final model → evaluate.

    Args:
        audio_dir (str): Path to directory with renamed WAV files.
        n_splits (int): Number of Stratified K-Fold splits.
        test_size (float): Fraction of data held out for final test evaluation.
        random_state (int): Reproducibility seed.
        num_rounds (int): Max XGBoost boosting rounds per fold.
    """
    os.makedirs(OUT_MODELS, exist_ok=True)
    os.makedirs(OUT_FIGURES, exist_ok=True)

    params = {
        "objective":   "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "eta":         0.1,
        "max_depth":   4,
        "lambda":      1.0,   # L2 regularization
        "verbosity":   0,
    }

    X, y = load_and_augment(audio_dir)

    # Hold out 20% as a test set that never touches training
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 5-fold CV to get a reliable performance estimate
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_y_true, all_y_pred, fold_aucs = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val)):
        print(f"\nFold {fold + 1}/{n_splits}")
        X_tr, X_vl = X_train_val[train_idx], X_train_val[val_idx]
        y_tr, y_vl = y_train_val[train_idx], y_train_val[val_idx]

        model, mean, std = train_fold(X_tr, y_tr, X_vl, y_vl, params, num_rounds)

        dval  = xgb.DMatrix((X_vl - mean) / std)
        proba = model.predict(dval)
        preds = (proba > 0.5).astype(int)

        fold_auc = roc_auc_score(y_vl, proba)
        fold_f1  = f1_score(y_vl, preds)
        print(f"  AUC-ROC: {fold_auc:.4f}  |  F1: {fold_f1:.4f}")

        fold_aucs.extend([fold_auc])
        all_y_true.extend(y_vl)
        all_y_pred.extend(preds)

    print(f"\nMean CV AUC-ROC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    plot_confusion_matrix(
        confusion_matrix(all_y_true, all_y_pred),
        "Aggregated Confusion Matrix — Validation Folds",
        save_path=os.path.join(OUT_FIGURES, "cv_confusion_matrix.png"),
    )

    # Train final model on full train+val split, evaluate on held-out test
    mean_final = np.mean(X_train_val, axis=0)
    std_final  = np.std(X_train_val,  axis=0) + 1e-8
    dtrain_final = xgb.DMatrix((X_train_val - mean_final) / std_final, label=y_train_val)
    final_model = xgb.train(params, dtrain_final, num_rounds)

    dtest = xgb.DMatrix((X_test - mean_final) / std_final)
    test_proba = final_model.predict(dtest)
    test_preds = (test_proba > 0.5).astype(int)

    test_acc = accuracy_score(y_test, test_preds) * 100
    test_f1  = f1_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_proba)

    print(f"\n{'='*40}")
    print(f"Test Set Accuracy : {test_acc:.2f}%")
    print(f"Test Set F1 Score : {test_f1:.4f}")
    print(f"Test Set AUC-ROC  : {test_auc:.4f}")
    print(f"{'='*40}")
    print(classification_report(y_test, test_preds, target_names=["Normal", "Disorder"]))

    plot_confusion_matrix(
        confusion_matrix(y_test, test_preds),
        "Test Set Confusion Matrix",
        save_path=os.path.join(OUT_FIGURES, "test_confusion_matrix.png"),
    )

    # Save model and normalization params
    model_path    = os.path.join(OUT_MODELS, "voice_disorder_model.json")
    norm_path     = os.path.join(OUT_MODELS, "mean_std_values.npy")
    final_model.save_model(model_path)
    np.save(norm_path, np.array([mean_final, std_final]))
    print(f"\nModel saved → {model_path}")
    print(f"Norm params saved → {norm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VOCA voice disorder classifier")
    parser.add_argument("--audio_dir", required=True, help="Path to renamed WAV files")
    parser.add_argument("--n_splits",  type=int,   default=5,    help="K-Fold splits")
    parser.add_argument("--test_size", type=float, default=0.2,  help="Test set fraction")
    parser.add_argument("--seed",      type=int,   default=42,   help="Random seed")
    parser.add_argument("--rounds",    type=int,   default=100,  help="Max XGBoost rounds")
    args = parser.parse_args()

    main(
        audio_dir=args.audio_dir,
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.seed,
        num_rounds=args.rounds,
    )
