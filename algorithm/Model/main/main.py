# Original development version of the training pipeline.
# See src/train.py for the cleaned implementation with argparse and docstrings.

import os
import librosa
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import warnings
import xgboost as xgb

warnings.filterwarnings("ignore")


class VoiceDisorderClassifier:
    def __init__(self, audio_dir, test_size=0.2, n_splits=5, random_state=42):
        self.audio_dir = audio_dir
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'error'],
            'eta': 0.1,
            'max_depth': 4,
            'lambda': 1.0,
            'verbosity': 0
        }
        self.num_rounds = 100

    @staticmethod
    def add_noise(y, noise_factor=0.02):
        noise = np.random.randn(len(y))
        return y + noise_factor * noise

    @staticmethod
    def shift_pitch(y, sr, n_steps=2):
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    @staticmethod
    def extract_mel_spectrogram(y, sr=16000, n_mels=128, fixed_shape=(128, 100)):
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_resized = cv2.resize(mel_db, fixed_shape, interpolation=cv2.INTER_AREA)
        return mel_resized.flatten()

    def load_and_augment_data(self):
        normal_files, disorder_files = [], []
        for file_name in os.listdir(self.audio_dir):
            if file_name.endswith(".wav"):
                if file_name.startswith("N_"):
                    normal_files.append(file_name)
                elif file_name.startswith("D_"):
                    disorder_files.append(file_name)

        # Balance classes dynamically — use min count across both classes
        n = min(len(normal_files), len(disorder_files))
        selected_normal_files = normal_files[:n]
        selected_disorder_files = disorder_files[:n]

        original_audio_files, original_labels = [], []
        augmented_audio_files, augmented_labels = [], []

        for file_name in selected_normal_files + selected_disorder_files:
            file_path = os.path.join(self.audio_dir, file_name)
            y, sr = librosa.load(file_path, sr=16000)

            label = 0 if file_name.startswith("N_") else 1
            original_audio_files.append(y)
            original_labels.append(label)

            augmented_audio_files.append(y)
            augmented_labels.append(label)

            augmented_audio_files.append(self.add_noise(y))
            augmented_labels.append(label)

            augmented_audio_files.append(self.shift_pitch(y, sr, n_steps=2))
            augmented_labels.append(label)

        X_augmented = np.array([self.extract_mel_spectrogram(y) for y in augmented_audio_files])
        y_augmented = np.array(augmented_labels)

        normal_count = len([label for label in original_labels if label == 0])
        disorder_count = len([label for label in original_labels if label == 1])
        augmented_normal_count = len([label for label in augmented_labels if label == 0])
        augmented_disorder_count = len([label for label in augmented_labels if label == 1])

        print("\n### Data Summary ###")
        print(f"Data Before Augmentation: Normal = {normal_count}, Disorder = {disorder_count}")
        print(f"Data After Augmentation: Normal = {augmented_normal_count}, Disorder = {augmented_disorder_count}")

        return X_augmented, y_augmented

    def train_model(self, X_train, y_train, X_val, y_val):
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        evals_result = {}
        model = xgb.train(
            self.params, dtrain, self.num_rounds,
            evals=[(dtrain, 'train'), (dval, 'eval')],
            early_stopping_rounds=10, evals_result=evals_result, verbose_eval=False
        )
        return model, evals_result

    def evaluate_on_test_set(self, model, X_test, y_test, X_train_val):
        mean = np.mean(X_train_val, axis=0)
        std = np.std(X_train_val, axis=0)
        X_test_normalized = (X_test - mean) / std

        dtest = xgb.DMatrix(X_test_normalized)
        y_test_pred_proba = model.predict(dtest)
        y_test_pred = (y_test_pred_proba > 0.5).astype(int)

        test_accuracy = accuracy_score(y_test, y_test_pred) * 100
        test_f1 = f1_score(y_test, y_test_pred)

        conf_matrix = confusion_matrix(y_test, y_test_pred)
        self.plot_confusion_matrix(conf_matrix, "Confusion Matrix - Test Set")
        return test_accuracy, test_f1

    @staticmethod
    def plot_confusion_matrix(conf_matrix, title):
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['Normal', 'Disorder'], yticklabels=['Normal', 'Disorder'])
        plt.title(title)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    def aggregated_confusion_matrix(self, y_true, y_pred):
        conf_matrix = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(conf_matrix, "Aggregated Confusion Matrix - Validation Set")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train VOCA XGBoost classifier")
    parser.add_argument("--audio_dir", required=True, help="Path to renamed WAV files")
    args = parser.parse_args()

    classifier = VoiceDisorderClassifier(args.audio_dir)

    X_augmented, y_augmented = classifier.load_and_augment_data()

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_augmented, y_augmented, test_size=0.2, stratify=y_augmented, random_state=42
    )

    skf = StratifiedKFold(n_splits=classifier.n_splits, shuffle=True, random_state=classifier.random_state)

    all_y_true, all_y_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val)):
        print(f"\nFold {fold + 1}")
        X_train_fold, X_val_fold = X_train_val[train_idx], X_train_val[val_idx]
        y_train_fold, y_val_fold = y_train_val[train_idx], y_train_val[val_idx]

        model, _ = classifier.train_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold)

        mean = np.mean(X_train_fold, axis=0)
        std = np.std(X_train_fold, axis=0)
        X_val_normalized = (X_val_fold - mean) / std

        dval = xgb.DMatrix(X_val_normalized)
        y_val_pred = (model.predict(dval) > 0.5).astype(int)

        all_y_true.extend(y_val_fold)
        all_y_pred.extend(y_val_pred)

    classifier.aggregated_confusion_matrix(all_y_true, all_y_pred)

    dtrain_final = xgb.DMatrix((X_train_val - np.mean(X_train_val, axis=0)) / np.std(X_train_val, axis=0),
                               label=y_train_val)
    final_model = xgb.train(classifier.params, dtrain_final, classifier.num_rounds)
    final_model.save_model("voice_disorder_model.json")

    test_accuracy, test_f1 = classifier.evaluate_on_test_set(final_model, X_test, y_test, X_train_val)
    print(f"\nTest Set Accuracy: {test_accuracy:.2f}%")
    print(f"Test Set F1 Score: {test_f1:.4f}")


if __name__ == "__main__":
    main()