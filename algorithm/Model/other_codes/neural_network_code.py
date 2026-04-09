# Experimental neural network approach using jitter, shimmer, and HNR features.
# This was the first architecture tried — overfitting on the clinical dataset led
# to switching to XGBoost with Mel-Spectrogram features. Kept for reference.

import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Augmentation functions
def add_noise(y, noise_factor=0.02):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def shift_pitch(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def stretch_time(y, stretch_factor=0.8):
    return librosa.effects.time_stretch(y, rate=stretch_factor)

# Feature extraction functions
def extract_jitter(y):
    frame_diffs = np.diff(y)
    return np.std(frame_diffs) / np.mean(frame_diffs) if np.mean(frame_diffs) != 0 else 0

def extract_shimmer(y):
    return np.std(np.abs(y)) / np.mean(np.abs(y)) if np.mean(np.abs(y)) != 0 else 0

def extract_hnr(y, sr):
    hnr = librosa.effects.harmonic(y, margin=8)
    noise = y - hnr
    noise_energy = np.sum(noise**2)
    hnr_energy = np.sum(hnr**2)
    return 10 * np.log10(hnr_energy / (noise_energy + 1e-10))

# Function to extract jitter, shimmer, and HNR features
def extract_features(y, sr=16000):
    jitter = extract_jitter(y)
    shimmer = extract_shimmer(y)
    hnr = extract_hnr(y, sr)
    return [jitter, shimmer, hnr]

import argparse
parser = argparse.ArgumentParser(description="Train NN voice disorder classifier (jitter/shimmer/HNR)")
parser.add_argument("--audio_dir", required=True, help="Path to renamed WAV files")
args = parser.parse_args()
audio_dir = args.audio_dir

# Load, label & balance data
normal_files = []
disorder_files = []

for file_name in os.listdir(audio_dir):
    if file_name.endswith(".wav"):
        if file_name.startswith("N_"):
            normal_files.append(file_name)
        elif file_name.startswith("D_"):
            disorder_files.append(file_name)

# Balance classes dynamically
n = min(len(normal_files), len(disorder_files))
selected_normal_files = normal_files[:n]
selected_disorder_files = disorder_files[:n]

audio_data = []
labels = []
for file_name in selected_normal_files + selected_disorder_files:
    file_path = os.path.join(audio_dir, file_name)
    y, sr = librosa.load(file_path, sr=16000)

    # Label data
    label = 0 if file_name.startswith("N_") else 1
    audio_data.append(extract_features(y, sr))
    labels.append(label)

    # Augment data
    audio_data.append(extract_features(add_noise(y), sr))
    labels.append(label)

    audio_data.append(extract_features(shift_pitch(y, sr, n_steps=2), sr))
    labels.append(label)

    audio_data.append(extract_features(stretch_time(y, stretch_factor=0.8), sr))
    labels.append(label)

# Convert to numpy arrays
X = np.array(audio_data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#normalizing features
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

#NN arhcitecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

y_pred = (model.predict(X_test) > 0.5).astype("int32")
f1 = f1_score(y_test, y_pred)
accuracy = np.mean(y_pred == y_test) * 100

print(f"F1 Score: {f1}")
print(f"Overall Model Accuracy: {accuracy:.2f}%")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Disorder'], yticklabels=['Normal', 'Disorder'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

model.save('voice_disorder_model.h5')

print(f"Classes loaded — Normal: {len(selected_normal_files)}, Disorder: {len(selected_disorder_files)}")
print(f"After augmentation — Normal: {len([l for l in labels if l == 0])}, Disorder: {len([l for l in labels if l == 1])}")
