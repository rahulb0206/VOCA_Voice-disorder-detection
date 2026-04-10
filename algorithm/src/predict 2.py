"""
predict.py — Run inference on a single audio file using the trained XGBoost model.

Loads the saved model and normalization parameters, extracts Mel-Spectrogram
features from the input audio, and returns a prediction with confidence scores
and key acoustic measurements.

Usage:
    python algorithm/src/predict.py --audio path/to/recording.wav

Expected model artifacts:
    outputs/models/voice_disorder_model.json
    outputs/models/mean_std_values.npy
"""

import os
import argparse
import numpy as np
import librosa
import cv2
import xgboost as xgb

DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "models", "voice_disorder_model.json")
DEFAULT_NORM  = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "models", "mean_std_values.npy")


class VoiceDisorderPredictor:
    """
    Inference wrapper for the trained XGBoost voice disorder classifier.

    Loads the model once at init and exposes a single predict() method
    that handles all preprocessing internally.

    Args:
        model_path (str): Path to the saved XGBoost model (.json).
        mean_std_path (str): Path to the saved normalization array (.npy).

    Example:
        >>> predictor = VoiceDisorderPredictor(model_path, norm_path)
        >>> result = predictor.predict("patient_recording.wav")
        >>> print(result["prediction"], result["probability_disorder"])
    """

    def __init__(self, model_path=DEFAULT_MODEL, mean_std_path=DEFAULT_NORM):
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        norm = np.load(mean_std_path)
        self.mean, self.std = norm[0], norm[1]

    @staticmethod
    def extract_mel_spectrogram(y, sr=16000, n_mels=128, fixed_shape=(128, 100)):
        """
        Extract a fixed-size flattened Mel-Spectrogram.

        Args:
            y (np.ndarray): Audio time-series.
            sr (int): Sample rate.
            n_mels (int): Number of Mel frequency bands.
            fixed_shape (tuple): (width, height) to resize spectrogram to.

        Returns:
            np.ndarray: 1D feature vector of length 12800.
        """
        mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return cv2.resize(mel_db, fixed_shape, interpolation=cv2.INTER_AREA).flatten()

    @staticmethod
    def extract_acoustic_features(y, sr):
        """
        Extract three interpretable acoustic measurements for the result payload.

        These aren't used as model inputs — just returned alongside the prediction
        to give clinicians additional context.

        Args:
            y (np.ndarray): Audio time-series.
            sr (int): Sample rate.

        Returns:
            dict: fundamental_frequency (Hz), tone (spectral centroid Hz), volume (RMS).
        """
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch = float(np.mean(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0

        tone   = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        volume = float(np.mean(librosa.feature.rms(y=y)))

        return {
            "fundamental_frequency": round(pitch, 2),
            "tone":   round(tone,   2),
            "volume": round(volume, 4),
        }

    def predict(self, file_path):
        """
        Predict whether a voice recording indicates a disorder.

        Args:
            file_path (str): Path to the audio file (.wav, .mp3, .ogg, .m4a).

        Returns:
            dict: {
                "prediction": "Normal" | "Disorder",
                "probability_normal": float (0–100),
                "probability_disorder": float (0–100),
                "fundamental_frequency": float (Hz),
                "tone": float (Hz),
                "volume": float
            }

        Raises:
            ValueError: If the audio file cannot be loaded or processed.

        Example:
            >>> result = predictor.predict("voice_sample.wav")
            >>> print(f"{result['prediction']} — {result['probability_disorder']}% disorder probability")
        """
        try:
            y, sr = librosa.load(file_path, sr=16000)
        except Exception as e:
            raise ValueError(f"Could not load audio file '{file_path}': {e}")

        features = self.extract_mel_spectrogram(y, sr)
        features_norm = (features - self.mean) / (self.std + 1e-8)

        prob_disorder = float(self.model.predict(xgb.DMatrix(np.array([features_norm])))[0])
        prob_normal   = 1.0 - prob_disorder
        label         = "Disorder" if prob_disorder > 0.5 else "Normal"

        acoustic = self.extract_acoustic_features(y, sr)

        return {
            "prediction":            label,
            "probability_normal":    round(prob_normal   * 100, 2),
            "probability_disorder":  round(prob_disorder * 100, 2),
            **acoustic,
        }


def main():
    parser = argparse.ArgumentParser(description="Run VOCA voice disorder inference")
    parser.add_argument("--audio",      required=True,        help="Path to audio file")
    parser.add_argument("--model",      default=DEFAULT_MODEL, help="Path to model .json")
    parser.add_argument("--norm",       default=DEFAULT_NORM,  help="Path to mean_std .npy")
    args = parser.parse_args()

    predictor = VoiceDisorderPredictor(model_path=args.model, mean_std_path=args.norm)

    try:
        result = predictor.predict(args.audio)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print("\n=== VOCA Voice Disorder Prediction ===")
    print(f"  Prediction          : {result['prediction']}")
    print(f"  P(Normal)           : {result['probability_normal']:.1f}%")
    print(f"  P(Disorder)         : {result['probability_disorder']:.1f}%")
    print(f"  Fundamental Freq    : {result['fundamental_frequency']} Hz")
    print(f"  Spectral Centroid   : {result['tone']} Hz")
    print(f"  RMS Volume          : {result['volume']}")


if __name__ == "__main__":
    main()
