# Optional audio cleaning — silence trim, noise reduction, volume normalisation.
# Explored but not used in the final pipeline. See src/preprocess.py clean_audio().

import numpy as np
import librosa
import noisereduce as nr
from pydub import AudioSegment
import soundfile as sf

# Function to remove silence, reduce noise, and normalize volume
def preprocess_audio(audio_file, output_file):

    y, sr = librosa.load(audio_file, sr=16000)
    y, _ = librosa.effects.trim(y, top_db=20)  # Remove Silence - adjust top_db for silence sensitivity
    y_denoised = nr.reduce_noise(y=y, sr=sr)  # Noise Reduction - reduces background noise
    
    # Volume Normalization - Converting to AudioSegment for normalization with pydub
    y_audio_segment = AudioSegment(
        y_denoised.tobytes(),
        frame_rate=sr,
        sample_width=y_denoised.dtype.itemsize,
        channels=1
    )
    normalized_audio = y_audio_segment.apply_gain(-y_audio_segment.dBFS)  # Normalize to 0 dBFS
    
    normalized_audio.export(output_file, format="wav")
    y_normalized, _ = librosa.load(output_file, sr=sr)
    return y_normalized, sr

audio_file = 'path/to/input_audio.wav'
output_file = 'path/to/processed_audio.wav'
processed_audio, sr = preprocess_audio(audio_file, output_file)
