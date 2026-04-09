"""
preprocess.py — Audio preprocessing utilities for VOCA.

Two distinct preprocessing stages:

1. organise_dataset()  — Reads the PVQD Excel metadata and renames/copies
   raw audio files into the {LABEL}_{GENDER}_{INDEX}.wav convention that
   train.py expects.

2. clean_audio()       — Optional signal-level cleaning: silence trimming,
   noise reduction, and volume normalization. Not used in the final pipeline
   (augmentation in train.py is sufficient), but available if raw recordings
   have significant background noise.

Usage:
    # Stage 1: rename raw PVQD files
    python algorithm/src/preprocess.py organise \
        --excel path/to/dataset.xlsx \
        --audio_in path/to/PVQD/Audio_Files/ \
        --audio_out path/to/renamed_audio_files/

    # Stage 2: clean a single file
    python src/preprocess.py clean \
        --input  path/to/input.wav \
        --output path/to/cleaned.wav
"""

import os
import shutil
import argparse
import numpy as np
import librosa
import soundfile as sf

# Optional deps — only needed for clean_audio()
try:
    import noisereduce as nr
    from pydub import AudioSegment
    _CLEAN_AVAILABLE = True
except ImportError:
    _CLEAN_AVAILABLE = False


# ── Stage 1 — Dataset organisation ───────────────────────────────────────────

def organise_dataset(excel_path, audio_in_dir, audio_out_dir, log_csv=None):
    """
    Rename and copy PVQD audio files using metadata from the dataset spreadsheet.

    Output naming convention: {LABEL}_{GENDER}_{INDEX}.wav
    Example: N_M_1.wav (Normal, Male, index 1), D1_F_5.wav (Spasmodic Dysphonia, Female, 5)

    Labels:
        N  = Normal
        D1 = Spasmodic Dysphonia
        D2 = Vocal Nodules
        D3 = Vocal Fold Paralysis

    Args:
        excel_path (str): Path to dataset.xlsx with columns: file_id, disorder, gender, age.
        audio_in_dir (str): Directory containing original PVQD audio files.
        audio_out_dir (str): Destination directory for renamed files.
        log_csv (str, optional): If given, saves a rename log CSV here.

    Returns:
        int: Number of files successfully renamed and copied.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for organise_dataset. Run: pip install pandas openpyxl")

    df = pd.read_excel(excel_path)
    os.makedirs(audio_out_dir, exist_ok=True)

    counters = {f"{label}_{gender}": 1 for label in ["N", "D1", "D2", "D3"] for gender in ["M", "F"]}
    renamed = []

    for _, row in df.iterrows():
        file_id = str(row["file_id"])
        label   = str(row["disorder"])
        gender  = str(row["gender"])
        age     = row.get("age", "")

        # Find the matching file by partial name match (PVQD filenames vary in format)
        matches = [f for f in os.listdir(audio_in_dir) if file_id.lower() in f.lower()]
        if not matches:
            continue

        key          = f"{label}_{gender}"
        new_fname    = f"{key}_{counters.get(key, 1)}.wav"
        src          = os.path.join(audio_in_dir, matches[0])
        dst          = os.path.join(audio_out_dir, new_fname)

        shutil.copy2(src, dst)
        counters[key] = counters.get(key, 1) + 1
        renamed.append({"original": matches[0], "renamed": new_fname, "age": age, "gender": gender, "disorder": label})

    print(f"Renamed and copied {len(renamed)} files → {audio_out_dir}")

    if log_csv and renamed:
        pd.DataFrame(renamed).to_csv(log_csv, index=False)
        print(f"Rename log saved → {log_csv}")

    return len(renamed)


# ── Stage 2 — Signal cleaning ─────────────────────────────────────────────────

def clean_audio(input_path, output_path, top_db=20):
    """
    Clean a single audio file: trim silence, reduce noise, normalise volume.

    This was explored early in the project but not used in the final pipeline —
    the augmentation strategy in train.py was sufficient for generalization, and
    adding noise reduction on top of augmentation felt like it was cleaning away
    useful signal variation. Kept here for completeness.

    Requires: noisereduce, pydub

    Args:
        input_path (str): Path to raw audio file.
        output_path (str): Path to save the cleaned file.
        top_db (float): Silence trim threshold in dB (higher = more aggressive).

    Returns:
        tuple: (y_normalized, sr) — cleaned signal and sample rate.

    Raises:
        ImportError: If noisereduce or pydub are not installed.
    """
    if not _CLEAN_AVAILABLE:
        raise ImportError(
            "clean_audio requires noisereduce and pydub.\n"
            "Install with: pip install noisereduce pydub"
        )

    y, sr = librosa.load(input_path, sr=16000)

    # Trim leading/trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)

    # Reduce stationary background noise
    y_denoised = nr.reduce_noise(y=y_trimmed, sr=sr)

    # Normalise volume to 0 dBFS using pydub
    seg = AudioSegment(
        y_denoised.tobytes(),
        frame_rate=sr,
        sample_width=y_denoised.dtype.itemsize,
        channels=1,
    )
    seg_norm = seg.apply_gain(-seg.dBFS)
    seg_norm.export(output_path, format="wav")

    y_out, _ = librosa.load(output_path, sr=sr)
    return y_out, sr


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VOCA audio preprocessing utilities")
    sub = parser.add_subparsers(dest="command")

    # organise subcommand
    p_org = sub.add_parser("organise", help="Rename PVQD files using dataset metadata")
    p_org.add_argument("--excel",     required=True, help="Path to dataset.xlsx")
    p_org.add_argument("--audio_in",  required=True, help="Directory with raw PVQD audio files")
    p_org.add_argument("--audio_out", required=True, help="Destination directory for renamed files")
    p_org.add_argument("--log_csv",   default=None,  help="Optional path to save rename log CSV")

    # clean subcommand
    p_clean = sub.add_parser("clean", help="Clean a single audio file")
    p_clean.add_argument("--input",  required=True, help="Input audio file")
    p_clean.add_argument("--output", required=True, help="Output cleaned file")
    p_clean.add_argument("--top_db", type=float, default=20, help="Silence trim threshold (dB)")

    args = parser.parse_args()

    if args.command == "organise":
        organise_dataset(args.excel, args.audio_in, args.audio_out, log_csv=args.log_csv)

    elif args.command == "clean":
        y, sr = clean_audio(args.input, args.output, top_db=args.top_db)
        print(f"Cleaned audio saved → {args.output}  ({len(y)/sr:.2f}s @ {sr} Hz)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
