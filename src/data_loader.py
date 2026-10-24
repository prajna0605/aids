"""
ECG Insight AI - Data Loader Module
Handles loading MIT-BIH Arrhythmia Database records from PhysioNet,
caching locally, parsing clinical annotations according to AAMI EC57 standards,
loading user-uploaded custom ECG files, and generating synthetic ECG records.
"""

import os
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import wfdb

# MIT-BIH Arrhythmia Database Curated Records for Demo & Evaluation
PRESET_RECORDS = {
    "100": {
        "title": "Record 100 — Baseline Normal Sinus Rhythm",
        "description": "Predominantly normal sinus rhythm with very few isolated ectopic beats. Ideal as a clean baseline.",
        "primary_arrhythmia": "Normal Sinus Rhythm",
        "recommended_lead": 0,
    },
    "106": {
        "title": "Record 106 — Frequent PVCs (Ventricular Ectopy)",
        "description": "Abundant Premature Ventricular Contractions (PVCs) appearing in isolated and couplet patterns.",
        "primary_arrhythmia": "Premature Ventricular Contractions (PVC)",
        "recommended_lead": 0,
    },
    "119": {
        "title": "Record 119 — Ventricular Bigeminy & Paired PVCs",
        "description": "Strong alternating pattern of normal beats and PVCs (bigeminy) providing high-yield anomaly signals.",
        "primary_arrhythmia": "Ventricular Bigeminy",
        "recommended_lead": 0,
    },
    "200": {
        "title": "Record 200 — Complex Ventricular Arrhythmias",
        "description": "Rich variety of Ventricular Tachycardia runs, couplets, and polymorphic ectopic beats.",
        "primary_arrhythmia": "Ventricular Tachycardia / Couplets",
        "recommended_lead": 0,
    },
    "208": {
        "title": "Record 208 — Frequent PVCs & Ventricular Couplets",
        "description": "Extremely frequent ventricular ectopy with varying coupling intervals and morphology changes.",
        "primary_arrhythmia": "Couplets & Bigeminy",
        "recommended_lead": 0,
    },
    "209": {
        "title": "Record 209 — Atrial Premature Contractions (PAC)",
        "description": "Features Supraventricular / Atrial premature beats (PACs) with slight timing variations.",
        "primary_arrhythmia": "Atrial Premature Beats (PAC)",
        "recommended_lead": 0,
    },
}

# AAMI EC57 Heartbeat Classification Mapping
# Non-ectopic (Normal-like)
NORMAL_BEAT_SYMBOLS = {"N", "L", "R", "e", "j"}
# Ventricular Ectopic Beats (VEB)
VENTRICULAR_BEAT_SYMBOLS = {"V", "E", "!", "r"}
# Supraventricular Ectopic Beats (SVEB)
SUPRAVENTRICULAR_BEAT_SYMBOLS = {"A", "a", "J", "S"}
# Fusion or Paced or Unknown
OTHER_ANOMALOUS_SYMBOLS = {"F", "/", "f", "Q", "?"}

# Combined set of all clinically anomalous beat symbols
ALL_ANOMALOUS_SYMBOLS = VENTRICULAR_BEAT_SYMBOLS | SUPRAVENTRICULAR_BEAT_SYMBOLS | OTHER_ANOMALOUS_SYMBOLS

# Non-beat annotation symbols (rhythm changes, artifact, measurement markers)
NON_BEAT_SYMBOLS = {"+", "~", "|", "x", "[", "]", '"', "s", "t", "p", "D", "T"}

ANNOTATION_DESCRIPTIONS = {
    "N": "Normal Beat",
    "L": "Left Bundle Branch Block Beat",
    "R": "Right Bundle Branch Block Beat",
    "e": "Atrial Escape Beat",
    "j": "Nodal (Junctional) Escape Beat",
    "V": "Premature Ventricular Contraction (PVC)",
    "E": "Ventricular Escape Beat",
    "A": "Atrial Premature Beat (PAC)",
    "a": "Aberrated Atrial Premature Beat",
    "J": "Nodal (Junctional) Premature Beat",
    "S": "Supraventricular Premature Beat",
    "F": "Fusion of Ventricular and Normal Beat",
    "/": "Paced Beat",
    "f": "Fusion of Paced and Normal Beat",
    "Q": "Unclassifiable Beat",
    "+": "Rhythm Change Note",
    "~": "Signal Quality / Noise Marker",
}


def get_cache_dir() -> Path:
    """Returns local cache directory path for downloaded records."""
    cache_path = Path(__file__).resolve().parent.parent / "data" / "mitdb"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def load_mitdb_record(
    record_name: str,
    channel: int = 0,
    max_seconds: Optional[int] = 120,
    pn_dir: str = "mitdb",
) -> Dict[str, Any]:
    """
    Downloads or loads an MIT-BIH Arrhythmia Database record using WFDB.
    Caches downloaded records locally to accelerate subsequent loads.

    Args:
        record_name: MIT-BIH record ID (e.g., '100', '106', '119')
        channel: ECG channel/lead index (default: 0)
        max_seconds: Number of seconds to load (None for entire record, default: 120s)
        pn_dir: PhysioNet database directory

    Returns:
        Dictionary containing raw signal, sampling frequency, lead name, and beat annotations.
    """
    cache_dir = get_cache_dir()
    local_rec_path = cache_dir / record_name

    # Check if local record files exist (.dat and .hea)
    dat_file = cache_dir / f"{record_name}.dat"
    hea_file = cache_dir / f"{record_name}.hea"
    atr_file = cache_dir / f"{record_name}.atr"

    has_local = dat_file.exists() and hea_file.exists() and atr_file.exists()

    # Determine max sample count if max_seconds provided
    # MIT-BIH sampling rate is 360 Hz
    fs_est = 360
    sampto = int(max_seconds * fs_est) if max_seconds else None

    # Fast npz cache check: check exact match or any available cached slice
    npz_cache_dir = Path(__file__).resolve().parent.parent / "data" / "cache_npz"
    npz_cache_dir.mkdir(parents=True, exist_ok=True)
    npz_file = npz_cache_dir / f"{record_name}_{max_seconds}_{channel}.npz"

    # Search candidates: exact match first, then any cached file for this record
    candidate_files = []
    if npz_file.exists():
        candidate_files.append(npz_file)
    else:
        candidate_files.extend(list(npz_cache_dir.glob(f"{record_name}_*_{channel}.npz")))

    for cand in candidate_files:
        try:
            cached = np.load(str(cand), allow_pickle=True)
            sig = cached["signal"]
            ann_s = cached["ann_samples"]
            ann_sym = cached["ann_symbols"]
            ann_lab = cached["ann_labels"]
            ann_desc = cached["ann_descriptions"]

            if sampto and len(sig) > sampto:
                sig = sig[:sampto]
                mask = ann_s < sampto
                ann_s = ann_s[mask]
                ann_sym = ann_sym[mask]
                ann_lab = ann_lab[mask]
                ann_desc = ann_desc[mask]

            return {
                "record_name": str(cached["record_name"]),
                "signal": sig,
                "fs": int(cached["fs"]),
                "channel": int(cached["channel"]),
                "lead_name": str(cached["lead_name"]),
                "total_samples": len(sig),
                "duration_sec": len(sig) / int(cached["fs"]),
                "ann_samples": ann_s,
                "ann_symbols": ann_sym,
                "ann_labels": ann_lab,
                "ann_descriptions": ann_desc.tolist(),
                "is_synthetic": False,
            }
        except Exception:
            continue

    if has_local:
        try:
            record = wfdb.rdrecord(str(local_rec_path), sampto=sampto)
            annotation = wfdb.rdann(str(local_rec_path), "atr", sampto=sampto)
        except Exception:
            record = wfdb.rdrecord(record_name, pn_dir=pn_dir, sampto=sampto)
            try:
                annotation = wfdb.rdann(record_name, "atr", pn_dir=pn_dir, sampto=sampto)
            except Exception:
                annotation = None
    else:
        # Stream on-demand slice directly from PhysioNet via HTTP range request
        record = wfdb.rdrecord(record_name, pn_dir=pn_dir, sampto=sampto)
        try:
            annotation = wfdb.rdann(record_name, "atr", pn_dir=pn_dir, sampto=sampto)
        except Exception:
            annotation = None

    fs = int(record.fs)
    signal = record.p_signal[:, channel].astype(np.float64)
    lead_name = record.sig_name[channel] if record.sig_name else f"Lead {channel}"

    # Extract & decode annotations
    ann_samples = []
    ann_symbols = []
    ann_labels = []  # 0 for normal, 1 for anomaly
    ann_descriptions = []

    if annotation is not None:
        for s_idx, sym in zip(annotation.sample, annotation.symbol):
            # Only consider valid samples within range and valid heartbeat annotations
            if s_idx < len(signal) and sym not in NON_BEAT_SYMBOLS:
                ann_samples.append(int(s_idx))
                ann_symbols.append(str(sym))
                is_anomaly = 1 if sym in ALL_ANOMALOUS_SYMBOLS else 0
                ann_labels.append(is_anomaly)
                desc = ANNOTATION_DESCRIPTIONS.get(sym, f"Beat type: {sym}")
                ann_descriptions.append(desc)

    result = {
        "record_name": record_name,
        "signal": signal,
        "fs": fs,
        "channel": channel,
        "lead_name": lead_name,
        "total_samples": len(signal),
        "duration_sec": len(signal) / fs,
        "ann_samples": np.array(ann_samples, dtype=int),
        "ann_symbols": np.array(ann_symbols, dtype=object),
        "ann_labels": np.array(ann_labels, dtype=int),
        "ann_descriptions": ann_descriptions,
        "is_synthetic": False,
    }

    try:
        np.savez_compressed(
            str(npz_file),
            record_name=record_name,
            signal=signal,
            fs=fs,
            channel=channel,
            lead_name=lead_name,
            total_samples=len(signal),
            duration_sec=len(signal) / fs,
            ann_samples=result["ann_samples"],
            ann_symbols=result["ann_symbols"],
            ann_labels=result["ann_labels"],
            ann_descriptions=np.array(ann_descriptions, dtype=object),
        )
    except Exception:
        pass

    return result


def generate_synthetic_ecg(
    duration_sec: int = 60,
    fs: int = 360,
    heart_rate: float = 75.0,
    noise_level: float = 0.04,
    anomaly_prob: float = 0.18,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Generates a realistic synthetic ECG time series with customizable Premature
    Ventricular Contractions (PVCs), baseline wander, and high-frequency noise.
    Provides offline demo support and reproducible benchmark scenarios.

    Args:
        duration_sec: Length of generated recording in seconds
        fs: Sampling rate in Hz
        heart_rate: Base beats per minute
        noise_level: Amplitude of Gaussian noise and 50Hz hum
        anomaly_prob: Probability of injecting an ectopic beat (PVC)
        random_state: Random seed for repeatability

    Returns:
        Structured dataset dictionary compatible with MIT-BIH format.
    """
    np.random.seed(random_state)
    total_samples = int(duration_sec * fs)
    time = np.linspace(0, duration_sec, total_samples, endpoint=False)
    signal = np.zeros(total_samples, dtype=np.float64)

    # Base RR interval in seconds
    mean_rr = 60.0 / heart_rate

    # Generate sequence of beat timestamps
    current_time = 0.6  # initial offset
    beat_times = []
    beat_types = []  # 'N' or 'V'

    while current_time < duration_sec - 0.5:
        is_pvc = np.random.rand() < anomaly_prob and len(beat_times) > 0
        if is_pvc:
            # PVC is premature: shorter RR interval (60-70% of normal), followed by compensatory pause
            rr = mean_rr * np.random.uniform(0.60, 0.72)
            current_time += rr
            if current_time >= duration_sec - 0.5:
                break
            beat_times.append(current_time)
            beat_types.append("V")
            # Compensatory pause for next beat
            current_time += mean_rr * np.random.uniform(1.25, 1.40)
        else:
            # Normal beat with slight physiological HRV jitter
            rr = mean_rr * np.random.uniform(0.95, 1.05)
            current_time += rr
            if current_time >= duration_sec - 0.5:
                break
            beat_times.append(current_time)
            beat_types.append("N")

    ann_samples = []
    ann_symbols = []
    ann_labels = []
    ann_descriptions = []

    # Synthesize waveforms for each beat
    for b_time, b_type in zip(beat_times, beat_types):
        r_idx = int(b_time * fs)
        if r_idx >= total_samples:
            continue

        ann_samples.append(r_idx)
        ann_symbols.append(b_type)
        is_ano = 1 if b_type == "V" else 0
        ann_labels.append(is_ano)
        ann_descriptions.append(
            "Premature Ventricular Contraction (Synthetic PVC)" if is_ano else "Normal Sinus Beat (Synthetic)"
        )

        if b_type == "N":
            # Normal P-QRS-T complex
            # P wave: small positive bump ~160ms before R
            p_center = b_time - 0.16
            signal += 0.20 * np.exp(-((time - p_center) ** 2) / (2 * (0.025**2)))

            # Q wave: small negative deflection ~40ms before R
            q_center = b_time - 0.04
            signal -= 0.18 * np.exp(-((time - q_center) ** 2) / (2 * (0.012**2)))

            # R wave: tall sharp positive peak
            signal += 1.25 * np.exp(-((time - b_time) ** 2) / (2 * (0.015**2)))

            # S wave: negative deflection ~40ms after R
            s_center = b_time + 0.04
            signal -= 0.35 * np.exp(-((time - s_center) ** 2) / (2 * (0.014**2)))

            # T wave: broader positive bump ~220ms after R
            t_center = b_time + 0.22
            signal += 0.35 * np.exp(-((time - t_center) ** 2) / (2 * (0.045**2)))

        else:
            # Ventricular Ectopic Beat (PVC):
            # No preceding P wave, wide bizarre biphasic/inverted QRS complex, discordant inverted T wave
            # Widened notched main deflection
            signal += 1.60 * np.exp(-((time - b_time) ** 2) / (2 * (0.042**2)))
            # Deep wide S deflection
            s_center = b_time + 0.08
            signal -= 1.10 * np.exp(-((time - s_center) ** 2) / (2 * (0.038**2)))
            # Discordant inverted wide T wave
            t_center = b_time + 0.26
            signal -= 0.45 * np.exp(-((time - t_center) ** 2) / (2 * (0.065**2)))

    # Add realistic baseline wander (low frequency breathing drift 0.25 Hz + 0.1 Hz)
    baseline_wander = 0.25 * np.sin(2 * np.pi * 0.22 * time) + 0.12 * np.cos(2 * np.pi * 0.08 * time)

    # Add 50 Hz powerline interference
    powerline = (noise_level * 0.6) * np.sin(2 * np.pi * 50.0 * time)

    # Add physiological high-frequency electromyographic (EMG) noise
    emg_noise = np.random.normal(0, noise_level, total_samples)

    final_signal = signal + baseline_wander + powerline + emg_noise

    return {
        "record_name": "SYNTH-75",
        "signal": final_signal,
        "fs": fs,
        "channel": 0,
        "lead_name": "Synthetic Lead II",
        "total_samples": total_samples,
        "duration_sec": duration_sec,
        "ann_samples": np.array(ann_samples, dtype=int),
        "ann_symbols": np.array(ann_symbols, dtype=object),
        "ann_labels": np.array(ann_labels, dtype=int),
        "ann_descriptions": ann_descriptions,
        "is_synthetic": True,
    }


def get_custom_file_preview(file_bytes: bytes) -> Dict[str, Any]:
    """
    Scans uploaded file headers and first rows to return metadata,
    detected delimiter, column names, and preview dataframe.
    """
    try:
        # Read small sample for speed and low memory
        preview_buf = io.BytesIO(file_bytes[: min(len(file_bytes), 256 * 1024)])
        df_preview = pd.read_csv(preview_buf, sep=None, engine="python", nrows=20)
        numeric_cols = list(df_preview.select_dtypes(include=[np.number]).columns)
        all_cols = list(df_preview.columns)
        return {
            "success": True,
            "preview_df": df_preview.head(5),
            "numeric_columns": numeric_cols,
            "all_columns": all_cols,
            "file_size_bytes": len(file_bytes),
            "file_size_mb": len(file_bytes) / (1024 * 1024),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_size_bytes": len(file_bytes),
            "file_size_mb": len(file_bytes) / (1024 * 1024),
        }


def load_custom_ecg_csv(
    file_bytes: bytes,
    fs: int = 360,
    filename: str = "custom.csv",
    selected_column: Optional[str] = None,
    start_second: float = 0.0,
    duration_allowance_sec: Optional[float] = 120.0,
    max_samples_allowance: Optional[int] = 500_000,
    downsample_factor: int = 1,
) -> Dict[str, Any]:
    """
    Parses a user-uploaded CSV/TXT ECG signal with configurable Data Allowance controls:
    - Time-window allowance slicing [start_second, start_second + duration_allowance_sec]
    - Max samples allowance constraint (protects UI from browser memory lag)
    - Decimation/Downsampling factor for high-frequency or long Holter monitors
    - Column lead selection

    Returns structured dictionary with metadata on original vs allowed data slice.
    """
    # Load data
    df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")

    # Pick or validate column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("Uploaded file does not contain any numeric ECG voltage columns.")

    if selected_column and selected_column in df.columns:
        signal_col = selected_column
    else:
        signal_col = numeric_cols[0]

    raw_signal = df[signal_col].to_numpy(dtype=np.float64)

    # Drop NaNs or infinities
    valid_mask = np.isfinite(raw_signal)
    raw_signal = raw_signal[valid_mask]

    orig_total_samples = len(raw_signal)
    orig_total_duration_sec = orig_total_samples / fs

    # 1. Apply Time Window Allowance Slicing
    start_idx = int(start_second * fs)
    if start_idx >= orig_total_samples:
        start_idx = 0

    if duration_allowance_sec and duration_allowance_sec > 0:
        end_idx = start_idx + int(duration_allowance_sec * fs)
        signal_slice = raw_signal[start_idx:min(orig_total_samples, end_idx)]
    else:
        signal_slice = raw_signal[start_idx:]

    # 2. Apply Max Samples Allowance Safeguard
    if max_samples_allowance and len(signal_slice) > max_samples_allowance:
        signal_slice = signal_slice[:max_samples_allowance]

    # 3. Apply Decimation Allowance Factor if requested (>1)
    effective_fs = fs
    if downsample_factor > 1:
        signal_slice = signal_slice[::downsample_factor]
        effective_fs = max(10, fs // downsample_factor)

    final_samples = len(signal_slice)
    final_duration = final_samples / effective_fs
    memory_mb = (signal_slice.nbytes) / (1024 * 1024)

    return {
        "record_name": filename,
        "signal": signal_slice,
        "fs": effective_fs,
        "original_fs": fs,
        "channel": 0,
        "lead_name": str(signal_col),
        "total_samples": final_samples,
        "duration_sec": final_duration,
        "ann_samples": np.array([], dtype=int),
        "ann_symbols": np.array([], dtype=object),
        "ann_labels": np.array([], dtype=int),
        "ann_descriptions": [],
        "is_synthetic": False,
        "is_user_uploaded": True,
        # Data allowance analytics
        "allowance_meta": {
            "raw_file_size_mb": len(file_bytes) / (1024 * 1024),
            "orig_total_samples": orig_total_samples,
            "orig_total_duration_sec": orig_total_duration_sec,
            "processed_samples": final_samples,
            "processed_duration_sec": final_duration,
            "start_second": start_second,
            "duration_allowance_sec": duration_allowance_sec,
            "downsample_factor": downsample_factor,
            "memory_usage_mb": memory_mb,
            "available_columns": list(numeric_cols),
        },
    }
