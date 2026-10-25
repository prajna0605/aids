"""
ECG Insight AI - Data Loader Module
====================================
Unified data ingestion engine supporting:
1. PhysioNet MIT-BIH Arrhythmia Database (.dat, .hea, .atr) with local caching
2. AAMI EC57 Clinical Annotation standard decoding
3. User-uploaded custom ECG files (both continuous 1D recordings and 2D AAMI beat matrices)
4. Configurable Data Allowance (time windowing, sample limits, decimation)
5. Offline synthetic ECG generation with realistic P-QRS-T morphology & PVC injection
"""

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb

# ==============================================================================
# 1. CLINICAL PRESETS & AAMI EC57 CLASSIFICATION STANDARDS
# ==============================================================================

PRESET_RECORDS: Dict[str, Dict[str, Any]] = {
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

# AAMI EC57 Beat Classes
NORMAL_BEAT_SYMBOLS = {"N", "L", "R", "e", "j"}
VENTRICULAR_BEAT_SYMBOLS = {"V", "E", "!", "r"}
SUPRAVENTRICULAR_BEAT_SYMBOLS = {"A", "a", "J", "S"}
OTHER_ANOMALOUS_SYMBOLS = {"F", "/", "f", "Q", "?"}
ALL_ANOMALOUS_SYMBOLS = VENTRICULAR_BEAT_SYMBOLS | SUPRAVENTRICULAR_BEAT_SYMBOLS | OTHER_ANOMALOUS_SYMBOLS
NON_BEAT_SYMBOLS = {"+", "~", "|", "x", "[", "]", '"', "s", "t", "p", "D", "T"}

ANNOTATION_DESCRIPTIONS: Dict[str, str] = {
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

# Mapping for Kaggle/AAMI integer labels (0=N, 1=SVEB/A, 2=VEB/V, 3=F, 4=Q)
AAMI_LABEL_TO_SYMBOL = {0.0: "N", 1.0: "A", 2.0: "V", 3.0: "F", 4.0: "Q"}
AAMI_LABEL_TO_DESC = {
    0.0: "Normal sinus beat",
    1.0: "Premature atrial contraction (PAC)",
    2.0: "Premature ventricular contraction (PVC)",
    3.0: "Fusion beat",
    4.0: "Paced / unclassifiable beat",
}


# ==============================================================================
# 2. CACHING & RECORD PACKAGING HELPERS
# ==============================================================================

def get_cache_dir() -> Path:
    """Returns local cache directory path for downloaded raw MIT-BIH records."""
    cache_path = Path(__file__).resolve().parent.parent / "data" / "mitdb"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def get_npz_cache_dir() -> Path:
    """Returns directory path for compressed parsed NPZ records."""
    npz_dir = Path(__file__).resolve().parent.parent / "data" / "cache_npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    return npz_dir


def _create_record_dict(
    record_name: str,
    signal: np.ndarray,
    fs: int,
    lead_name: str,
    ann_samples: Optional[np.ndarray] = None,
    ann_symbols: Optional[np.ndarray] = None,
    ann_labels: Optional[np.ndarray] = None,
    ann_descriptions: Optional[List[str]] = None,
    is_synthetic: bool = False,
    is_user_uploaded: bool = False,
    original_fs: Optional[int] = None,
    channel: int = 0,
    allowance_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Unified record constructor ensuring consistent schema across all loaders."""
    total_samples = len(signal)
    duration_sec = total_samples / fs if fs > 0 else 0.0

    return {
        "record_name": str(record_name),
        "signal": signal,
        "fs": int(fs),
        "original_fs": int(original_fs) if original_fs is not None else int(fs),
        "channel": int(channel),
        "lead_name": str(lead_name),
        "total_samples": total_samples,
        "duration_sec": duration_sec,
        "ann_samples": ann_samples if ann_samples is not None else np.array([], dtype=int),
        "ann_symbols": ann_symbols if ann_symbols is not None else np.array([], dtype=object),
        "ann_labels": ann_labels if ann_labels is not None else np.array([], dtype=int),
        "ann_descriptions": ann_descriptions if ann_descriptions is not None else [],
        "is_synthetic": is_synthetic,
        "is_user_uploaded": is_user_uploaded,
        "allowance_meta": allowance_meta or {},
    }


# ==============================================================================
# 3. MIT-BIH DATABASE LOADER & ANNOTATION PARSER
# ==============================================================================

def _parse_wfdb_annotations(
    annotation: Any, max_len: int
) -> Tuple[List[int], List[str], List[int], List[str]]:
    """Decodes WFDB annotation records into AAMI beat categories."""
    ann_samples, ann_symbols, ann_labels, ann_desc = [], [], [], []
    if annotation is None:
        return ann_samples, ann_symbols, ann_labels, ann_desc

    for s_idx, sym in zip(annotation.sample, annotation.symbol):
        if s_idx < max_len and sym not in NON_BEAT_SYMBOLS:
            ann_samples.append(int(s_idx))
            ann_symbols.append(str(sym))
            is_anomaly = 1 if sym in ALL_ANOMALOUS_SYMBOLS else 0
            ann_labels.append(is_anomaly)
            ann_desc.append(ANNOTATION_DESCRIPTIONS.get(sym, f"Beat type: {sym}"))

    return ann_samples, ann_symbols, ann_labels, ann_desc


def load_mitdb_record(
    record_name: str,
    channel: int = 0,
    max_seconds: Optional[int] = 120,
    pn_dir: str = "mitdb",
) -> Dict[str, Any]:
    """
    Downloads or loads an MIT-BIH Arrhythmia Database record using WFDB.
    Accelerated with local compressed NPZ caching.

    Args:
        record_name: MIT-BIH record ID (e.g., '100', '106', '119')
        channel: ECG channel/lead index (default: 0)
        max_seconds: Number of seconds to load (None for full record, default: 120s)
        pn_dir: PhysioNet database directory

    Returns:
        Structured dataset dictionary compatible with pipeline & UI.
    """
    fs_est = 360
    sampto = int(max_seconds * fs_est) if max_seconds else None
    npz_dir = get_npz_cache_dir()
    npz_file = npz_dir / f"{record_name}_{max_seconds}_{channel}.npz"

    # 1. Fast Cache Check: Load pre-extracted slice from NPZ if available
    candidate_files = [npz_file] if npz_file.exists() else list(npz_dir.glob(f"{record_name}_*_{channel}.npz"))
    for cand in candidate_files:
        try:
            cached = np.load(str(cand), allow_pickle=True)
            sig = cached["signal"]
            ann_s = cached["ann_samples"]
            ann_sym = cached["ann_symbols"]
            ann_lab = cached["ann_labels"]
            ann_desc = cached["ann_descriptions"].tolist()

            if sampto and len(sig) > sampto:
                sig = sig[:sampto]
                mask = ann_s < sampto
                ann_s = ann_s[mask]
                ann_sym = ann_sym[mask]
                ann_lab = ann_lab[mask]
                ann_desc = [ann_desc[i] for i, m in enumerate(mask) if m]

            return _create_record_dict(
                record_name=cached["record_name"],
                signal=sig,
                fs=int(cached["fs"]),
                lead_name=str(cached["lead_name"]),
                ann_samples=ann_s,
                ann_symbols=ann_sym,
                ann_labels=ann_lab,
                ann_descriptions=ann_desc,
                channel=channel,
            )
        except Exception:
            continue

    # 2. WFDB Extraction: Read from local cache or stream via PhysioNet
    cache_dir = get_cache_dir()
    local_rec_path = cache_dir / record_name
    has_local = (cache_dir / f"{record_name}.dat").exists() and (cache_dir / f"{record_name}.hea").exists()

    rec_target = str(local_rec_path) if has_local else record_name
    rec_kwargs = {"sampto": sampto} if has_local else {"pn_dir": pn_dir, "sampto": sampto}

    try:
        record = wfdb.rdrecord(rec_target, **rec_kwargs)
        annotation = wfdb.rdann(rec_target, "atr", **rec_kwargs)
    except Exception:
        # Fallback to direct PhysioNet streaming if local file was corrupt
        record = wfdb.rdrecord(record_name, pn_dir=pn_dir, sampto=sampto)
        try:
            annotation = wfdb.rdann(record_name, "atr", pn_dir=pn_dir, sampto=sampto)
        except Exception:
            annotation = None

    fs = int(record.fs)
    signal = record.p_signal[:, channel].astype(np.float64)
    lead_name = record.sig_name[channel] if record.sig_name else f"Lead {channel}"

    # 3. Parse AAMI EC57 Annotations
    ann_samples, ann_symbols, ann_labels, ann_descriptions = _parse_wfdb_annotations(annotation, len(signal))

    # 4. Save to NPZ Cache for future ultra-fast loads
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
            ann_samples=np.array(ann_samples, dtype=int),
            ann_symbols=np.array(ann_symbols, dtype=object),
            ann_labels=np.array(ann_labels, dtype=int),
            ann_descriptions=np.array(ann_descriptions, dtype=object),
        )
    except Exception:
        pass

    return _create_record_dict(
        record_name=record_name,
        signal=signal,
        fs=fs,
        lead_name=lead_name,
        ann_samples=np.array(ann_samples, dtype=int),
        ann_symbols=np.array(ann_symbols, dtype=object),
        ann_labels=np.array(ann_labels, dtype=int),
        ann_descriptions=ann_descriptions,
        channel=channel,
    )


# ==============================================================================
# 4. SYNTHETIC ECG GENERATOR
# ==============================================================================

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
    """
    np.random.seed(random_state)
    total_samples = int(duration_sec * fs)
    time = np.linspace(0, duration_sec, total_samples, endpoint=False)
    signal = np.zeros(total_samples, dtype=np.float64)

    mean_rr = 60.0 / heart_rate
    current_time = 0.6
    beat_times, beat_types = [], []

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
            current_time += mean_rr * np.random.uniform(1.25, 1.40)
        else:
            rr = mean_rr * np.random.uniform(0.95, 1.05)
            current_time += rr
            if current_time >= duration_sec - 0.5:
                break
            beat_times.append(current_time)
            beat_types.append("N")

    ann_samples, ann_symbols, ann_labels, ann_descriptions = [], [], [], []

    def _wave(t_center: float, amp: float, width: float) -> np.ndarray:
        return amp * np.exp(-((time - t_center) ** 2) / (2 * (width**2)))

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
            signal += _wave(b_time - 0.16, 0.20, 0.025)   # P wave (atrial depolarization)
            signal += _wave(b_time - 0.04, -0.18, 0.012)  # Q wave (septal deflection)
            signal += _wave(b_time, 1.25, 0.015)          # R peak (ventricular depolarization)
            signal += _wave(b_time + 0.04, -0.35, 0.014)  # S wave
            signal += _wave(b_time + 0.22, 0.35, 0.045)   # T wave (ventricular repolarization)
        else:
            # Ventricular Ectopic Beat (PVC): wide QRS spike, deep S wave, inverted T wave
            signal += _wave(b_time, 1.60, 0.042)          # Wide abnormal R spike
            signal += _wave(b_time + 0.08, -1.10, 0.038)  # Deep wide S deflection
            signal += _wave(b_time + 0.26, -0.45, 0.065)  # Inverted T wave


    # Baseline wander (respiratory drift) & high-frequency interference
    baseline_wander = 0.25 * np.sin(2 * np.pi * 0.22 * time) + 0.12 * np.cos(2 * np.pi * 0.08 * time)
    powerline = (noise_level * 0.6) * np.sin(2 * np.pi * 50.0 * time)
    emg_noise = np.random.normal(0, noise_level, total_samples)

    final_signal = signal + baseline_wander + powerline + emg_noise

    return _create_record_dict(
        record_name="SYNTH-75",
        signal=final_signal,
        fs=fs,
        lead_name="Synthetic Lead II",
        ann_samples=np.array(ann_samples, dtype=int),
        ann_symbols=np.array(ann_symbols, dtype=object),
        ann_labels=np.array(ann_labels, dtype=int),
        ann_descriptions=ann_descriptions,
        is_synthetic=True,
    )


# ==============================================================================
# 5. CUSTOM CSV INGESTION & AAMI MATRIX RECONSTRUCTION
# ==============================================================================

def _read_csv_smart(buf: io.BytesIO, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, bool, bool]:
    """
    Intelligently inspects uploaded CSV bytes to detect:
    1. Headerless numeric data vs text header row
    2. AAMI pre-segmented heartbeat matrix (e.g. 187/188 columns) vs continuous multi-lead signal
    Returns (dataframe, is_heartbeat_matrix, is_numeric_header).
    """
    raw_head = buf.read(64 * 1024)
    buf.seek(0)
    first_line = raw_head.split(b"\n")[0].decode("utf-8", errors="ignore").strip()

    delimiter = ","
    for d in [",", ";", "\t", " "]:
        if d in first_line:
            delimiter = d
            break

    tokens = [t.strip() for t in first_line.split(delimiter) if t.strip()]
    is_numeric_header = False
    if len(tokens) > 0:
        try:
            [float(t) for t in tokens]
            is_numeric_header = True
        except ValueError:
            is_numeric_header = False

    header_arg = None if is_numeric_header else "infer"
    try:
        df = pd.read_csv(buf, sep=delimiter, header=header_arg, nrows=nrows)
    except Exception:
        buf.seek(0)
        df = pd.read_csv(buf, sep=None, header=header_arg, nrows=nrows, engine="python")

    num_cols = df.shape[1]
    is_heartbeat_matrix = (num_cols in (187, 188, 140, 141)) or (num_cols >= 50 and is_numeric_header)

    if is_numeric_header:
        if is_heartbeat_matrix:
            last_col = df.iloc[:, -1]
            unique_last = np.unique(last_col.dropna())
            has_lbl = (num_cols in (188, 141)) or (
                len(unique_last) <= 8 and np.all(np.isin(unique_last, [0.0, 1.0, 2.0, 3.0, 4.0, 0, 1, 2, 3, 4]))
            )
            n_samples = (num_cols - 1) if has_lbl else num_cols
            clean_cols = [f"Sample_{i}" for i in range(n_samples)]
            if has_lbl:
                clean_cols.append("Class_Label")
            df.columns = clean_cols
        else:
            df.columns = [f"Lead_{i+1}" for i in range(num_cols)]

    return df, is_heartbeat_matrix, is_numeric_header


def get_custom_file_preview(file_bytes: bytes) -> Dict[str, Any]:
    """
    Scans uploaded file headers and first rows to return metadata,
    detected delimiter, column names, and preview dataframe.
    """
    try:
        preview_buf = io.BytesIO(file_bytes[: min(len(file_bytes), 256 * 1024)])
        df_preview, is_beat_matrix, _ = _read_csv_smart(preview_buf, nrows=20)

        if is_beat_matrix:
            numeric_cols = ["Continuous Rhythm Strip (Reconstructed from Beats)", "Sample_0", "Sample_1"]
            all_cols = numeric_cols
        else:
            numeric_cols = list(df_preview.select_dtypes(include=[np.number]).columns)
            all_cols = list(df_preview.columns)

        return {
            "success": True,
            "preview_df": df_preview.head(5),
            "numeric_columns": numeric_cols,
            "all_columns": all_cols,
            "is_heartbeat_matrix": is_beat_matrix,
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


def _reconstruct_rhythm_from_beats(
    df: pd.DataFrame,
    fs: int,
    filename: str,
    file_bytes_len: int,
    start_second: float = 0.0,
    duration_allowance_sec: Optional[float] = 120.0,
    downsample_factor: int = 1,
) -> Dict[str, Any]:
    """
    Vectorized reconstruction of a continuous ECG rhythm strip from pre-segmented AAMI heartbeat rows
    (e.g., mitbih_test.csv). Accurately extracts ground-truth class labels from column 188.
    """
    num_cols = df.shape[1]
    last_col = df.iloc[:, -1]
    unique_last = np.unique(last_col.dropna())
    has_labels = (num_cols in (188, 141)) or (
        len(unique_last) <= 8 and np.all(np.isin(unique_last, [0.0, 1.0, 2.0, 3.0, 4.0, 0, 1, 2, 3, 4]))
    )
    samples_per_beat = (num_cols - 1) if has_labels else num_cols

    # Natural resting cardiac stride ~72-75 BPM (~0.80 seconds per beat)
    target_stride = max(samples_per_beat + 20, int(0.80 * fs))
    spacing = target_stride - samples_per_beat
    beat_stride = target_stride

    target_samples = (
        int(duration_allowance_sec * fs)
        if (duration_allowance_sec and duration_allowance_sec > 0)
        else (len(df) * beat_stride)
    )
    # 1. Clean incomplete/truncated rows and extract beat voltage matrix
    df_clean = df.dropna(subset=df.columns[:samples_per_beat])
    if len(df_clean) == 0:
        df_clean = df.fillna(0.0)

    beats_needed = min(len(df_clean), max(1, target_samples // beat_stride))
    start_beat = min(max(0, len(df_clean) - 1), int(start_second * fs / beat_stride))
    end_beat = min(len(df_clean), start_beat + beats_needed)

    beats_data = df_clean.iloc[start_beat:end_beat, :samples_per_beat].to_numpy(dtype=np.float64)
    beats_data = np.nan_to_num(beats_data, nan=0.0)
    n_beats = len(beats_data)
    raw_labels = df_clean.iloc[start_beat:end_beat, samples_per_beat].to_numpy(dtype=np.float64) if has_labels else None

    # 2. High-performance vectorized waveform assembly
    # Append flat resting baseline (equal to last beat sample) to each beat, then flatten
    padding = np.repeat(beats_data[:, -1:], spacing, axis=1)
    signal_slice = np.nan_to_num(np.hstack([beats_data, padding]).ravel(), nan=0.0)

    # 3. Vectorized R-peak and annotation calculation
    r_peaks_relative = np.argmax(beats_data, axis=1)
    beat_offsets = np.arange(n_beats) * beat_stride
    ann_samples = beat_offsets + r_peaks_relative

    ann_symbols = []
    ann_labels = []
    ann_descriptions = []

    if has_labels and raw_labels is not None:
        # Check dataset labeling convention:
        # Kaggle ECG5000 / UCR: 1.0=Normal, 0.0=Abnormal
        # MIT-BIH AAMI standard: 0.0=Normal, >0=Abnormal
        is_ecg5000 = (num_cols == 141) and set(np.unique(raw_labels)).issubset({0.0, 1.0, 0, 1})
        for lbl in raw_labels:
            if is_ecg5000:
                is_ano = 1 if lbl == 0.0 else 0
                ann_symbols.append("N" if is_ano == 0 else "V")
                ann_labels.append(is_ano)
                ann_descriptions.append("Normal sinus beat" if is_ano == 0 else "Arrhythmia / Ectopic beat")
            else:
                ann_symbols.append(AAMI_LABEL_TO_SYMBOL.get(lbl, "V" if lbl > 0 else "N"))
                ann_labels.append(1 if lbl > 0 else 0)
                ann_descriptions.append(AAMI_LABEL_TO_DESC.get(lbl, "Ectopic beat"))

    # 4. Optional Decimation
    effective_fs = fs
    if downsample_factor > 1:
        signal_slice = signal_slice[::downsample_factor]
        effective_fs = max(10, fs // downsample_factor)
        ann_samples = ann_samples // downsample_factor

    final_samples = len(signal_slice)
    final_duration = final_samples / effective_fs
    memory_mb = signal_slice.nbytes / (1024 * 1024)

    allowance_meta = {
        "raw_file_size_mb": file_bytes_len / (1024 * 1024),
        "orig_total_samples": len(df) * beat_stride,
        "orig_total_duration_sec": (len(df) * beat_stride) / fs,
        "processed_samples": final_samples,
        "processed_duration_sec": final_duration,
        "start_second": start_second,
        "duration_allowance_sec": duration_allowance_sec,
        "downsample_factor": downsample_factor,
        "memory_usage_mb": memory_mb,
        "available_columns": ["Continuous Rhythm Strip (Reconstructed from Beats)"],
    }

    return _create_record_dict(
        record_name=filename,
        signal=signal_slice,
        fs=effective_fs,
        lead_name="AAMI Lead (Continuous Rhythm)",
        ann_samples=np.array(ann_samples, dtype=int),
        ann_symbols=np.array(ann_symbols, dtype=object),
        ann_labels=np.array(ann_labels, dtype=int),
        ann_descriptions=ann_descriptions,
        is_user_uploaded=True,
        original_fs=fs,
        allowance_meta=allowance_meta,
    )


def _load_continuous_timeseries(
    df: pd.DataFrame,
    fs: int,
    filename: str,
    file_bytes_len: int,
    selected_column: Optional[str] = None,
    start_second: float = 0.0,
    duration_allowance_sec: Optional[float] = 120.0,
    max_samples_allowance: Optional[int] = 500_000,
    downsample_factor: int = 1,
) -> Dict[str, Any]:
    """Slices continuous multi-lead time series and applies allowance constraints."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("Uploaded file does not contain any numeric ECG voltage columns.")

    # Filter out obvious time/index/timestamp columns when auto-selecting default ECG lead
    candidate_cols = [c for c in numeric_cols if not any(t in str(c).lower() for t in ("time", "index", "timestamp", "sec"))]
    if len(candidate_cols) == 0:
        candidate_cols = numeric_cols

    signal_col = selected_column if (selected_column and selected_column in df.columns) else candidate_cols[0]

    # Clean lead name if numeric or scientific notation
    lead_name = str(signal_col)
    if "e+" in lead_name or "e-" in lead_name or lead_name.replace(".", "", 1).isdigit():
        lead_name = "Lead 1"

    raw_signal = df[signal_col].to_numpy(dtype=np.float64)
    raw_signal = raw_signal[np.isfinite(raw_signal)]

    orig_total_samples = len(raw_signal)
    orig_total_duration_sec = orig_total_samples / fs

    # 1. Time Window Slicing
    start_idx = int(start_second * fs) if (0 <= int(start_second * fs) < orig_total_samples) else 0
    end_idx = (
        min(orig_total_samples, start_idx + int(duration_allowance_sec * fs))
        if (duration_allowance_sec and duration_allowance_sec > 0)
        else orig_total_samples
    )
    signal_slice = raw_signal[start_idx:end_idx]

    # 2. Max Samples Allowance Safeguard
    if max_samples_allowance and len(signal_slice) > max_samples_allowance:
        signal_slice = signal_slice[:max_samples_allowance]

    # 3. Decimation
    effective_fs = fs
    if downsample_factor > 1:
        signal_slice = signal_slice[::downsample_factor]
        effective_fs = max(10, fs // downsample_factor)

    final_samples = len(signal_slice)
    final_duration = final_samples / effective_fs
    memory_mb = signal_slice.nbytes / (1024 * 1024)

    allowance_meta = {
        "raw_file_size_mb": file_bytes_len / (1024 * 1024),
        "orig_total_samples": orig_total_samples,
        "orig_total_duration_sec": orig_total_duration_sec,
        "processed_samples": final_samples,
        "processed_duration_sec": final_duration,
        "start_second": start_second,
        "duration_allowance_sec": duration_allowance_sec,
        "downsample_factor": downsample_factor,
        "memory_usage_mb": memory_mb,
        "available_columns": list(numeric_cols),
    }

    return _create_record_dict(
        record_name=filename,
        signal=signal_slice,
        fs=effective_fs,
        lead_name=lead_name,
        is_user_uploaded=True,
        original_fs=fs,
        allowance_meta=allowance_meta,
    )


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
    Parses a user-uploaded CSV/TXT ECG signal with configurable Data Allowance controls.
    Automatically detects whether the file is a continuous recording or a pre-segmented
    AAMI heartbeat matrix, routing to the specialized high-performance loader.
    """
    df, is_beat_matrix, _ = _read_csv_smart(io.BytesIO(file_bytes))

    # Detect if we should reconstruct continuous rhythm strip from heartbeat rows
    is_reconstruct_mode = is_beat_matrix and (
        selected_column is None
        or "Continuous" in str(selected_column)
        or "Rhythm" in str(selected_column)
        or str(selected_column) in ("Sample_0", "0", "0.0")
        or "1.0000" in str(selected_column)
    )

    if is_reconstruct_mode:
        return _reconstruct_rhythm_from_beats(
            df=df,
            fs=fs,
            filename=filename,
            file_bytes_len=len(file_bytes),
            start_second=start_second,
            duration_allowance_sec=duration_allowance_sec,
            downsample_factor=downsample_factor,
        )

    return _load_continuous_timeseries(
        df=df,
        fs=fs,
        filename=filename,
        file_bytes_len=len(file_bytes),
        selected_column=selected_column,
        start_second=start_second,
        duration_allowance_sec=duration_allowance_sec,
        max_samples_allowance=max_samples_allowance,
        downsample_factor=downsample_factor,
    )
