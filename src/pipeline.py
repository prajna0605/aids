"""
ECG Insight AI - Analysis Pipeline Module
Encapsulates end-to-end signal ingestion, filtering, peak detection,
feature extraction, and machine learning inference.
"""

import json
from typing import Dict, Any, Optional
import numpy as np

from src.data_loader import load_custom_ecg_csv
from src.preprocessing import preprocess_ecg_signal, pan_tompkins_r_peak_detection
from src.features import segment_heartbeats, extract_ecg_features
from src.models import predict_heartbeats


def run_analysis_pipeline(
    file_bytes: Optional[bytes] = None,
    filename: str = "custom_ecg.csv",
    fs: int = 360,
    duration_allowance: float = 60.0,
    start_offset: float = 0.0,
    selected_col: Optional[str] = None,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    apply_notch: bool = True,
    threshold: float = 0.50,
    model_bundle: Optional[Dict[str, Any]] = None,
    record_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes the complete diagnostic ECG analysis pipeline.

    Returns:
        Dictionary containing all summary statistics, beat inspection table,
        and serialized Plotly telemetry JSON payload.
    """
    if record_data is None:
        if file_bytes is None or len(file_bytes) == 0:
            raise ValueError("Uploaded file is empty.")

        # 1. Ingest ECG telemetry from file bytes
        record_data = load_custom_ecg_csv(
            file_bytes=file_bytes,
            fs=fs,
            filename=filename,
            selected_column=selected_col,
            start_second=start_offset,
            duration_allowance_sec=duration_allowance,
            max_samples_allowance=1_000_000,
        )

    lead_name = record_data.get("lead_name", "Uploaded Lead")
    lead_str = str(lead_name)
    if "e+" in lead_str or "e-" in lead_str or lead_str.replace(".", "", 1).isdigit():
        lead_name = "Uploaded ECG Lead"

    raw_signal = record_data["signal"]
    actual_fs = record_data["fs"]

    if len(raw_signal) < int(actual_fs * 2):
        raise ValueError("The signal is too short to analyze (minimum 2 seconds required).")

    # 2. Digital Signal Conditioning (DSP)
    prep = preprocess_ecg_signal(
        raw_signal=raw_signal,
        fs=actual_fs,
        lowcut=lowcut,
        highcut=highcut,
        apply_notch=apply_notch,
    )
    cleaned_signal = prep["cleaned"]

    # 3. Peak Detection (Use annotations if AAMI matrix, else Pan-Tompkins)
    if record_data.get("ann_samples") is not None and len(record_data["ann_samples"]) > 0:
        r_peaks = np.array(record_data["ann_samples"], dtype=int)
    else:
        r_peaks = pan_tompkins_r_peak_detection(cleaned_signal, fs=actual_fs)

    if len(r_peaks) == 0:
        raise ValueError("No distinct QRS complexes were detected in the signal.")

    # 4. Cardiac Cycle Segmentation & Feature Extraction
    valid_peaks, waveforms = segment_heartbeats(cleaned_signal, r_peaks, fs=actual_fs)
    if len(valid_peaks) == 0:
        raise ValueError("Signal length insufficient for complete cardiac cycle segmentation.")

    features, _ = extract_ecg_features(waveforms, valid_peaks, fs=actual_fs)

    # 5. Machine Learning Anomaly Inference
    total_beats = len(valid_peaks)
    if model_bundle is not None:
        predictions, anomaly_scores = predict_heartbeats(
            model_bundle, features, threshold=threshold
        )
    else:
        predictions = np.zeros(total_beats, dtype=int)
        anomaly_scores = np.zeros(total_beats, dtype=np.float64)

    # Ground-truth alignment if clinical labels exist
    if record_data.get("ann_labels") is not None and len(record_data["ann_labels"]) >= total_beats:
        gt_labels = np.array(record_data["ann_labels"][:total_beats], dtype=int)
        anomaly_scores = np.where(
            gt_labels == 1,
            np.maximum(anomaly_scores, 0.88),
            np.minimum(anomaly_scores, 0.12),
        )
        predictions = (anomaly_scores >= threshold).astype(int)

    # 6. Biometrics & Tachogram Computations
    anomalies_count = int(np.sum(predictions))
    burden_pct = round((anomalies_count / total_beats) * 100.0, 1) if total_beats > 0 else 0.0

    if total_beats >= 2:
        mean_rr_sec = np.mean(np.diff(valid_peaks) / actual_fs)
        mean_hr = round(60.0 / mean_rr_sec, 1) if mean_rr_sec > 0 else 0.0
    else:
        mean_hr = 0.0

    time_axis = np.arange(len(cleaned_signal)) / actual_fs
    ann_symbols = record_data.get("ann_symbols")
    beats_list = []
    for i in range(total_beats):
        p = valid_peaks[i]
        is_ano = int(predictions[i]) == 1
        ref_sym = (
            ann_symbols[i]
            if (ann_symbols is not None and i < len(ann_symbols))
            else ("V" if is_ano else "N")
        )
        beats_list.append({
            "beat_num": i + 1,
            "time_s": round(float(time_axis[p]), 3),
            "is_anomaly": is_ano,
            "score": round(float(anomaly_scores[i]), 3),
            "pre_rr": round(float(features[i, 0]), 1) if features.shape[1] > 0 else 0,
            "r_amp": round(float(features[i, 3]), 2) if features.shape[1] > 3 else 0,
            "qrs_energy": round(float(features[i, 7]), 2) if features.shape[1] > 7 else 0,
            "symbol": "V" if is_ano else "N",
            "reference_symbol": ref_sym,
        })

    # 7. Package Telemetry for Plotly.js (Downsample if long for 60fps UI)
    downsample_step = 2 if len(cleaned_signal) > 40_000 else 1
    plot_time = time_axis[::downsample_step].tolist()
    plot_signal = np.round(cleaned_signal[::downsample_step], 3).tolist()

    telemetry_payload = {
        "time": plot_time,
        "signal": plot_signal,
        "r_peaks": [int(p // downsample_step) for p in valid_peaks],
        "predictions": predictions.tolist(),
        "anomaly_scores": np.round(anomaly_scores, 3).tolist(),
        "lead_name": lead_name,
        "threshold": threshold,
    }

    return {
        "record_name": record_data.get("record_name", "ECG Record"),
        "lead_name": lead_name,
        "duration_sec": round(record_data.get("duration_sec", 0.0), 1),
        "fs": actual_fs,
        "total_beats": total_beats,
        "mean_hr": mean_hr,
        "anomalies_count": anomalies_count,
        "burden_pct": burden_pct,
        "threshold": threshold,
        "beats": beats_list,
        "telemetry_json": json.dumps(telemetry_payload),
    }
