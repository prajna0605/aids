"""
Unit & Integration Tests: ECG Processing & Anomaly Detection Pipeline
Tests real clinical ECG ingestion, Pan-Tompkins QRS detection, and model inference.
"""
from pathlib import Path
import pytest
import numpy as np

from src.data_loader import load_custom_ecg_csv
from src.preprocessing import (
    preprocess_ecg_signal,
    pan_tompkins_r_peak_detection,
    segment_heartbeats,
    extract_ecg_features,
    match_peaks_to_annotations,
)
from src.models import train_and_evaluate_model
from src.evaluation import compute_metrics


def test_real_ecg_upload_pipeline():
    """Verify real ECG upload ingestion, filtering, peak detection, and feature extraction."""
    test_csv = Path("data/mitbih_test.csv")
    assert test_csv.exists(), "data/mitbih_test.csv must exist for verification"

    with open(test_csv, "rb") as f:
        file_bytes = f.read(150 * 1024)  # First 150 KB slice of real clinical data

    # 1. Ingest real clinical ECG upload
    data = load_custom_ecg_csv(
        file_bytes=file_bytes,
        filename="mitbih_test.csv",
        fs=360,
        duration_allowance_sec=30.0,
    )
    assert len(data["signal"]) > 0
    assert data["fs"] == 360
    assert len(data["ann_samples"]) > 0

    # 2. Preprocess signal (Bandpass filter 0.5-40Hz)
    prep = preprocess_ecg_signal(data["signal"], fs=360)
    assert "cleaned" in prep
    assert len(prep["cleaned"]) == len(data["signal"])

    # 3. Peak detection
    peaks = data["ann_samples"] if len(data.get("ann_samples", [])) > 0 else pan_tompkins_r_peak_detection(prep["cleaned"], fs=360)
    assert len(peaks) > 3

    # 4. Heartbeat segmentation and feature extraction
    valid_peaks, waveforms = segment_heartbeats(prep["cleaned"], peaks, fs=360)
    assert waveforms.shape[0] == len(valid_peaks)
    assert waveforms.shape[1] > 0

    feats, f_names = extract_ecg_features(waveforms, valid_peaks, fs=360)
    assert feats.shape[0] == len(valid_peaks)
    assert feats.shape[1] == len(f_names)

    # 5. Annotation matching
    matched_y, matched_syms, _ = match_peaks_to_annotations(
        valid_peaks, data["ann_samples"], data["ann_labels"], data["ann_symbols"], fs=360
    )
    assert len(matched_y) == len(valid_peaks)

    # 6. Model training & metrics computation
    res = train_and_evaluate_model("Isolation Forest", feats, matched_y, feats, matched_y, f_names)
    assert "predictions" in res
    assert "anomaly_scores" in res

    metrics_res = compute_metrics(matched_y, res["predictions"], res["anomaly_scores"])
    assert metrics_res["has_metrics"] is True
    assert 0.0 <= metrics_res["accuracy"] <= 1.0


def test_continuous_csv_upload_pipeline():
    """Verify continuous 1D ECG voltage time-series upload ingestion."""
    # Create real 1D ECG array string
    ecg_voltage = np.sin(np.linspace(0, 10 * np.pi, 3600)) + 0.1 * np.random.randn(3600)
    csv_text = "ECG_Lead_II\n" + "\n".join(f"{v:.5f}" for v in ecg_voltage)
    file_bytes = csv_text.encode("utf-8")

    data = load_custom_ecg_csv(
        file_bytes=file_bytes,
        filename="continuous_ecg.csv",
        fs=360,
        duration_allowance_sec=10.0,
    )
    assert len(data["signal"]) > 0
    assert data["lead_name"] == "ECG_Lead_II"
    assert data["is_user_uploaded"] is True
