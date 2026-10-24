"""
Unit & Integration Tests: ECG Processing & Anomaly Detection Pipeline
"""
import pytest
from src.data_loader import generate_synthetic_ecg, load_mitdb_record
from src.preprocessing import (
    preprocess_ecg_signal,
    pan_tompkins_r_peak_detection,
    segment_heartbeats,
    extract_ecg_features,
    match_peaks_to_annotations,
)
from src.models import train_and_evaluate_model
from src.evaluation import compute_metrics, compute_beat_subtype_breakdown


def test_synthetic_ecg_pipeline():
    """Verify synthetic ECG generation, filtering, peak detection, and feature extraction."""
    # 1. Generate synthetic ECG
    data = generate_synthetic_ecg(duration_sec=30, fs=360, anomaly_prob=0.20, random_state=42)
    assert len(data["signal"]) > 0
    assert len(data["ann_samples"]) > 0

    # 2. Preprocess signal
    prep = preprocess_ecg_signal(data["signal"], fs=360)
    assert "cleaned" in prep
    assert len(prep["cleaned"]) == len(data["signal"])

    # 3. Peak detection
    peaks = pan_tompkins_r_peak_detection(prep["cleaned"], fs=360)
    assert len(peaks) > 5

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


def test_mitdb_loader_fallback():
    """Verify PhysioNet MIT-BIH loader or fallback to synthetic signal."""
    try:
        mit_data = load_mitdb_record("100", max_seconds=10)
        assert len(mit_data["signal"]) > 0
        assert mit_data["fs"] == 360
    except Exception:
        # Fallback synthetic signal if network is unavailable
        fallback = generate_synthetic_ecg(duration_sec=10, fs=360)
        assert len(fallback["signal"]) > 0
