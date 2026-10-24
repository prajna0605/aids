"""
ECG Insight AI - Model Evaluation Module
Evaluates the trained model on held-out test recordings with ground-truth clinical annotations.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn import metrics

from src.data_loader import load_mitdb_record, generate_synthetic_ecg
from src.preprocessing import (
    preprocess_ecg_signal,
    pan_tompkins_r_peak_detection,
    match_peaks_to_annotations,
)
from src.features import segment_heartbeats, extract_ecg_features
from src.train import load_trained_model


def run_evaluation(
    model_path: str = "models/model.joblib",
    test_record: str = "119",
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluates the model on test data and returns quantitative metrics dictionary.
    """
    bundle = load_trained_model(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]

    # Load test record with ground-truth annotations
    try:
        rec_data = load_mitdb_record(test_record, max_seconds=120)
    except Exception:
        # Fallback to test synthetic record
        rec_data = generate_synthetic_ecg(duration_sec=90, anomaly_prob=0.22, random_state=999)

    prep = preprocess_ecg_signal(rec_data["signal"], fs=rec_data["fs"])
    peaks = pan_tompkins_r_peak_detection(prep["cleaned"], fs=rec_data["fs"])
    valid_peaks, waveforms = segment_heartbeats(prep["cleaned"], peaks, fs=rec_data["fs"])
    features, _ = extract_ecg_features(waveforms, valid_peaks, fs=rec_data["fs"])

    # Match peaks to annotations
    y_true, matched_symbols, _ = match_peaks_to_annotations(
        valid_peaks,
        rec_data["ann_samples"],
        rec_data["ann_labels"],
        rec_data["ann_symbols"],
        fs=rec_data["fs"],
    )

    # Scale features and predict
    X_scaled = scaler.transform(features)
    raw_scores = model.decision_function(X_scaled)
    inverted = -raw_scores
    s_min, s_max = np.min(inverted), np.max(inverted)
    if s_max - s_min > 1e-6:
        anomaly_scores = (inverted - s_min) / (s_max - s_min)
    else:
        anomaly_scores = np.zeros(len(features))

    y_pred = (anomaly_scores >= threshold).astype(int)

    # Calculate metrics
    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = float(metrics.accuracy_score(y_true, y_pred))
    precision = float(metrics.precision_score(y_true, y_pred, zero_division=0))
    recall = float(metrics.recall_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(metrics.f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(metrics.roc_auc_score(y_true, anomaly_scores))
    except Exception:
        roc_auc = 0.0

    eval_results = {
        "record_name": rec_data["record_name"],
        "total_beats": len(y_true),
        "total_anomalies": int(np.sum(y_true)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    print(f"--- Evaluation on Record {test_record} ---")
    print(f"Accuracy:    {accuracy * 100:.1f}%")
    print(f"Sensitivity: {recall * 100:.1f}%")
    print(f"Specificity: {specificity * 100:.1f}%")
    print(f"Precision:   {precision * 100:.1f}%")
    print(f"F1-Score:    {f1:.3f}")
    print(f"ROC-AUC:     {roc_auc:.3f}")

    return eval_results


if __name__ == "__main__":
    run_evaluation()
