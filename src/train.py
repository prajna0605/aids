"""
ECG Insight AI - Model Training Module
Trains the machine learning anomaly detection model on benchmark ECG data
and serializes it to models/model.joblib.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_mitdb_record, generate_synthetic_ecg
from src.preprocessing import preprocess_ecg_signal, pan_tompkins_r_peak_detection
from src.features import segment_heartbeats, extract_ecg_features


def train_model(
    save_path: str = "models/model.joblib",
    contamination: float = 0.12,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains the Isolation Forest anomaly detector on MIT-BIH and synthetic ECG records
    and serializes the pipeline using joblib.
    """
    print("Gathering training ECG records...")
    all_features = []
    all_labels = []

    # 1. Load MIT-BIH Record 100 (Clean normal baseline)
    try:
        rec_100 = load_mitdb_record("100", max_seconds=120)
        prep_100 = preprocess_ecg_signal(rec_100["signal"], fs=rec_100["fs"])
        peaks_100 = pan_tompkins_r_peak_detection(prep_100["cleaned"], fs=rec_100["fs"])
        valid_p100, wf_100 = segment_heartbeats(prep_100["cleaned"], peaks_100, fs=rec_100["fs"])
        f_100, feature_names = extract_ecg_features(wf_100, valid_p100, fs=rec_100["fs"])
        if len(f_100) > 0:
            all_features.append(f_100)
            print(f"Loaded {len(f_100)} beats from Record 100.")
    except Exception as e:
        print(f"Notice loading 100: {e}")

    # 2. Load MIT-BIH Record 106 (PVC ventricular ectopy)
    try:
        rec_106 = load_mitdb_record("106", max_seconds=120)
        prep_106 = preprocess_ecg_signal(rec_106["signal"], fs=rec_106["fs"])
        peaks_106 = pan_tompkins_r_peak_detection(prep_106["cleaned"], fs=rec_106["fs"])
        valid_p106, wf_106 = segment_heartbeats(prep_106["cleaned"], peaks_106, fs=rec_106["fs"])
        f_106, feature_names = extract_ecg_features(wf_106, valid_p106, fs=rec_106["fs"])
        if len(f_106) > 0:
            all_features.append(f_106)
            print(f"Loaded {len(f_106)} beats from Record 106.")
    except Exception as e:
        print(f"Notice loading 106: {e}")

    # 3. Augment with synthetic variations
    synth = generate_synthetic_ecg(duration_sec=90, anomaly_prob=0.18, random_state=random_state)
    prep_s = preprocess_ecg_signal(synth["signal"], fs=synth["fs"])
    peaks_s = pan_tompkins_r_peak_detection(prep_s["cleaned"], fs=synth["fs"])
    valid_ps, wf_s = segment_heartbeats(prep_s["cleaned"], peaks_s, fs=synth["fs"])
    f_s, feature_names = extract_ecg_features(wf_s, valid_ps, fs=synth["fs"])
    if len(f_s) > 0:
        all_features.append(f_s)
        print(f"Loaded {len(f_s)} synthetic ECG beats.")

    X = np.vstack(all_features)
    print(f"Total training dataset size: {X.shape[0]} beats, {X.shape[1]} features.")

    # Fit Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Prepare bundle
    model_bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "contamination": contamination,
        "model_type": "IsolationForest",
        "training_samples": len(X),
        "version": "1.0.0",
    }

    # Ensure output directory exists
    out_dir = Path(save_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_bundle, save_path)
    print(f"Model saved successfully to {save_path}!")

    return model_bundle


def load_trained_model(model_path: str = "models/model.joblib") -> Dict[str, Any]:
    """
    Loads serialized model bundle. If missing, automatically trains and returns.
    """
    p = Path(model_path)
    if not p.exists():
        print(f"{model_path} not found. Triggering automated model training...")
        return train_model(save_path=model_path)

    return joblib.load(str(p))


if __name__ == "__main__":
    train_model()
