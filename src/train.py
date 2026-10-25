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
    contamination: float = 0.10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains the machine learning anomaly detector on comprehensive multi-dataset ECG benchmarks:
    1. Kaggle ECG5000 / UCR benchmark (healthy normal heartbeats and abnormal arrhythmias)
    2. MIT-BIH Arrhythmia train corpus (AAMI classes N, S, V, F, Q)
    3. PhysioNet clinical records (Record 100 normal, 106 PVC, 119 bigeminy, 200 tachycardia)
    4. Synthetic physiological baseline variations
    
    Serializes a high-performance Hybrid Ensemble (RandomForest + IsolationForest) with StandardScaler.
    """
    import pandas as pd
    from src.data_loader import _reconstruct_rhythm_from_beats

    print("Gathering comprehensive multi-dataset ECG training corpus...")
    all_features = []
    all_labels = []

    # 1. ECG5000 Kaggle / UCR benchmark dataset
    ecg5000_path = Path("data/ecg5000.csv")
    if ecg5000_path.exists():
        try:
            # Normal heartbeats (rows 0 to 800)
            df_norm = pd.read_csv(ecg5000_path, header=None, nrows=800)
            rec_n = _reconstruct_rhythm_from_beats(df_norm, fs=360, filename="ecg5000.csv", file_bytes_len=1000)
            p_n = preprocess_ecg_signal(rec_n["signal"], fs=360)
            vp_n, wf_n = segment_heartbeats(p_n["cleaned"], np.array(rec_n["ann_samples"], dtype=int), fs=360)
            f_n, feature_names = extract_ecg_features(wf_n, vp_n, fs=360)
            if len(f_n) > 0:
                all_features.append(f_n)
                all_labels.append(np.zeros(len(vp_n), dtype=int))
                print(f"Loaded {len(f_n)} normal beats from ECG5000.")

            # Abnormal heartbeats (rows 3000 to 3800)
            df_ab = pd.read_csv(ecg5000_path, header=None, skiprows=3000, nrows=800)
            rec_ab = _reconstruct_rhythm_from_beats(df_ab, fs=360, filename="ecg5000.csv", file_bytes_len=1000)
            p_ab = preprocess_ecg_signal(rec_ab["signal"], fs=360)
            vp_ab, wf_ab = segment_heartbeats(p_ab["cleaned"], np.array(rec_ab["ann_samples"], dtype=int), fs=360)
            f_ab, _ = extract_ecg_features(wf_ab, vp_ab, fs=360)
            if len(f_ab) > 0:
                all_features.append(f_ab)
                all_labels.append(np.ones(len(vp_ab), dtype=int))
                print(f"Loaded {len(f_ab)} arrhythmia beats from ECG5000.")
        except Exception as e:
            print(f"Notice loading ECG5000: {e}")

    # 2. MIT-BIH Training CSV (classes 0=Normal, 1..4=Arrhythmias)
    mit_csv = Path("data/mitbih_train.csv")
    if mit_csv.exists():
        try:
            df_mit = pd.read_csv(mit_csv, header=None, nrows=1500)
            rec_mit = _reconstruct_rhythm_from_beats(df_mit, fs=360, filename="mitbih_train.csv", file_bytes_len=1000)
            p_mit = preprocess_ecg_signal(rec_mit["signal"], fs=360)
            vp_mit, wf_mit = segment_heartbeats(p_mit["cleaned"], np.array(rec_mit["ann_samples"], dtype=int), fs=360)
            f_mit, _ = extract_ecg_features(wf_mit, vp_mit, fs=360)
            lbl_mit = np.array(rec_mit["ann_labels"][:len(vp_mit)], dtype=int)
            if len(f_mit) > 0:
                all_features.append(f_mit)
                all_labels.append(lbl_mit)
                print(f"Loaded {len(f_mit)} beats from MIT-BIH training CSV.")
        except Exception as e:
            print(f"Notice loading MIT-BIH CSV: {e}")

    # 3. PhysioNet benchmark records: 100 (Normal), 106 (PVC), 119 (Bigeminy), 200 (V-Tach)
    for rec_id in ["100", "106", "119", "200"]:
        try:
            rec = load_mitdb_record(rec_id, max_seconds=120)
            prep = preprocess_ecg_signal(rec["signal"], fs=rec["fs"])
            peaks = np.array(rec["ann_samples"], dtype=int) if len(rec.get("ann_samples", [])) > 0 else pan_tompkins_r_peak_detection(prep["cleaned"], fs=rec["fs"])
            valid_p, wf = segment_heartbeats(prep["cleaned"], peaks, fs=rec["fs"])
            f, feat_names = extract_ecg_features(wf, valid_p, fs=rec["fs"])
            feature_names = feat_names
            lbls = np.array(rec["ann_labels"][:len(valid_p)], dtype=int) if len(rec.get("ann_labels", [])) >= len(valid_p) else np.zeros(len(valid_p), dtype=int)
            if len(f) > 0:
                all_features.append(f)
                all_labels.append(lbls)
                print(f"Loaded {len(f)} beats from MIT-BIH Record {rec_id}.")
        except Exception as e:
            print(f"Notice loading Record {rec_id}: {e}")

    # 4. Augment with synthetic variations
    try:
        synth = generate_synthetic_ecg(duration_sec=90, anomaly_prob=0.18, random_state=random_state)
        prep_s = preprocess_ecg_signal(synth["signal"], fs=synth["fs"])
        peaks_s = np.array(synth["ann_samples"], dtype=int)
        valid_ps, wf_s = segment_heartbeats(prep_s["cleaned"], peaks_s, fs=synth["fs"])
        f_s, _ = extract_ecg_features(wf_s, valid_ps, fs=synth["fs"])
        lbl_s = np.array(synth["ann_labels"][:len(valid_ps)], dtype=int)
        if len(f_s) > 0:
            all_features.append(f_s)
            all_labels.append(lbl_s)
            print(f"Loaded {len(f_s)} synthetic ECG beats.")
    except Exception as e:
        print(f"Notice loading synthetic: {e}")

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    print(f"Total training dataset size: {X.shape[0]} beats (Normal: {np.sum(y == 0)}, Arrhythmia: {np.sum(y == 1)}), {X.shape[1]} features.")

    # Fit Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Supervised Random Forest Classifier
    classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    classifier.fit(X_scaled, y)

    # Train Unsupervised Isolation Forest on normal baseline beats
    X_normal = X_scaled[y == 0] if np.sum(y == 0) > 10 else X_scaled
    iso_model = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso_model.fit(X_normal)

    # Prepare bundle
    model_bundle = {
        "model": iso_model,
        "classifier": classifier,
        "scaler": scaler,
        "feature_names": feature_names,
        "contamination": contamination,
        "model_type": "Hybrid-Ensemble (RandomForest + IsolationForest)",
        "training_samples": len(X),
        "normal_samples": int(np.sum(y == 0)),
        "arrhythmia_samples": int(np.sum(y == 1)),
        "version": "2.0.0",
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
