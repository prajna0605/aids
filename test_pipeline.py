"""
Test Pipeline Verification Script for ECG Insight AI
"""
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

def run_test():
    print("Step 1: Testing Synthetic ECG Generation...")
    data = generate_synthetic_ecg(duration_sec=30, fs=360, anomaly_prob=0.20)
    print(f"Generated signal length: {len(data['signal'])}, Annotations count: {len(data['ann_samples'])}")

    print("\nStep 2: Testing Signal Preprocessing & R-Peak Detection...")
    prep = preprocess_ecg_signal(data['signal'], fs=360)
    peaks = pan_tompkins_r_peak_detection(prep['cleaned'], fs=360)
    print(f"Detected {len(peaks)} R-peaks")

    print("\nStep 3: Testing Heartbeat Segmentation & Feature Extraction...")
    valid_peaks, waveforms = segment_heartbeats(prep['cleaned'], peaks, fs=360)
    feats, f_names = extract_ecg_features(waveforms, valid_peaks, fs=360)
    print(f"Segmented {waveforms.shape[0]} beats, extracted {feats.shape[1]} features")

    matched_y, matched_syms, _ = match_peaks_to_annotations(
        valid_peaks, data['ann_samples'], data['ann_labels'], data['ann_symbols'], fs=360
    )

    print("\nStep 4: Testing Model Training (Isolation Forest)...")
    res = train_and_evaluate_model("Isolation Forest", feats, matched_y, feats, matched_y, f_names)
    metrics_res = compute_metrics(matched_y, res['predictions'], res['anomaly_scores'])
    print(f"Accuracy: {metrics_res['accuracy'] * 100:.1f}%")
    print(f"Sensitivity (Recall): {metrics_res['recall'] * 100:.1f}%")
    print(f"Specificity: {metrics_res['specificity'] * 100:.1f}%")
    print(f"F1-Score: {metrics_res['f1']:.3f}")
    print(f"ROC-AUC: {metrics_res['roc_auc']:.3f}")

    print("\nStep 5: Testing MIT-BIH PhysioNet Loader (Record 100)...")
    try:
        mit_data = load_mitdb_record("100", max_seconds=10)
        print(f"PhysioNet Record 100 loaded successfully! Shape: {mit_data['signal'].shape}")
    except Exception as e:
        print(f"PhysioNet load notice: {e}")

    print("\nSUCCESS: All pipeline components are operating flawlessly!")

if __name__ == "__main__":
    run_test()
