"""
ECG Insight AI - Evaluation Module
Computes comprehensive machine learning & biomedical validation metrics:
- Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score
- ROC-AUC and PR-AUC (Average Precision)
- Confusion Matrix (TP, FP, TN, FN)
- ROC and Precision-Recall Curves
- Arrhythmia subtype detection breakdown (e.g., PVCs, PACs, Normal)
- Clinical engineering placement report generator
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn import metrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes standard biomedical machine learning metrics.
    Handles edge cases like single-class test splits gracefully.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "has_metrics": False,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "f1": 0.0,
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    # Ensure binary integer format
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = float(metrics.accuracy_score(y_true, y_pred))
    precision = float(metrics.precision_score(y_true, y_pred, zero_division=0))
    recall = float(metrics.recall_score(y_true, y_pred, zero_division=0))  # Sensitivity
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(metrics.f1_score(y_true, y_pred, zero_division=0))

    # ROC AUC and PR AUC
    roc_auc = 0.0
    pr_auc = 0.0
    roc_curve_data = None
    pr_curve_data = None

    if y_scores is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(metrics.roc_auc_score(y_true, y_scores))
            fpr, tpr, roc_thresh = metrics.roc_curve(y_true, y_scores)
            roc_curve_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresh.tolist()}
        except Exception:
            roc_auc = 0.0

        try:
            pr_auc = float(metrics.average_precision_score(y_true, y_scores))
            precisions, recalls, pr_thresh = metrics.precision_recall_curve(y_true, y_scores)
            pr_curve_data = {"precision": precisions.tolist(), "recall": recalls.tolist()}
        except Exception:
            pr_auc = 0.0

    return {
        "has_metrics": True,
        "total_beats": len(y_true),
        "total_anomalies": int(np.sum(y_true)),
        "anomaly_prevalence": float(np.mean(y_true) * 100.0),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,  # Sensitivity
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "roc_curve": roc_curve_data,
        "pr_curve": pr_curve_data,
    }


def compute_beat_subtype_breakdown(
    ann_symbols: np.ndarray,
    y_pred: np.ndarray,
    ann_descriptions: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Computes detection breakdown across each annotated ECG beat type
    (e.g., 'N' normal vs 'V' premature ventricular contraction vs 'A' atrial).
    """
    unique_symbols = np.unique(ann_symbols)
    results = []

    for sym in unique_symbols:
        mask = ann_symbols == sym
        total_count = int(np.sum(mask))
        if total_count == 0:
            continue
        flagged_count = int(np.sum(y_pred[mask] == 1))
        detection_rate = (flagged_count / total_count) * 100.0

        desc = sym
        if ann_descriptions and len(ann_descriptions) == len(ann_symbols):
            # Pick first description for this symbol
            indices = np.where(mask)[0]
            if len(indices) > 0:
                desc = ann_descriptions[indices[0]]

        results.append({
            "symbol": str(sym),
            "description": desc,
            "total_count": total_count,
            "flagged_anomalous": flagged_count,
            "detection_rate_pct": round(detection_rate, 1),
        })

    return sorted(results, key=lambda x: x["total_count"], reverse=True)


def generate_placement_report_md(
    record_name: str,
    model_name: str,
    metrics_dict: Dict[str, Any],
    n_beats: int,
    mean_hr: float,
    threshold: float,
) -> str:
    """
    Generates a professional Markdown evaluation report suitable for
    interviews, placement portfolios, and project documentation.
    """
    cm = metrics_dict.get("confusion_matrix", {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    acc = metrics_dict.get("accuracy", 0.0) * 100
    sens = metrics_dict.get("recall", 0.0) * 100
    spec = metrics_dict.get("specificity", 0.0) * 100
    prec = metrics_dict.get("precision", 0.0) * 100
    f1 = metrics_dict.get("f1", 0.0)
    roc_auc = metrics_dict.get("roc_auc", 0.0)

    report = f"""# ECG Insight AI — Performance & Placement Portfolio Report

**Generated System Evaluation Summary**
* **Target Dataset / Record**: PhysioNet MIT-BIH {record_name}
* **Model Deployed**: {model_name}
* **Decision Threshold**: {threshold:.2f}
* **Analyzed Heartbeats**: {n_beats} beats
* **Average Heart Rate**: {mean_hr:.1f} BPM

---

## 1. Executive Summary & Placement Value
This project demonstrates end-to-end competencies in biomedical data science, time-series signal processing, machine learning pipeline engineering, and interactive web visualization. By leveraging clinical annotations from the MIT-BIH Arrhythmia Database, the system automatically detects abnormal cardiac morphology (such as Premature Ventricular Contractions) without human intervention.

## 2. Quantitative Performance Metrics
| Metric | Score | Clinical / ML Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **{acc:.1f}%** | Overall fraction of correct beat classifications |
| **Sensitivity (Recall)** | **{sens:.1f}%** | Critical ability to catch true cardiac anomalies |
| **Specificity** | **{spec:.1f}%** | Ability to avoid false alarms on normal sinus beats |
| **Precision (PPV)** | **{prec:.1f}%** | Reliability of raised clinical alerts |
| **F1-Score** | **{f1:.3f}** | Balanced harmonic mean for imbalanced data |
| **ROC-AUC** | **{roc_auc:.3f}** | Discriminative power across all decision thresholds |

## 3. Confusion Matrix Breakdown
* **True Positives (TP)**: {cm['tp']} arrhythmias accurately flagged
* **True Negatives (TN)**: {cm['tn']} normal beats correctly cleared
* **False Positives (FP)**: {cm['fp']} normal beats flagged as anomalies (false alarms)
* **False Negatives (FN)**: {cm['fn']} arrhythmias missed

## 4. Key Engineering Takeaways for Technical Interviews
1. **Signal Conditioning**: Butterworth zero-phase bandpass (0.5–40 Hz) and notch filtering remove baseline wander (respiration) and 50 Hz powerline hum without temporal phase distortion.
2. **QRS Detection**: Pan-Tompkins pipeline (differentiation, squaring, moving integration, physiological refractory window) robustly tracks R-peaks.
3. **Feature Engineering**: Morphological wave amplitudes (R, Q, S deflections), area under QRS, statistical kurtosis/skewness, and timing metrics (pre/post RR ratio) capture ectopy characteristics.
4. **Unsupervised vs. Supervised**: Isolation Forest isolates ectopic beat signatures because anomalous waveforms occupy sparse, distant branches in tree partitions.

*Disclaimer: Educational research prototype. Not intended for direct medical diagnostic use.*
"""
    return report
