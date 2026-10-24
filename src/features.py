"""
ECG Insight AI - Feature Engineering Module
Extracts morphological, statistical, and Heart Rate Variability (HRV) timing features
from segmented ECG heartbeats.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np


def segment_heartbeats(
    cleaned_signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int = 360,
    before_sec: float = 0.25,  # 250ms before R-peak (P & Q waves)
    after_sec: float = 0.40,   # 400ms after R-peak (S & T waves)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segments single heartbeats around each detected R-peak.

    Returns:
        valid_peaks: R-peak indices with complete surrounding windows.
        beat_waveforms: 2D array of shape (n_beats, samples_per_beat).
    """
    n_before = int(before_sec * fs)
    n_after = int(after_sec * fs)
    beat_length = n_before + n_after

    valid_peaks = []
    beat_waveforms = []

    for r in r_peaks:
        if r - n_before >= 0 and r + n_after <= len(cleaned_signal):
            waveform = cleaned_signal[r - n_before : r + n_after]
            # Baseline correct by referencing the start of PR segment to zero
            baseline_ref = np.median(waveform[: max(1, int(0.08 * fs))])
            waveform = waveform - baseline_ref
            beat_waveforms.append(waveform)
            valid_peaks.append(r)

    if len(beat_waveforms) == 0:
        return np.array([], dtype=int), np.empty((0, beat_length), dtype=np.float64)

    return np.array(valid_peaks, dtype=int), np.array(beat_waveforms, dtype=np.float64)


def extract_ecg_features(
    beat_waveforms: np.ndarray,
    r_peaks: np.ndarray,
    fs: int = 360,
    before_sec: float = 0.25,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts 30 biomedical features per beat:
    - HRV Timing: Pre-RR interval, Post-RR interval, Local RR Ratio
    - Morphological: R-peak amplitude, Q-deflection, S-deflection, Peak-to-Peak,
      QRS Area/Energy, T-wave amplitude, ST segment elevation/depression
    - Statistical Moments: Variance, Skewness, Kurtosis, Zero crossings
    - Spatial Shape: 16 downsampled morphology bins

    Returns:
        features: 2D numpy array of shape (n_beats, n_features)
        feature_names: List of feature names
    """
    n_beats = len(beat_waveforms)
    if n_beats == 0:
        return np.empty((0, 0)), []

    n_before = int(before_sec * fs)

    # Compute RR intervals in milliseconds
    rr_intervals_ms = np.diff(r_peaks) * (1000.0 / fs)
    pre_rr = np.zeros(n_beats)
    post_rr = np.zeros(n_beats)

    for i in range(n_beats):
        if i == 0:
            pre_rr[i] = rr_intervals_ms[0] if len(rr_intervals_ms) > 0 else 800.0
        else:
            pre_rr[i] = (r_peaks[i] - r_peaks[i - 1]) * (1000.0 / fs)

        if i < n_beats - 1:
            post_rr[i] = (r_peaks[i + 1] - r_peaks[i]) * (1000.0 / fs)
        else:
            post_rr[i] = pre_rr[i]

    # Moving average RR (5-beat window) for premature contraction detection
    moving_avg_rr = np.zeros(n_beats)
    for i in range(n_beats):
        w_start = max(0, i - 2)
        w_end = min(n_beats, i + 3)
        moving_avg_rr[i] = np.mean(pre_rr[w_start:w_end])

    rr_ratio = pre_rr / np.maximum(moving_avg_rr, 1.0)

    base_feature_names = [
        "Pre_RR_ms",
        "Post_RR_ms",
        "RR_Ratio",
        "R_Amplitude",
        "Q_Deflection",
        "S_Deflection",
        "Peak_to_Peak",
        "QRS_Energy",
        "T_Amplitude",
        "ST_Level",
        "Beat_Variance",
        "Beat_Skewness",
        "Beat_Kurtosis",
        "Zero_Crossings",
    ]
    n_morphology_bins = 16
    morphology_bin_names = [f"Morph_Bin_{i}" for i in range(n_morphology_bins)]
    all_feature_names = base_feature_names + morphology_bin_names

    feature_matrix = []

    for i in range(n_beats):
        wf = beat_waveforms[i]
        r_pt = n_before

        r_amp = wf[r_pt]

        # Q point
        q_window = wf[max(0, r_pt - int(0.05 * fs)) : r_pt]
        q_amp = np.min(q_window) if len(q_window) > 0 else 0.0

        # S point
        s_window = wf[r_pt : min(len(wf), r_pt + int(0.06 * fs))]
        s_amp = np.min(s_window) if len(s_window) > 0 else 0.0

        p2p = r_amp - s_amp

        # QRS energy
        qrs_wf = wf[max(0, r_pt - int(0.06 * fs)) : min(len(wf), r_pt + int(0.08 * fs))]
        qrs_energy = np.sum(qrs_wf**2)

        # T wave amplitude
        t_window = wf[min(len(wf), r_pt + int(0.12 * fs)) : min(len(wf), r_pt + int(0.30 * fs))]
        t_amp = np.max(t_window) if len(t_window) > 0 else 0.0

        # ST level
        st_window = wf[min(len(wf), r_pt + int(0.06 * fs)) : min(len(wf), r_pt + int(0.12 * fs))]
        st_level = np.mean(st_window) if len(st_window) > 0 else 0.0

        # Statistical moments
        variance = np.var(wf)
        mean_val = np.mean(wf)
        std_val = np.std(wf) + 1e-6
        skewness = np.mean(((wf - mean_val) / std_val) ** 3)
        kurtosis = np.mean(((wf - mean_val) / std_val) ** 4) - 3.0
        zero_crossings = np.sum(np.diff(np.signbit(wf)))

        # Downsample waveform into 16 bins for shape profile
        bin_size = len(wf) // n_morphology_bins
        morph_bins = [np.mean(wf[b * bin_size : (b + 1) * bin_size]) for b in range(n_morphology_bins)]

        row = [
            pre_rr[i],
            post_rr[i],
            rr_ratio[i],
            r_amp,
            q_amp,
            s_amp,
            p2p,
            qrs_energy,
            t_amp,
            st_level,
            variance,
            skewness,
            kurtosis,
            float(zero_crossings),
        ] + morph_bins

        feature_matrix.append(row)

    return np.array(feature_matrix, dtype=np.float64), all_feature_names
