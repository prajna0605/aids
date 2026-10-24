"""
ECG Insight AI - Preprocessing Module
Implements biomedical digital signal processing:
- Butterworth bandpass filtering (0.5 - 40 Hz) for baseline wander & EMG noise removal
- Notch filtering (50 Hz / 60 Hz) for power-line interference suppression
- Pan-Tompkins style R-peak detection & RR interval calculation
- Heartbeat segmentation (P-QRS-T windows)
- Comprehensive feature extraction (morphological, statistical, and HRV timing features)
- Ground-truth annotation matching
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import signal as sp_signal


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: int = 360,
    order: int = 3,
) -> np.ndarray:
    """
    Applies zero-phase digital Butterworth bandpass filter.
    Eliminates baseline wander (<0.5 Hz) and high-frequency muscle noise (>40 Hz).
    """
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    b, a = sp_signal.butter(order, [low, high], btype="bandpass")
    # filtfilt ensures zero phase distortion, preserving ECG peak timing
    filtered = sp_signal.filtfilt(b, a, data)
    return filtered


def notch_filter(
    data: np.ndarray,
    notch_freq: float = 50.0,
    quality_factor: float = 30.0,
    fs: int = 360,
) -> np.ndarray:
    """
    Applies an IIR notch filter to reject 50 Hz or 60 Hz AC electrical grid hum.
    """
    nyq = 0.5 * fs
    if notch_freq >= nyq:
        return data
    freq_norm = notch_freq / nyq
    b, a = sp_signal.iirnotch(freq_norm, quality_factor)
    filtered = sp_signal.filtfilt(b, a, data)
    return filtered


def preprocess_ecg_signal(
    raw_signal: np.ndarray,
    fs: int = 360,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    apply_notch: bool = True,
    notch_freq: float = 50.0,
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Full preprocessing pipeline:
    Raw -> Bandpass -> Notch -> Normalized signal.
    Returns intermediate stages for visualization in the Signal Lab tab.
    """
    # 1. Bandpass filter
    bandpass_sig = butter_bandpass_filter(raw_signal, lowcut=lowcut, highcut=highcut, fs=fs)

    # 2. Notch filter
    if apply_notch and notch_freq < (fs / 2):
        notch_sig = notch_filter(bandpass_sig, notch_freq=notch_freq, fs=fs)
    else:
        notch_sig = bandpass_sig.copy()

    # 3. Normalization (Z-score or Robust Scaler)
    if normalize:
        std_val = np.std(notch_sig)
        if std_val < 1e-6:
            std_val = 1e-6
        norm_sig = (notch_sig - np.median(notch_sig)) / std_val
    else:
        norm_sig = notch_sig.copy()

    return {
        "raw": raw_signal,
        "bandpass": bandpass_sig,
        "notch": notch_sig,
        "cleaned": norm_sig,
    }


def pan_tompkins_r_peak_detection(
    cleaned_signal: np.ndarray,
    fs: int = 360,
    refractory_sec: float = 0.25,
) -> np.ndarray:
    """
    Pan-Tompkins style QRS detection algorithm:
    1. Bandpass filter (5-15 Hz focusing on QRS energy)
    2. Five-point derivative to isolate slopes
    3. Non-linear squaring to amplify large deflections
    4. Moving-window integration (150ms window)
    5. Adaptive threshold peak detection with physiological refractory constraint
    """
    # 1. Bandpass 5-15 Hz (QRS dominant frequencies)
    nyq = 0.5 * fs
    b, a = sp_signal.butter(2, [5.0 / nyq, min(0.95, 15.0 / nyq)], btype="bandpass")
    filtered = sp_signal.filtfilt(b, a, cleaned_signal)

    # 2. Derivative
    # Five-point central difference derivative
    derivative = np.gradient(filtered)

    # 3. Squaring
    squared = derivative**2

    # 4. Moving window integration (~120ms window)
    window_len = int(0.12 * fs)
    if window_len < 1:
        window_len = 1
    kernel = np.ones(window_len) / window_len
    integrated = np.convolve(squared, kernel, mode="same")

    # 5. Peak finding on integrated signal with refractory period
    min_distance = int(refractory_sec * fs)  # Human heart cannot beat faster than ~240 bpm (250ms)
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)

    init_peaks, _ = sp_signal.find_peaks(
        integrated,
        height=threshold,
        distance=min_distance,
    )

    # 6. Search back in cleaned_signal for true local maximum of R-peak within +/- 80ms of integrated peak
    search_window = int(0.08 * fs)
    refined_r_peaks = []

    for p in init_peaks:
        left = max(0, p - search_window)
        right = min(len(cleaned_signal), p + search_window)
        if right > left:
            local_r = left + np.argmax(cleaned_signal[left:right])
            # Avoid duplicate peaks
            if len(refined_r_peaks) == 0 or (local_r - refined_r_peaks[-1] > min_distance // 2):
                refined_r_peaks.append(local_r)

    return np.array(refined_r_peaks, dtype=int)


def segment_heartbeats(
    cleaned_signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int = 360,
    before_sec: float = 0.25,  # 250ms before R (captures P and Q)
    after_sec: float = 0.40,   # 400ms after R (captures S and T)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segments single heartbeats around each detected R-peak.
    Returns:
        valid_peaks: R-peak indices that have complete waveforms
        beat_waveforms: 2D array of shape (N_beats, samples_per_beat)
    """
    n_before = int(before_sec * fs)
    n_after = int(after_sec * fs)
    beat_length = n_before + n_after

    valid_peaks = []
    beat_waveforms = []

    for r in r_peaks:
        if r - n_before >= 0 and r + n_after <= len(cleaned_signal):
            waveform = cleaned_signal[r - n_before : r + n_after]
            # Baseline correct by shifting beat median or start to 0
            waveform = waveform - np.median(waveform[: int(0.08 * fs)])
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
    Extracts comprehensive time-domain, morphological, statistical, and
    heart rate variability (HRV) timing features for each segmented beat.

    Features extracted per beat:
    - Pre-RR interval (ms)
    - Post-RR interval (ms)
    - Local RR ratio (pre_RR / local_moving_average_RR)
    - R-peak amplitude
    - Q-deflection depth
    - S-deflection depth
    - Peak-to-peak amplitude (R - S)
    - QRS area / energy
    - T-wave peak amplitude
    - ST-segment mean level
    - Waveform statistical moments: Variance, Skewness, Kurtosis
    - Waveform zero crossings count
    - Waveform vector downsampled representation (16 morphological shape bins)
    """
    n_beats = len(beat_waveforms)
    if n_beats == 0:
        return np.empty((0, 0)), []

    n_before = int(before_sec * fs)
    feature_list = []

    # Calculate RR intervals in milliseconds
    rr_intervals_ms = np.diff(r_peaks) * (1000.0 / fs)
    # Pre-RR and Post-RR for each beat
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

    # Calculate moving average RR (window of 5 beats) for local premature contraction detection
    moving_avg_rr = np.zeros(n_beats)
    for i in range(n_beats):
        w_start = max(0, i - 2)
        w_end = min(n_beats, i + 3)
        moving_avg_rr[i] = np.mean(pre_rr[w_start:w_end])

    rr_ratio = pre_rr / np.maximum(moving_avg_rr, 1.0)

    # Feature definitions
    feature_names = [
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
    # Add 16 downsampled morphology bins for shape signature
    n_morphology_bins = 16
    morphology_bin_names = [f"Morph_Bin_{i}" for i in range(n_morphology_bins)]
    all_feature_names = feature_names + morphology_bin_names

    for i in range(n_beats):
        wf = beat_waveforms[i]
        r_pt = n_before

        # R-peak amplitude
        r_amp = wf[r_pt]

        # Q-point: minimum in [R - 50ms, R]
        q_window = wf[max(0, r_pt - int(0.05 * fs)) : r_pt]
        q_amp = np.min(q_window) if len(q_window) > 0 else 0.0

        # S-point: minimum in [R, R + 60ms]
        s_window = wf[r_pt : min(len(wf), r_pt + int(0.06 * fs))]
        s_amp = np.min(s_window) if len(s_window) > 0 else 0.0

        # Peak to peak
        p2p = r_amp - s_amp

        # QRS energy
        qrs_wf = wf[max(0, r_pt - int(0.06 * fs)) : min(len(wf), r_pt + int(0.08 * fs))]
        qrs_energy = np.sum(qrs_wf**2)

        # T-wave amplitude: maximum in [R + 120ms, R + 300ms]
        t_window = wf[min(len(wf), r_pt + int(0.12 * fs)) : min(len(wf), r_pt + int(0.30 * fs))]
        t_amp = np.max(t_window) if len(t_window) > 0 else 0.0

        # ST level: average in [R + 60ms, R + 120ms]
        st_window = wf[min(len(wf), r_pt + int(0.06 * fs)) : min(len(wf), r_pt + int(0.12 * fs))]
        st_level = np.mean(st_window) if len(st_window) > 0 else 0.0

        # Statistical moments
        variance = np.var(wf)
        mean_val = np.mean(wf)
        std_val = np.std(wf) + 1e-6
        skewness = np.mean(((wf - mean_val) / std_val) ** 3)
        kurtosis = np.mean(((wf - mean_val) / std_val) ** 4) - 3.0

        # Zero crossings
        zero_crossings = np.sum(np.diff(np.signbit(wf)))

        # Downsample waveform into 16 bins for spatial shape encoding
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

        feature_list.append(row)

    return np.array(feature_list, dtype=np.float64), all_feature_names


def match_peaks_to_annotations(
    detected_r_peaks: np.ndarray,
    ann_samples: np.ndarray,
    ann_labels: np.ndarray,
    ann_symbols: np.ndarray,
    fs: int = 360,
    tolerance_ms: float = 75.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Associates each detected R-peak with its closest ground-truth annotation.

    Returns:
        matched_labels: 0 for normal, 1 for anomaly, -1 for unmatched
        matched_symbols: annotation character symbol (e.g., 'N', 'V', 'A')
        time_offsets_ms: timing difference between detection and annotation
    """
    n_peaks = len(detected_r_peaks)
    matched_labels = np.zeros(n_peaks, dtype=int)
    matched_symbols = np.array(["Unknown"] * n_peaks, dtype=object)
    time_offsets_ms = np.zeros(n_peaks, dtype=np.float64)

    if len(ann_samples) == 0:
        return matched_labels, matched_symbols, time_offsets_ms

    tolerance_samples = int((tolerance_ms / 1000.0) * fs)

    for i, peak in enumerate(detected_r_peaks):
        diffs = np.abs(ann_samples - peak)
        min_idx = np.argmin(diffs)
        min_diff = diffs[min_idx]

        if min_diff <= tolerance_samples:
            matched_labels[i] = ann_labels[min_idx]
            matched_symbols[i] = ann_symbols[min_idx]
            time_offsets_ms[i] = (peak - ann_samples[min_idx]) * (1000.0 / fs)
        else:
            # Beat was detected by algorithm but not present in reference annotation
            matched_labels[i] = 0
            matched_symbols[i] = "Unmatched"
            time_offsets_ms[i] = 0.0

    return matched_labels, matched_symbols, time_offsets_ms
