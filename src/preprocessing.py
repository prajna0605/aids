"""
ECG Insight AI - Biomedical Signal Preprocessing
================================================
Cleans noisy raw ECG recordings using digital filters and detects R-peaks:
1. Butterworth Bandpass Filter (0.5 - 40 Hz): Removes breathing wander (<0.5 Hz) & muscle tremor (>40 Hz)
2. Notch Filter (50 Hz / 60 Hz): Removes electrical powerline interference
3. Normalization: Z-score normalization for consistent signal amplitude
4. Pan-Tompkins Algorithm: Robust R-peak detection based on QRS energy
"""

from typing import Dict
import numpy as np
from scipy import signal as sp_signal

# Re-export feature extraction functions from src.features for seamless backward compatibility
from src.features import (
    segment_heartbeats,
    extract_ecg_features,
    match_peaks_to_annotations,
)


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: int = 360,
    order: int = 3,
) -> np.ndarray:
    """Applies zero-phase Butterworth bandpass filter to remove baseline wander and high-frequency noise."""
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    b, a = sp_signal.butter(order, [low, high], btype="bandpass")
    return sp_signal.filtfilt(b, a, data)


def notch_filter(
    data: np.ndarray,
    notch_freq: float = 50.0,
    quality_factor: float = 30.0,
    fs: int = 360,
) -> np.ndarray:
    """Applies an IIR notch filter to suppress 50 Hz or 60 Hz electrical grid hum."""
    nyq = 0.5 * fs
    if notch_freq >= nyq:
        return data
    b, a = sp_signal.iirnotch(notch_freq / nyq, quality_factor)
    return sp_signal.filtfilt(b, a, data)


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
    Standard preprocessing pipeline:
    Raw ECG -> Bandpass Filter -> 50Hz Notch Filter -> Normalized Signal.
    """
    # 1. Bandpass filter
    bandpass_sig = butter_bandpass_filter(raw_signal, lowcut=lowcut, highcut=highcut, fs=fs)

    # 2. Notch filter for AC grid interference
    notch_sig = notch_filter(bandpass_sig, notch_freq=notch_freq, fs=fs) if (apply_notch and notch_freq < fs / 2) else bandpass_sig.copy()

    # 3. Z-score Normalization
    if normalize:
        std_val = max(1e-6, float(np.std(notch_sig)))
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
    Pan-Tompkins QRS Detection:
    1. 5-15 Hz QRS energy bandpass
    2. Central difference derivative
    3. Non-linear squaring
    4. Moving window integration (~120ms)
    5. Adaptive thresholding with physiological refractory period constraint (min 250ms between beats)
    """
    # 1. Bandpass focusing on QRS spike energy
    nyq = 0.5 * fs
    b, a = sp_signal.butter(2, [5.0 / nyq, min(0.95, 15.0 / nyq)], btype="bandpass")
    filtered = sp_signal.filtfilt(b, a, cleaned_signal)

    # 2. Derivative & 3. Squaring
    derivative = np.gradient(filtered)
    squared = derivative**2

    # 4. Moving window integration
    window_len = max(1, int(0.12 * fs))
    integrated = np.convolve(squared, np.ones(window_len) / window_len, mode="same")

    # 5. Peak finding with human refractory constraint (>250ms between beats)
    min_distance = int(refractory_sec * fs)
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    init_peaks, _ = sp_signal.find_peaks(integrated, height=threshold, distance=min_distance)

    # 6. Refine true peak position by finding local maximum in cleaned signal
    search_window = int(0.08 * fs)
    refined_peaks = []
    for p in init_peaks:
        left = max(0, p - search_window)
        right = min(len(cleaned_signal), p + search_window)
        if right > left:
            local_r = left + int(np.argmax(cleaned_signal[left:right]))
            if not refined_peaks or (local_r - refined_peaks[-1] > min_distance // 2):
                refined_peaks.append(local_r)

    return np.array(refined_peaks, dtype=int)
