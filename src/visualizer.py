"""
ECG Insight AI - Visualizer Module
Produces high-performance, clinical telemetry-grade interactive Plotly visualizations:
1. Continuous ECG waveform with highlighted anomalous beats & bounding bands
2. RR Interval Tachogram & HRV time series
3. Multi-stage Signal Preprocessing comparison (Raw vs Filtered vs Energy)
4. Power Spectral Density (FFT) frequency spectrum
5. Beat Morphology Superposition (Normal vs Anomalous P-QRS-T overlay)
6. Poincaré Plot (RR_n vs RR_{n+1})
7. Confusion Matrix Heatmap & ROC / PR curves
8. Anomaly Score Distribution Histogram
"""

from typing import Dict, List, Optional, Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Theme Palette (Clean Minimalist White Theme)
PALETTE = {
    "bg": "#ffffff",
    "card_bg": "#ffffff",
    "grid": "#f1f5f9",
    "ecg_trace": "#0284c7",      # Clean medical ocean blue
    "ecg_raw": "#94a3b8",        # Slate gray
    "normal_beat": "#10b981",    # Emerald green
    "anomaly_beat": "#ef4444",   # Alert crimson red
    "warning": "#f59e0b",        # Amber
    "text": "#0f172a",          # Deep slate charcoal
    "text_muted": "#64748b",     # Muted slate
}


def plot_ecg_interactive(
    time: np.ndarray,
    signal: np.ndarray,
    r_peaks: np.ndarray,
    predictions: np.ndarray,
    anomaly_scores: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    ann_symbols: Optional[np.ndarray] = None,
    title: str = "ECG Waveform Telemetry — Real-Time Anomaly Highlighting",
    fs: int = 360,
    lead_name: str = "MLII",
) -> go.Figure:
    """
    Creates the main clinical ECG telemetry strip with glowing line trace,
    green normal beat markers, and highlighted red anomaly flags.
    """
    fig = go.Figure()

    # 1. ECG Signal Trace
    fig.add_trace(
        go.Scatter(
            x=time,
            y=signal,
            mode="lines",
            name=f"ECG ({lead_name})",
            line=dict(color=PALETTE["ecg_trace"], width=1.5),
            hoverinfo="x+y",
        )
    )

    # 2. Extract beat information for annotations
    if len(r_peaks) > 0 and len(predictions) == len(r_peaks):
        norm_mask = predictions == 0
        ano_mask = predictions == 1

        # Normal Beats
        if np.any(norm_mask):
            fig.add_trace(
                go.Scatter(
                    x=time[r_peaks[norm_mask]],
                    y=signal[r_peaks[norm_mask]],
                    mode="markers",
                    name="Normal Beat",
                    marker=dict(
                        symbol="circle",
                        size=9,
                        color=PALETTE["normal_beat"],
                        line=dict(color="#ffffff", width=1),
                    ),
                    text=[
                        f"Beat #{idx}<br>Status: Normal<br>Score: {score:.3f}"
                        for idx, score in zip(np.where(norm_mask)[0], anomaly_scores[norm_mask])
                    ],
                    hoverinfo="text+x+y",
                )
            )

        # Anomalous Beats (Red alert markers with larger size)
        if np.any(ano_mask):
            fig.add_trace(
                go.Scatter(
                    x=time[r_peaks[ano_mask]],
                    y=signal[r_peaks[ano_mask]],
                    mode="markers+text",
                    name="Flagged Anomaly",
                    marker=dict(
                        symbol="diamond",
                        size=13,
                        color=PALETTE["anomaly_beat"],
                        line=dict(color="#ffffff", width=1.5),
                    ),
                    text=["⚠" for _ in range(np.sum(ano_mask))],
                    textposition="top center",
                    textfont=dict(color=PALETTE["anomaly_beat"], size=14),
                    hovertext=[
                        f"<b>POTENTIAL ANOMALY</b><br>Beat #{idx}<br>Score: {score:.3f}<br>Ref: {sym if sym is not None else 'N/A'}"
                        for idx, score, sym in zip(
                            np.where(ano_mask)[0],
                            anomaly_scores[ano_mask],
                            ann_symbols[ano_mask] if ann_symbols is not None else [None] * np.sum(ano_mask),
                        )
                    ],
                    hoverinfo="text+x+y",
                )
            )

            # Add highlighted vertical shaded regions around flagged beats
            # Shaded window around ectopic beat (-150ms to +250ms)
            half_win = 0.20
            for pk in r_peaks[ano_mask]:
                t_center = time[pk]
                fig.add_vrect(
                    x0=max(0, t_center - half_win),
                    x1=min(time[-1], t_center + half_win),
                    fillcolor=PALETTE["anomaly_beat"],
                    opacity=0.15,
                    layer="below",
                    line_width=0,
                )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(color=PALETTE["text"], size=16),
        ),
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["card_bg"],
        font=dict(color=PALETTE["text"], family="Inter, Roboto, sans-serif"),
        xaxis=dict(
            title="Time (seconds)",
            gridcolor=PALETTE["grid"],
            zerolinecolor=PALETTE["grid"],
            rangeslider=dict(visible=True, thickness=0.08, bgcolor="#f8fafc"),
        ),
        yaxis=dict(
            title="Amplitude (mV / Normalized)",
            gridcolor=PALETTE["grid"],
            zerolinecolor=PALETTE["grid"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor=PALETTE["grid"],
            borderwidth=1,
        ),
        hovermode="closest",
        margin=dict(l=50, r=30, t=60, b=40),
        height=480,
    )

    return fig


def plot_tachogram_and_scores(
    r_peaks: np.ndarray,
    time: np.ndarray,
    predictions: np.ndarray,
    anomaly_scores: np.ndarray,
    fs: int = 360,
    threshold: float = 0.50,
) -> go.Figure:
    """
    Subplot showing RR Interval Tachogram (top) and Continuous Anomaly Scores (bottom).
    Demonstrates compensatory pauses and heart rate dynamics.
    """
    if len(r_peaks) < 2:
        return go.Figure()

    beat_times = time[r_peaks]
    rr_intervals_ms = np.diff(r_peaks) * (1000.0 / fs)
    rr_times = beat_times[1:]
    rr_preds = predictions[1:]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "<b>RR Interval Tachogram (ms)</b> — Premature Contractions & Pauses",
            "<b>Model Anomaly Score Stream [0, 1]</b>",
        ),
    )

    # RR Tachogram
    norm_rr = rr_preds == 0
    ano_rr = rr_preds == 1

    if np.any(norm_rr):
        fig.add_trace(
            go.Scatter(
                x=rr_times[norm_rr],
                y=rr_intervals_ms[norm_rr],
                mode="markers+lines",
                name="Normal RR",
                line=dict(color="#3b82f6", width=1),
                marker=dict(color=PALETTE["normal_beat"], size=6),
            ),
            row=1,
            col=1,
        )
    if np.any(ano_rr):
        fig.add_trace(
            go.Scatter(
                x=rr_times[ano_rr],
                y=rr_intervals_ms[ano_rr],
                mode="markers",
                name="Anomalous RR",
                marker=dict(color=PALETTE["anomaly_beat"], size=9, symbol="diamond"),
            ),
            row=1,
            col=1,
        )

    # Anomaly Scores
    fig.add_trace(
        go.Scatter(
            x=beat_times,
            y=anomaly_scores,
            mode="lines+markers",
            name="Anomaly Score",
            line=dict(color="#f59e0b", width=1.5),
            marker=dict(
                size=6,
                color=[PALETTE["anomaly_beat"] if p == 1 else PALETTE["normal_beat"] for p in predictions],
            ),
        ),
        row=2,
        col=1,
    )

    # Decision threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Threshold ({threshold:.2f})",
        annotation_position="top right",
        row=2,
        col=1,
    )

    fig.update_layout(
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["card_bg"],
        font=dict(color=PALETTE["text"], family="Inter, Roboto, sans-serif"),
        height=420,
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"], title="Time (s)", row=2, col=1)
    fig.update_xaxes(gridcolor=PALETTE["grid"], row=1, col=1)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title="RR (ms)", row=1, col=1)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title="Score", range=[-0.05, 1.05], row=2, col=1)

    return fig


def plot_preprocessing_stages(
    time: np.ndarray,
    stages: Dict[str, np.ndarray],
    max_seconds: float = 10.0,
    fs: int = 360,
) -> go.Figure:
    """
    Compares Raw Signal vs Butterworth Bandpass vs Cleaned Signal over a 10s window.
    """
    max_samples = int(max_seconds * fs)
    t = time[:max_samples]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "1. Raw Unfiltered Signal (Baseline Wander + Noise)",
            "2. Bandpass Filtered (0.5 – 40 Hz Butterworth)",
            "3. Notch Filtered & Z-Score Normalized Output",
        ),
    )

    fig.add_trace(
        go.Scatter(x=t, y=stages["raw"][:max_samples], line=dict(color="#f87171", width=1.2), name="Raw"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=stages["bandpass"][:max_samples], line=dict(color="#fbbf24", width=1.2), name="Bandpass"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=stages["cleaned"][:max_samples], line=dict(color="#34d399", width=1.4), name="Cleaned"),
        row=3,
        col=1,
    )

    fig.update_layout(
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["card_bg"],
        font=dict(color=PALETTE["text"], family="Inter, Roboto, sans-serif"),
        height=520,
        showlegend=False,
        margin=dict(l=50, r=30, t=50, b=40),
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"], title="Time (s)", row=3, col=1)
    fig.update_xaxes(gridcolor=PALETTE["grid"], row=1, col=1)
    fig.update_xaxes(gridcolor=PALETTE["grid"], row=2, col=1)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title="Raw (mV)", row=1, col=1)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title="Filt (mV)", row=2, col=1)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title="Norm (Z)", row=3, col=1)

    return fig


def plot_morphology_overlay(
    beat_waveforms: np.ndarray,
    predictions: np.ndarray,
    fs: int = 360,
    before_sec: float = 0.25,
    max_beats_to_draw: int = 40,
) -> go.Figure:
    """
    Overlays segmented normal beats vs anomalous beats to visually reveal
    pathological morphological distortions (e.g. widened QRS, inverted T).
    """
    if len(beat_waveforms) == 0:
        return go.Figure()

    beat_len = beat_waveforms.shape[1]
    beat_time_ms = (np.arange(beat_len) - int(before_sec * fs)) * (1000.0 / fs)

    norm_beats = beat_waveforms[predictions == 0]
    ano_beats = beat_waveforms[predictions == 1]

    fig = go.Figure()

    # Draw individual normal beats with low opacity
    for i in range(min(len(norm_beats), max_beats_to_draw)):
        fig.add_trace(
            go.Scatter(
                x=beat_time_ms,
                y=norm_beats[i],
                mode="lines",
                line=dict(color="rgba(0, 230, 118, 0.12)", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Draw individual anomalous beats with low opacity
    for i in range(min(len(ano_beats), max_beats_to_draw)):
        fig.add_trace(
            go.Scatter(
                x=beat_time_ms,
                y=ano_beats[i],
                mode="lines",
                line=dict(color="rgba(255, 23, 68, 0.20)", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Draw median waveforms (Bold)
    if len(norm_beats) > 0:
        median_norm = np.median(norm_beats, axis=0)
        fig.add_trace(
            go.Scatter(
                x=beat_time_ms,
                y=median_norm,
                mode="lines",
                name=f"Median Normal Beat (N={len(norm_beats)})",
                line=dict(color=PALETTE["normal_beat"], width=3.5),
            )
        )

    if len(ano_beats) > 0:
        median_ano = np.median(ano_beats, axis=0)
        fig.add_trace(
            go.Scatter(
                x=beat_time_ms,
                y=median_ano,
                mode="lines",
                name=f"Median Flagged Ectopic (N={len(ano_beats)})",
                line=dict(color=PALETTE["anomaly_beat"], width=3.5, dash="dash"),
            )
        )

    fig.add_vline(x=0, line_dash="dot", line_color="#94a3b8", annotation_text="R-Peak")

    fig.update_layout(
        title="<b>Heartbeat Morphology Overlay</b> — Normal vs. Anomalous Complexes",
        xaxis=dict(title="Time relative to R-peak (ms)", gridcolor=PALETTE["grid"]),
        yaxis=dict(title="Amplitude (Normalized)", gridcolor=PALETTE["grid"]),
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["card_bg"],
        font=dict(color=PALETTE["text"], family="Inter, Roboto, sans-serif"),
        height=440,
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right"),
        margin=dict(l=50, r=30, t=50, b=40),
    )

    return fig


def plot_poincare(r_peaks: np.ndarray, predictions: np.ndarray, fs: int = 360) -> go.Figure:
    """
    Creates a Poincaré Plot (RR_n vs RR_{n+1}) highlighting autonomic HRV dispersion.
    """
    if len(r_peaks) < 3:
        return go.Figure()

    rr_ms = np.diff(r_peaks) * (1000.0 / fs)
    rr_n = rr_ms[:-1]
    rr_np1 = rr_ms[1:]
    preds = predictions[1:-1]

    fig = go.Figure()

    # Normal scatter
    norm_mask = preds == 0
    if np.any(norm_mask):
        fig.add_trace(
            go.Scatter(
                x=rr_n[norm_mask],
                y=rr_np1[norm_mask],
                mode="markers",
                name="Normal Beats",
                marker=dict(color=PALETTE["normal_beat"], size=7, opacity=0.7),
            )
        )

    # Anomalous scatter
    ano_mask = preds == 1
    if np.any(ano_mask):
        fig.add_trace(
            go.Scatter(
                x=rr_n[ano_mask],
                y=rr_np1[ano_mask],
                mode="markers",
                name="Anomalous Scatter",
                marker=dict(color=PALETTE["anomaly_beat"], size=10, symbol="diamond"),
            )
        )

    # Line of identity
    min_val = min(np.min(rr_n), np.min(rr_np1)) * 0.9
    max_val = max(np.max(rr_n), np.max(rr_np1)) * 1.1
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Line of Identity",
            line=dict(color="#64748b", dash="dot"),
        )
    )

    fig.update_layout(
        title="<b>Poincaré Plot (RRₙ vs RRₙ₊₁)</b> — HRV Geometry",
        xaxis=dict(title="RRₙ (ms)", gridcolor=PALETTE["grid"]),
        yaxis=dict(title="RRₙ₊₁ (ms)", gridcolor=PALETTE["grid"]),
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["card_bg"],
        font=dict(color=PALETTE["text"], family="Inter, Roboto, sans-serif"),
        height=400,
        margin=dict(l=50, r=30, t=50, b=40),
    )

    return fig


def plot_confusion_matrix_heatmap(cm_dict: Dict[str, int]) -> go.Figure:
    """
    Renders an elegant confusion matrix heatmap.
    """
    tn = cm_dict.get("tn", 0)
    fp = cm_dict.get("fp", 0)
    fn = cm_dict.get("fn", 0)
    tp = cm_dict.get("tp", 0)

    z = [[tn, fp], [fn, tp]]
    text = [
        [f"TN: {tn}<br>(Normal Cleared)", f"FP: {fp}<br>(False Alarm)"],
        [f"FN: {fn}<br>(Anomaly Missed)", f"TP: {tp}<br>(Anomaly Detected)"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted Normal", "Predicted Anomaly"],
            y=["Actual Normal", "Actual Anomaly"],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=14, color="#ffffff"),
            colorscale=[[0, "#1e293b"], [0.5, "#2563eb"], [1.0, "#059669"]],
            showscale=False,
        )
    )

    fig.update_layout(
        title="<b>Confusion Matrix Heatmap</b>",
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["card_bg"],
        font=dict(color=PALETTE["text"], family="Inter, Roboto, sans-serif"),
        height=350,
        margin=dict(l=60, r=40, t=50, b=40),
    )

    return fig


def plot_roc_and_pr_curves(
    roc_data: Optional[Dict[str, List[float]]],
    pr_data: Optional[Dict[str, List[float]]],
    roc_auc: float = 0.0,
    pr_auc: float = 0.0,
) -> go.Figure:
    """
    Subplots displaying the ROC Curve and Precision-Recall Curve.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"<b>ROC Curve</b> (AUC = {roc_auc:.3f})",
            f"<b>Precision-Recall Curve</b> (PR-AUC = {pr_auc:.3f})",
        ),
    )

    # ROC
    if roc_data:
        fig.add_trace(
            go.Scatter(
                x=roc_data["fpr"],
                y=roc_data["tpr"],
                mode="lines",
                name="ROC",
                line=dict(color="#00f0ff", width=2.5),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guess",
            line=dict(color="#64748b", dash="dash"),
        ),
        row=1,
        col=1,
    )

    # PR Curve
    if pr_data:
        fig.add_trace(
            go.Scatter(
                x=pr_data["recall"],
                y=pr_data["precision"],
                mode="lines",
                name="PR Curve",
                line=dict(color="#a855f7", width=2.5),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["card_bg"],
        font=dict(color=PALETTE["text"], family="Inter, Roboto, sans-serif"),
        height=380,
        showlegend=False,
        margin=dict(l=50, r=30, t=50, b=40),
    )
    fig.update_xaxes(title="False Positive Rate", gridcolor=PALETTE["grid"], row=1, col=1)
    fig.update_yaxes(title="True Positive Rate (Recall)", gridcolor=PALETTE["grid"], row=1, col=1)
    fig.update_xaxes(title="Recall", gridcolor=PALETTE["grid"], row=1, col=2)
    fig.update_yaxes(title="Precision", gridcolor=PALETTE["grid"], row=1, col=2)

    return fig
