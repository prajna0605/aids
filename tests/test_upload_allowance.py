"""
Unit & Integration Tests: High-Capacity Data Upload Allowance & Processing Controls
"""
import io
import numpy as np
import pandas as pd
import pytest
from src.data_loader import get_custom_file_preview, load_custom_ecg_csv


def test_custom_ecg_upload_allowance():
    """Verify large CSV allowance, column detection, window slicing, and decimation."""
    t = np.linspace(0, 100.0, 36000)
    lead_mlii = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 3.5 * t)
    lead_v5 = np.cos(2 * np.pi * 1.2 * t)

    df = pd.DataFrame({"time_s": t, "Lead_MLII": lead_mlii, "Lead_V5": lead_v5})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    file_bytes = buf.getvalue()

    # 1. Preview
    prev = get_custom_file_preview(file_bytes)
    assert "Lead_MLII" in prev["numeric_columns"]
    assert "Lead_V5" in prev["numeric_columns"]
    assert prev["file_size_mb"] > 0

    # 2. Window allowance slicing (30 seconds)
    rec_30 = load_custom_ecg_csv(
        file_bytes=file_bytes,
        fs=360,
        filename="patient_telemetry.csv",
        selected_column="Lead_MLII",
        start_second=10.0,
        duration_allowance_sec=30.0,
    )
    assert rec_30["duration_sec"] <= 30.5
    assert rec_30["total_samples"] > 0
    assert rec_30["allowance_meta"]["memory_usage_mb"] > 0

    # 3. Decimation allowance (2x downsampling)
    rec_deci = load_custom_ecg_csv(
        file_bytes=file_bytes,
        fs=360,
        filename="patient_telemetry.csv",
        selected_column="Lead_V5",
        start_second=0.0,
        duration_allowance_sec=60.0,
        downsample_factor=2,
    )
    assert rec_deci["fs"] == 180  # 360 / 2
    assert rec_deci["total_samples"] > 0
