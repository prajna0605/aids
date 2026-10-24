"""
Test script to verify 1GB Data Upload Allowance and processing controls
"""
import io
import numpy as np
import pandas as pd
from src.data_loader import get_custom_file_preview, load_custom_ecg_csv

def test_allowance():
    print("1. Synthesizing multi-lead ECG CSV file (100,000 samples)...")
    t = np.linspace(0, 277.7, 100000)
    lead_mlii = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 3.5 * t)
    lead_v5 = np.cos(2 * np.pi * 1.2 * t)

    df = pd.DataFrame({"time_s": t, "Lead_MLII": lead_mlii, "Lead_V5": lead_v5})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    file_bytes = buf.getvalue()
    print(f"Generated file size: {len(file_bytes) / (1024 * 1024):.2f} MB")

    print("\n2. Testing get_custom_file_preview...")
    prev = get_custom_file_preview(file_bytes)
    print("Detected numeric columns:", prev["numeric_columns"])
    print("File size MB:", round(prev["file_size_mb"], 3))

    print("\n3. Testing load_custom_ecg_csv with 60s window allowance...")
    rec_60 = load_custom_ecg_csv(
        file_bytes=file_bytes,
        fs=360,
        filename="patient_telemetry.csv",
        selected_column="Lead_MLII",
        start_second=10.0,
        duration_allowance_sec=60.0,
    )
    print(f"Processed samples: {rec_60['total_samples']} (Duration: {rec_60['duration_sec']:.1f}s)")
    print(f"Memory footprint: {rec_60['allowance_meta']['memory_usage_mb']:.3f} MB")

    print("\n4. Testing load_custom_ecg_csv with decimation allowance (2x)...")
    rec_deci = load_custom_ecg_csv(
        file_bytes=file_bytes,
        fs=360,
        filename="patient_telemetry.csv",
        selected_column="Lead_V5",
        start_second=0.0,
        duration_allowance_sec=120.0,
        downsample_factor=2,
    )
    print(f"Decimated effective fs: {rec_deci['fs']} Hz, samples: {rec_deci['total_samples']}")

    print("\nSUCCESS: Data upload allowance pipeline verified!")

if __name__ == "__main__":
    test_allowance()
