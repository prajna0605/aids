"""
Unit and Integration Tests for ECG Insight AI Flask Application
"""

import os
import sys
import io
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app
from src.train import load_trained_model


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_page(client):
    """Verifies that the home page loads successfully with HTML content."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"ECG Insight AI" in response.data
    assert b"PhysioNet MIT-BIH Database" in response.data


def test_health_api(client):
    """Verifies that the API healthcheck returns 200 and correct configuration."""
    response = client.get("/api/health")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "healthy"
    assert json_data["max_upload_allowance_mb"] == 1024


def test_evaluation_page(client):
    """Verifies that the evaluation and benchmark page loads with metrics."""
    response = client.get("/evaluation")
    assert response.status_code == 200
    assert b"Model Evaluation &amp; Benchmark Performance" in response.data or b"Model Evaluation" in response.data
    assert b"Confusion Matrix Heatmap" in response.data


def test_export_report_endpoint(client):
    """Verifies that the markdown placement portfolio report can be downloaded."""
    response = client.get("/api/export_report")
    assert response.status_code == 200
    assert b"ECG Insight AI" in response.data
    assert response.mimetype == "text/markdown"


def test_analyze_synthetic(client):
    """Verifies end-to-end inference on synthetic ECG simulation."""
    payload = {
        "dataset_mode": "synthetic",
        "synth_hr": "75",
        "synth_duration": "20",
        "synth_ano_prob": "15",
        "synth_noise": "3",
        "decision_threshold": "50",
        "filter_bandpass": "0.5-40",
        "apply_notch": "true",
    }
    response = client.post("/analyze", data=payload)
    assert response.status_code == 200
    assert b"Telemetry Analysis" in response.data
    assert b"Analyzed Heartbeats" in response.data


def test_analyze_preset_record(client):
    """Verifies end-to-end inference on pre-cached MIT-BIH Record 100."""
    payload = {
        "dataset_mode": "preset",
        "preset_record": "100",
        "preset_duration": "20",
        "preset_lead": "0",
        "decision_threshold": "50",
        "filter_bandpass": "0.5-40",
        "apply_notch": "true",
    }
    response = client.post("/analyze", data=payload)
    assert response.status_code == 200
    assert b"Record 100" in response.data


def test_analyze_custom_upload_allowance(client):
    """Verifies custom file upload within the 1,024 MB allowance."""
    from src.data_loader import generate_synthetic_ecg
    synth = generate_synthetic_ecg(duration_sec=15, fs=360, anomaly_prob=0.15)
    df = pd.DataFrame({"Lead_MLII": synth["signal"]})
    csv_bytes = io.BytesIO()
    df.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)

    data = {
        "dataset_mode": "upload",
        "custom_file": (csv_bytes, "patient_upload.csv"),
        "custom_fs": "360",
        "allowance_mode": "window",
        "window_duration": "10",
        "window_offset": "0",
        "decision_threshold": "50",
    }

    response = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"patient_upload.csv" in response.data


def test_model_joblib_integrity():
    """Verifies that the serialized model bundle exists and contains required keys."""
    assert Path("models/model.joblib").exists()
    bundle = load_trained_model("models/model.joblib")
    assert "model" in bundle
    assert "scaler" in bundle
    assert "feature_names" in bundle
    assert len(bundle["feature_names"]) == 30
