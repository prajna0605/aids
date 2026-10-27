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
    assert b"Upload" in response.data


def test_health_api(client):
    """Verifies that the API healthcheck returns 200 and correct configuration."""
    response = client.get("/api/health")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "healthy"
    assert json_data["max_upload_allowance_mb"] == 1024


def test_favicon(client):
    """Verifies that the favicon endpoint returns 200 OK without 404 error."""
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    assert "image" in response.content_type


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


def test_analyze_uploaded_mitbih_file(client):
    """Verifies end-to-end inference on uploaded mitbih_test.csv slice."""
    with open("data/mitbih_test.csv", "rb") as f:
        file_bytes = f.read(100 * 1024)  # First 100 KB
    data = {
        "custom_file": (io.BytesIO(file_bytes), "mitbih_test.csv"),
        "custom_fs": "360",
        "window_duration": "20",
        "decision_threshold": "50",
    }
    response = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"Telemetry Analysis" in response.data
    assert b"Analyzed Heartbeats" in response.data


def test_analyze_no_file_redirect(client):
    """Verifies that submitting without a file redirects back to home with an alert."""
    response = client.post("/analyze", data={}, follow_redirects=True)
    assert response.status_code == 200
    assert b"Please select an ECG file" in response.data or b"No file" in response.data


def test_analyze_custom_upload_allowance(client):
    """Verifies custom file upload within the 1,024 MB allowance using real ECG data."""
    with open("data/mitbih_test.csv", "rb") as f:
        file_bytes = f.read(50 * 1024)
    csv_bytes = io.BytesIO(file_bytes)

    data = {
        "custom_file": (csv_bytes, "patient_upload.csv"),
        "custom_fs": "360",
        "window_duration": "10",
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


def test_analyze_preset_mode(client):
    """Verifies 1-click analysis on preset demo record 119."""
    data = {"dataset_mode": "preset", "decision_threshold": "50"}
    response = client.post("/analyze", data=data)
    assert response.status_code == 200
    assert b"Telemetry Analysis" in response.data
    assert b"MIT-BIH_Record_119_Bigeminy" in response.data


def test_analyze_synthetic_mode(client):
    """Verifies analysis on mathematically generated synthetic ECG."""
    data = {"dataset_mode": "synthetic", "window_duration": "30", "decision_threshold": "50"}
    response = client.post("/analyze", data=data)
    assert response.status_code == 200
    assert b"Telemetry Analysis" in response.data
    assert b"SYNTH" in response.data or b"Synthetic" in response.data


def test_download_clinical_pdf(client):
    """Verifies that the clinical PDF report is generated and served with application/pdf mimetype."""
    response = client.get("/report/benchmark_119/pdf")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/pdf"
    assert b"%PDF" in response.data[:10]
    assert len(response.data) > 10_000



