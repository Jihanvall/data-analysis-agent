import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api


FAKE_PLAN = {
    "steps": [
        {"column": "a", "action": "fill_missing_mean", "reason": "test plan"},
        {"column": "target", "action": "none", "reason": "target column"},
    ]
}


@pytest.fixture(autouse=True)
def mock_gemini(monkeypatch):
    monkeypatch.setattr("src.pipeline.get_preprocessing_plan", lambda report, target, task_type: FAKE_PLAN)


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture
def sample_csv():
    df = pd.DataFrame({
        "a": [1, 2, None, 4, 5, 6, 7, 8, 9, 10] * 3,
        "target": [0, 1] * 15,
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_returns_session_id(client, sample_csv):
    response = client.post(
        "/analyze",
        files={"file": ("data.csv", sample_csv, "text/csv")},
        data={"target": "target", "task_type": "classification"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "session_id" in body
    assert "model_name" in body
    assert "inspection_report" in body


def test_analyze_invalid_task_type(client, sample_csv):
    response = client.post(
        "/analyze",
        files={"file": ("data.csv", sample_csv, "text/csv")},
        data={"target": "target", "task_type": "banana"},
    )
    assert response.status_code == 400


def test_analyze_missing_target_for_classification(client, sample_csv):
    response = client.post(
        "/analyze",
        files={"file": ("data.csv", sample_csv, "text/csv")},
        data={"task_type": "classification"},
    )
    assert response.status_code == 400


def test_download_unknown_session_returns_404(client):
    response = client.get("/download/does-not-exist")
    assert response.status_code == 404


def test_full_round_trip_download_matches_session(client, sample_csv):
    analyze_response = client.post(
        "/analyze",
        files={"file": ("data.csv", sample_csv, "text/csv")},
        data={"target": "target", "task_type": "classification"},
    )
    session_id = analyze_response.json()["session_id"]

    download_response = client.get(f"/download/{session_id}")
    assert download_response.status_code == 200
    assert download_response.headers["content-type"].startswith("text/csv")

    downloaded_df = pd.read_csv(io.BytesIO(download_response.content))
    assert "target" in downloaded_df.columns
    assert len(downloaded_df) > 0