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

def _analyze(client, sample_csv):
    response = client.post(
        "/analyze",
        files={"file": ("data.csv", sample_csv, "text/csv")},
        data={"target": "target", "task_type": "classification"},
    )
    assert response.status_code == 200
    return response.json()


def _new_customers_csv(rows=3, **overrides):
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0][:rows],
    })
    for column, values in overrides.items():
        df[column] = values
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def test_analyze_returns_required_columns_and_prediction_column(client, sample_csv):
    body = _analyze(client, sample_csv)
    assert body["required_columns"] == ["a"]
    assert body["prediction_column"] == "predicted_target"


def test_predict_returns_preview_and_prediction_id(client, sample_csv):
    session_id = _analyze(client, sample_csv)["session_id"]

    response = client.post(
        "/predict",
        data={"session_id": session_id},
        files={"file": ("new.csv", _new_customers_csv(), "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["n_rows"] == 3
    assert body["prediction_column"] == "predicted_target"
    assert len(body["preview"]) == 3
    assert "predicted_target" in body["preview"][0]
    assert body["warnings"] == []
    assert "prediction_id" in body


def test_predict_unknown_session_returns_404(client):
    response = client.post(
        "/predict",
        data={"session_id": "does-not-exist"},
        files={"file": ("new.csv", _new_customers_csv(), "text/csv")},
    )
    assert response.status_code == 404


def test_predict_missing_required_column_returns_400(client, sample_csv):
    session_id = _analyze(client, sample_csv)["session_id"]
    wrong = io.BytesIO(b"other\n1\n2\n")

    response = client.post(
        "/predict",
        data={"session_id": session_id},
        files={"file": ("new.csv", wrong, "text/csv")},
    )

    assert response.status_code == 400
    assert "a" in response.json()["detail"]


def test_predict_empty_file_returns_400(client, sample_csv):
    session_id = _analyze(client, sample_csv)["session_id"]
    empty = io.BytesIO(b"a\n")

    response = client.post(
        "/predict",
        data={"session_id": session_id},
        files={"file": ("new.csv", empty, "text/csv")},
    )

    assert response.status_code == 400


def test_predict_ignores_extra_target_column_with_warning(client, sample_csv):
    session_id = _analyze(client, sample_csv)["session_id"]

    response = client.post(
        "/predict",
        data={"session_id": session_id},
        files={"file": ("new.csv", _new_customers_csv(target=[0, 1, 0]), "text/csv")},
    )

    assert response.status_code == 200
    assert any("target" in w for w in response.json()["warnings"])


def test_download_predictions_unknown_id_returns_404(client):
    response = client.get("/download-predictions/does-not-exist")
    assert response.status_code == 404


def test_predict_then_download_round_trip(client, sample_csv):
    session_id = _analyze(client, sample_csv)["session_id"]

    predict_response = client.post(
        "/predict",
        data={"session_id": session_id},
        files={"file": ("new.csv", _new_customers_csv(), "text/csv")},
    )
    prediction_id = predict_response.json()["prediction_id"]

    download = client.get(f"/download-predictions/{prediction_id}")
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("text/csv")

    downloaded = pd.read_csv(io.BytesIO(download.content))
    assert len(downloaded) == 3
    assert "predicted_target" in downloaded.columns

def test_analyze_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_MB", 0)
    response = client.post(
        "/analyze",
        files={"file": ("big.csv", b"a,b\n1,2\n", "text/csv")},
        data={"target": "b", "task_type": "classification"},
    )
    assert response.status_code == 413
    assert "too large" in response.json()["detail"]

def test_predict_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_MB", 0)
    response = client.post(
        "/predict",
        data={"session_id": "anything"},
        files={"file": ("big.csv", b"a,b\n1,2\n", "text/csv")},
    )
    assert "too large" in response.json()["detail"]