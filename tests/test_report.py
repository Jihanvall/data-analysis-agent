import os

from src.report import generate_report, save_report

FAKE_RESULT = {
    "plan": {"steps": [{"column": "id", "action": "drop_column", "reason": "Identifier"}]},
    "log": ["applied drop_column on 'id'"],
    "model_name": "random_forest",
    "cv_scores": {"random_forest": 0.83, "logistic_regression": 0.71},
    "test_metrics": {"accuracy": 0.85},
}


def test_report_contains_model_name():
    text = generate_report(FAKE_RESULT, "churn")
    assert "random_forest" in text


def test_report_marks_best_model():
    text = generate_report(FAKE_RESULT, "churn")
    assert "random_forest: 0.83 ✅" in text


def test_report_contains_metrics():
    text = generate_report(FAKE_RESULT, "churn")
    assert "accuracy" in text


def test_save_report_creates_file(tmp_path):
    text = generate_report(FAKE_RESULT, "churn")
    path = save_report(text, "churn", folder=str(tmp_path))
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        assert f.read() == text