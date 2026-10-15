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

TUNED_RESULT = {
    "plan": {"steps": []},
    "log": [],
    "model_name": "random_forest",
    "cv_scores": {
        "scoring_used": "f1_weighted",
        "scores": {"random_forest": 0.83, "logistic_regression": 0.71},
        "tuning": {
            "best_params": {"n_estimators": 200, "max_depth": 10},
            "score_before": 0.83,
            "score_after": 0.87,
        },
    },
    "test_metrics": {"accuracy": 0.85},
}

def test_report_shows_tuning_section_when_present():
    text = generate_report(TUNED_RESULT, "churn")
    assert "## Hyperparameter Tuning" in text
    assert "Score before: 0.83" in text
    assert "Score after: 0.87" in text
    assert "n_estimators=200" in text

def test_report_has_no_tuning_section_without_tuning():
    text = generate_report(FAKE_RESULT, "churn")
    assert "Hyperparameter Tuning" not in text

def test_report_has_no_tuning_section_when_scores_not_improved():
    result = {
        **TUNED_RESULT,
        "cv_scores": {
            "scoring_used": "f1_weighted",
            "scores": {"random_forest": 0.83},
        },
    }
    assert "Hyperparameter Tuning" not in generate_report(result, "churn")

def test_report_shows_importance_section_when_present():
    result = {
        **FAKE_RESULT,
        "feature_importance": [{"feature": "age", "importance": 0.12}],
    }
    text = generate_report(result, "churn")
    assert "## Most Influential Columns" in text
    assert "- age: 0.12" in text

def test_report_has_no_importance_section_without_data():
    assert "Most Influential Columns" not in generate_report(FAKE_RESULT, "churn")