import pandas as pd
import pytest

from src.inspection import inspect_data
from src.decision import get_preprocessing_plan, ALLOWED_ACTIONS


@pytest.fixture(scope="module")
def sample_report():
    df = pd.DataFrame({
        "id": range(20),
        "age": [25, 30, None, 200] + [35] * 16,
        "city": ["a", "b"] * 10,
        "churn": [0, 1] * 10,
    })
    return inspect_data(df, target="churn")


@pytest.fixture(scope="module")
def sample_plan(sample_report):
    return get_preprocessing_plan(sample_report, "churn", "classification")


def test_plan_has_steps_key(sample_plan):
    assert "steps" in sample_plan
    assert isinstance(sample_plan["steps"], list)


def test_all_actions_are_allowed(sample_plan):
    for step in sample_plan["steps"]:
        assert step["action"] in ALLOWED_ACTIONS


def test_every_step_has_required_fields(sample_plan):
    for step in sample_plan["steps"]:
        assert "column" in step
        assert "action" in step
        assert "reason" in step


def test_target_column_not_dropped_or_outlier_removed(sample_plan):
    for step in sample_plan["steps"]:
        if step["column"] == "churn":
            assert step["action"] not in {"drop_column", "remove_outliers"}


def test_invalid_task_type_raises(sample_report):
    with pytest.raises(ValueError):
        get_preprocessing_plan(sample_report, "churn", "not_a_real_task")