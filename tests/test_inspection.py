import pandas as pd

from src.inspection import inspect_data


def test_basic_counts():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "z", "w"]})
    report = inspect_data(df)
    assert report["n_rows"] == 4
    assert report["n_columns"] == 2


def test_missing_pct():
    df = pd.DataFrame({"a": [1, 2, None, 4]})
    report = inspect_data(df)
    assert report["columns"]["a"]["missing_pct"] == 25.0
    assert isinstance(report["columns"]["a"]["missing_pct"], float)


def test_no_missing_gives_zero():
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    report = inspect_data(df)
    assert report["columns"]["a"]["missing_pct"] == 0.0


def test_outlier_count_numeric_column():
    df = pd.DataFrame({"a": [10, 11, 12, 13, 1000]})
    report = inspect_data(df)
    assert report["columns"]["a"]["outlier_count"] >= 1


def test_no_outlier_key_for_text_column():
    df = pd.DataFrame({"a": ["x", "y", "z"]})
    report = inspect_data(df)
    assert "outlier_count" not in report["columns"]["a"]


def test_target_info_included():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    report = inspect_data(df, target="target")
    assert "target" in report
    assert report["target"]["column"] == "target"


def test_sample_values_present():
    df = pd.DataFrame({"a": [1, 2, 3]})
    report = inspect_data(df)
    assert len(report["columns"]["a"]["sample_values"]) <= 3