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

def test_missingness_evidence_detects_relation_to_target():
    n= 100
    df = pd.DataFrame({
        "income": [50000] * n,
        "churn": [0] * n,
    })
    df.loc[df.index[:40], "churn"] = 1
    df.loc[df.index[:35], "income"] = None
    report = inspect_data(df, target="churn")
    evidence = report["columns"]["income"]["missingness_evidence"]
    assert evidence["target_distribution_when_missing"]["1"] > 0.8

def test_missingness_evidence_absent_when_no_missing_values():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "churn": [0, 1, 0, 1]})
    report = inspect_data(df, target="churn")
    assert "missingness_evidence" not in report["columns"]["a"]

def test_n_missing_count():
    df = pd.DataFrame({"a": [1, None, 3, None, 5]})
    report = inspect_data(df)
    assert report["columns"]["a"]["n_missing"] == 2

def test_inspection_detects_date_column():
    df = pd.DataFrame({
        "d": ["2024-03-15", "2024-03-16", "2024-04-01", "2024-05-20"],
        "y": [0, 1, 0, 1],
    })
    report = inspect_data(df, target="y")
    assert report["columns"]["d"]["looks_like_date"] is True
    assert report["columns"]["y"]["looks_like_date"] is False

def test_inspection_ignores_plain_text():
    df = pd.DataFrame({"c": ["a", "b", "c", "d"], "y": [0, 1, 0, 1]})
    assert inspect_data(df, target="y")["columns"]["c"]["looks_like_date"] is False