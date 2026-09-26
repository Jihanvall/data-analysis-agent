import pandas as pd

from src.preprocessing import fit_plan, transform


def test_drop_column():
    df = pd.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30]})
    plan = {"steps": [{"column": "id", "action": "drop_column"}]}
    fitted, _ = fit_plan(df, plan)
    out, _ = transform(df, fitted)
    assert "id" not in out.columns


def test_fill_missing_median():
    df = pd.DataFrame({"a": [1, 2, None, 100]})
    plan = {"steps": [{"column": "a", "action": "fill_missing_median"}]}
    fitted, _ = fit_plan(df, plan)
    out, _ = transform(df, fitted)
    assert out["a"].isna().sum() == 0


def test_target_column_protected():
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    plan = {"steps": [{"column": "target", "action": "drop_column"}]}
    fitted, _ = fit_plan(df, plan, target="target")
    out, _ = transform(df, fitted)
    assert "target" in out.columns


def test_unknown_column_skipped():
    df = pd.DataFrame({"a": [1, 2, 3]})
    plan = {"steps": [{"column": "ghost", "action": "drop_column"}]}
    fitted, _ = fit_plan(df, plan)
    out, _ = transform(df, fitted)
    assert list(out.columns) == ["a"]


def test_encode_categorical_consistent_columns():
    train = pd.DataFrame({"city": ["a", "b", "a", "b"]})
    test = pd.DataFrame({"city": ["a", "a"]})
    plan = {"steps": [{"column": "city", "action": "encode_categorical"}]}
    fitted, _ = fit_plan(train, plan)
    train_out, _ = transform(train, fitted)
    test_out, _ = transform(test, fitted)
    assert list(train_out.columns) == list(test_out.columns)


def test_remove_outliers_train_drops_rows():
    df = pd.DataFrame({"a": [10, 11, 12, 13, 1000]})
    plan = {"steps": [{"column": "a", "action": "remove_outliers"}]}
    fitted, _ = fit_plan(df, plan)
    out, _ = transform(df, fitted, drop_outlier_rows=True)
    assert len(out) < len(df)


def test_remove_outliers_test_clips_instead_of_dropping():
    df = pd.DataFrame({"a": [10, 11, 12, 13, 1000]})
    plan = {"steps": [{"column": "a", "action": "remove_outliers"}]}
    fitted, _ = fit_plan(df, plan)
    out, _ = transform(df, fitted, drop_outlier_rows=False)
    assert len(out) == len(df)