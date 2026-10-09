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

def test_add_missing_indicator_marks_missing_rows():
    df = pd.DataFrame({"age": [25, None, 30, None]})
    plan = {"steps": [{"column": "age", "action": "add_missing_indicator"}]}
    fitted, _ = fit_plan(df, plan)
    out, _= transform(df, fitted)
    assert out["age_was_missing"].tolist() == [0, 1, 0, 1]

def test_indicator_runs_before_fill():
    df = pd.DataFrame({"age": [25, None, 30]})
    plan = {
        "steps": [
            {"column": "age", "action": "fill_missing_mean"},
            {"column": "age", "action": "add_missing_indicator"},
        ]
    }
    fitted, _ = fit_plan(df, plan)
    out, _ = transform(df, fitted)
    assert out["age_was_missing"].tolist() == [0, 1, 0]

def test_drop_rows_with_missing_removes_rows_in_train():
    df = pd.DataFrame({"city": ["a", None, "b", "a"]})
    plan = {"steps": [{"column": "city", "action": "drop_rows_with_missing"}]}
    fitted, _ = fit_plan(df, plan)
    out, _ = transform(df, fitted, drop_outlier_rows=True)
    assert len(out) == 3
    assert out["city"].isna().sum() == 0

def test_drop_rows_with_missing_fills_instead_in_test():
    train = pd.DataFrame({"city": ["a", "a", "b", "a"]})
    test = pd.DataFrame({"city": ["a", None]})
    plan = {"steps": [{"column": "city", "action": "drop_rows_with_missing"}]}
    fitted, _ = fit_plan(train, plan)
    out, _ = transform(test, fitted, drop_outlier_rows=False)
    assert len(out) == 2
    assert out["city"].isna().sum() == 0

DATE_PLAN = {"steps": [{"column": "d", "action": "extract_date_features", "reason": "date"}]}

def make_date_df(dates):
    return pd.DataFrame({"d": dates, "y": [0, 1, 0, 1][: len(dates)]})

def test_extract_date_features_creates_four_columns_and_drops_original():
    df = make_date_df(["2024-03-15", "2024-03-16", "2024-04-01", "2024-05-20"])
    fitted, _ = fit_plan(df, DATE_PLAN, target="y")
    out, _ = transform(df,fitted)
    assert "d" not in out.columns
    for part in ["year", "month", "day", "dayofweek"]:
        assert f"d_{part}" in out.columns
    assert out.loc[0, "d_year"] == 2024
    assert out.loc[0, "d_month"] == 3
    assert out.loc[0, "d_day"] == 15
    assert out.loc[0, "d_dayofweek"] == 4

def test_unreadable_date_uses_value_learned_from_train():
   train = make_date_df(["2024-03-15", "2024-03-20", "2024-03-25", "2024-05-01"])
   fitted, _ = fit_plan(train, DATE_PLAN, target="y")
   new = pd.DataFrame({"d": ["not a date"], "y": [0]})
   out, _ = transform(new, fitted, drop_outlier_rows=False)
   assert out.loc[0, "d_month"] == 3
   assert not out.isna().any().any()
