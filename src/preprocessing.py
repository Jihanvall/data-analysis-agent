import pandas as pd

ACTION_ORDER = {
    "drop_column": 0,
    "fill_missing_mean": 1,
    "fill_missing_median": 1,
    "fill_missing_mode": 1,
    "remove_outliers": 2,
    "encode_categorical": 3,
}


def _drop_column(df, col):
    return df.drop(columns=[col])


def _fill_mean(df, col):
    df[col] = df[col].fillna(df[col].mean())
    return df


def _fill_median(df, col):
    df[col] = df[col].fillna(df[col].median())
    return df


def _fill_mode(df, col):
    mode = df[col].mode()
    if len(mode) > 0:
        df[col] = df[col].fillna(mode.iloc[0])
    return df


def _remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[col] >= lower) & (df[col] <= upper)]


def _encode_categorical(df, col):
    return pd.get_dummies(df, columns=[col], drop_first=True, dtype=int)


ACTIONS = {
    "drop_column": _drop_column,
    "fill_missing_mean": _fill_mean,
    "fill_missing_median": _fill_median,
    "fill_missing_mode": _fill_mode,
    "remove_outliers": _remove_outliers,
    "encode_categorical": _encode_categorical,
}

NUMERIC_ONLY = {"fill_missing_mean", "fill_missing_median", "remove_outliers"}


def apply_plan(df, plan, target=None):
    df = df.copy()
    log = []

    if target is not None:
        before = len(df)
        df = df.dropna(subset=[target])
        if len(df) < before:
            log.append(f"dropped {before - len(df)} rows with missing target")

    steps = [s for s in plan.get("steps", []) if s.get("action") in ACTIONS]
    steps.sort(key=lambda s: ACTION_ORDER[s["action"]])

    for step in steps:
        col = step.get("column")
        action = step["action"]

        if col not in df.columns:
            log.append(f"skipped {action} on '{col}': column not found")
            continue
        if col == target:
            log.append(f"skipped {action} on '{col}': target column is protected")
            continue
        if action in NUMERIC_ONLY and not pd.api.types.is_numeric_dtype(df[col]):
            log.append(f"skipped {action} on '{col}': column is not numeric")
            continue

        rows_before = len(df)
        df = ACTIONS[action](df, col)
        removed = rows_before - len(df)
        suffix = f" ({removed} rows removed)" if removed else ""
        log.append(f"applied {action} on '{col}'{suffix}")

    return df, log