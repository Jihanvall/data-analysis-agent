import pandas as pd

ACTION_ORDER = {
    "drop_column": 0,
    "fill_missing_mean": 1,
    "fill_missing_median": 1,
    "fill_missing_mode": 1,
    "remove_outliers": 2,
    "encode_categorical": 3,
}

NUMERIC_ONLY = {"fill_missing_mean", "fill_missing_median", "remove_outliers"}


def fit_plan(df, plan, target=None):
    """Learn parameters (fill values, outlier bounds, categories) from training data only."""
    params = {}
    log = []

    steps = [s for s in plan.get("steps", []) if s.get("action") in ACTION_ORDER]
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

        if action == "drop_column":
            params[(col, action)] = True
        elif action == "fill_missing_mean":
            params[(col, action)] = float(df[col].mean())
        elif action == "fill_missing_median":
            params[(col, action)] = float(df[col].median())
        elif action == "fill_missing_mode":
            mode = df[col].mode()
            params[(col, action)] = mode.iloc[0] if len(mode) > 0 else None
        elif action == "remove_outliers":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            params[(col, action)] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        elif action == "encode_categorical":
            params[(col, action)] = sorted(df[col].dropna().unique().tolist())

        log.append(f"learned {action} on '{col}'")

    return {"steps": steps, "params": params}, log


def transform(df, fitted, drop_outlier_rows=True):
    """Apply previously learned parameters to any DataFrame (train, test, or new data)."""
    df = df.copy()
    log = []

    for step in fitted["steps"]:
        col = step.get("column")
        action = step["action"]
        key = (col, action)

        if key not in fitted["params"] or col not in df.columns:
            log.append(f"skipped {action} on '{col}': not learned or column missing")
            continue

        value = fitted["params"][key]

        if action == "drop_column":
            df = df.drop(columns=[col])
        elif action in ("fill_missing_mean", "fill_missing_median"):
            df[col] = df[col].fillna(value)
        elif action == "fill_missing_mode":
            if value is not None:
                df[col] = df[col].fillna(value)
        elif action == "remove_outliers":
            if drop_outlier_rows:
                lower, upper = value
                rows_before = len(df)
                df = df[(df[col] >= lower) & (df[col] <= upper)]
                log.append(f"applied {action} on '{col}' ({rows_before - len(df)} rows removed)")
                continue
            else:
                lower, upper = value
                df[col] = df[col].clip(lower, upper)
        elif action == "encode_categorical":
            known = fitted["params"][key]
            df[col] = pd.Categorical(df[col], categories=known)
            df = pd.get_dummies(df, columns=[col], drop_first=True, dtype=int)

        log.append(f"applied {action} on '{col}'")

    return df, log