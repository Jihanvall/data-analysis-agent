import pandas as pd

from src.preprocessing import transform


def required_columns(bundle):
    dropped = {
        col for (col, action) in bundle["fitted"]["params"] if action == "drop_column"
    }
    return [c for c in bundle["input_columns"] if c not in dropped]


def prediction_column_name(bundle):
    if bundle["task_type"] == "clustering":
        return "cluster"
    return f"predicted_{bundle['target']}"


def _find_unseen_categories(bundle, df):
    unseen = {}
    affected = pd.Series(False, index=df.index)

    for (col, action), known in bundle["fitted"]["params"].items():
        if action != "encode_categorical" or col not in df.columns:
            continue
        is_new = df[col].notna() & ~df[col].isin(known)
        if is_new.any():
            unseen[col] = sorted({str(v) for v in df.loc[is_new, col]})
            affected |= is_new

    return unseen, int(affected.sum())


def predict_with_bundle(bundle, new_df):
    df = new_df.reset_index(drop=True)
    target = bundle["target"]
    warnings = []

    missing = [c for c in required_columns(bundle) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if target is not None and target in df.columns:
        warnings.append(
            f"Column '{target}' is the training target, so it was ignored for prediction."
        )

    known_inputs = set(bundle["input_columns"])
    extra = [c for c in df.columns if c not in known_inputs and c != target]
    if extra:
        warnings.append(f"Columns not used by the model were ignored: {extra}")

    unseen, n_rows = _find_unseen_categories(bundle, df)
    if unseen:
        warnings.append(
            f"{n_rows} row(s) contain values the model did not see during training {unseen}; "
            "their predictions are less reliable."
        )

    processed, _ = transform(df, bundle["fitted"], drop_outlier_rows=False)

    absent = [c for c in bundle["feature_columns"] if c not in processed.columns]
    if absent:
        raise ValueError(f"Preprocessing did not produce these model columns: {absent}")

    X = processed[bundle["feature_columns"]]

    still_missing = X.columns[X.isna().any()].tolist()
    if still_missing:
        raise ValueError(
            f"These columns contain missing values that the saved preprocessing cannot fill: {still_missing}"
        )

    out = df.copy()
    out[prediction_column_name(bundle)] = bundle["model"].predict(X)
    return out, warnings