import pandas as pd


def _target_info(series):
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
        return {
            "type": "continuous",
            "min": float(series.min()),
            "mean": round(float(series.mean()), 3),
            "max": float(series.max()),
        }
    dist = series.value_counts(normalize=True).round(3)
    return {
        "type": "categorical",
        "distribution": {str(k): float(v) for k, v in dist.items()},
    }


def _missingness_evidence(df, col, target):
    is_missing = df[col].isna()
    if is_missing.sum() == 0 or is_missing.sum() == len(df):
        return None

    target_series = df[target]
    if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 20:
        mean_missing = round(float(target_series[is_missing].mean()), 3)
        mean_present = round(float(target_series[~is_missing].mean()), 3)
        return {
            "target_mean_when_missing": mean_missing,
            "target_mean_when_present": mean_present,
        }

    dist_missing = target_series[is_missing].value_counts(normalize=True).round(3)
    dist_present = target_series[~is_missing].value_counts(normalize=True).round(3)
    return {
        "target_distribution_when_missing": {str(k): float(v) for k, v in dist_missing.items()},
        "target_distribution_when_present": {str(k): float(v) for k, v in dist_present.items()},
    }


def inspect_data(df, target=None):
    report = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": {},
    }

    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "missing_pct": round(float(df[col].isna().mean() * 100), 2),
            "n_missing": int(df[col].isna().sum()),
            "n_unique": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(3).astype(str).tolist(),
        }

        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower) | (df[col] > upper)][col]
            col_info["outlier_count"] = int(outliers.count())

        if target is not None and col != target and col_info["missing_pct"] > 0:
            evidence = _missingness_evidence(df, col, target)
            if evidence is not None:
                col_info["missingness_evidence"] = evidence

        report["columns"][col] = col_info

    if target is not None:
        report["target"] = {"column": target, **_target_info(df[target].dropna())}

    return report