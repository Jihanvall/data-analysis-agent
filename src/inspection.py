import pandas as pd 


def _target_info(series):
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
        return {
            "type": "continious",
            "min": float(series.min()),
            "mean": round(float(series.mean()), 3),
            "max": float(series.max()),
        }
    dist = series.value_counts(normalize=True).round(3)
    return {
        "type": "categorical",
        "distribution": {str(k): float(v) for k, v in dist.items()},
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

        report["columns"][col] = col_info

    if target is not None:
        report["target"] = {"column": target, **_target_info(df[target].dropna())}

    return report