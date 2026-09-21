import pandas as pd 

def inspect_data(df):
    report = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": {},
    }

    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "missing_pct": round(float(df[col].isna().mean() * 100), 2),
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower) | (df[col] > upper)][col]
            col_info["outlier_count"] = int(outliers.count())
        report["columns"][col] = col_info

    return report 