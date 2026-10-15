from sklearn.inspection import permutation_importance

MAX_ROWS = 300
TOP_N = 10

def compute_importance(model, X_test, y_test, task_type):
    if task_type == "clustering" or y_test is None:
        return []

    scoring = "f1_weighted" if task_type == "classification" else "r2"
    try:
        X = X_test.reset_index(drop=True)
        y = y_test.reset_index(drop=True)
        X = X.sample(n=min(MAX_ROWS, len(X)), random_state=42)
        y = y.loc[X.index]
        result = permutation_importance(
            model, X, y, scoring=scoring, n_repeats=3, random_state=42
        )
    except Exception:
        return []

    pairs = sorted(
        zip(X.columns, result.importances_mean), key=lambda p: p[1], reverse=True
    )
    return [
        {"feature": str(name), "importance": round(float(value), 4)}
        for name, value in pairs[:TOP_N]
    ]