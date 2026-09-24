import time

import pandas as pd
from sklearn.model_selection import train_test_split

from src.inspection import inspect_data
from src.decision import get_preprocessing_plan
from src.preprocessing import fit_plan, transform
from src.training import select_and_train, evaluate


def get_plan_with_retry(report, target, task_type, max_attempts=3, wait_seconds=5):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return get_preprocessing_plan(report, target, task_type)
        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                time.sleep(wait_seconds)
    raise RuntimeError(
        f"Could not get a preprocessing plan from Gemini after {max_attempts} attempts: {last_error}"
    )


def run_pipeline(csv_path, target, task_type, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)

    stratify = df[target] if task_type == "classification" else None
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )

    report = inspect_data(train, target=target)
    plan = get_plan_with_retry(report, target, task_type)

    fitted, fit_log = fit_plan(train, plan, target)
    train_out, train_log = transform(train, fitted)
    test_out, test_log = transform(test, fitted, drop_outlier_rows=False)

    X_train = train_out.drop(columns=[target])
    y_train = train_out[target]
    X_test = test_out.drop(columns=[target])
    y_test = test_out[target]

    model, model_name, cv_scores = select_and_train(X_train, y_train, task_type)
    metrics = evaluate(model, X_test, y_test, task_type)

    return {
        "model": model,
        "model_name": model_name,
        "cv_scores": cv_scores,
        "test_metrics": metrics,
        "plan": plan,
        "log": fit_log + train_log + test_log,
    }