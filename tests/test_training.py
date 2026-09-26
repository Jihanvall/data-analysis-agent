import numpy as np
import pandas as pd

from src.training import select_and_train, evaluate


def _classification_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
    })
    y = (X["a"] + X["b"] > 0).astype(int)
    return X, y


def _regression_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
    })
    y = 3 * X["a"] - 2 * X["b"] + rng.normal(0, 0.1, n)
    return X, y


def test_classification_returns_valid_model_name():
    X, y = _classification_data()
    model, name, scores = select_and_train(X, y, "classification")
    assert name in {"logistic_regression", "random_forest", "gradient_boosting"}
    assert set(scores.keys()) == {"logistic_regression", "random_forest", "gradient_boosting"}


def test_classification_model_can_predict():
    X, y = _classification_data()
    model, name, scores = select_and_train(X, y, "classification")
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_classification_evaluate_returns_metrics():
    X, y = _classification_data()
    model, name, scores = select_and_train(X, y, "classification")
    metrics = evaluate(model, X, y, "classification")
    assert "f1_weighted" in metrics
    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_regression_returns_valid_model_name():
    X, y = _regression_data()
    model, name, scores = select_and_train(X, y, "regression")
    assert name in {"linear_regression", "random_forest", "gradient_boosting"}


def test_regression_evaluate_returns_metrics():
    X, y = _regression_data()
    model, name, scores = select_and_train(X, y, "regression")
    metrics = evaluate(model, X, y, "regression")
    assert "r2" in metrics
    assert "mae" in metrics


def test_clustering_returns_kmeans():
    X, _ = _classification_data()
    model, name, scores = select_and_train(X, None, "clustering")
    assert name == "kmeans"
    assert scores == {}


def test_clustering_evaluate_returns_n_clusters():
    X, _ = _classification_data()
    model, name, scores = select_and_train(X, None, "clustering")
    metrics = evaluate(model, X, None, "clustering")
    assert "n_clusters" in metrics


def test_invalid_task_type_raises():
    X, y = _classification_data()
    try:
        select_and_train(X, y, "not_a_real_task")
        assert False, "expected ValueError"
    except ValueError:
        pass