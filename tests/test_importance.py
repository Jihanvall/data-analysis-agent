import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier

from src.importance import compute_importance

def _data(n=200):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "signal": rng.normal(size=n),
        "noise": rng.normal(size=n),
    })
    y = (X["signal"] > 0).astype(int)
    return X, y

def test_signal_column_ranks_first():
    X, y = _data()
    model = RandomForestClassifier(random_state=0).fit(X, y)
    result = compute_importance(model, X, y, "classification")
    assert result[0]["feature"] == "signal"
    assert result[0]["importance"] > 0

def test_clustering_returns_empty_list():
    X, _ = _data()
    assert compute_importance(None, X, None, "clustering") == []