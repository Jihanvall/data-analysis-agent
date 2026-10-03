import joblib
import numpy as np
import pandas as pd 
import pytest 

import src.pipeline as pipeline_module
from src.model_io import REQUIRED_KEYS, load_bundle, save_bundle

PLAN = {
    "steps": [
        {"column": "a", "action": "fill_missing_mean", "reason": "test"},
        {"column": "city", "action": "encode_categorical", "reason": "test"},
    ]
}

def _make_csv(folder):
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "city": rng.choice(["x", "y", "z"], n),
        "target": rng.integers(0, 2, n),
    })
    df.loc[df.sample(6, random_state=1).index, "a"] = None
    path = folder / "data.csv"
    df.to_csv(path, index=False)
    return str(path)

@pytest.fixture(scope="module")
def result(tmp_path_factory):
    path = _make_csv(tmp_path_factory.mktemp("data"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            pipeline_module,
            "get_preprocessing_plan",
            lambda report, target, task_type: PLAN,
        )
        return pipeline_module.run_pipeline(path, target="target", task_type="classification")

def test_bundle_has_all_required_keys(result):
    assert REQUIRED_KEYS <= set(result["bundle"])

def test_bundle_records_feature_and_input_columns(result):
    bundle = result["bundle"]
    assert "target" not in bundle["feature_columns"]
    assert "target" not in bundle["input_columns"]
    assert set(bundle["input_columns"]) == {"a", "city"}
    assert any(c.startswith("city_") for c in bundle["feature_columns"])

def test_save_and_load_round_trip(result, tmp_path):
    path = save_bundle(result["bundle"], tmp_path/ "model.joblib")
    loaded = load_bundle(path)

    assert loaded["model_name"] == result["bundle"]["model_name"]
    assert loaded["feature_columns"] == result["bundle"]["feature_columns"]

    X = result["train_data"].drop(columns=["target"])
    expected = result["bundle"]["model"].predict(X)
    actual = loaded["model"].predict(X)
    assert (expected == actual).all()

def test_loaded_bundle_keeps_learned_parameters(result, tmp_path):
    path = save_bundle(result["bundle"], tmp_path / "model.joblib")
    loaded = load_bundle(path)
    params = loaded["fitted"]["params"]
    assert ("a", "fill_missing_mean") in params
    assert ("city", "encode_categorical") in params

def test_load_rejects_invalid_bundle(tmp_path):
    path = tmp_path / "bad.joblib"
    joblib.dump({"model": "x"}, path)
    with pytest.raises(ValueError):
        load_bundle(path)

def test_load_warns_on_sklearn_version_mismatch(result, tmp_path):
    bundle = dict(result["bundle"])
    bundle["sklearn_version"] = "0.0.1"
    path = save_bundle(bundle, tmp_path / "old.joblib")
    with pytest.warns(UserWarning):
        load_bundle(path)