import numpy as np
import pandas as pd
import pytest

import src.pipeline as pipeline_module
from src.predict import predict_with_bundle

PLAN = {
    "steps": [
        {"column": "id", "action": "drop_column", "reason": "test"},
        {"column": "a", "action": "add_missing_indicator", "reason": "test"},
        {"column": "a", "action": "fill_missing_mean", "reason": "test"},
        {"column": "city", "action": "encode_categorical", "reason": "test"},
    ]
}


def _make_df():
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "id": range(n),
        "a": rng.normal(0, 1, n),
        "b": rng.normal(10, 2, n),
        "city": rng.choice(["x", "y", "z"], n),
        "target": rng.integers(0, 2, n),
    })
    df.loc[df.sample(6, random_state=1).index, "a"] = None
    return df


def _new_rows(**overrides):
    df = pd.DataFrame({
        "id": [101, 102, 103],
        "a": [0.5, -0.2, 1.1],
        "b": [9.0, 11.0, 10.5],
        "city": ["x", "y", "z"],
    })
    for column, values in overrides.items():
        df[column] = values
    return df


def _train(folder, df, target, task_type):
    path = folder / "data.csv"
    df.to_csv(path, index=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            pipeline_module,
            "get_preprocessing_plan",
            lambda report, target, task_type: PLAN,
        )
        return pipeline_module.run_pipeline(str(path), target=target, task_type=task_type)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    folder = tmp_path_factory.mktemp("classification")
    return _train(folder, _make_df(), "target", "classification")


@pytest.fixture(scope="module")
def trained_clustering(tmp_path_factory):
    folder = tmp_path_factory.mktemp("clustering")
    return _train(folder, _make_df().drop(columns=["target"]), None, "clustering")


def test_adds_prediction_column(trained):
    out, warnings = predict_with_bundle(trained["bundle"], _new_rows())
    assert "predicted_target" in out.columns
    assert len(out) == 3
    assert warnings == []


def test_fills_missing_values_using_saved_parameters(trained):
    out, _ = predict_with_bundle(trained["bundle"], _new_rows(a=[None, 0.1, 0.2]))
    assert len(out) == 3
    assert out["predicted_target"].notna().all()


def test_dropped_columns_are_not_required(trained):
    out, warnings = predict_with_bundle(trained["bundle"], _new_rows().drop(columns=["id"]))
    assert len(out) == 3
    assert warnings == []


def test_missing_required_column_raises(trained):
    with pytest.raises(ValueError, match="city"):
        predict_with_bundle(trained["bundle"], _new_rows().drop(columns=["city"]))


def test_target_column_is_ignored_with_warning(trained):
    out, warnings = predict_with_bundle(trained["bundle"], _new_rows(target=[0, 1, 0]))
    assert any("target" in w for w in warnings)
    assert out["target"].tolist() == [0, 1, 0]
    assert "predicted_target" in out.columns


def test_unseen_category_is_reported_but_still_predicted(trained):
    out, warnings = predict_with_bundle(trained["bundle"], _new_rows(city=["w", "x", "y"]))
    assert len(out) == 3
    assert any("['w']" in w for w in warnings)


def test_extra_columns_are_ignored_with_warning(trained):
    out, warnings = predict_with_bundle(trained["bundle"], _new_rows(notes=["p", "q", "r"]))
    assert len(out) == 3
    assert any("notes" in w for w in warnings)


def test_missing_value_without_saved_fill_raises(trained):
    with pytest.raises(ValueError, match="missing values"):
        predict_with_bundle(trained["bundle"], _new_rows(b=[None, 10.0, 11.0]))


def test_matches_training_time_preprocessing(trained):
    original = _make_df().drop(columns=["target"])
    out, _ = predict_with_bundle(trained["bundle"], original)

    X_train = trained["train_data"].drop(columns=["target"])
    expected = trained["bundle"]["model"].predict(X_train)
    actual = out.loc[X_train.index, "predicted_target"].to_numpy()
    assert (expected == actual).all()


def test_clustering_returns_cluster_column(trained_clustering):
    out, _ = predict_with_bundle(trained_clustering["bundle"], _new_rows())
    assert "cluster" in out.columns
    assert len(out) == 3