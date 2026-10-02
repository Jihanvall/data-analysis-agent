import numpy as np
import pandas as pd 
import pytest 

import src.pipeline as pipeline_module

FAKE_PLAN = {"steps": []}

@pytest.fixture(autouse=True)
def mock_gemini(monkeypatch):
    monkeypatch.setattr(
        pipeline_module, "get_preprocessing_plan", lambda report, target, task_type: FAKE_PLAN
    )

def test_run_pipeline_drops_rows_with_missing_target(tmp_path):
    n = 60
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "target": rng.normal(50, 10, n),
    })
    df.loc[df.sample(5, random_state=1).index, "target"] = None

    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    result = pipeline_module.run_pipeline(str(csv_path), target="target", task_type="regression")

    assert result["train_data"]["target"].isna().sum() == 0
    assert result["test_data"]["target"].isna().sum() == 0