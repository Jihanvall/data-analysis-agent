from src import explain

class FakeResponse:
    text = "  The model works well.  "

class FakeModels:
    def generate_content(self, **kwargs):
        return FakeResponse()

class FakeClient:
    models = FakeModels()

RESULT = {
    "model_name": "random_forest",
    "cv_scores": {"scoring_used": "f1_weighted", "scores": {"random_forest": 0.9}},
    "test_metrics": {"accuracy": 0.95},
    "plan": {"steps": [{"column": "id", "action": "drop_column", "reason": "id"}]},
}

def test_explain_returns_stripped_text(monkeypatch):
    monkeypatch.setattr(explain.genai, "Client", lambda api_key=None: FakeClient())
    assert explain.explain_results(RESULT, "classification") == "The model works well."

def test_explain_returns_none_when_gemini_fails(monkeypatch):
    def boom(api_key=None):
        raise RuntimeError("no network")

    monkeypatch.setattr(explain.genai, "Client", boom)
    assert explain.explain_results(RESULT, "classification") is None