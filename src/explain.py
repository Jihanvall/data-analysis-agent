import json
import os

from dotenv import load_dotenv
from google import genai 
from google.genai import types 

load_dotenv()

MODEL = "gemini-2.5-flash"

EXPLAIN_PROMPT = """You are a data scientist explaining machine learning results to a non-technical reader.
You receive a JSON object with the task type, the best model, the scoring metric used for model selection, the test metrics, and the preprocessing steps applied.
Write 3 to 5 short sentences in plain English:
- Say what the system did to the data.
- Say how good the model is, using the test metrics.
- Mention one caution, for example a small dataset, class imbalance, or a metric that looks too perfect.
Do not invent numbers that are not in the input. Do not use markdown or bullet points."""

def explain_results(result, task_type):
    payload = {
        "task_type": task_type,
        "best_model": result["model_name"],
        "scoring_used": (result["cv_scores"] or {}).get("scoring_used"),
        "test_metrics": result["test_metrics"],
        "preprocessing_steps": [
            {"column": s.get("column"), "action": s.get("action")}
            for s in result["plan"].get("steps", [])
        ],
    }
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model = MODEL,
            contents=json.dumps(payload, default=float),
            config=types.GenerateContentConfig(system_instruction=EXPLAIN_PROMPT)
        )
        text = (response.text or "").strip()
        return text or None
    except Exception:
        return None 