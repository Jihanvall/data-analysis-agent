import json
import os 

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-2.5-flash"

TASK_TYPES = {"classification", "regression", "clustering"}

ALLOWED_ACTIONS = {
    "drop_column",
    "fill_missing_mean",
    "fill_missing_median",
    "fill_missing_mode",
    "remove_outliers",
    "encode_categorical",
    "none",
}

SYSTEM_PROMPT = """You are a senior data scientist. You receive a JSON object with a task_type, a target_column (null for clustering), and a dataset inspection report.
Choose preprocessing steps that fit the task and the data. Follow these rules:
- Use only column names that appear in the report. Never invent columns.
- Never drop the target column and never remove outliers in it. Use action "none" for it.
- Drop identifier-like columns (n_unique equals n_rows, or the name suggests an id).
- Drop any column with more than 50 percent missing values.
- Drop columns that look like dates or free text (no date handling is available).
- Text columns (dtype object or str): use encode_categorical only if n_unique is 15 or less, otherwise drop_column.
- Numeric columns with missing values: use fill_missing_median if outlier_count is greater than 0, otherwise fill_missing_mean. Text columns with missing values: use fill_missing_mode before encoding.
- Ignore outlier_count for columns with n_unique of 10 or less, they are binary or categorical numbers.
- Do not remove outliers when they may be the signal (fraud, anomaly detection, rare events).
- A column may have several steps. Steps run in the order you list them: fill missing values first, then remove outliers, then encode.
- Omit columns that need no change. Keep each reason under 15 words.
Return a JSON object in this exact shape:
{"steps": [{"column": "<column name or null>", "action": "<one of: drop_column, fill_missing_mean, fill_missing_median, fill_missing_mode, remove_outliers, encode_categorical, none>", "reason": "<short reason>"}]}
"""

def get_preprocessing_plan(report, target, task_type):
    if task_type not in TASK_TYPES:
        raise ValueError(f"task_type must be one of {sorted(TASK_TYPES)}")

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    payload = {"task_type": task_type, "target_column": target, "report": report}

    response = client.models.generate_content(
        model=MODEL,
        contents=json.dumps(payload),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
        ),
    )
    plan = json.loads(response.text)

    safe_steps = []
    for step in plan.get("steps", []):
        if step.get("action") not in ALLOWED_ACTIONS:
            continue
        if target is not None and step.get("column") ==target and step.get("action") in {"drop_column", "remove_outliers"}:
            continue
        safe_steps.append(step)

    return {"steps": safe_steps}