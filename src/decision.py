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
    "extract_date_features",
    "add_missing_indicator",
    "fill_missing_mean",
    "fill_missing_median",
    "fill_missing_mode",
    "drop_rows_with_missing",
    "remove_outliers",
    "encode_categorical",
    "none",
}

SYSTEM_PROMPT = """You are a senior data scientist. You receive a JSON object with a task_type, a target_column (null for clustering), and a dataset inspection report.
Choose preprocessing steps that fit the task and the data. Follow these rules:
- Use only column names that appear in the report. Never invent columns.
- Never drop the target column and never remove outliers in it. Use action "none" for it.
- Drop identifier-like columns (n_unique equals n_rows, or the name suggests an id), but never a column with looks_like_date true.
- Drop any column with more than 60 percent missing values.
- Columns with looks_like_date true: always use extract_date_features, even when every value is unique, because unique dates are normal. Use drop_column only if missing values exceed 60 percent.
- Drop columns that look like free text.
- Text columns (dtype object or str): use encode_categorical only if n_unique is 15 or less, otherwise drop_column.
- Each column with missing values has n_missing and may have missingness_evidence. The evidence is reliable only when n_missing is at least 30. If n_missing is below 30, ignore the evidence completely.
- If n_missing is at least 30 and the target distribution (or target mean) differs by more than 15 percentage points in any class between when_missing and when_present, the missingness may not be random. In that case first use add_missing_indicator, then fill the column, and mention possible non-random missingness in the reason.
- Otherwise, if missing_pct is 5 or less, use drop_rows_with_missing.
- Otherwise, fill missing values. Numeric columns use fill_missing_median if outlier_count is greater than 0, otherwise fill_missing_mean. Text columns use fill_missing_mode.
- Ignore outlier_count for columns with n_unique of 10 or less, they are binary or categorical numbers.
- Do not remove outliers when they may be the signal (fraud, anomaly detection, rare events).
- A column may have several steps. Steps run in a safe order automatically: drop, indicator, fill, outliers, encode.
- Omit columns that need no change. Keep each reason under 20 words.
Return a JSON object in this exact shape:
{"steps": [{"column": "<column name or null>", "action": "<one of: drop_column, extract_date_features, add_missing_indicator, fill_missing_mean, fill_missing_median, fill_missing_mode, drop_rows_with_missing, remove_outliers, encode_categorical, none>", "reason": "<short reason>"}]}
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