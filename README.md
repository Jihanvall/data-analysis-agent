# Autonomous Data Analysis Agent

An AI-powered tool that takes a raw CSV file and automatically inspects it, decides on a preprocessing plan using the Gemini API, applies that plan safely, trains and selects the best machine learning model, and generates a human-readable report — all without manual data cleaning.

## How it works

1. **Inspection** (`src/inspection.py`) — scans the dataset and builds a report: row/column counts, types, missing value percentages and counts, outlier counts, sample values, target column distribution, and missingness evidence (whether missing values in a column correlate with the target, a signal that missingness may not be random).
2. **Decision** (`src/decision.py`) — sends the inspection report to the Gemini API, which returns a structured preprocessing plan (column, action, reason) chosen from a fixed set of safe actions.
3. **Preprocessing** (`src/preprocessing.py`) — applies the plan with `fit`/`transform`, learning parameters (fill values, outlier bounds, categories) from training data only, to avoid data leakage into the test set. Available actions include dropping columns, filling missing values (mean/median/mode), adding a missing-value indicator column, dropping rows with missing values, removing outliers, and encoding categorical columns.
4. **Training** (`src/training.py`) — trains several candidate models (Logistic Regression, Random Forest, Gradient Boosting for classification; equivalents for regression; KMeans for clustering), selects the best one via cross-validation, and automatically switches to a minority-class-focused metric when the target is highly imbalanced.
5. **Report** (`src/report.py`) — turns the results into a Markdown report and saves it under `reports/`.

All of this is tied together by `run_pipeline()` in `src/pipeline.py`.

## Why an LLM decides, not just runs

The LLM (Gemini) only *chooses* preprocessing steps from a fixed, safe list  it never executes anything directly. The code validates every step: unknown actions are ignored, the target column is always protected, and unknown columns are skipped. This keeps decisions explainable while keeping execution safe and deterministic.

## Setup

1. Clone the repository and create a virtual environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com), and create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

## Running the pipeline

```python
from src.pipeline import run_pipeline

result = run_pipeline("data/your_file.csv", target="your_target_column", task_type="classification")

print(result["model_name"], result["test_metrics"])
```

`task_type` can be `"classification"`, `"regression"`, or `"clustering"`.

This returns a dictionary with the trained model, cross-validation scores, test metrics, the preprocessing plan, and an execution log. A Markdown report is also generated via `src/report.py` and can be saved with `save_report()`.

## Tests

```
pytest -v
```

Tests cover inspection, preprocessing, training, and report generation offline. `test_decision.py` makes real calls to the Gemini API and requires a valid `.env` file.

## Project structure

```
src/
  inspection.py      # data inspection report
  decision.py         # Gemini-based preprocessing plan
  preprocessing.py    # fit/transform preprocessing (leak-free)
  training.py         # model selection and evaluation
  report.py           # Markdown report generation
  pipeline.py          # ties everything together
tests/                 # pytest test suite
reports/               # generated reports
data/                  # input datasets (not tracked in git)
```

## Example

The pipeline was tested end-to-end on the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 0.17% fraud). Gemini correctly avoided removing outliers from the anonymized `V1`–`V28` columns, recognizing they may carry the fraud signal rather than noise. See `reports/` for a generated example report.