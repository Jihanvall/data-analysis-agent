# Autonomous Data Analysis Agent

![CI](https://github.com/Jihanvall/data-analysis-agent/actions/workflows/ci.yml/badge.svg)

An AI-powered tool that takes a raw CSV file and automatically inspects it, decides on a preprocessing plan using the Gemini API, applies that plan safely, trains and selects the best machine learning model, and generates a human-readable report — all without manual data cleaning.

## How it works

1. **Inspection** (`src/inspection.py`) — scans the dataset and builds a report: row/column counts, types, missing value percentages and counts, outlier counts, sample values, date-like columns, target column distribution, and missingness evidence (whether missing values in a column correlate with the target, a signal that missingness may not be random).
2. **Decision** (`src/decision.py`) — sends the inspection report to the Gemini API, which returns a structured preprocessing plan (column, action, reason) chosen from a fixed set of safe actions.
3. **Preprocessing** (`src/preprocessing.py`) — applies the plan with `fit`/`transform`, learning parameters (fill values, outlier bounds, categories, date fallbacks) from training data only, to avoid data leakage into the test set.
4. **Training** (`src/training.py`) — trains several candidate models (Logistic Regression, Random Forest, Gradient Boosting for classification; equivalents for regression; KMeans for clustering), selects the best one via cross-validation, and automatically switches to a minority-class-focused metric when the target is highly imbalanced. The best model is then tuned with a small randomized search, and the tuned version is kept only if its cross-validation score improves.
5. **Explanation** (`src/explain.py`) — asks Gemini for a short plain-language summary of the results. If the call fails, the analysis still completes without a summary.
6. **Report** (`src/report.py`) — turns the results into a Markdown report and saves it under `reports/`.
7. **Prediction** (`src/predict.py`, `src/model_io.py`) — the trained model is saved together with its preprocessing steps, so new CSV files can be predicted with exactly the same cleaning.

All of this is tied together by `run_pipeline()` in `src/pipeline.py`.

## Why an LLM decides, not just runs

The LLM (Gemini) only *chooses* preprocessing steps from a fixed, safe list  it never executes anything directly. The code validates every step: unknown actions are ignored, the target column is always protected, and unknown columns are skipped. This keeps decisions explainable while keeping execution safe and deterministic.

## Preprocessing actions

| Action | What it does |
| --- | --- |
| `drop_column` | Removes identifier-like or mostly empty columns |
| `extract_date_features` | Replaces a date column with year, month, day and day-of-week |
| `add_missing_indicator` | Adds a `<column>_was_missing` flag |
| `fill_missing_mean` / `fill_missing_median` / `fill_missing_mode` | Fills missing values with a value learned from training data |
| `drop_rows_with_missing` | Drops rows with missing values in training, fills them in test and new data |
| `remove_outliers` | Drops outlier rows in training, clips them in test and new data |
| `encode_categorical` | One-hot encodes text columns with the categories seen in training |

## Setup

1. Clone the repository and create a virtual environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com), and create a `.env` file in the project root (see `.env.example`):

```
GEMINI_API_KEY=your_key_here
```

## Configuration

These optional environment variables control the web API:

| Variable | Default | Meaning |
| --- | --- | --- |
| `GEMINI_API_KEY` | none | Gemini API key (required) |
| `MAX_UPLOAD_MB` | `20` | Maximum size of an uploaded CSV |
| `SESSION_TTL_SECONDS` | `3600` | How long sessions and predictions stay in memory |
| `MAX_STORE_ENTRIES` | `20` | Maximum entries per in-memory store; the oldest are removed first |
| `ALLOWED_ORIGINS` | `http://localhost:5173` | Comma-separated list of allowed CORS origins |

## Running the pipeline

There are three ways to run the pipeline.

### 1. From Python directly

```python
from src.pipeline import run_pipeline

result = run_pipeline("data/your_file.csv", target="your_target_column", task_type="classification")

print(result["model_name"], result["test_metrics"])
```

`task_type` can be `"classification"`, `"regression"`, or `"clustering"`.

This returns a dictionary with the trained model, cross-validation scores, test metrics, the preprocessing plan, an execution log, and a bundle used for prediction. A Markdown report is also generated via `src/report.py` and can be saved with `save_report()`.

### 2. From the command line (CLI)

```
python run.py --file data/your_file.csv --target your_target_column --task classification --name your_dataset_name
```

`--target` is required unless `--task` is `clustering`. The best model, test metrics, and saved report path are printed to the terminal.

### 3. From the web interface (FastAPI + React)

This starts a local API server and a browser-based UI for uploading a file, viewing results, and predicting on new data.

Terminal 1 — start the backend:

```
uvicorn api:app --reload
```

Terminal 2 — start the frontend:

```
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173` in your browser, upload a CSV file, enter the target column, choose a task type, and click Analyze. After the analysis, the "Predict on new data" section lists the required columns and accepts a new CSV.

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | Health check |
| POST | `/analyze` | Runs the full pipeline on an uploaded CSV |
| GET | `/download/{session_id}` | Downloads the cleaned data |
| POST | `/predict` | Predicts on a new CSV using the model of a session |
| GET | `/download-predictions/{prediction_id}` | Downloads the predictions |

Interactive documentation is available at `http://127.0.0.1:8000/docs` while the backend is running.

## Docker

Build and run the backend in a container:

```
docker build -t data-agent .
docker run --rm -p 8000:8000 --env-file .env data-agent
```

The API is then available at `http://127.0.0.1:8000`. Configuration is done through the environment variables listed above.

## Tests

```
pytest -v
```

Tests cover inspection, preprocessing, training, prediction, the API, and report generation offline. The tests in `test_decision.py` call the real Gemini API, so they are skipped automatically when `GEMINI_API_KEY` is not set. The free Gemini tier allows only a few requests per minute, so if you see a quota error, run `pytest tests/test_decision.py` on its own.

Tests also run automatically on every push and pull request through GitHub Actions.

## Project structure

```
api.py                 # FastAPI backend
run.py                 # command line entry point
src/
  inspection.py        # data inspection report
  decision.py          # Gemini-based preprocessing plan
  preprocessing.py     # fit/transform preprocessing (leak-free)
  training.py          # model selection, tuning and evaluation
  explain.py           # plain-language summary of the results
  report.py            # Markdown report generation
  model_io.py          # save/load the model bundle
  predict.py           # prediction on new data
  pipeline.py          # ties everything together
frontend/              # React app
tests/                 # pytest test suite
reports/               # generated reports (not tracked in git)
data/                  # input datasets (not tracked in git)
```

## Example

The pipeline was tested end-to-end on the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 0.17% fraud). Gemini correctly avoided removing outliers from the anonymized `V1`–`V28` columns, recognizing they may carry the fraud signal rather than noise.

## Notes

- Uploaded model files are never loaded; models live only in the server memory.
- Stored sessions are kept in memory, so they are lost when the server restarts. This fits a single server; use a shared store such as Redis for several servers.