import os
import json
import shutil 
import tempfile
import uuid
import time
from collections import OrderedDict

import pandas as pd

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from src.pipeline import run_pipeline
from src.predict import prediction_column_name, required_columns, predict_with_bundle
from src.explain import explain_results

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
MAX_STORE_ENTRIES = int(os.getenv("MAX_STORE_ENTRIES", "20"))

class ExpiringStore:
    def __init__(self, ttl_seconds, max_entries):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._items = OrderedDict()

    def _purge(self):
        now = time.monotonic()
        expired = [k for k, (t, _) in self._items.items() if now - t > self.ttl_seconds]
        for k in expired:
            del self._items[k]

    def __setitem__(self, key, value):
        self._purge()
        self._items[key] = (time.monotonic(), value)
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def __getitem__(self, key):
        self._purge()
        return self._items[key][1]

    def __contains__(self, key):
        self._purge()
        return key in self._items

    def __len__(self):
        self._purge()
        return len(self._items)

    def clear(self):
        self._items.clear()

    

app = FastAPI(title="Autonomous Data Analysis Agent API")
cleaned_data_store = ExpiringStore(SESSION_TTL_SECONDS, MAX_STORE_ENTRIES)
bundle_store = ExpiringStore(SESSION_TTL_SECONDS, MAX_STORE_ENTRIES)
predictions_store = ExpiringStore(SESSION_TTL_SECONDS, MAX_STORE_ENTRIES)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))

def check_upload_size(file: UploadFile) -> None:
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    file.file.seek(0, os.SEEK_END)
    size = file.file.tell()
    file.file.seek(0)
    if size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File is too large ({size / 1024 / 1024:.1f} MB). "
                F"The maximum allowed size is {MAX_UPLOAD_MB} MB."
            ),
        )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    target: str = Form(None),
    task_type: str = Form(...),
):
    if task_type not in {"classification", "regression", "clustering"}:
        raise HTTPException(status_code=400, detail="Invalid task_type")
    if task_type != "clustering" and not target:
        raise HTTPException(status_code=400, detail="target is required unless task_type is clustering")

    check_upload_size(file)
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = run_pipeline(tmp_path, target=target, task_type=task_type)

        combined = pd.concat([result["train_data"], result["test_data"]], ignore_index=True)
        session_id = str(uuid.uuid4())
        cleaned_data_store[session_id] = combined
        bundle_store[session_id] = result["bundle"]
        explanation = explain_results(result, task_type)

        return {
            "model_name": result["model_name"],
            "cv_scores": result["cv_scores"],
            "test_metrics": result["test_metrics"],
            "plan": result["plan"],
            "log": result["log"],
            "inspection_report": result["inspection_report"],
            "required_columns": required_columns(result["bundle"]),
            "prediction_column": prediction_column_name(result["bundle"]),
            "session_id": session_id,
            "explanation": explanation,
        }  
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@app.get("/download/{session_id}")
def download_cleaned_data(session_id: str):
    if session_id not in cleaned_data_store:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    df = cleaned_data_store[session_id]
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cleaned_data.csv"},
    )

@app.post("/predict")
async def predict(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    check_upload_size(file)

    if session_id not in bundle_store:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    bundle = bundle_store[session_id]

    try:
        new_df = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as CSV")

    if new_df.empty:
        raise HTTPException(status_code=400, detail="The uploaded file has no rows")

    try:
        out, warnings = predict_with_bundle(bundle, new_df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    prediction_id = str(uuid.uuid4())
    predictions_store[prediction_id] = out

    return {
        "prediction_id": prediction_id,
        "prediction_column": prediction_column_name(bundle),
        "n_rows": len(out),
        "preview": json.loads(out.head(50).to_json(orient="records")),
        "warnings": warnings,
    }

@app.get("/download-predictions/{prediction_id}")
def download_predictions(prediction_id: str):
    if prediction_id not in predictions_store:
        raise HTTPException(status_code=404, detail="Predictions not found or expired")

    csv_bytes = predictions_store[prediction_id].to_csv(index=False).encode("utf-8")

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )