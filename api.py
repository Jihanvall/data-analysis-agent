import os
import json
import shutil 
import tempfile
import uuid

import pandas as pd

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from src.pipeline import run_pipeline
from src.predict import prediction_column_name, required_columns, predict_with_bundle

app = FastAPI(title="Autonomous Data Analysis Agent API")
cleaned_data_store: dict[str, pd.DataFrame] = {}
bundle_store: dict[str, dict] = {}
predictions_store: dict[str, pd.DataFrame] = {}

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
                f"File is too large ({size / 1024 / 1024:.1f} MB)."
                F"the maximum allowed size is {MAX_UPLOAD_MB} MB."
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