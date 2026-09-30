import os
import shutil 
import tempfile
import uuid

import pandas as pd

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from src.pipeline import run_pipeline

app = FastAPI(title="Autonomous Data Analysis Agent API")
cleaned_data_store: dict[str, pd.DataFrame] = {}

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

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = run_pipeline(tmp_path, target=target, task_type=task_type)

        combined = pd.concat([result["train_data"], result["test_data"]], ignore_index=True)
        session_id = str(uuid.uuid4())
        cleaned_data_store[session_id] = combined

        return {
            "model_name": result["model_name"],
            "cv_scores": result["cv_scores"],
            "test_metrics": result["test_metrics"],
            "plan": result["plan"],
            "log": result["log"],
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