import os
import shutil 
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.pipeline import run_pipeline

app = FastAPI(title="Autonomous Data Analysis Agent API")

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

        return {
            "model_name": result["model_name"],
            "cv_scores": result["cv_scores"],
            "test_metrics": result["test_metrics"],
            "plan": result["plan"],
            "log": result["log"],
        }  
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)