import os
from datetime import datetime

def generate_report(result, dataset_name="dataset"):
    lines = []
    lines.append(f"# Data Analysis Report: {dataset_name}")
    lines.append("")

    lines.append("## Preprocessing Plan")
    lines.append("")
    for step in result["plan"].get("steps", []):
        col = step.get("column")
        action = step.get("action")
        reason = step.get("reason", "")
        lines.append(f"- **{col}** → `{action}`: {reason}")
    lines.append("")

    lines.append("## Applied Steps Log")
    lines.append("")
    for entry in result["log"]:
        lines.append(f"- {entry}")
    lines.append("")

    lines.append("## Model Selection")
    lines.append("")
    lines.append(f"**Best model:** `{result['model_name']}`")
    lines.append("")
    cv = result["cv_scores"]
    if "scoring_used" in cv:
        lines.append(f"Cross-validation scores (metric: `{cv['scoring_used']}`):")
        lines.append("")
        for name, score in cv["scores"].items():
            marker = " ✅" if name == result["model_name"] else ""
            lines.append(f"- {name}: {round(score, 4)}{marker}")
    else:
        lines.append("Cross-validation scores:")
        lines.append("")
        for name, score in cv.items():
            marker = " ✅" if name == result["model_name"] else ""
            lines.append(f"- {name}: {round(score, 4)}{marker}")
    lines.append("")

    tuning = cv.get("tuning") if "scoring_used" in cv else None
    if tuning:
        lines.append("## Hyperparameter Tuning")
        lines.append("")
        lines.append(
            "The best model was tuned with a randomized search, and the tuned "
            "version was kept because it scored higher in cross-validation."
        )
        lines.append("")
        lines.append(f"- Score before: {round(tuning['score_before'], 4)}")
        lines.append(f"- Score after: {round(tuning['score_after'], 4)}")
        params = ", ".join(f"{k}={v}" for k, v in tuning["best_params"].items())
        lines.append(f"- Best parameteres: {params}")
        lines.append("")
        
    lines.append("## Test Set Performance")
    lines.append("")
    for metric, value in result["test_metrics"].items():
        lines.append(f"- **{metric}**: {round(value, 4)}")
    lines.append("")

    return "\n".join(lines)

def save_report(report_text, dataset_name="dataset", folder="reports"):
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset_name)
    filename = f"{safe_name}_{timestamp}.md"
    filepath = os.path.join(folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)

    return filepath