import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Cell,
} from "recharts";
import "./App.css";
import DataOverview from "./DataOverview";
import PredictionSection from "./PredictionSection";

const API_URL = "http://127.0.0.1:8000/analyze";

const ACTION_STYLES = {
  drop_column: { icon: "🗑️", label: "Drop column", tone: "danger" },
  add_missing_indicator: { icon: "", label: "Add missing indicator", tone: "warning" },
  fill_missing_mean: { icon: "", label: "Fill with mean", tone: "success" },
  fill_missing_median: { icon: "", label: "Fill with median", tone: "success" },
  fill_missing_mode: { icon: "", label: "Fill with mode", tone: "success" },
  drop_rows_with_missing: { icon: "", label: "Drop rows with missing", tone: "danger" },
  remove_outliers: { icon: "", label: "Remove outliers", tone: "purple" },
  encode_categorical: { icon: "", label: "Encode categorical", tone: "info" },
  none: { icon: "⏸", label: "No change", tone: "muted" },
};

const RATIO_METRICS = new Set(["f1_weighted", "f1_minority_class", "accuracy", "r2"]);

function metricTone(value) {
  if (value >= 0.9) return "success";
  if (value >= 0.7) return "warning";
  return "danger";
}

function prettify(text) {
  return String(text).replaceAll("_", " ");
}

function App() {
  const [file, setFile] = useState(null);
  const [target, setTarget] = useState("");
  const [taskType, setTaskType] = useState("classification");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);

    if (!file) {
      setError("Please choose a CSV file.");
      return;
    }
    if (taskType !== "clustering" && !target) {
      setError("Target column is required unless task is clustering.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    if (taskType !== "clustering") formData.append("target", target);
    formData.append("task_type", taskType);

    setLoading(true);
    try {
      const response = await fetch(API_URL, { method: "POST", body: formData });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Something went wrong.");
      }
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    window.open(`http://127.0.0.1:8000/download/${result.session_id}`, "_blank");
  };
  const cv = result?.cv_scores ?? {};
  const cvScores = cv.scores ?? cv;
  const scoringName = cv.scoring_used ?? "cross-validation";
  const cvData = Object.entries(cvScores)
    .filter(([, v]) => typeof v === "number")
    .map(([key, score]) => ({
      key,
      name: prettify(key),
      score: Number(score.toFixed(4)),
    }));
  const minScore = cvData.length ? Math.min(0, ...cvData.map((d) => d.score)) : 0;

  return (
    <div className="app">
      <header className="hero">
        <h1>Autonomous Data Analysis Agent</h1>
        <p>Upload a CSV, and the agent inspects, cleans, trains, and reports.</p>
      </header>

      <form className="card" onSubmit={handleSubmit}>
        <label
          className={`dropzone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept=".csv"
            hidden
            onChange={(e) => setFile(e.target.files[0] ?? null)}
          />
          <span className="dropzone-icon">📂</span>
          <span className="dropzone-text">
            {file ? file.name : "Click or drop a CSV file here"}
          </span>
        </label>

        <div className="fields">
          <label className="field">
            Target column
            <input
              type="text"
              value={target}
              disabled={taskType === "clustering"}
              onChange={(e) => setTarget(e.target.value)}
              placeholder={taskType === "clustering" ? "Not needed for clustering" : "e.g. churn"}
            />
          </label>

          <label className="field">
            Task type
            <select value={taskType} onChange={(e) => setTaskType(e.target.value)}>
              <option value="classification">classification</option>
              <option value="regression">regression</option>
              <option value="clustering">clustering</option>
            </select>
          </label>
        </div>

        <button className="submit" type="submit" disabled={loading}>
          {loading && <span className="spinner" />}
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <>
          <section className="card model-card">
            <span className="model-icon"></span>
            <div>
              <div className="model-label">Best model</div>
              <div className="model-name">{prettify(result.model_name)}</div>
            </div>
            <button className="download-btn" onClick={handleDownload}>
               ⬇ Download cleaned data
            </button>
          </section>

          <section className="card">
            <h3>Test metrics</h3>
            <div className="metrics">
              {Object.entries(result.test_metrics).map(([key, value]) => {
                const isRatio = RATIO_METRICS.has(key);
                const tone = isRatio ? metricTone(value) : "info";
                return (
                  <div className={`metric tone-${tone}`} key={key}>
                    <span className="metric-label">{prettify(key)}</span>
                    <span className="metric-value">
                      {Number(value).toFixed(isRatio ? 4 : 3)}
                    </span>
                    {isRatio && (
                      <div className="bar">
                        <div
                          className="bar-fill"
                          style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
                        />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </section>

          {cvData.length > 0 && (
            <section className="card">
              <h3>
                Model comparison <span className="hint">({prettify(scoringName)})</span>
              </h3>
              <div className="chart">
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={cvData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2a2f4a" vertical={false} />
                    <XAxis dataKey="name" stroke="#9aa3c7" tick={{ fontSize: 12 }} />
                    <YAxis stroke="#9aa3c7" tick={{ fontSize: 12 }} domain={[minScore, 1]} />
                    <Tooltip
                      cursor={{ fill: "rgba(124, 92, 255, 0.08)" }}
                      formatter={(v) => Number(v).toFixed(4)}
                      contentStyle={{
                        background: "#181c2e",
                        border: "1px solid #2a2f4a",
                        borderRadius: 8,
                      }}
                    />
                    <Bar dataKey="score" radius={[6, 6, 0, 0]}>
                      {cvData.map((d) => (
                        <Cell
                          key={d.key}
                          fill={d.key === result.model_name ? "#7c5cff" : "#3a4270"}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>
          )}
          {result.inspection_report && (
            <DataOverview report={result.inspection_report} />
          )}
          <section className="card">
            <h3>Preprocessing plan</h3>
            <ul className="steps">
              {result.plan.steps.map((step, i) => {
                const style = ACTION_STYLES[step.action] || {
                  icon: "⚙️",
                  label: step.action,
                  tone: "muted",
                };
                return (
                  <li className={`step tone-${style.tone}`} key={i}>
                    <span className="step-icon">{style.icon}</span>
                    <div>
                      <div className="step-head">
                        <strong>{step.column ?? "—"}</strong>
                        <span className="badge">{style.label}</span>
                      </div>
                      <p className="step-reason">{step.reason}</p>
                    </div>
                  </li>
                );
              })}
            </ul>
          </section>

          <details className="card log">
            <summary>Execution log ({result.log.length} entries)</summary>
            <ul>
              {result.log.map((entry, i) => (
                <li key={i}>{entry}</li>
              ))}
            </ul>
          </details>

          <PredictionSection
            key={result.session_id}
            sessionId={result.session_id}
            requiredColumns={result.required_columns}
            predictionColumn={result.prediction_column}
          />
        </>
      )}
    </div>
  );
}

export default App;