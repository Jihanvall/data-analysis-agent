import { useState, useEffect, useRef } from "react";
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
  extract_date_features: { icon: "", label: "Extract date features", tone: "info"},
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

const STAGES = [
  "Inspection your data",
  "Asking Gemini for a plan",
  "Preprocessing safely",
  "Training and comparing models",
  "Evaluating and writing the summary",
];

const STAGE_MS = 4500;

function UploadIcon({ done }) {
  return (
    <svg className={`upload-svg ${done ? "is-done" : ""}`} viewBox="0 0 64 64" width="64" height="64" aria-hidden="true">
      <defs>
        <linearGradient id="ug" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="64" y2="64">
          <stop offset="0%" stopColor="#7c5cff" />
          <stop offset="100%" stopColor="#2aa8e6" />
        </linearGradient>
      </defs>
      {done ? (
        <>
          <circle className="done-circle" cx="32" cy="32" r="28" fill="rgba(23,166,115,0.12)" stroke="#17a673" strokeWidth="3" />
          <path className="done-check" d="M20 33l8 8 16-17" fill="none" stroke="#17a673" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
        </>
      ) : (
        <>
          <circle className="upload-ring" cx="32" cy="32" r="28" fill="none" stroke="url(#ug)" strokeWidth="3" strokeDasharray="6 8" />
          <g className="upload-arrow" stroke="url(#ug)" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" fill="none">
            <path d="M32 42V22" />
            <path d="M23 30l9-9 9 9" />
          </g>
        </>
      )}
    </svg>
  );
}

function Stages({ active }) {
  return (
    <ol className="stages">
      {STAGES.map((label, i) => (
        <li key={label} className={i < active ? "done" : i === active ? "current" : ""}>
          <span className="stage-dot">{i < active ? "✓" : i + 1}</span>
          {label}
        </li>
      ))}
    </ol>
  );
}

function CountUp({ value, decimals }) {
  const [shown, setShown] = useState(0);
  const frame = useRef();
  useEffect(() => {
    const start = performance.now();
    const duration = 900;
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration);
      setShown(value * (1 - Math.pow(1 - t, 3)));
      if (t < 1) frame.current = requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame.current);
  }, [value]);
  return <>{Number(shown).toFixed(decimals)}</>;
}
function App() {
  const [file, setFile] = useState(null);
  const [target, setTarget] = useState("");
  const [taskType, setTaskType] = useState("classification");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [stage, setStage] = useState(0);

  useEffect(() => {
    if (!loading) return;
    setStage(0);
    const id = setInterval(() => {
      setStage((s) => Math.min(s + 1, STAGES.length - 1));
    }, STAGE_MS);
    return () => clearInterval(id);
  }, [loading]);

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
          <span className="dropzone-icon"><UploadIcon done={!!file} /></span>
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
        {loading && <Stages active={stage} />}
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
          
          {result.explanation && (
            <section className="card">
              <h3>Summary</h3>
              <p>{result.explanation}</p>
            </section>
          )}
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
                    <CountUp value={Number(value)} decimals={isRatio ? 4 : 3} />
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
  <CartesianGrid strokeDasharray="3 3" stroke="#dde2f5" vertical={false} />
  <XAxis dataKey="name" stroke="#5f6a96" tick={{ fontSize: 12 }} />
  <YAxis stroke="#5f6a96" tick={{ fontSize: 12 }} domain={[minScore, 1]} />
  <Tooltip
    cursor={{ fill: "rgba(124, 92, 255, 0.08)" }}
    formatter={(v) => Number(v).toFixed(4)}
    contentStyle={{
      background: "#ffffff",
      border: "1px solid #dde2f5",
      borderRadius: 8,
    }}
  />
  <Bar dataKey="score" radius={[6, 6, 0, 0]}>
    {cvData.map((d) => (
      <Cell
        key={d.key}
        fill={d.key === result.model_name ? "#7c5cff" : "#b9c0e6"}
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