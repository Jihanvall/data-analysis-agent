import { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [target, setTarget] = useState("");
  const [taskType, setTaskType] = useState("classification");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

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
    formData.append("target", target);
    formData.append("task_type", taskType);

    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        body: formData,
      });
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

  return (
    <div className="app">
      <h1>Autonomous Data Analysis Agent</h1>

      <form onSubmit={handleSubmit}>
        <label>
          CSV file
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files[0])}
          />
        </label>

        <label>
          Target column
          <input
            type="text"
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            placeholder="e.g. churn"
          />
        </label>

        <label>
          Task type
          <select value={taskType} onChange={(e) => setTaskType(e.target.value)}>
            <option value="classification">classification</option>
            <option value="regression">regression</option>
            <option value="clustering">clustering</option>
          </select>
        </label>

        <button type="submit" disabled={loading}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result">
          <h2>Best model: {result.model_name}</h2>

          <h3>Test metrics</h3>
          <ul>
            {Object.entries(result.test_metrics).map(([key, value]) => (
              <li key={key}>
                {key}: {value.toFixed(4)}
              </li>
            ))}
          </ul>

          <h3>Preprocessing plan</h3>
          <ul>
            {result.plan.steps.map((step, i) => (
              <li key={i}>
                <strong>{step.column}</strong> → {step.action} ({step.reason})
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;