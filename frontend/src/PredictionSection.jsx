import { useState } from "react";

const API_URL = "http://127.0.0.1:8000";

export default function PredictionSection({
  sessionId,
  requiredColumns,
  predictionColumn,
}) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const pickFile = (f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError("");
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("session_id", sessionId);
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        const detail =
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail);
        throw new Error(detail || "Prediction failed");
      }
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const previewColumns =
    result && result.preview.length > 0 ? Object.keys(result.preview[0]) : [];

  return (
    <section className="pred-section">
      <h2>Predict on new data</h2>
      <p className="pred-intro">
        Upload a CSV of new rows. The trained model and the same preprocessing
        steps will be applied to it.
      </p>

      <div className="pred-columns">
        <span className="pred-label">Required columns:</span>
        {requiredColumns.map((c) => (
          <span key={c} className="pred-tag">
            {c}
          </span>
        ))}
      </div>

      <label
        className={`pred-drop ${dragOver ? "pred-drop-active" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          pickFile(e.dataTransfer.files[0]);
        }}
      >
        <input
          type="file"
          accept=".csv"
          hidden
          onChange={(e) => pickFile(e.target.files[0])}
        />
        {file ? file.name : "Drop a CSV here or click to choose"}
      </label>

      <button
        className="pred-button"
        onClick={handlePredict}
        disabled={!file || loading}
      >
        {loading ? "Predicting..." : "Predict"}
      </button>

      {error && <div className="pred-error">{error}</div>}

      {result && (
        <div className="pred-result">
          {result.warnings.length > 0 && (
            <div className="pred-warnings">
              {result.warnings.map((w, i) => (
                <div key={i}>{w}</div>
              ))}
            </div>
          )}

          <div className="pred-result-head">
            <span>
              {result.n_rows} rows predicted (showing the first{" "}
              {result.preview.length})
            </span>
            <a
              className="pred-download"
              href={`${API_URL}/download-predictions/${result.prediction_id}`}
            >
              Download predictions
            </a>
          </div>

          <div className="pred-table-wrap">
            <table className="pred-table">
              <thead>
                <tr>
                  {previewColumns.map((c) => (
                    <th
                      key={c}
                      className={c === result.prediction_column ? "pred-col" : ""}
                    >
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.preview.map((row, i) => (
                  <tr key={i}>
                    {previewColumns.map((c) => (
                      <td
                        key={c}
                        className={c === result.prediction_column ? "pred-col" : ""}
                      >
                        {String(row[c])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}