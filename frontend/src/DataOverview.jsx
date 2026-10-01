import { missingTone, missingnessSignal, outlierText } from "./overviewUtils";

function SummaryCards({ report }) {
  const columns = Object.values(report.columns);
  const withMissing = columns.filter((c) => c.missing_pct > 0).length;
  const linked = columns.filter((c) => missingnessSignal(c) !== null).length;

  const cards = [
    { label: "rows", value: report.n_rows, tone: "info" },
    { label: "columns", value: report.n_columns, tone: "info" },
    { label: "with missing values", value: withMissing, tone: withMissing ? "warning" : "success" },
    { label: "linked to target", value: linked, tone: linked ? "danger" : "success" },
  ];

  return (
    <div className="metrics compact">
      {cards.map((card) => (
        <div className={`metric tone-${card.tone}`} key={card.label}>
          <span className="metric-label">{card.label}</span>
          <span className="metric-value">{card.value}</span>
        </div>
      ))}
    </div>
  );
}

function NeedsAttention({ report }) {
  const items = Object.entries(report.columns)
    .map(([name, info]) => ({ name, info, signal: missingnessSignal(info) }))
    .filter(({ info, signal }) => info.missing_pct > 0 || signal)
    .sort((a, b) => b.info.missing_pct - a.info.missing_pct);

  if (items.length === 0) return null;

  return (
    <div className="attention">
      <h4>Needs attention</h4>
      <ul className="attention-list">
        {items.map(({ name, info, signal }) => (
          <li className={`attention-item tone-${missingTone(info.missing_pct)}`} key={name}>
            <div className="attention-head">
              <strong>{name}</strong>
              <span className="pill">
                {info.missing_pct}% missing ({info.n_missing} rows)
              </span>
            </div>
            <div className="bar">
              <div
                className="bar-fill"
                style={{ width: `${Math.min(100, info.missing_pct)}%` }}
              />
            </div>
            {signal && (
              <p className="attention-note">
                ⚠ Missing values look linked to the target: class {signal.cls} makes up{" "}
                {Math.round(signal.a * 100)}% of the rows where this column is missing,
                versus {Math.round(signal.b * 100)}% in the rest. The missingness may not
                be random.
              </p>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function ColumnsTable({ report }) {
  return (
    <details className="all-columns">
      <summary>Show all columns ({report.n_columns})</summary>
      <div className="table-wrap">
        <table className="overview">
          <thead>
            <tr>
              <th>Column</th>
              <th>Type</th>
              <th>Missing</th>
              <th>Unique</th>
              <th>Outliers</th>
              <th>Sample</th>
              <th>Linked to target</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(report.columns).map(([name, info]) => {
              const isTarget = report.target?.column === name;
              return (
                <tr key={name}>
                  <td>
                    <strong>{name}</strong>
                    {isTarget && <span className="badge tone-purple">target</span>}
                  </td>
                  <td>{info.dtype}</td>
                  <td>
                    <span className={`pill tone-${missingTone(info.missing_pct)}`}>
                      {info.missing_pct}%
                    </span>
                  </td>
                  <td>{info.n_unique}</td>
                  <td>{outlierText(info)}</td>
                  <td className="sample">{info.sample_values.join(", ")}</td>
                  <td>{missingnessSignal(info) ? "⚠ yes" : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </details>
  );
}

function DataOverview({ report }) {
  return (
    <section className="card">
      <h3>
        Data overview <span className="hint">(read only)</span>
      </h3>
      <SummaryCards report={report} />
      <NeedsAttention report={report} />
      <ColumnsTable report={report} />
    </section>
  );
}

export default DataOverview;