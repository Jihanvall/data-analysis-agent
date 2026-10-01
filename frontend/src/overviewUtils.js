const SIGNAL_THRESHOLD = 0.15;
const MIN_MISSING_FOR_SIGNAL = 30;

export function missingTone(pct) {
  if (pct === 0) return "success";
  if (pct <= 5) return "warning";
  return "danger";
}

export function outlierText(info) {
  if (info.outlier_count === undefined || info.n_unique <= 10) return "—";
  return String(info.outlier_count);
}

export function missingnessSignal(info) {
  const evidence = info.missingness_evidence;
  if (!evidence || info.n_missing < MIN_MISSING_FOR_SIGNAL) return null;

  const whenMissing = evidence.target_distribution_when_missing;
  const whenPresent = evidence.target_distribution_when_present;
  if (!whenMissing || !whenPresent) return null;

  const classes = new Set([...Object.keys(whenMissing), ...Object.keys(whenPresent)]);
  const rows = [...classes].map((cls) => {
    const a = whenMissing[cls] ?? 0;
    const b = whenPresent[cls] ?? 0;
    return { cls, a, b, diff: Math.round(Math.abs(a - b) * 1000) / 1000 };
  });
  if (rows.length === 0) return null;

  rows.sort((x, y) => y.diff - x.diff || y.a - y.b - (x.a - x.b));
  return rows[0].diff > SIGNAL_THRESHOLD ? rows[0] : null;
}