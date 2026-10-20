import { useEffect, useState } from 'react'

type Row = Record<string, string | number | boolean | null>
type Metadata = { counts?: Record<string, number>; important_limitation?: string; timing_limitation?: string }
type ResultsData = { paper_reference: Row[]; three_tier_protocol: Row[]; three_tier_metadata: Metadata; binary_research_benchmark: Row[]; binary_deployed_pdf_model: Row[] }
const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'
const percent = (value: unknown) => typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : '—'

function ResultCards({ rows }: { rows: Row[] }) {
  return <div className="modelGrid">{rows.map(row => <section className="card" key={String(row.model)}><p className="eyebrow">{String(row.model).replaceAll('_', ' ')}</p><strong className="resultNumber">{percent(row.accuracy)}</strong><p>Accuracy</p>{row.status && <small>{String(row.status)}</small>}</section>)}</div>
}
function CompactComparison({ rows }: { rows: Row[] }) {
  return <div className="table"><table><thead><tr><th>Model</th><th>Accuracy</th><th>Macro-F1</th><th>Balanced accuracy</th></tr></thead><tbody>{rows.map(row => <tr key={String(row.model)}><td>{String(row.model).replaceAll('_', ' ')}</td><td>{percent(row.accuracy)}</td><td>{percent(row.macro_f1)}</td><td>{percent(row.balanced_accuracy)}</td></tr>)}</tbody></table></div>
}
function MetricTable({ row, setup }: { row?: Row; setup: string }) {
  if (!row) return <p className="notice">Saved deployed-model metrics are unavailable.</p>
  const metrics = [['Accuracy', row.accuracy], ['Precision', row.precision], ['Recall', row.recall], ['F1-score', row.f1], ['ROC-AUC', row.roc_auc], ['PR-AUC', row.pr_auc]]
  return <section className="evaluationBlock"><p>{setup}</p><div className="binaryMetrics">{metrics.map(([label, value]) => <div key={String(label)}><span>{label}</span><b>{percent(value)}</b></div>)}</div></section>
}

export default function Results() {
  const [data, setData] = useState<ResultsData | null>(null); const [error, setError] = useState('')
  useEffect(() => { fetch(`${API}/demo-results`).then(response => response.ok ? response.json() : Promise.reject()).then(setData).catch(() => setError('Results are unavailable. Start the FastAPI backend.')) }, [])
  if (error) return <main><p className="notice error">{error}</p></main>
  if (!data) return <main><p>Loading verified results…</p></main>
  const counts = data.three_tier_metadata.counts ?? {}; const deployed = data.binary_deployed_pdf_model.find(row => row.model === 'deployable_hgb_v2'); const research = data.binary_research_benchmark.find(row => row.model === 'hist_gradient_boosting'); const baseline = data.three_tier_protocol.find(row => row.model === 'majority_baseline'); const forest = data.three_tier_protocol.find(row => row.model === 'random_forest')
  return <main><p className="eyebrow">Verified experiment outputs</p><h1 className="pageTitle">Results</h1><p className="lede">Two distinct tasks were evaluated: a base-paper-inspired three-tier study of already highly cited patents, and binary prediction of whether a patent reaches at least 10 forward citations within three years. Their accuracy figures are not directly comparable.</p>
    <section className="resultSection"><div className="sectionHeader"><div><p className="eyebrow">Experiment 1</p><h2>Three-Tier Classification — Base-Paper-Inspired</h2></div></div><p className="lede">An approximate implementation inspired by the paper&apos;s C0, C1, and C2 methodology; it is not an exact reproduction.</p><div className="tierCards"><div><b>1,149</b><span>High-citation cohort</span></div><div><b>C0 · {counts.C0 ?? 997}</b><span>10–40 citations</span></div><div><b>C1 · {counts.C1 ?? 86}</b><span>41–62 citations</span></div><div><b>C2 · {counts.C2 ?? 66}</b><span>63 or more citations</span></div><div><b>230</b><span>Fixed stratified test records</span></div></div><section className="summaryResult"><p className="eyebrow">Three-Tier Classification Result</p><div><p><span>Accuracy</span><b>{percent(forest?.accuracy)}</b></p><p><span>Majority-class baseline</span><b>{percent(baseline?.accuracy)}</b></p><p><span>Macro-F1 · Random Forest</span><b>{percent(forest?.macro_f1)}</b></p><p><span>Balanced accuracy</span><b>{percent(forest?.balanced_accuracy)}</b></p></div><p className="summaryExplanation">The model&apos;s accuracy matches the majority-class baseline. It predicted C0 for every test record and failed to identify C1 and C2.</p></section><h3 className="comparisonHeading">Compact model comparison</h3><CompactComparison rows={data.three_tier_protocol} /><p className="notice warning">{data.three_tier_metadata.important_limitation}</p><p className="notice">{data.three_tier_metadata.timing_limitation}</p></section>
    <section className="resultSection"><div className="sectionHeader"><div><p className="eyebrow">Reference paper</p><h2>Reported, not PatentPulse measurements</h2></div><p>This is an approximate reproduction. Some features and experimental details from the original paper were unavailable, so results are not directly comparable.</p></div><ResultCards rows={data.paper_reference} /></section>
    <section className="resultSection"><div className="sectionHeader"><div><p className="eyebrow">Binary citation prediction</p><h2>Research and PDF model evaluations</h2></div></div><p className="lede">Both estimate whether a patent reaches the project’s high-citation threshold within three years, but they use different feature sets.</p><h3>Experiment 2 — Research Binary HGB</h3><MetricTable row={research} setup="Fixed 80/20 held-out test. This broader research benchmark includes backward citations and title/abstract word counts; backward citations are timing-uncertain and not reliably extractable from a PDF." /><h3>Experiment 3 — PDF-Safe HGB v2 · Production Prediction Model</h3><MetricTable row={deployed} setup="Fixed 80/20 held-out test. This is the primary uploaded-PDF model because its document feature set excludes backward citations and family size." /><p className="notice">The research HGB is retained as a benchmark. The PDF-safe HGB v2 remains the production prediction model; neither binary evaluation is combined with the three-tier study.</p></section>
  </main>
}
