import { useEffect, useState } from 'react'

type Row = Record<string, string | number | boolean | null>
type ResultsData = { paper_reference: Row[]; paper_note: string; binary_research_benchmark: Row[]; binary_deployed_pdf_model: Row[]; chronological: Row[] }
const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

function Table({ rows }: { rows: Row[] }) {
  const columns = Object.keys(rows[0] || {}).filter(key => ['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'status', 'cv_accuracy_mean'].includes(key))
  return <div className="table"><table><thead><tr>{columns.map(column => <th key={column}>{column.replaceAll('_', ' ')}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index}>{columns.map(column => <td key={column}>{typeof row[column] === 'number' && column !== 'status' ? `${(Number(row[column]) * 100).toFixed(1)}%` : String(row[column] ?? '')}</td>)}</tr>)}</tbody></table></div>
}

export default function Results() {
  const [data, setData] = useState<ResultsData | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { fetch(`${API}/demo-results`).then(response => response.ok ? response.json() : Promise.reject()).then(setData).catch(() => setError('Demo results are unavailable. Start the FastAPI backend.')) }, [])
  if (error) return <main><p className="notice error">{error}</p></main>
  if (!data) return <main><p>Loading verified results...</p></main>
  return <main><p className="eyebrow">Demo results</p><h2 className="title">Two classification tasks.<br/>Never one score.</h2><section className="card"><h3>A. Comparison with the base paper</h3><p>{data.paper_note}</p><Table rows={data.paper_reference} /></section><section className="card"><h3>B. PatentPulse binary citation prediction</h3><p>Predicts whether a patent receives at least 10 forward citations within three years. The research benchmark and deployed PDF predictor use different feature configurations and are kept separate.</p><h4>Research benchmark</h4><Table rows={data.binary_research_benchmark} /><h4>Deployed PDF model</h4><Table rows={data.binary_deployed_pdf_model} /></section>{data.chronological.length > 0 && <section className="card"><h3>Chronological evaluation</h3><p>Older publications train the model and later publications test it. This temporal result is separate from the random split.</p><Table rows={data.chronological} /></section>}<section className="card"><h3>How to read the metrics</h3><p>Precision is how often positive predictions are correct. Recall is how many truly highly cited patents were found. F1 balances both; ROC-AUC assesses score ranking. Accuracy alone is insufficient for imbalanced classes.</p></section></main>
}
