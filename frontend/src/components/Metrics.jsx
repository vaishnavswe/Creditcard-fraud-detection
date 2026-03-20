import React from "react";
import "../styles/App.css";

const METRICS = [
  {
    key: "precision",
    label: "Precision",
    description: "True fraud detections out of all positive predictions",
  },
  {
    key: "recall",
    label: "Recall",
    description: "True fraud cases found out of all actual fraud cases",
  },
  {
    key: "f1",
    label: "F1 Score",
    description: "Balanced view of precision and recall",
  },
  {
    key: "pr_auc",
    label: "PR-AUC",
    description: "Ranking quality for fraud detection on imbalanced data",
  },
];

export default function Metrics({ data, detailed = false }) {
  return (
    <section className="metrics-grid">
      {METRICS.map((metric) => (
        <article className="metric-card" key={metric.key}>
          <span className="metric-label">{metric.label}</span>
          <span className="metric-value">
            {data[metric.key] !== undefined ? Number(data[metric.key]).toFixed(3) : "—"}
          </span>
          {detailed && <p className="metric-note">{metric.description}</p>}
        </article>
      ))}
    </section>
  );
}
