import React from "react";
import Metrics from "../components/Metrics";
import SectionHeader from "../components/SectionHeader";

export default function ModelMetricsSection({ metrics }) {
  return (
    <section className="page-section">
      <SectionHeader
        title="Model Metrics"
        subtitle="Classification performance and alert tradeoffs"
      />

      <Metrics data={metrics} detailed />

      <div className="model-explain-grid">
        <article className="dashboard-card explain-card">
          <h3>Precision</h3>
          <p>Measures how many flagged transactions are actually fraud.</p>
        </article>
        <article className="dashboard-card explain-card">
          <h3>Recall</h3>
          <p>Measures how many real fraud cases the model successfully captures.</p>
        </article>
        <article className="dashboard-card explain-card">
          <h3>F1 Score</h3>
          <p>Balances precision and recall for a single quality indicator.</p>
        </article>
        <article className="dashboard-card explain-card">
          <h3>PR-AUC</h3>
          <p>Shows ranking quality for imbalanced fraud detection tasks.</p>
        </article>
      </div>

      <article className="dashboard-card performance-summary">
        <h3>Performance summary</h3>
        <ul>
          <li>High recall indicates most fraud cases are captured.</li>
          <li>Lower precision reflects class imbalance and false positives.</li>
          <li>PR-AUC highlights meaningful fraud ranking quality for analysts.</li>
        </ul>
      </article>

      <article className="dashboard-card threshold-placeholder">
        <h3>Threshold and confusion matrix notes</h3>
        <p>
          Current threshold is 0.50. You can add confusion matrix and threshold sweep
          analysis here in future iterations.
        </p>
      </article>
    </section>
  );
}
