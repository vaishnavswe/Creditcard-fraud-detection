import React from "react";

const RISK_BANDS = [
  { key: "low", label: "Low Risk", className: "risk-low" },
  { key: "medium", label: "Medium Risk", className: "risk-medium" },
  { key: "high", label: "High Risk", className: "risk-high" },
];

export default function RiskBandSummary({ bands }) {
  return (
    <section className="risk-band-grid">
      {RISK_BANDS.map((band) => (
        <article key={band.key} className={`risk-band-card ${band.className}`}>
          <p>{band.label}</p>
          <strong>{bands[band.key] || 0}</strong>
        </article>
      ))}
    </section>
  );
}
