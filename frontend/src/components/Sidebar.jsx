import React from "react";
import "../styles/Sidebar.css";

const NAV_ITEMS = [
  { key: "overview", label: "Overview" },
  { key: "alerts", label: "Alerts" },
  { key: "analytics", label: "Analytics" },
];

export default function Sidebar({ activeSection, onChange }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-mark">FM</div>
        <div>
          <h1>Fraud Monitor</h1>
          <p>Risk analytics console</p>
        </div>
      </div>

      <nav className="sidebar-nav" aria-label="Dashboard sections">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.key}
            type="button"
            className={`nav-item ${activeSection === item.key ? "active" : ""}`}
            onClick={() => onChange(item.key)}
          >
            {item.label}
          </button>
        ))}
      </nav>
    </aside>
  );
}
