import React from "react";
import "../styles/Table.css";

const FILTERS = [
  { key: "all", label: "All" },
  { key: "fraud", label: "Fraud only" },
  { key: "normal", label: "Normal only" },
];

function getRiskBand(probability) {
  if (probability < 0.4) return "low";
  if (probability < 0.7) return "medium";
  return "high";
}

function getFilteredRows(rows, filter) {
  if (filter === "fraud") {
    return rows.filter((row) => row.true_label === 1);
  }
  if (filter === "normal") {
    return rows.filter((row) => row.true_label === 0);
  }
  return rows;
}

export default function AlertsTable({
  rows,
  title = "Alerts",
  subtitle,
  limit,
  showFilters = true,
}) {
  const [activeFilter, setActiveFilter] = React.useState("all");
  const sortedRows = React.useMemo(
    () => [...rows].sort((a, b) => b.probability - a.probability),
    [rows]
  );
  const filteredRows = React.useMemo(
    () => getFilteredRows(sortedRows, activeFilter),
    [sortedRows, activeFilter]
  );
  const displayRows = limit ? filteredRows.slice(0, limit) : filteredRows;

  if (!rows.length) {
    return (
      <article className="dashboard-card alerts-card">
        <div className="card-head">
          <h3>{title}</h3>
          <p>No alerts to display.</p>
        </div>
      </article>
    );
  }

  return (
    <article className="dashboard-card alerts-card">
      <div className="card-head alerts-head">
        <div>
          <h3>{title}</h3>
          <p>
            {subtitle || `${filteredRows.length} transactions ranked by fraud probability`}
          </p>
        </div>

        {showFilters && (
          <div className="filter-group" role="tablist" aria-label="Alert filters">
            {FILTERS.map((filter) => (
              <button
                key={filter.key}
                type="button"
                className={`filter-btn ${activeFilter === filter.key ? "active" : ""}`}
                onClick={() => setActiveFilter(filter.key)}
              >
                {filter.label}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="table-scroll">
        <table className="alerts-table">
          <thead>
            <tr>
              <th>Transaction ID</th>
              <th>Fraud Score</th>
              <th>Status</th>
              <th>Risk Band</th>
              <th>Time</th>
              <th className="align-right">Amount</th>
            </tr>
          </thead>
          <tbody>
            {displayRows.map((row) => {
              const isFraud = row.true_label === 1;
              const riskBand = getRiskBand(row.probability);

              return (
                <tr key={`${row.index}-${row.Time}`} className={isFraud ? "fraud-row" : ""}>
                  <td className="mono">#{row.index}</td>
                  <td className="probability-cell">{row.probability.toFixed(4)}</td>
                  <td>
                    <span className={`status-badge ${isFraud ? "fraud" : "normal"}`}>
                      {isFraud ? "Fraud" : "Normal"}
                    </span>
                  </td>
                  <td>
                    <span className={`risk-badge ${riskBand}`}>{riskBand}</span>
                  </td>
                  <td>{Math.round(row.Time)}</td>
                  <td className="align-right">${row.Amount.toFixed(2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </article>
  );
}
