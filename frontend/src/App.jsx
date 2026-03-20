import React, { useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import OverviewSection from "./sections/OverviewSection";
import AlertsSection from "./sections/AlertsSection";
import AnalyticsSection from "./sections/AnalyticsSection";
import "./styles/App.css";
import "./styles/Table.css";
import "./styles/Sidebar.css";


function parseCSV(text) {
  const lines = text.trim().split("\n");
  if (lines.length < 2) return [];
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const row = {};
    headers.forEach((h, i) => {
      row[h.trim()] = values[i]?.trim() ?? "";
    });
    return row;
  });
}

function toNumber(value, fallback = 0) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function normalizeAlerts(rows) {
  return rows
    .map((row) => ({
      index: row.index,
      probability: toNumber(row.probability),
      true_label: toNumber(row.true_label),
      Time: toNumber(row.Time),
      Amount: toNumber(row.Amount),
    }))
    .sort((a, b) => b.probability - a.probability);
}

function buildScoreDistribution(rows) {
  const buckets = Array.from({ length: 10 }, (_, i) => ({
    label: `${(i / 10).toFixed(1)}-${((i + 1) / 10).toFixed(1)}`,
    count: 0,
    color: i >= 7 ? "#DC2626" : i >= 4 ? "#D97706" : "#16A34A",
  }));

  rows.forEach((row) => {
    const idx = Math.min(9, Math.floor(row.probability * 10));
    buckets[idx].count += 1;
  });

  return buckets;
}

function buildBreakdown(rows) {
  const fraud = rows.filter((row) => row.true_label === 1).length;
  const normal = rows.length - fraud;

  return [
    { name: "Normal", value: normal },
    { name: "Fraud", value: fraud },
  ];
}

function buildRiskBands(rows) {
  const bands = { low: 0, medium: 0, high: 0 };

  rows.forEach((row) => {
    if (row.probability < 0.4) {
      bands.low += 1;
    } else if (row.probability < 0.7) {
      bands.medium += 1;
    } else {
      bands.high += 1;
    }
  });

  return bands;
}

function buildTimeSeries(rows, bucketCount = 8) {
  if (!rows.length) return [];

  const times = rows.map((row) => row.Time);
  const minTime = Math.min(...times);
  const maxTime = Math.max(...times);

  if (minTime === maxTime) {
    return [{ label: `${minTime}`, count: rows.length }];
  }

  const range = maxTime - minTime;
  const step = range / bucketCount;
  const buckets = Array.from({ length: bucketCount }, (_, idx) => {
    const start = minTime + idx * step;
    const end = idx === bucketCount - 1 ? maxTime : start + step;
    return {
      start,
      end,
      label: `${Math.round(start)}-${Math.round(end)}`,
      count: 0,
    };
  });

  rows.forEach((row) => {
    const rawIndex = step === 0 ? 0 : Math.floor((row.Time - minTime) / step);
    const index = Math.min(bucketCount - 1, rawIndex);
    buckets[index].count += 1;
  });

  return buckets.map((bucket) => ({ label: bucket.label, count: bucket.count }));
}

function buildTopByAmount(rows) {
  return rows
    .filter((row) => row.probability >= 0.5)
    .sort((a, b) => b.Amount - a.Amount)
    .slice(0, 10)
    .map((row) => ({
      name: `#${row.index}`,
      amount: row.Amount,
      probability: row.probability,
    }));
}

function buildAmountSummary(rows) {
  if (!rows.length) {
    return { avg: 0, median: 0, total: 0 };
  }

  const amounts = rows.map((row) => row.Amount).sort((a, b) => a - b);
  const total = amounts.reduce((sum, amount) => sum + amount, 0);
  const avg = total / amounts.length;

  const middle = Math.floor(amounts.length / 2);
  const median =
    amounts.length % 2 === 0
      ? (amounts[middle - 1] + amounts[middle]) / 2
      : amounts[middle];

  return { avg, median, total };
}

export default function App() {
  const [metrics, setMetrics] = useState(null);
  const [alerts, setAlerts] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeSection, setActiveSection] = useState("overview");

  useEffect(() => {
    let mounted = true;

    async function fetchData() {
      try {
        const [metricsRes, alertsRes] = await Promise.all([
          fetch("/metrics.json"),
          fetch("/top_alerts.csv"),
        ]);

        if (!metricsRes.ok) throw new Error("Failed to load metrics.json");
        if (!alertsRes.ok) throw new Error("Failed to load top_alerts.csv");

        const metricsData = await metricsRes.json();
        const alertsText = await alertsRes.text();
        const alertsData = normalizeAlerts(parseCSV(alertsText));

        if (mounted) {
          setMetrics(metricsData);
          setAlerts(alertsData);
          setError(null);
        }
      } catch (err) {
        if (mounted) {
          setError(err.message || "Unable to load dashboard data.");
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    }

    fetchData();
    return () => {
      mounted = false;
    };
  }, []);

  const scoreDistribution = alerts ? buildScoreDistribution(alerts) : [];
  const breakdownData = alerts ? buildBreakdown(alerts) : [];
  const riskBands = alerts ? buildRiskBands(alerts) : { low: 0, medium: 0, high: 0 };
  const timeSeries = alerts ? buildTimeSeries(alerts) : [];
  const topByAmount = alerts ? buildTopByAmount(alerts) : [];
  const amountSummary = alerts
    ? buildAmountSummary(alerts)
    : { avg: 0, median: 0, total: 0 };

  const contentBySection = {
    overview: (
      <OverviewSection
        metrics={metrics}
        alerts={alerts}
        riskBands={riskBands}
        scoreDistribution={scoreDistribution}
        breakdownData={breakdownData}
      />
    ),
    alerts: <AlertsSection alerts={alerts} />,
    analytics: (
      <AnalyticsSection
        scoreDistribution={scoreDistribution}
        timeSeries={timeSeries}
        topByAmount={topByAmount}
        breakdownData={breakdownData}
        amountSummary={amountSummary}
      />
    ),
  };

  return (
    <div className="dashboard-shell">
      <Sidebar activeSection={activeSection} onChange={setActiveSection} />

      <main className="dashboard-main">
        {loading && <p className="loading-text">Loading dashboard data...</p>}

        {!loading && error && (
          <div className="error-banner">
            <p>
              {error} Place <strong>metrics.json</strong> and <strong>top_alerts.csv</strong>
              inside <strong>frontend/public/</strong>.
            </p>
          </div>
        )}

        {!loading && !error && metrics && alerts && contentBySection[activeSection]}
      </main>
    </div>
  );
}
