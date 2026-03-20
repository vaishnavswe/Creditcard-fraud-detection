import React from "react";
import {
  Bar,
  BarChart,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Metrics from "../components/Metrics";
import AlertsTable from "../components/AlertsTable";
import RiskBandSummary from "../components/RiskBandSummary";
import SectionHeader from "../components/SectionHeader";

const BREAKDOWN_COLORS = {
  Normal: "#16A34A",
  Fraud: "#DC2626",
};

export default function OverviewSection({
  metrics,
  alerts,
  riskBands,
  scoreDistribution,
  breakdownData,
}) {
  return (
    <section className="page-section">
      <SectionHeader
        title="Overview"
        subtitle="Real-time fraud monitoring and model summary"
      />

      <Metrics data={metrics} detailed={false} />

      <div className="chart-grid two-col">
        <article className="dashboard-card chart-card">
          <div className="card-head">
            <h3>Fraud score distribution</h3>
            <p>Probability buckets across ranked transactions</p>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={scoreDistribution}>
                <XAxis dataKey="label" tick={{ fill: "#5B6B93", fontSize: 12 }} />
                <YAxis tick={{ fill: "#5B6B93", fontSize: 12 }} allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                  {scoreDistribution.map((entry) => (
                    <Cell key={entry.label} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="dashboard-card chart-card">
          <div className="card-head">
            <h3>Fraud vs normal breakdown</h3>
            <p>Class composition of flagged transactions</p>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={breakdownData}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={64}
                  outerRadius={90}
                  paddingAngle={2}
                >
                  {breakdownData.map((entry) => (
                    <Cell key={entry.name} fill={BREAKDOWN_COLORS[entry.name]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </article>
      </div>

      <RiskBandSummary bands={riskBands} />

      <AlertsTable
        rows={alerts}
        title="Top alerts preview"
        subtitle="Highest-risk transactions for immediate analyst review"
        limit={10}
        showFilters={false}
      />
    </section>
  );
}
