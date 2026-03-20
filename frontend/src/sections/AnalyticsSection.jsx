import React from "react";
import {
  Area,
  AreaChart,
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
import SectionHeader from "../components/SectionHeader";

const BREAKDOWN_COLORS = {
  Normal: "#16A34A",
  Fraud: "#DC2626",
};

export default function AnalyticsSection({
  scoreDistribution,
  timeSeries,
  topByAmount,
  breakdownData,
  amountSummary,
}) {
  return (
    <section className="page-section">
      <SectionHeader
        title="Analytics"
        subtitle="Pattern and trend analysis for fraud activity"
      />

      <div className="chart-grid two-col">
        <article className="dashboard-card chart-card">
          <div className="card-head">
            <h3>Fraud score distribution</h3>
            <p>Risk concentration across fraud scores</p>
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
            <h3>Alerts over time</h3>
            <p>Distribution of flagged transactions by time range</p>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={timeSeries}>
                <XAxis dataKey="label" tick={{ fill: "#5B6B93", fontSize: 12 }} />
                <YAxis tick={{ fill: "#5B6B93", fontSize: 12 }} allowDecimals={false} />
                <Tooltip />
                <Area
                  type="monotone"
                  dataKey="count"
                  stroke="#1D4ED8"
                  fill="#1D4ED8"
                  fillOpacity={0.16}
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="dashboard-card chart-card">
          <div className="card-head">
            <h3>Top risky transactions by amount</h3>
            <p>High-risk events with the largest transaction values</p>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={topByAmount} layout="vertical" margin={{ left: 8, right: 8 }}>
                <XAxis type="number" tick={{ fill: "#5B6B93", fontSize: 12 }} />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={80}
                  tick={{ fill: "#5B6B93", fontSize: 12 }}
                />
                <Tooltip />
                <Bar dataKey="amount" radius={[0, 8, 8, 0]}>
                  {topByAmount.map((entry) => (
                    <Cell
                      key={entry.name}
                      fill={entry.probability >= 0.7 ? "#DC2626" : "#2563EB"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="dashboard-card chart-card">
          <div className="card-head">
            <h3>Fraud vs normal comparison</h3>
            <p>Observed label mix from ranked alerts</p>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={breakdownData}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={58}
                  outerRadius={86}
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

      <article className="dashboard-card amount-summary-card">
        <div className="card-head">
          <h3>Amount distribution summary</h3>
          <p>Aggregate transaction amounts across the current alert set</p>
        </div>
        <div className="amount-summary-grid">
          <div>
            <p>Average amount</p>
            <strong>${amountSummary.avg.toFixed(2)}</strong>
          </div>
          <div>
            <p>Median amount</p>
            <strong>${amountSummary.median.toFixed(2)}</strong>
          </div>
          <div>
            <p>Total amount</p>
            <strong>${amountSummary.total.toFixed(2)}</strong>
          </div>
        </div>
      </article>
    </section>
  );
}
