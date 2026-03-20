import React from "react";
import AlertsTable from "../components/AlertsTable";
import SectionHeader from "../components/SectionHeader";

export default function AlertsSection({ alerts }) {
  return (
    <section className="page-section">
      <SectionHeader
        title="Alerts"
        subtitle="Ranked transactions by fraud probability"
      />

      <AlertsTable
        rows={alerts}
        title="Investigation queue"
        subtitle="Prioritized by fraud score to support rapid triage"
        showFilters
      />
    </section>
  );
}
