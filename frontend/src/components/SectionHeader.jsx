import React from "react";

export default function SectionHeader({ title, subtitle }) {
  return (
    <header className="section-header">
      <h2>{title}</h2>
      <p>{subtitle}</p>
    </header>
  );
}
