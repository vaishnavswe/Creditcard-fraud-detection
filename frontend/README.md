# Fraud Detection Frontend

React dashboard showing fraud detection metrics and top alerts.  
No backend — reads static JSON/CSV files from `public/`.

## Setup

```bash
# Make sure the ML outputs are in public/
# (from the project root)
copy outputs\metrics.json frontend\public\
copy outputs\top_alerts.csv frontend\public\

# Install & run
cd frontend
npm install
npm start
```

Open [http://localhost:3000](http://localhost:3000).

## What you'll see

1. **Metric cards** — Precision, Recall, F1 Score, PR-AUC
2. **Alerts table** — Top 200 transactions ranked by fraud probability (fraud rows highlighted in orange)
