# Credit Card Fraud Detection — Simple ML + React

A minimal project: **Logistic Regression** detects fraud, **React** shows results. No backend.

## Project Structure

```
data/creditcard.csv        ← Kaggle dataset
src/train.py               ← Train model & generate outputs
src/predict.py             ← Re-predict on full dataset
src/utils.py               ← Shared helpers
outputs/                   ← Generated: metrics, alerts, model files
frontend/                  ← React dashboard (reads static files)
```

## Quick Start

### 1. Python — Train the model

```bash
# Create & activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Train
python src/train.py

# (Optional) Re-predict on full dataset
python src/predict.py
```

**Outputs saved to `outputs/`:**
- `metrics.json` — precision, recall, f1, PR-AUC
- `top_alerts.csv` — top 200 transactions ranked by fraud probability
- `model.joblib` / `scaler.joblib` — serialized model & scaler

### 2. Copy outputs to frontend

```bash
copy outputs\metrics.json frontend\public\
copy outputs\top_alerts.csv frontend\public\
```

### 3. React — View dashboard

```bash
cd frontend
npm install
npm start
```

Open [http://localhost:3000](http://localhost:3000) — you'll see metric cards and a table of top fraud alerts.

## Dataset

Download **creditcard.csv** from Kaggle and place it at `data/creditcard.csv`.  
Label column: `Class` (1 = fraud, 0 = normal).
