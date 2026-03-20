import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_data(csv_path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load the credit-card CSV. Exit with a helpful message if missing."""
    if not os.path.exists(csv_path):
        print(f"\n[ERROR] Dataset not found at '{csv_path}'.")
        print("  -> Download it from Kaggle and place it at:")
        print(f"     {os.path.abspath(csv_path)}\n")
        raise SystemExit(1)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save a dict as a pretty-printed JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {path}")
