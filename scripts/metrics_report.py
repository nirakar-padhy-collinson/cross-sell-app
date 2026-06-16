from __future__ import annotations

import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from utils.data_loader import DATA_PATH, DATASET_VERSION, load_or_create_data, validate_history_schema
from utils.monitoring import portfolio_report


def main() -> None:
    df = load_or_create_data(DATA_PATH)
    issues = validate_history_schema(df)
    report = {
        "dataset_version": DATASET_VERSION,
        "data_path": str(DATA_PATH),
        "schema_issues": issues,
        "portfolio": portfolio_report(df),
    }
    out_path = APP_DIR / "artifacts" / "portfolio_metrics_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
