from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from scoring.ml_engine import MLRecommendationEngine
from utils.data_loader import DATA_PATH, DATASET_VERSION, generate_synthetic_dataset, validate_history_schema


def main() -> None:
    df = generate_synthetic_dataset(DATA_PATH, n_rows=900, seed=42, include_golden=True)
    issues = validate_history_schema(df)
    if issues:
        raise SystemExit("Generated data failed validation: " + "; ".join(issues))

    print(f"Reset cross-sell demo data to {DATA_PATH}")
    print(f"Dataset version: {DATASET_VERSION}")
    print(f"Rows: {len(df)}")
    print(f"Targetable rate: {df['historical_decision'].isin(['RM Priority Lead', 'Campaign Target', 'Digital Nurture']).mean():.2%}")
    print(f"Suppression rate: {(df['historical_decision'] == 'Suppress').mean():.2%}")
    print(f"Holdout rows: {int(df['holdout_control_flag'].sum())}")

    training_metrics = MLRecommendationEngine(model_dir=str(APP_DIR / "artifacts")).train(df)
    print(
        "Retrained ML artifact: "
        f"rows={training_metrics['rows_trained']:.0f}, "
        f"conversion_rate={training_metrics['conversion_rate']:.2%}"
    )


if __name__ == "__main__":
    main()
