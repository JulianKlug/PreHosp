from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from rl_recommender import AnalgesiaRecommender, prepare_analgesia_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the analgesia dosing recommender")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("../temp_data/trauma_categories_Rega Pain Study15.09.2025_v2.xlsx"),
        help="Path to the pre-hospital dataset (.xlsx)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to store trained artifacts",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_analgesia_dataset(args.data_path)
    dataset = prepared.data
    metadata = prepared.metadata

    train_df, test_df = train_test_split(
        dataset,
        test_size=args.test_size,
        stratify=dataset[metadata["target_column"]],
        random_state=args.seed,
    )

    agent = AnalgesiaRecommender(
        numeric_features=metadata["numeric_features"],
        categorical_features=metadata["categorical_features"],
        action_table=metadata["action_table"],
        random_state=args.seed,
    )
    agent.fit(train_df)
    metrics = agent.evaluate(train_df, test_df)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    action_mapping_path = output_dir / "action_mapping.csv"
    metadata["action_table"].to_csv(action_mapping_path, index=False)

    summary = _build_dataset_summary(dataset, metadata)
    summary_path = output_dir / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    sample_recos = (
        agent.recommend(test_df[metadata["feature_columns"]].head(25))
        .reset_index(drop=True)
    )
    sample_cases = test_df.head(25).reset_index(drop=True)
    result_preview = pd.concat([sample_cases, sample_recos], axis=1)
    preview_path = output_dir / "sample_recommendations.csv"
    result_preview.to_csv(preview_path, index=False)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved action mapping to {action_mapping_path}")
    print(f"Saved dataset summary to {summary_path}")
    print(f"Saved sample recommendations to {preview_path}")


def _build_dataset_summary(dataset, metadata: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_cases": int(len(dataset)),
        "success_rate": float(dataset[metadata["target_column"]].mean()),
        "fentanyl_usage_rate": float((dataset["fentanyl_total_mg"] > 0).mean()),
        "ketamine_usage_rate": float((dataset["ketamine_total_mg"] > 0).mean()),
        "reward_column": metadata["target_column"],
        "num_actions": int(metadata["num_actions"]),
    }

    return summary


if __name__ == "__main__":
    main()
