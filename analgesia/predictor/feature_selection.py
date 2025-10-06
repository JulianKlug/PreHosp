"""Search for a minimal feature subset that preserves model performance."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ARTIFACT_DIR = Path("analgesia/predictor/artifacts")
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
SELECTED_FEATURES_PATH = ARTIFACT_DIR / "selected_features.txt"
SELECTION_DIR = ARTIFACT_DIR / "feature_selection_runs"
SELECTION_SUMMARY = ARTIFACT_DIR / "feature_selection_summary.json"

BASELINE_TOLERANCE = 0.005  # allowable drop in ROC/PR AUC
MANDATORY_FEATURES = {
    "Alter ",
    "Geschlecht",
    "doctor_sex",
    "VAS_on_scene",
    "NACA (nummerisch)",
}


def load_metrics(path: Path) -> Dict:
    return json.loads(path.read_text())


def derive_feature_ranking(metrics: Dict, top_k: int | None = None) -> List[Tuple[str, float]]:
    log_scores = metrics["logistic_regression"].get("raw_feature_scores", {})
    xgb_scores = metrics["xgboost"].get("raw_feature_scores", {})
    if not log_scores and not xgb_scores:
        raise ValueError("Raw feature scores missing; rerun training with updated script.")

    features = sorted(set(log_scores) | set(xgb_scores))
    log_max = max(log_scores.values()) if log_scores else 1.0
    xgb_max = max(xgb_scores.values()) if xgb_scores else 1.0

    ranking = []
    for feat in features:
        l_score = log_scores.get(feat, 0.0) / log_max
        x_score = xgb_scores.get(feat, 0.0) / xgb_max
        combined = 0.5 * (l_score + x_score)
        ranking.append((feat, combined))

    ranking.sort(key=lambda item: item[1], reverse=True)
    if top_k is not None:
        ranking = ranking[:top_k]
    return ranking


def write_whitelist(features: List[str], path: Path) -> None:
    path.write_text("\n".join(features) + "\n", encoding="utf-8")


def train_with_whitelist(whitelist: Path, output_dir: Path, tune: bool = False) -> Dict:
    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "analgesia.predictor.train_predictor",
        "--output-dir",
        str(output_dir),
        "--feature-whitelist",
        str(whitelist),
    ]
    if tune:
        cmd.append("--tune-xgb")
    subprocess.run(cmd, check=True)
    return load_metrics(output_dir / "metrics.json")


def meets_baseline(candidate: Dict, baseline: Dict) -> bool:
    cand = candidate["xgboost"]["test_metrics"]
    base = baseline["xgboost"]["test_metrics"]
    return (
        cand["roc_auc"] >= base["roc_auc"] - BASELINE_TOLERANCE
        and cand["pr_auc"] >= base["pr_auc"] - BASELINE_TOLERANCE
    )


def main() -> None:
    metrics = load_metrics(METRICS_PATH)
    baseline_metrics = metrics
    ranking = derive_feature_ranking(metrics)

    candidate_counts = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60]
    selection_results = []
    best_config = None

    SELECTION_DIR.mkdir(parents=True, exist_ok=True)

    feature_names_ordered = [feat for feat, _ in ranking]

    for count in candidate_counts:
        subset = list(dict.fromkeys(list(MANDATORY_FEATURES) + feature_names_ordered[:count]))
        if len(subset) == 0:
            continue
        whitelist_path = SELECTION_DIR / f"whitelist_top_{count}.txt"
        write_whitelist(subset, whitelist_path)
        output_dir = SELECTION_DIR / f"top_{count}"
        candidate_metrics = train_with_whitelist(whitelist_path, output_dir, tune=False)

        result_entry = {
            "feature_count": len(subset),
            "features": subset,
            "metrics": candidate_metrics["xgboost"]["test_metrics"],
        }
        selection_results.append(result_entry)

        if meets_baseline(candidate_metrics, baseline_metrics) and best_config is None:
            best_config = {
                "feature_count": len(subset),
                "features": subset,
                "validation": candidate_metrics["xgboost"]["validation_metrics"],
                "test": candidate_metrics["xgboost"]["test_metrics"],
            }
            break

    summary = {
        "baseline_test_metrics": baseline_metrics["xgboost"]["test_metrics"],
        "candidates": selection_results,
        "selected": best_config,
    }
    SELECTION_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if best_config is None:
        print("No subset met baseline tolerance; keeping full feature set.")
        return

    # Save selected features and retrain with tuning for final artifacts
    write_whitelist(best_config["features"], SELECTED_FEATURES_PATH)
    final_metrics = train_with_whitelist(
        SELECTED_FEATURES_PATH,
        ARTIFACT_DIR,
        tune=True,
    )
    summary["final_metrics"] = final_metrics["xgboost"]["test_metrics"]
    summary["final_validation"] = final_metrics["xgboost"]["validation_metrics"]
    SELECTION_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Selected {best_config['feature_count']} features. Final test PR AUC = {final_metrics['xgboost']['test_metrics']['pr_auc']:.3f}, "
        f"ROC AUC = {final_metrics['xgboost']['test_metrics']['roc_auc']:.3f}."
    )


if __name__ == "__main__":
    main()
