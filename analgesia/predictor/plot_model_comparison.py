"""Generate comparative plots for logistic regression and XGBoost analgesia models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import load
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from .train_predictor import (
    augment_physician_features,
    drop_low_information_columns,
    load_filtered_dataset,
    tidy_location_features,
    detect_feature_roles,
)

sns.set_theme(style="whitegrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot model comparison charts")
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("analgesia/predictor/artifacts/metrics.json"),
        help="Path to JSON metrics output",
    )
    parser.add_argument(
        "--logistic-model",
        type=Path,
        default=Path("analgesia/predictor/artifacts/logistic_regression.joblib"),
        help="Path to fitted logistic regression pipeline",
    )
    parser.add_argument(
        "--xgb-model",
        type=Path,
        default=Path("analgesia/predictor/artifacts/xgboost_classifier.joblib"),
        help="Path to fitted XGBoost pipeline",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("analgesia/temp_data/trauma_categories_Rega Pain Study15.09.2025_v2.xlsx"),
        help="Source Excel dataset",
    )
    parser.add_argument(
        "--available-columns",
        type=Path,
        default=Path("analgesia/predictor/available_columns.md"),
        help="Markdown describing prehospital columns",
    )
    parser.add_argument(
        "--doctor-roster-path",
        type=Path,
        default=Path("analgesia/temp_data/Liste Notärzte-1.xlsx"),
        help="Physician roster with sex annotations",
    )
    parser.add_argument(
        "--doctor-metadata-path",
        type=Path,
        default=Path("analgesia/temp_data/final_complete_extractions_20251001_190933.xlsx"),
        help="Physician metadata with birth year and specialisations",
    )
    parser.add_argument(
        "--reference-year",
        type=int,
        default=2025,
        help="Reference year for age derivation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analgesia/predictor/artifacts/plots"),
        help="Directory to store generated plots",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size used during training",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Validation split fraction relative to the training fold",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for deterministic splitting",
    )
    return parser.parse_args()


def load_metrics(metrics_path: Path) -> dict:
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def plot_metric_bars(metrics: dict[str, dict[str, float]], output: Path) -> None:
    metric_keys = [
        "roc_auc",
        "average_precision",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "brier",
    ]
    labels = list(metrics)
    comparison_metrics = metric_keys[:-1]  # exclude brier for separate plot
    indices = np.arange(len(comparison_metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(indices - width / 2, [metrics[labels[0]][m] for m in comparison_metrics], width, label=labels[0])
    ax.bar(indices + width / 2, [metrics[labels[1]][m] for m in comparison_metrics], width, label=labels[1])
    ax.set_xticks(indices)
    ax.set_xticklabels([m.replace("_", " ").title() for m in comparison_metrics], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)

    # Brier score plot (lower is better)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, [metrics[label]["brier"] for label in labels], color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Brier Score (lower is better)")
    ax.set_title("Brier Score Comparison")
    fig.tight_layout()
    fig.savefig(output.with_name("brier_score_comparison.png"), dpi=300)
    plt.close(fig)


def bootstrap_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    kind: str,
    n_bootstrap: int = 500,
    random_state: int = 42,
    grid_size: int = 100,
):
    rng = np.random.default_rng(random_state)
    grid = np.linspace(0, 1, grid_size)
    curves = []
    aucs = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        y_sample = y_true[indices]
        prob_sample = y_prob[indices]
        if len(np.unique(y_sample)) < 2:
            continue
        if kind == "roc":
            fpr, tpr, _ = roc_curve(y_sample, prob_sample)
            aucs.append(roc_auc_score(y_sample, prob_sample))
            fpr = np.concatenate(([0.0], fpr, [1.0]))
            tpr = np.concatenate(([0.0], tpr, [1.0]))
            interp = np.interp(grid, fpr, tpr)
        else:
            precision, recall, _ = precision_recall_curve(y_sample, prob_sample)
            aucs.append(auc(recall, precision))
            recall_rev = recall[::-1]
            precision_rev = precision[::-1]
            # ensure bounds
            recall_rev[0], precision_rev[0] = 0.0, precision_rev[0]
            recall_rev[-1], precision_rev[-1] = 1.0, precision_rev[-1]
            interp = np.interp(grid, recall_rev, precision_rev)
        curves.append(interp)

    if not curves:
        return grid, None, None, None, (np.nan, np.nan)

    curves = np.array(curves)
    mean = curves.mean(axis=0)
    lower = np.percentile(curves, 2.5, axis=0)
    upper = np.percentile(curves, 97.5, axis=0)
    auc_ci = np.percentile(aucs, [2.5, 97.5]) if aucs else (np.nan, np.nan)
    return grid, mean, lower, upper, auc_ci


def plot_curves(
    X_test,
    y_test,
    logistic_pipeline,
    xgb_pipeline,
    output_dir: Path,
) -> None:
    y_true = np.asarray(y_test)
    y_prob_log = logistic_pipeline.predict_proba(X_test)[:, 1]
    y_prob_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]

    # ROC
    fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    auc_log = roc_auc_score(y_test, y_prob_log)
    auc_xgb = roc_auc_score(y_test, y_prob_xgb)

    roc_grid_log, _, roc_lower_log, roc_upper_log, auc_ci_log = bootstrap_curve(
        y_true, y_prob_log, kind="roc"
    )
    roc_grid_xgb, _, roc_lower_xgb, roc_upper_xgb, auc_ci_xgb = bootstrap_curve(
        y_true, y_prob_xgb, kind="roc"
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    label_log = f"Logistic (AUC {auc_log:.3f} [{auc_ci_log[0]:.3f}, {auc_ci_log[1]:.3f}])"
    label_xgb = f"XGBoost (AUC {auc_xgb:.3f} [{auc_ci_xgb[0]:.3f}, {auc_ci_xgb[1]:.3f}])"
    ax.plot(fpr_log, tpr_log, label=label_log, color="#1f77b4")
    ax.plot(fpr_xgb, tpr_xgb, label=label_xgb, color="#ff7f0e")
    if roc_lower_log is not None:
        ax.fill_between(roc_grid_log, roc_lower_log, roc_upper_log, color="#1f77b4", alpha=0.15)
    if roc_lower_xgb is not None:
        ax.fill_between(roc_grid_xgb, roc_lower_xgb, roc_upper_xgb, color="#ff7f0e", alpha=0.15)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_comparison.png", dpi=300)
    plt.close(fig)

    # Precision-Recall
    precision_log, recall_log, _ = precision_recall_curve(y_test, y_prob_log)
    precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)

    pr_auc_log = auc(recall_log, precision_log)
    pr_auc_xgb = auc(recall_xgb, precision_xgb)
    pr_grid_log, _, pr_lower_log, pr_upper_log, pr_ci_log = bootstrap_curve(
        y_true, y_prob_log, kind="pr"
    )
    pr_grid_xgb, _, pr_lower_xgb, pr_upper_xgb, pr_ci_xgb = bootstrap_curve(
        y_true, y_prob_xgb, kind="pr"
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    label_log_pr = f"Logistic (PR AUC {pr_auc_log:.3f} [{pr_ci_log[0]:.3f}, {pr_ci_log[1]:.3f}])"
    label_xgb_pr = f"XGBoost (PR AUC {pr_auc_xgb:.3f} [{pr_ci_xgb[0]:.3f}, {pr_ci_xgb[1]:.3f}])"
    ax.plot(recall_log, precision_log, label=label_log_pr, color="#1f77b4")
    ax.plot(recall_xgb, precision_xgb, label=label_xgb_pr, color="#ff7f0e")
    if pr_lower_log is not None:
        ax.fill_between(pr_grid_log, pr_lower_log, pr_upper_log, color="#1f77b4", alpha=0.15)
    if pr_lower_xgb is not None:
        ax.fill_between(pr_grid_xgb, pr_lower_xgb, pr_upper_xgb, color="#ff7f0e", alpha=0.15)
    baseline = y_true.mean()
    ax.hlines(baseline, xmin=0, xmax=1, linestyles="--", color="grey", alpha=0.5, label=f"Baseline = {baseline:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_dir / "pr_comparison.png", dpi=300)
    plt.close(fig)


def plot_logistic_feature_importance(contribs: dict, output_dir: Path, top_n: int = 10) -> None:
    top_positive = contribs.get("top_positive", [])[:top_n]
    top_negative = contribs.get("top_negative", [])[:top_n]
    top_magnitude = contribs.get("top_magnitude", [])[:top_n]

    if top_positive:
        features = [item["feature"] for item in top_positive][::-1]
        values = [item["coefficient"] for item in top_positive][::-1]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(features, values, color="#1f77b4")
        ax.set_xlabel("Coefficient")
        ax.set_title("Logistic Regression – Top Positive Coefficients")
        fig.tight_layout()
        fig.savefig(output_dir / "logistic_top_positive.png", dpi=300)
        plt.close(fig)

    if top_negative:
        features = [item["feature"] for item in top_negative]
        values = [item["coefficient"] for item in top_negative]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(features, values, color="#d62728")
        ax.set_xlabel("Coefficient")
        ax.set_title("Logistic Regression – Top Negative Coefficients")
        fig.tight_layout()
        fig.savefig(output_dir / "logistic_top_negative.png", dpi=300)
        plt.close(fig)

    if top_magnitude:
        features = [item["feature"] for item in top_magnitude][::-1]
        values = [item["abs_coefficient"] for item in top_magnitude][::-1]
        colors = ["#9467bd" for _ in values]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(features, values, color=colors)
        ax.set_xlabel("|Coefficient|")
        ax.set_title("Logistic Regression – Largest Absolute Effects")
        fig.tight_layout()
        fig.savefig(output_dir / "logistic_top_magnitude.png", dpi=300)
        plt.close(fig)


def plot_xgb_feature_importance(contribs: dict, output_dir: Path, top_n: int = 15) -> None:
    importance = contribs.get("importance", [])[:top_n]
    if not importance:
        return
    features = [item["feature"] for item in importance][::-1]
    values = [item["importance"] for item in importance][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(features, values, color="#ff7f0e")
    ax.set_xlabel("Gain Importance")
    ax.set_title("XGBoost – Top Feature Importances")
    fig.tight_layout()
    fig.savefig(output_dir / "xgboost_feature_importance.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_data = load_metrics(args.metrics_path)
    model_metrics = {
        "Logistic Regression": metrics_data["logistic_regression"]["test_metrics"],
        "XGBoost": metrics_data["xgboost"]["test_metrics"],
    }

    plot_metric_bars(model_metrics, args.output_dir / "metric_comparison.png")

    # Load data and preprocess to recreate the test split
    features, target, _ = load_filtered_dataset(args.data_path, args.available_columns)
    features = augment_physician_features(
        features,
        args.doctor_roster_path,
        args.doctor_metadata_path,
        args.reference_year,
    )
    features = tidy_location_features(features)
    roles = detect_feature_roles(features.copy())
    features_model = features.copy()
    roles = drop_low_information_columns(features_model, roles)

    X_temp, X_test, y_temp, y_test = train_test_split(
        features_model,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=args.validation_size,
        random_state=args.random_state,
        stratify=y_temp,
    )

    logistic_pipeline = load(args.logistic_model)
    xgb_pipeline = load(args.xgb_model)

    plot_curves(X_test, y_test, logistic_pipeline, xgb_pipeline, args.output_dir)

    # Calibration curve
    y_true = np.asarray(y_test)
    prob_log = logistic_pipeline.predict_proba(X_test)[:, 1]
    prob_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, probs, color in [
        ("Logistic", prob_log, "#1f77b4"),
        ("XGBoost", prob_xgb, "#ff7f0e"),
    ]:
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", label=label, color=color)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.6)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(args.output_dir / "calibration_curve.png", dpi=300)
    plt.close(fig)

    plot_logistic_feature_importance(
        metrics_data["logistic_regression"].get("feature_contributions", {}),
        args.output_dir,
    )
    plot_xgb_feature_importance(
        metrics_data["xgboost"].get("feature_contributions", {}),
        args.output_dir,
    )

    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
