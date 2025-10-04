# Analgesia Prediction Tool

This module trains two supervised models to predict insufficient analgesia at hospital arrival (VAS_on_arrival > 3) using only prehospital data.

## Workflow

1. Activate the `prehosp` conda environment and make sure it includes `pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`, `openpyxl`, and `joblib`.
2. Ensure the Excel data file `analgesia/temp_data/trauma_categories_Rega Pain Study15.09.2025_v2.xlsx` is available.
3. Run the training script:

```bash
python -m analgesia.predictor.train_predictor --output-dir analgesia/predictor/artifacts
```

The command fits a logistic regression baseline and an XGBoost classifier, evaluates both, and stores:

- `metrics.json`: evaluation metrics, ROC curves (down-sampled), and feature summaries.
- `logistic_regression.joblib` and `xgboost_classifier.joblib`: serialized pipelines for downstream scoring.

Use `--multilabel-top-k` to adjust how many tokens are retained per multi-valued categorical column, `--validation-size` to control the internal validation split, and `--n-splits` to change cross-validation folds. Optional flags `--doctor-roster-path`, `--doctor-metadata-path`, and `--reference-year` override the default physician lookup sources when needed.

Running the trainer now produces:

- `metrics.json`: validation and test metrics (including PR-AUC) plus cross-validation summaries.
- `feature_importance.json`: consolidated coefficient/importances for both models.
- `table1.csv` / `table1.md`: descriptive statistics for training, validation, and test cohorts.
- `plots/`: comparison and feature-importance figures (generated via `plot_model_comparison.py`).
