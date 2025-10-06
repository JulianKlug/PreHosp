"""Train analgesia prediction models using prehospital data features.

This script prepares the filtered study cohort, engineers model-ready features, and
trains both a logistic regression baseline and an XGBoost classifier. Evaluation
metrics and trained pipelines are written to the configured output directory.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from scipy.stats import loguniform, randint, uniform

from .transformers import MultiLabelTopKEncoder


COLUMN_ALIASES: Dict[str, str] = {
    "Alter": "Alter ",
    "SPO2 Einsatzart": "SPO2",
    "Aktuelles Ereignis Hauptdiagnose": "Aktuelles Ereignis",
    "(Be)-Atmung Lichtreaktion (li)": "(Be)-Atmung",
}

MULTI_LABEL_DELIMITERS = (";", ",")


@dataclass
class FeatureRoles:
    numeric: List[str]
    categorical: List[str]
    multilabel: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train analgesia prediction models")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("analgesia/temp_data/trauma_categories_Rega Pain Study15.09.2025_v2.xlsx"),
        help="Path to the master Excel dataset",
    )
    parser.add_argument(
        "--available-columns",
        type=Path,
        default=Path("analgesia/predictor/available_columns.md"),
        help="Markdown file enumerating features available pre-hospital",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analgesia/predictor/artifacts"),
        help="Directory where models and metrics will be saved",
    )
    parser.add_argument(
        "--doctor-roster-path",
        type=Path,
        default=Path("analgesia/temp_data/Liste Notärzte-1.xlsx"),
        help="Path to roster file with physician sex information",
    )
    parser.add_argument(
        "--doctor-metadata-path",
        type=Path,
        default=Path("analgesia/temp_data/final_complete_extractions_20251001_190933.xlsx"),
        help="Path to metadata file with physician birth years and specialisations",
    )
    parser.add_argument(
        "--reference-year",
        type=int,
        default=2025,
        help="Reference year used to convert year of birth to age",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size (0-1)")
    parser.add_argument("--random-state", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument(
        "--multilabel-top-k",
        type=int,
        default=25,
        help="Number of tokens to retain per multi-label feature",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Proportion of the training fold reserved for validation",
    )
    parser.add_argument(
        "--tune-xgb",
        action="store_true",
        help="Run randomized hyperparameter search for the XGBoost model",
    )
    parser.add_argument(
        "--xgb-tuning-iterations",
        type=int,
        default=40,
        help="Number of parameter configurations sampled during XGBoost tuning",
    )
    parser.add_argument(
        "--xgb-tuning-scoring",
        type=str,
        default="average_precision",
        help="Scikit-learn scoring metric used during XGBoost tuning",
    )
    parser.add_argument(
        "--feature-whitelist",
        type=Path,
        default=None,
        help="Optional newline-delimited file listing raw feature columns to retain",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Cross-validation folds for metric estimation",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_allowed_columns(md_path: Path, dataset_columns: Sequence[str]) -> Tuple[List[str], List[str]]:
    text = md_path.read_text(encoding="utf-8")
    columns: List[str] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            if columns:
                break
            continue
        line = raw_line
        if ":" in line:
            _, line = line.split(":", 1)
        parts = [part.strip() for part in line.split("\t") if part.strip()]
        columns.extend(parts)
    resolved: List[str] = []
    missing: List[str] = []
    seen: set[str] = set()
    for column in columns:
        dataset_column = COLUMN_ALIASES.get(column, column)
        if dataset_column in dataset_columns and dataset_column not in seen:
            resolved.append(dataset_column)
            seen.add(dataset_column)
        else:
            missing.append(column)
    return resolved, missing


def load_filtered_dataset(
    data_path: Path, allowed_columns_path: Path
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
    logging.info("Loading dataset from %s", data_path)
    df = pd.read_excel(data_path, sheet_name="Grid")

    duplicate_key = "SNZ Ereignis Nr. "
    if duplicate_key in df.columns:
        original_size = len(df)
        df = df.drop_duplicates(subset=[duplicate_key])
        removed = original_size - len(df)
        if removed:
            logging.info("Dropped %s duplicate missions based on '%s'", removed, duplicate_key)
    feature_columns, missing = parse_allowed_columns(allowed_columns_path, df.columns)
    if missing:
        logging.warning("The following allowed columns were not found in the dataset: %s", ", ".join(missing))
    required = ["VAS_on_scene", "VAS_on_arrival", "Einteilung (reduziert)", "Einsatzart", "Alter "]
    for column in required:
        if column not in df.columns:
            raise KeyError(f"Required column '{column}' not present in dataset")

    filtered = df[
        (df["VAS_on_scene"] > 3)
        & (df["Einteilung (reduziert)"] == "Unfall")
        & (df["Einsatzart"] == "Primär")
        & (df["Alter "] >= 16)
        & df["VAS_on_arrival"].notna()
    ].copy()

    if duplicate_key in filtered.columns:
        filtered_before = len(filtered)
        filtered = filtered.drop_duplicates(subset=[duplicate_key]).copy()
        filtered_removed = filtered_before - len(filtered)
        if filtered_removed:
            logging.info(
                "Removed %s duplicate missions from filtered cohort using '%s'",
                filtered_removed,
                duplicate_key,
            )

    logging.info("Filtered cohort size: %s rows", len(filtered))

    features = filtered[feature_columns].copy()
    target = (filtered["VAS_on_arrival"] > 3).astype(int)

    summary = {
        "rows_total": int(len(df)),
        "rows_filtered": int(len(filtered)),
        "positives": int(target.sum()),
        "negatives": int((1 - target).sum()),
    }
    return features, target, summary


def clean_doctor_names(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s*\(.*\)$", "", regex=True)
        .str.strip()
    )


def load_doctor_lookup(
    roster_path: Path, metadata_path: Path, reference_year: int
) -> pd.DataFrame:
    roster = pd.read_excel(roster_path)
    roster = roster.rename(columns={"Sex m/w": "doctor_sex_raw"})
    roster["doctor_name"] = clean_doctor_names(roster["Mitglieder mit Einsatzfunktion"])
    roster["doctor_sex"] = (
        roster["doctor_sex_raw"].astype(str).str.strip().str.lower().map({"m": "male", "w": "female"})
    )
    roster_lookup = roster.dropna(subset=["doctor_name"]).drop_duplicates("doctor_name")

    metadata = pd.read_excel(metadata_path)
    metadata["doctor_name"] = (
        metadata["first_name"].fillna("").astype(str).str.strip()
        + " "
        + metadata["last_name"].fillna("").astype(str).str.strip()
    ).str.strip()
    metadata["doctor_name"] = metadata["doctor_name"].replace("", pd.NA)
    metadata["year_of_birth"] = pd.to_numeric(metadata["year_of_birth"], errors="coerce")
    metadata["doctor_age"] = reference_year - metadata["year_of_birth"]
    metadata.loc[metadata["year_of_birth"].isna(), "doctor_age"] = np.nan
    metadata["doctor_specialist_qualifications"] = (
        metadata["specialist_qualifications"].fillna("").astype(str)
    )
    metadata_lookup = metadata.dropna(subset=["doctor_name"]).drop_duplicates("doctor_name")

    doctor_lookup = roster_lookup[["doctor_name", "doctor_sex"]].merge(
        metadata_lookup[["doctor_name", "doctor_age", "doctor_specialist_qualifications"]],
        on="doctor_name",
        how="outer",
    )
    doctor_lookup["doctor_sex"] = doctor_lookup["doctor_sex"].fillna("unknown")
    doctor_lookup["doctor_specialist_qualifications"] = doctor_lookup[
        "doctor_specialist_qualifications"
    ].fillna("")
    return doctor_lookup.set_index("doctor_name")


def augment_physician_features(
    features: pd.DataFrame,
    roster_path: Path,
    metadata_path: Path,
    reference_year: int,
) -> pd.DataFrame:
    doctor_column = "Mitglieder mit Einsatzfunktion"
    if doctor_column not in features.columns:
        logging.warning("Doctor column '%s' not available; skipping augmentation", doctor_column)
        return features

    features = features.copy()
    cleaned_names = clean_doctor_names(features[doctor_column])
    doctor_lookup = load_doctor_lookup(roster_path, metadata_path, reference_year)

    sex_map = doctor_lookup["doctor_sex"].to_dict()
    age_map = doctor_lookup["doctor_age"].to_dict()
    specialty_map = doctor_lookup["doctor_specialist_qualifications"].to_dict()

    features["doctor_sex"] = cleaned_names.map(sex_map).fillna("unknown")
    features["doctor_age"] = cleaned_names.map(age_map)
    features["doctor_specialist_qualifications"] = cleaned_names.map(specialty_map).fillna("")

    missing_mask = ~cleaned_names.isin(doctor_lookup.index)
    if missing_mask.any():
        logging.info(
            "Doctor metadata missing for %s missions",
            int(cleaned_names[missing_mask].nunique()),
        )

    features.drop(columns=[doctor_column], inplace=True)
    return features


def tidy_location_features(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()

    if "PLZ" in features.columns or "PLZ4" in features.columns:
        combined = features.get("PLZ4")
        if combined is None:
            combined = features.get("PLZ")
        elif "PLZ" in features.columns:
            combined = combined.combine_first(features["PLZ"])
        features["PLZ"] = pd.to_numeric(combined, errors="coerce")

    drop_cols = [col for col in ["PLZ4", "Strassennummer", "Gemeinde"] if col in features.columns]
    if drop_cols:
        features = features.drop(columns=drop_cols)

    return features


def _coalesce_numeric_columns(
    features: pd.DataFrame, new_name: str, candidates: Sequence[str]
) -> pd.DataFrame:
    available = [column for column in candidates if column in features.columns]
    if not available:
        return features

    combined: pd.Series | None = None
    for column in available:
        series = pd.to_numeric(features[column], errors="coerce")
        if combined is None:
            combined = series
        else:
            combined = combined.combine_first(series)

    if combined is None:
        return features

    features[new_name] = combined
    for column in available:
        if column != new_name and column in features.columns:
            features.drop(columns=column, inplace=True)
    return features


def _normalize_multilabel_text(series: pd.Series) -> pd.Series:
    splitter = re.compile(r"[;,/\\n]+")

    def normalize_entry(value: object) -> str | float:
        if value is None or pd.isna(value):
            return np.nan
        text = str(value).strip()
        if not text:
            return np.nan
        tokens = [token.strip() for token in splitter.split(text) if token.strip()]
        if not tokens:
            return np.nan
        unique_tokens: List[str] = []
        seen_lower: set[str] = set()
        for token in tokens:
            lowered = token.lower()
            if lowered not in seen_lower:
                unique_tokens.append(token)
                seen_lower.add(lowered)
        return "; ".join(unique_tokens)

    normalized = series.apply(normalize_entry)
    return normalized


def _join_unique_tokens(tokens: Iterable[str]) -> str | float:
    unique_tokens: List[str] = []
    seen_lower: set[str] = set()
    for token in tokens:
        cleaned = str(token).strip()
        if not cleaned or cleaned == "-" or cleaned == "--:--:--":
            continue
        lowered = cleaned.lower()
        if lowered not in seen_lower:
            unique_tokens.append(cleaned)
            seen_lower.add(lowered)
    if not unique_tokens:
        return np.nan
    return "; ".join(unique_tokens)


def _extract_venous_access_features(series: pd.Series) -> pd.DataFrame:
    splitter = re.compile(r"[;\n]+")

    counts: List[int] = []
    types: List[str | float] = []
    locations: List[str | float] = []
    sides: List[str | float] = []
    sizes: List[str | float] = []
    preexisting: List[str | float] = []
    inserted_by: List[str | float] = []

    for value in series:
        if value is None or pd.isna(value):
            counts.append(0)
            types.append(np.nan)
            locations.append(np.nan)
            sides.append(np.nan)
            sizes.append(np.nan)
            preexisting.append(np.nan)
            inserted_by.append(np.nan)
            continue

        text = str(value).strip()
        if not text or text == "-":
            counts.append(0)
            types.append(np.nan)
            locations.append(np.nan)
            sides.append(np.nan)
            sizes.append(np.nan)
            preexisting.append(np.nan)
            inserted_by.append(np.nan)
            continue

        access_entries = [entry.strip() for entry in splitter.split(text) if entry.strip()]
        parsed_entries: List[List[str]] = []
        for entry in access_entries:
            parts = [part.strip() for part in entry.split(",")]
            if not any(parts):
                continue
            while len(parts) < 7:
                parts.append("")
            parsed_entries.append(parts[:7])

        counts.append(len(parsed_entries))
        if not parsed_entries:
            types.append(np.nan)
            locations.append(np.nan)
            sides.append(np.nan)
            sizes.append(np.nan)
            preexisting.append(np.nan)
            inserted_by.append(np.nan)
            continue

        type_tokens = [parts[0] for parts in parsed_entries if len(parts) > 0]
        location_tokens = [parts[1] for parts in parsed_entries if len(parts) > 1]
        side_tokens = [parts[2] for parts in parsed_entries if len(parts) > 2]
        size_tokens = [parts[3] for parts in parsed_entries if len(parts) > 3]
        preexisting_tokens = [parts[4] for parts in parsed_entries if len(parts) > 4]
        inserted_by_tokens = [parts[5] for parts in parsed_entries if len(parts) > 5]

        types.append(_join_unique_tokens(type_tokens))
        locations.append(_join_unique_tokens(location_tokens))
        sides.append(_join_unique_tokens(side_tokens))
        sizes.append(_join_unique_tokens(size_tokens))
        preexisting.append(_join_unique_tokens(preexisting_tokens))
        inserted_by.append(_join_unique_tokens(inserted_by_tokens))

    return pd.DataFrame(
        {
            "venous_access_count": counts,
            "venous_access_types": types,
            "venous_access_locations": locations,
            "venous_access_sides": sides,
            "venous_access_sizes": sizes,
            "venous_access_preexisting": preexisting,
            "venous_access_inserted_by": inserted_by,
        },
        index=series.index,
    )


def apply_feature_modifications(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()

    features = _coalesce_numeric_columns(
        features,
        "heart_rate",
        (
            "heart_rate",
            "HF",
            "HF ",
            "HR",
        ),
    )

    features = _coalesce_numeric_columns(
        features,
        "diastolic",
        (
            "diastolic",
            "IBD Diastolisch ",
            "IBD Diastolisch",
            "IBDDiastolisch",
            "NIBD Diastolisch",
            "NIBD Diastolisch ",
            "NIBD Diastolisch10",
        ),
    )

    features = _coalesce_numeric_columns(
        features,
        "systolic",
        (
            "systolic",
            "IBD Systolisch ",
            "IBD Systolisch",
            "NIBD Systolisch",
            "NIBD Systolisch ",
        ),
    )

    features = _coalesce_numeric_columns(
        features,
        "naca",
        (
            "naca",
            "NACA (nummerisch)",
            "NACA",
        ),
    )

    if "Weitere Massnahmen" in features.columns:
        features["Weitere Massnahmen"] = _normalize_multilabel_text(features["Weitere Massnahmen"])

    if "Zugänge" in features.columns:
        access_features = _extract_venous_access_features(features["Zugänge"])
        features = features.drop(columns=["Zugänge"]).join(access_features)

    drop_columns = [
        "Detail",
        "ICD-Code der Hauptdiagnose",
        "Alle ICD-Codes",
        "Institution",
        "Kategorie (reduziert)",
        "Ort3",
        "Strasse",
        "Sauerstoffabgaben",
        "Sprache",
        "Weitere Diagnosen",
        "Zeitpunkt, Messart, Wert",
        "Zeitpunkt",
        "Messart",
        "Wert",
        "(Be)-Atmung / Beatmung / Befund Atmung",
        "(Be)-Atmung",
        "Beatmung",
        "Befund Atmung",
        "Auffälligkeiten",
        "Arme",
        "Ereignisort2",
        "PLZ",
        "PLZ4",
    ]

    existing_drop_columns = [column for column in drop_columns if column in features.columns]
    if existing_drop_columns:
        features = features.drop(columns=existing_drop_columns)
        logging.info("Dropped %s columns per feature modifications", len(existing_drop_columns))

    return features


def detect_feature_roles(
    features: pd.DataFrame,
    numeric_ratio_threshold: float = 0.9,
    multilabel_ratio_threshold: float = 0.3,
) -> FeatureRoles:
    numeric: List[str] = []
    categorical: List[str] = []
    multilabel: List[str] = []

    for column in features.columns:
        series = features[column]
        if series.notna().sum() == 0:
            continue
        if pd.api.types.is_bool_dtype(series):
            features[column] = series.astype(int)
            numeric.append(column)
            continue
        if pd.api.types.is_numeric_dtype(series):
            numeric.append(column)
            continue
        as_string = series.astype(str)
        coerced = pd.to_numeric(as_string.str.replace(",", "."), errors="coerce")
        total = series.notna().sum()
        numeric_fraction = coerced.notna().sum() / total if total else 0.0
        if numeric_fraction >= numeric_ratio_threshold:
            features[column] = coerced
            numeric.append(column)
            continue
        sample = as_string.dropna().sample(min(200, as_string.dropna().shape[0]), random_state=42) if as_string.dropna().shape[0] > 200 else as_string.dropna()
        if not sample.empty:
            delimiter_ratio = sample.str.contains("|".join(map(escape_regex, MULTI_LABEL_DELIMITERS))).mean()
        else:
            delimiter_ratio = 0.0
        if delimiter_ratio >= multilabel_ratio_threshold:
            multilabel.append(column)
        else:
            categorical.append(column)

    logging.info(
        "Detected %s numeric, %s categorical, and %s multi-label features",
        len(numeric),
        len(categorical),
        len(multilabel),
    )
    return FeatureRoles(numeric=numeric, categorical=categorical, multilabel=multilabel)


def escape_regex(delimiter: str) -> str:
    from re import escape

    return escape(delimiter)


def drop_low_information_columns(features: pd.DataFrame, roles: FeatureRoles) -> FeatureRoles:
    removed: List[str] = []
    for column in list(features.columns):
        if features[column].nunique(dropna=True) <= 1:
            features.drop(columns=column, inplace=True)
            removed.append(column)
    if removed:
        logging.info("Dropped %s low-variance columns", len(removed))
    numeric = [col for col in roles.numeric if col in features.columns]
    categorical = [col for col in roles.categorical if col in features.columns]
    multilabel = [col for col in roles.multilabel if col in features.columns]
    return FeatureRoles(numeric=numeric, categorical=categorical, multilabel=multilabel)


def format_mean_sd(series: pd.Series) -> str:
    if series is None or not hasattr(series, "to_numpy"):
        return "NA"
    numeric_series = pd.to_numeric(series, errors="coerce")
    if getattr(numeric_series, "notna", None) is None or numeric_series.notna().sum() == 0:
        return "NA"
    mean = numeric_series.mean()
    std = numeric_series.std()
    return f"{mean:.1f} ± {std:.1f}"


def format_percentage(series: pd.Series, positives: Iterable[str]) -> str:
    if series is None:
        return "NA"
    normalized = series.dropna().astype(str).str.strip().str.lower()
    if normalized.empty:
        return "NA"
    positives_lower = {value.lower() for value in positives}
    value = normalized.isin(positives_lower).mean() * 100
    return f"{value:.1f}%"


def build_split_table(splits: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for split_name, (X, y) in splits.items():
        row = {
            "Split": split_name,
            "n": int(len(y)),
            "Outcome %": f"{y.mean() * 100:.1f}%" if len(y) else "NA",
            "Age mean (SD)": format_mean_sd(X.get("Alter ")),
            "Female patients %": format_percentage(X.get("Geschlecht"), {"weiblich", "female", "w"}),
            "Female physicians %": format_percentage(X.get("doctor_sex"), {"female"}),
            "VAS_on_scene mean (SD)": format_mean_sd(X.get("VAS_on_scene")),
            "NACA mean (SD)": format_mean_sd(X.get("NACA (nummerisch)")),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def compute_raw_feature_scores(
    pipeline: Pipeline,
    available_columns: set[str],
    model_type: str,
) -> Dict[str, float]:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = get_preprocessed_feature_names(preprocessor)

    if model_type == "logistic" and hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    elif model_type == "xgboost" and hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        return {}

    scores: Dict[str, float] = {}
    for name, value in zip(feature_names, values):
        if value == 0:
            continue
        parts = name.split("__", 1)
        raw_name = parts[1] if len(parts) > 1 else parts[0]
        if raw_name not in available_columns and "_" in raw_name:
            candidate = raw_name.split("_")[0]
            if candidate in available_columns:
                raw_name = candidate
        if raw_name not in available_columns:
            continue
        scores[raw_name] = scores.get(raw_name, 0.0) + float(value)
    return scores


def tune_xgb_hyperparameters(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    n_iter: int,
    scoring: str,
) -> Tuple[Pipeline, Dict[str, object]]:
    param_distributions = {
        "model__n_estimators": randint(320, 560),
        "model__learning_rate": loguniform(0.02, 0.12),
        "model__max_depth": randint(3, 7),
        "model__min_child_weight": loguniform(1.0, 4.0),
        "model__subsample": uniform(0.65, 0.25),
        "model__colsample_bytree": uniform(0.5, 0.3),
        "model__gamma": loguniform(1e-4, 1.0),
        "model__reg_lambda": loguniform(0.5, 5.0),
        "model__reg_alpha": loguniform(1e-4, 0.5),
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=3,
        n_jobs=1,
        random_state=random_state,
        verbose=1,
        refit=True,
    )

    search.fit(X, y)

    tuned_pipeline: Pipeline = search.best_estimator_
    tuning_summary = {
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
    }
    return tuned_pipeline, tuning_summary


def get_preprocessed_feature_names(preprocessor: ColumnTransformer) -> np.ndarray:
    feature_names: List[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if transformer == "drop":
            continue
        if transformer == "passthrough":
            if isinstance(columns, (list, tuple, np.ndarray)):
                feature_names.extend([str(col) for col in columns])
            else:
                feature_names.append(str(columns))
            continue
        cols = list(columns) if isinstance(columns, (list, tuple, np.ndarray)) else [columns]
        if isinstance(transformer, Pipeline):
            final_estimator = transformer.steps[-1][1]
            if hasattr(final_estimator, "get_feature_names_out"):
                names = final_estimator.get_feature_names_out(cols)
            else:
                names = np.array(cols, dtype=object)
        elif hasattr(transformer, "get_feature_names_out"):
            names = transformer.get_feature_names_out(cols)
        else:
            names = np.array(cols, dtype=object)
        feature_names.extend([str(name) for name in names])
    return np.array(feature_names, dtype=object)


def build_preprocessor(
    roles: FeatureRoles,
    multilabel_top_k: int,
) -> ColumnTransformer:
    transformers: List[Tuple[str, object, Iterable[str]]] = []

    if roles.numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, roles.numeric))

    if roles.categorical:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, roles.categorical))

    if roles.multilabel:
        multilabel_pipeline = Pipeline(
            steps=[
                (
                    "encoder",
                    MultiLabelTopKEncoder(top_k=multilabel_top_k, delimiters=MULTI_LABEL_DELIMITERS),
                )
            ]
        )
        transformers.append(("multilabel", multilabel_pipeline, roles.multilabel))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def make_logistic_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def make_xgboost_pipeline(preprocessor: ColumnTransformer, scale_pos_weight: float, random_state: int) -> Pipeline:
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=0.0,
        reg_lambda=1.0,
        min_child_weight=2.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=random_state,
        n_jobs=4,
        verbosity=0,
        use_label_encoder=False,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def evaluate_pipeline(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    max_points = 400
    if len(fpr) > max_points:
        indices = np.linspace(0, len(fpr) - 1, max_points).astype(int)
        fpr = fpr[indices]
        tpr = tpr[indices]
        thresholds = thresholds[indices]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc_value = auc(recall[::-1], precision[::-1])
    return {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
        "pr_auc": float(pr_auc_value),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "brier": float(brier_score_loss(y_test, y_prob)),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
    }


def run_cross_validation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int,
    random_state: int,
) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "average_precision": "average_precision"},
        n_jobs=1,
        return_train_score=False,
    )
    return {
        "roc_auc_mean": float(scores["test_roc_auc"].mean()),
        "roc_auc_std": float(scores["test_roc_auc"].std()),
        "average_precision_mean": float(scores["test_average_precision"].mean()),
        "average_precision_std": float(scores["test_average_precision"].std()),
    }


def extract_feature_contributions(pipeline: Pipeline, top_n: int = 20) -> Dict[str, List[Dict[str, float]]]:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = get_preprocessed_feature_names(preprocessor)

    if hasattr(model, "coef_"):
        coefficients = model.coef_[0]
        sorted_idx = np.argsort(coefficients)
        top_negative = [
            {"feature": feature_names[i], "coefficient": float(coefficients[i])}
            for i in sorted_idx[:top_n]
        ]
        top_positive = [
            {"feature": feature_names[i], "coefficient": float(coefficients[i])}
            for i in sorted_idx[-top_n:][::-1]
        ]
        magnitude_idx = np.argsort(np.abs(coefficients))[::-1][:top_n]
        top_magnitude = [
            {
                "feature": feature_names[i],
                "coefficient": float(coefficients[i]),
                "abs_coefficient": float(abs(coefficients[i])),
            }
            for i in magnitude_idx
        ]
        return {
            "top_positive": top_positive,
            "top_negative": top_negative,
            "top_magnitude": top_magnitude,
        }

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1][:top_n]
        top_features = [
            {"feature": feature_names[i], "importance": float(importances[i])}
            for i in order
        ]
        return {"importance": top_features}

    return {}


def main() -> None:
    args = parse_args()
    configure_logging()

    features, target, summary = load_filtered_dataset(args.data_path, args.available_columns)
    features = augment_physician_features(
        features,
        args.doctor_roster_path,
        args.doctor_metadata_path,
        args.reference_year,
    )
    features = tidy_location_features(features)
    features = apply_feature_modifications(features)

    if args.feature_whitelist:
        whitelist_raw = [line for line in args.feature_whitelist.read_text(encoding="utf-8").splitlines() if line.strip()]
        normalized_map = {col: col for col in features.columns}
        normalized_map.update({col.strip(): col for col in features.columns})

        selected_columns: List[str] = []
        missing = []
        for entry in whitelist_raw:
            actual = normalized_map.get(entry)
            if actual and actual not in selected_columns:
                selected_columns.append(actual)
            else:
                missing.append(entry)

        if not selected_columns:
            raise ValueError("Feature whitelist does not match any available columns.")
        if missing:
            logging.warning("Whitelist columns not present and will be skipped: %s", ", ".join(sorted(set(missing))))
        features = features[selected_columns]

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

    scale_pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))

    logistic_preprocessor = build_preprocessor(roles, args.multilabel_top_k)
    logistic_pipeline = make_logistic_pipeline(logistic_preprocessor)

    xgb_preprocessor = build_preprocessor(roles, args.multilabel_top_k)
    xgb_pipeline = make_xgboost_pipeline(xgb_preprocessor, scale_pos_weight, args.random_state)

    logging.info("Training logistic regression baseline")
    logistic_pipeline.fit(X_train, y_train)

    # Fit baseline XGBoost to establish reference performance
    baseline_xgb_pipeline = make_xgboost_pipeline(build_preprocessor(roles, args.multilabel_top_k), scale_pos_weight, args.random_state)
    logging.info("Training baseline XGBoost classifier")
    baseline_xgb_pipeline.fit(X_train, y_train)
    baseline_val_metrics = evaluate_pipeline(baseline_xgb_pipeline, X_val, y_val)

    tuned_pipeline = baseline_xgb_pipeline
    xgb_tuning_summary: Dict[str, object | None] = {
        "enabled": args.tune_xgb,
        "accepted": False,
        "baseline_score": float(baseline_val_metrics.get(args.xgb_tuning_scoring, baseline_val_metrics["pr_auc"])),
        "best_score": None,
        "best_params": None,
        "scoring": args.xgb_tuning_scoring,
        "iterations": args.xgb_tuning_iterations if args.tune_xgb else 0,
    }

    if args.tune_xgb:
        logging.info("Running XGBoost hyperparameter tuning (%s iterations)", args.xgb_tuning_iterations)
        candidate_pipeline, tuning_summary = tune_xgb_hyperparameters(
            make_xgboost_pipeline(build_preprocessor(roles, args.multilabel_top_k), scale_pos_weight, args.random_state),
            X_train,
            y_train,
            args.random_state,
            args.xgb_tuning_iterations,
            args.xgb_tuning_scoring,
        )
        candidate_val_metrics = evaluate_pipeline(candidate_pipeline, X_val, y_val)
        tuned_score = candidate_val_metrics.get(args.xgb_tuning_scoring, candidate_val_metrics["pr_auc"])
        baseline_score = baseline_val_metrics.get(args.xgb_tuning_scoring, baseline_val_metrics["pr_auc"])

        xgb_tuning_summary.update(tuning_summary)
        xgb_tuning_summary["candidate_validation_score"] = float(tuned_score)

        if tuned_score >= baseline_score:
            logging.info("Accepted tuned XGBoost parameters (validation %s: %.3f >= baseline %.3f)", args.xgb_tuning_scoring, tuned_score, baseline_score)
            tuned_pipeline = candidate_pipeline
            xgb_tuning_summary["accepted"] = True
        else:
            logging.info("Retaining baseline XGBoost parameters (validation %s: %.3f < baseline %.3f)", args.xgb_tuning_scoring, tuned_score, baseline_score)

    xgb_pipeline = tuned_pipeline

    logging.info("Evaluating on validation and hold-out sets")
    logistic_val_metrics = evaluate_pipeline(logistic_pipeline, X_val, y_val)
    logistic_metrics = evaluate_pipeline(logistic_pipeline, X_test, y_test)
    xgb_val_metrics = evaluate_pipeline(xgb_pipeline, X_val, y_val)
    xgb_metrics = evaluate_pipeline(xgb_pipeline, X_test, y_test)

    available_cols_set = set(features_model.columns)
    logistic_raw_scores = compute_raw_feature_scores(logistic_pipeline, available_cols_set, "logistic")
    xgb_raw_scores = compute_raw_feature_scores(xgb_pipeline, available_cols_set, "xgboost")

    logging.info("Running cross-validation for logistic regression")
    logistic_cv = run_cross_validation(logistic_pipeline, X_train, y_train, args.n_splits, args.random_state)
    logging.info("Running cross-validation for XGBoost")
    xgb_cv = run_cross_validation(xgb_pipeline, X_train, y_train, args.n_splits, args.random_state)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dump(logistic_pipeline, output_dir / "logistic_regression.joblib")
    dump(xgb_pipeline, output_dir / "xgboost_classifier.joblib")

    results = {
        "data_summary": summary,
        "logistic_regression": {
            "test_metrics": logistic_metrics,
            "validation_metrics": logistic_val_metrics,
            "cross_validation": logistic_cv,
            "feature_contributions": extract_feature_contributions(logistic_pipeline),
            "raw_feature_scores": logistic_raw_scores,
        },
        "xgboost": {
            "test_metrics": xgb_metrics,
            "validation_metrics": xgb_val_metrics,
            "cross_validation": xgb_cv,
            "feature_contributions": extract_feature_contributions(xgb_pipeline),
            "tuning": xgb_tuning_summary,
            "raw_feature_scores": xgb_raw_scores,
        },
    }

    split_table = build_split_table(
        {
            "Training": (X_train, y_train),
            "Validation": (X_val, y_val),
            "Testing": (X_test, y_test),
        }
    )
    results["table1"] = split_table.to_dict(orient="records")

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logging.info("Saved metrics to %s", metrics_path)

    split_table.to_csv(output_dir / "table1.csv", index=False)
    (output_dir / "table1.md").write_text(dataframe_to_markdown(split_table), encoding="utf-8")

    importance_summary = {
        "logistic_regression": results["logistic_regression"]["feature_contributions"],
        "xgboost": results["xgboost"]["feature_contributions"],
    }
    (output_dir / "feature_importance.json").write_text(
        json.dumps(importance_summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
