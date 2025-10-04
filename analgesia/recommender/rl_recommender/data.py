from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

_MAIN_NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_REL_NS = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
_CELL_VALUE_RE = re.compile(r"([0-9]+[0-9,.]*)\s*(mg|mcg|µg|ug)", re.IGNORECASE)

FENT_BANDS: List[Dict[str, Any]] = [
    {"band": 0, "label": "0 mg", "lower": 0.0, "upper": 0.0, "suggested": 0.0},
    {"band": 1, "label": "≤0.10 mg", "lower": 0.0, "upper": 0.10, "suggested": 0.10},
    {"band": 2, "label": "0.10-0.20 mg", "lower": 0.10, "upper": 0.20, "suggested": 0.15},
    {"band": 3, "label": "0.20-0.30 mg", "lower": 0.20, "upper": 0.30, "suggested": 0.25},
    {"band": 4, "label": ">0.30 mg", "lower": 0.30, "upper": 1.00, "suggested": 0.35},
]

KET_BANDS: List[Dict[str, Any]] = [
    {"band": 0, "label": "0 mg", "lower": 0.0, "upper": 0.0, "suggested": 0.0},
    {"band": 1, "label": "≤25 mg", "lower": 0.0, "upper": 25.0, "suggested": 25.0},
    {"band": 2, "label": "25-50 mg", "lower": 25.0, "upper": 50.0, "suggested": 40.0},
    {"band": 3, "label": "50-75 mg", "lower": 50.0, "upper": 75.0, "suggested": 60.0},
    {"band": 4, "label": ">75 mg", "lower": 75.0, "upper": 300.0, "suggested": 90.0},
]


@dataclass
class PreparedDataset:
    data: pd.DataFrame
    metadata: Dict[str, Any]


def load_prehospital_table(
    workbook_path: Union[str, Path], sheet_index: int = 0
) -> pd.DataFrame:
    """Load an .xlsx worksheet without optional Excel dependencies.

    Parameters
    ----------
    workbook_path:
        Path to the Excel workbook. Only `.xlsx` files are supported.
    sheet_index:
        Zero-based sheet index to load.

    Returns
    -------
    pd.DataFrame
        Sheet contents represented as strings (numbers are not coerced).
    """

    workbook_path = Path(workbook_path)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    with zipfile.ZipFile(workbook_path) as archive:
        shared_strings = _read_shared_strings(archive)
        sheet_path = _resolve_sheet_path(archive, sheet_index)
        with archive.open(sheet_path) as sheet_file:
            header, rows = _read_sheet(sheet_file, shared_strings)
    return pd.DataFrame(rows, columns=header)


def prepare_analgesia_dataset(workbook_path: Union[str, Path]) -> PreparedDataset:
    """Load and preprocess the analgesia registry for RL training."""

    raw = load_prehospital_table(workbook_path, sheet_index=0)
    df = raw.copy()

    rename_map = {
        "Alter ": "age",
        "Geschlecht": "gender",
        "VAS_on_scene": "vas_on_scene",
        "VAS_on_arrival": "vas_on_arrival",
        "GCS": "gcs",
        "HR": "heart_rate",
        "SPO2": "spo2",
        "NIBD Systolisch": "nibp_systolic",
        "NIBD Diastolisch": "nibp_diastolic",
        "AF": "resp_rate",
        "Kategorie (reduziert)": "mechanism",
        "Einsatzart": "dispatch_type",
        "Bewusstseinlage": "consciousness",
        "Befund Atmung": "breathing_assessment",
        "Atemwegbefund": "airway_status",
        "Wochentag": "weekday",
        "Monat": "month",
        "Tag oder Nacht": "time_of_day",
        "NACA (nummerisch)": "naca_score",
        "Alle Medikamente detailliert": "medications",
        "SNZ Ereignis Nr. ": "case_id",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    numeric_columns = [
        "age",
        "vas_on_scene",
        "vas_on_arrival",
        "gcs",
        "heart_rate",
        "spo2",
        "nibp_systolic",
        "nibp_diastolic",
        "resp_rate",
        "naca_score",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = _to_numeric(df[column])

    # Basic cleaning
    if "age" in df.columns:
        df["age"] = df["age"].clip(lower=0, upper=110)
    if "vas_on_scene" in df.columns:
        df["vas_on_scene"] = df["vas_on_scene"].clip(lower=0, upper=10)
    if "vas_on_arrival" in df.columns:
        df["vas_on_arrival"] = df["vas_on_arrival"].clip(lower=0, upper=10)
    if "gcs" in df.columns:
        df["gcs"] = df["gcs"].clip(lower=3, upper=15)
    if "naca_score" in df.columns:
        df["naca_score"] = df["naca_score"].clip(lower=0, upper=7)

    df = df[df["vas_on_arrival"].notna()].copy()
    df = df[df["vas_on_scene"].notna()].copy()
    df = df[df["vas_on_scene"] >= 3].copy()

    med_series = df["medications"] if "medications" in df.columns else pd.Series([""] * len(df), index=df.index)
    fent_dose, ket_dose = _extract_medication_totals(med_series)
    df["fentanyl_total_mg"] = np.clip(fent_dose, 0.0, 1.0)
    df["ketamine_total_mg"] = np.clip(ket_dose, 0.0, 300.0)

    df["fent_band"] = df["fentanyl_total_mg"].apply(_map_fent_band)
    df["ket_band"] = df["ketamine_total_mg"].apply(_map_ket_band)

    num_ket_bands = len(KET_BANDS)
    df["action_idx"] = df["fent_band"] * num_ket_bands + df["ket_band"]

    df["reward"] = (df["vas_on_arrival"] < 3).astype(int)

    feature_columns = [
        col
        for col in [
            "age",
            "vas_on_scene",
            "gcs",
            "heart_rate",
            "spo2",
            "nibp_systolic",
            "nibp_diastolic",
            "resp_rate",
            "naca_score",
            "gender",
            "mechanism",
            "dispatch_type",
            "consciousness",
            "breathing_assessment",
            "airway_status",
            "weekday",
            "month",
            "time_of_day",
        ]
        if col in df.columns
    ]

    numeric_features = [
        col
        for col in [
            "age",
            "vas_on_scene",
            "gcs",
            "heart_rate",
            "spo2",
            "nibp_systolic",
            "nibp_diastolic",
            "resp_rate",
            "naca_score",
        ]
        if col in df.columns
    ]
    categorical_features = [col for col in feature_columns if col not in numeric_features]

    action_table = _build_action_table(df)

    metadata: Dict[str, Any] = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_columns": feature_columns,
        "fent_bands": FENT_BANDS,
        "ket_bands": KET_BANDS,
        "action_table": action_table,
        "num_actions": len(FENT_BANDS) * len(KET_BANDS),
        "target_column": "reward",
        "dose_columns": ["fentanyl_total_mg", "ketamine_total_mg"],
    }

    df = df[
        feature_columns
        + [
            "fentanyl_total_mg",
            "ketamine_total_mg",
            "fent_band",
            "ket_band",
            "action_idx",
            "reward",
            "vas_on_arrival",
        ]
        + (["case_id"] if "case_id" in df.columns else [])
    ].copy()

    return PreparedDataset(data=df.reset_index(drop=True), metadata=metadata)


def _read_shared_strings(archive: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    with archive.open("xl/sharedStrings.xml") as handle:
        root = ET.parse(handle).getroot()
        strings: List[str] = []
        for si in root.findall("main:si", _MAIN_NS):
            text_fragments = [node.text or "" for node in si.findall(".//main:t", _MAIN_NS)]
            strings.append("".join(text_fragments))
        return strings


def _resolve_sheet_path(archive: zipfile.ZipFile, sheet_index: int) -> str:
    with archive.open("xl/workbook.xml") as workbook_handle:
        workbook_root = ET.parse(workbook_handle).getroot()
        sheets = workbook_root.findall("main:sheets/main:sheet", _MAIN_NS)
        if sheet_index >= len(sheets):
            raise IndexError(f"Sheet index {sheet_index} out of range; workbook has {len(sheets)} sheets")
        sheet = sheets[sheet_index]
        rel_id = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")

    with archive.open("xl/_rels/workbook.xml.rels") as rel_handle:
        rel_root = ET.parse(rel_handle).getroot()
        for rel in rel_root.findall("rel:Relationship", _REL_NS):
            if rel.get("Id") == rel_id:
                target = rel.get("Target")
                if target.startswith("../"):
                    target = target[3:]
                return f"xl/{target}"
    raise ValueError(f"Could not resolve sheet path for rel id {rel_id}")


def _read_sheet(sheet_file, shared_strings: Sequence[str]) -> Tuple[List[str], List[List[str]]]:
    root = ET.parse(sheet_file).getroot()
    sheet_data = root.find("main:sheetData", _MAIN_NS)
    if sheet_data is None:
        return [], []
    rows = sheet_data.findall("main:row", _MAIN_NS)
    if not rows:
        return [], []

    header_cells = rows[0].findall("main:c", _MAIN_NS)
    header = [_cell_value(cell, shared_strings) for cell in header_cells]
    column_count = len(header)

    data_rows: List[List[str]] = []
    for row in rows[1:]:
        row_buffer = ["" for _ in range(column_count)]
        for cell in row.findall("main:c", _MAIN_NS):
            ref = cell.get("r", "")
            col_letters = _extract_column_letters(ref)
            if not col_letters:
                continue
            col_idx = _letters_to_index(col_letters)
            if col_idx >= column_count:
                continue
            row_buffer[col_idx] = _cell_value(cell, shared_strings)
        data_rows.append(row_buffer)
    return header, data_rows


def _cell_value(cell: ET.Element, shared_strings: Sequence[str]) -> str:
    value_node = cell.find("main:v", _MAIN_NS)
    if value_node is None:
        return ""
    cell_type = cell.get("t")
    raw_text = value_node.text or ""
    if cell_type == "s":
        index = int(raw_text)
        if 0 <= index < len(shared_strings):
            return shared_strings[index]
        return ""
    return raw_text


def _extract_column_letters(cell_reference: str) -> str:
    return "".join(ch for ch in cell_reference if ch.isalpha())


def _letters_to_index(letters: str) -> int:
    total = 0
    for char in letters:
        total = total * 26 + (ord(char.upper()) - ord("A") + 1)
    return total - 1


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", ".", regex=False)
    cleaned = cleaned.replace({"nan": np.nan, "": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def _extract_medication_totals(med_col: pd.Series) -> Tuple[pd.Series, pd.Series]:
    fent_values: List[float] = []
    ket_values: List[float] = []
    for entry in med_col.fillna(""):
        fent_total = 0.0
        ket_total = 0.0
        for chunk in (part.strip() for part in entry.split(";") if part.strip()):
            lowered = chunk.lower()
            match = _CELL_VALUE_RE.search(lowered)
            if not match:
                continue
            amount = float(match.group(1).replace(",", "."))
            unit = match.group(2).lower()
            scale = 1.0 if unit in {"mg"} else 0.001
            if "fentanyl" in lowered:
                fent_total += amount * scale
            if "ketamin" in lowered:
                ket_total += amount * scale
        fent_values.append(fent_total)
        ket_values.append(ket_total)
    return pd.Series(fent_values, index=med_col.index), pd.Series(ket_values, index=med_col.index)


def _map_fent_band(dose: float) -> int:
    if dose <= 0.0:
        return 0
    if dose <= 0.10:
        return 1
    if dose <= 0.20:
        return 2
    if dose <= 0.30:
        return 3
    return 4


def _map_ket_band(dose: float) -> int:
    if dose <= 0.0:
        return 0
    if dose <= 25.0:
        return 1
    if dose <= 50.0:
        return 2
    if dose <= 75.0:
        return 3
    return 4


def _build_action_table(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    grouped = (
        df.groupby(["fent_band", "ket_band"], dropna=False)
        .agg(
            action_idx=("action_idx", "first"),
            count=("reward", "size"),
            success_rate=("reward", "mean"),
            fent_median=("fentanyl_total_mg", "median"),
            ket_median=("ketamine_total_mg", "median"),
        )
        .reset_index()
    )
    grouped = grouped.set_index(["fent_band", "ket_band"])

    for fent_band in range(len(FENT_BANDS)):
        for ket_band in range(len(KET_BANDS)):
            key = (fent_band, ket_band)
            if key in grouped.index:
                row = grouped.loc[key]
                action_idx = int(row["action_idx"])
                count = int(row["count"])
                success_rate = float(row["success_rate"]) if count else np.nan
                fent_median = float(row["fent_median"]) if not np.isnan(row["fent_median"]) else None
                ket_median = float(row["ket_median"]) if not np.isnan(row["ket_median"]) else None
            else:
                action_idx = fent_band * len(KET_BANDS) + ket_band
                count = 0
                success_rate = np.nan
                fent_median = None
                ket_median = None
            fent_band_meta = FENT_BANDS[fent_band]
            ket_band_meta = KET_BANDS[ket_band]
            records.append(
                {
                    "action_idx": action_idx,
                    "fent_band": fent_band,
                    "ket_band": ket_band,
                    "fent_label": fent_band_meta["label"],
                    "ket_label": ket_band_meta["label"],
                    "historical_cases": count,
                    "historical_success_rate": success_rate,
                    "recommended_fent_mg": fent_median if fent_median is not None else fent_band_meta["suggested"],
                    "recommended_ket_mg": ket_median if ket_median is not None else ket_band_meta["suggested"],
                }
            )
    return pd.DataFrame(records)
