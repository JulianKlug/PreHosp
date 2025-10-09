import pandas as pd
from typing import Iterable, List
import numpy as np
import re


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

def get_multi_label_counts(data_df, multi_label_column):
    data_df[multi_label_column] = data_df[multi_label_column].replace(999, pd.NA)
    label_counter = {}
    # iterate through the rows
    for index, row in data_df.iterrows():
        # split by comma then strip spaces
        labels = [label.strip() for label in str(row[multi_label_column]).split(',')]
        # if label not in the dict, add it
        for label in labels:
            if label == 'nan' or label == '<NA>':
                continue
            if label not in label_counter:
                label_counter[label] = 1
            else:
                label_counter[label] += 1

    # sort the dictionary by value
    sorted_label_counter = dict(sorted(label_counter.items(), key=lambda item: item[1], reverse=True))
    return sorted_label_counter

