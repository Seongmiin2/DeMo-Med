"""Normalize MedCalc-Bench column names (which vary by dataset revision) into one
standard schema used everywhere else in this project.

Standard schema:
    id, patient_note, question, calculator_name, category, output_type,
    ground_truth_answer, lower_limit, upper_limit, relevant_entities,
    ground_truth_explanation
"""

import re

import pandas as pd

STANDARD_COLUMNS = [
    "id",
    "patient_note",
    "question",
    "calculator_name",
    "category",
    "output_type",
    "ground_truth_answer",
    "lower_limit",
    "upper_limit",
    "relevant_entities",
    "ground_truth_explanation",
]

# Known column name variants seen across MedCalc-Bench releases (raw HF columns
# use "Title Case With Spaces"; some community mirrors use snake_case already).
_CANDIDATES = {
    "id": ["row number", "note id", "id", "index", "row_id"],
    "patient_note": ["patient note", "note", "patient_note"],
    "question": ["question"],
    "calculator_name": ["calculator name", "calculator"],
    "category": ["category"],
    "output_type": ["output type"],
    "ground_truth_answer": ["ground truth answer", "answer", "ground_truth"],
    "lower_limit": ["lower limit", "lower_limit", "lower bound"],
    "upper_limit": ["upper limit", "upper_limit", "upper bound"],
    "relevant_entities": ["relevant entities", "relevant_entities"],
    "ground_truth_explanation": [
        "ground truth explanation",
        "explanation",
        "ground_truth_explanation",
    ],
}


def _normalize_key(name: str) -> str:
    """'Ground Truth Answer' / 'ground_truth_answer' -> 'ground truth answer'."""
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with columns renamed/reordered to STANDARD_COLUMNS.

    Any standard column that can't be found in the input is created as all-NaN,
    so downstream code can always rely on the full schema being present.
    Extra input columns that don't map to a standard field are dropped.
    """
    key_to_original = {_normalize_key(col): col for col in df.columns}

    result = pd.DataFrame(index=df.index)
    missing = []
    for target in STANDARD_COLUMNS:
        original_col = None
        for candidate in _CANDIDATES[target]:
            if candidate in key_to_original:
                original_col = key_to_original[candidate]
                break
        if original_col is not None:
            result[target] = df[original_col]
        else:
            result[target] = pd.NA
            missing.append(target)

    if result["id"].isna().all():
        result["id"] = range(len(result))

    if missing:
        print(f"[column_normalizer] warning: could not find source columns for {missing}")

    return result
