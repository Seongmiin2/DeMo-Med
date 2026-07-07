"""Helpers for scoring predictions against ground-truth answers.

MedCalc-Bench answers are either:
- numeric (optionally with a [lower_limit, upper_limit] tolerance band), or
- categorical/string (e.g. a risk category name), when lower/upper limit are absent.
"""

import re
from typing import Optional

from rapidfuzz import fuzz

# Matches ints/floats, including negatives and things like "12.5" inside "12.5 mL/min".
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_number(text) -> Optional[float]:
    """Pull the first number out of a string (predicted 'answer' field). None if not found."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    match = _NUMBER_RE.search(str(text))
    if not match:
        return None
    return float(match.group(0).replace(",", ""))


def is_number(text) -> bool:
    return extract_number(text) is not None


def is_within_range(value: float, lower_limit: float, upper_limit: float) -> bool:
    """Inclusive range check, tolerant of lower/upper being swapped."""
    lo, hi = min(lower_limit, upper_limit), max(lower_limit, upper_limit)
    return lo <= value <= hi


def numeric_matches(
    predicted_answer,
    ground_truth_answer,
    lower_limit=None,
    upper_limit=None,
    relative_tolerance: float = 0.05,
) -> bool:
    """True if the predicted numeric answer counts as correct.

    Uses [lower_limit, upper_limit] when both are present and numeric.
    Otherwise falls back to ground_truth_answer +/- relative_tolerance.
    """
    predicted_value = extract_number(predicted_answer)
    if predicted_value is None:
        return False

    if lower_limit is not None and upper_limit is not None:
        lower_value = extract_number(lower_limit)
        upper_value = extract_number(upper_limit)
        if lower_value is not None and upper_value is not None:
            return is_within_range(predicted_value, lower_value, upper_value)

    gt_value = extract_number(ground_truth_answer)
    if gt_value is None:
        return False
    if gt_value == 0:
        return predicted_value == 0
    return abs(predicted_value - gt_value) / abs(gt_value) <= relative_tolerance


def string_matches(predicted_answer, ground_truth_answer, fuzzy_threshold: float = 90.0) -> bool:
    """True if predicted/ground-truth strings match exactly (case-insensitive) or are a close fuzzy match."""
    if predicted_answer is None or ground_truth_answer is None:
        return False
    predicted_text = str(predicted_answer).strip().lower()
    gt_text = str(ground_truth_answer).strip().lower()
    if predicted_text == gt_text:
        return True
    return fuzz.ratio(predicted_text, gt_text) >= fuzzy_threshold


def answer_is_correct(
    predicted_answer,
    ground_truth_answer,
    lower_limit=None,
    upper_limit=None,
) -> bool:
    """Dispatch to numeric or string comparison depending on whether the ground truth looks numeric."""
    if is_number(ground_truth_answer):
        return numeric_matches(predicted_answer, ground_truth_answer, lower_limit, upper_limit)
    return string_matches(predicted_answer, ground_truth_answer)
