"""Helpers for scoring predictions against ground-truth answers.

MedCalc-Bench answers are either:
- numeric (optionally with a [lower_limit, upper_limit] tolerance band), or
- categorical/string (e.g. a risk category name), when lower/upper limit are absent.
"""

import re
from datetime import date, datetime
from typing import Optional

from rapidfuzz import fuzz

# Matches ints/floats, including negatives and things like "12.5" inside "12.5 mL/min".
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

_DATE_FORMATS = ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%Y/%m/%d", "%B %d, %Y", "%d %B %Y")


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


def parse_date(text) -> Optional[date]:
    """Parse a date string like '09/11/2014' using MedCalc-Bench's common formats. None if unparseable.

    Deliberately NOT done via _NUMBER_RE: pulling "the first number" out of a date
    (e.g. the "09" in "09/11/2014") would silently accept any wrong day/month/year
    that happens to start the same way.
    """
    if text is None:
        return None
    text = str(text).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def date_matches(predicted_answer, ground_truth_answer, lower_limit=None, upper_limit=None) -> bool:
    """True if the predicted date matches the ground truth (or falls in [lower_limit, upper_limit])."""
    predicted_date = parse_date(predicted_answer)
    if predicted_date is None:
        return False

    if lower_limit is not None and upper_limit is not None:
        lower_date = parse_date(lower_limit)
        upper_date = parse_date(upper_limit)
        if lower_date is not None and upper_date is not None:
            lower_date, upper_date = min(lower_date, upper_date), max(lower_date, upper_date)
            return lower_date <= predicted_date <= upper_date

    gt_date = parse_date(ground_truth_answer)
    return gt_date is not None and predicted_date == gt_date


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
    output_type=None,
) -> bool:
    """Dispatch to date / numeric / string comparison.

    MedCalc-Bench's own `output_type` column ("date" | "decimal" | "integer") is used
    when available, since guessing the answer type from the string is unreliable -
    e.g. extracting "09" out of the date "09/11/2014" would make it look numeric.
    """
    if output_type == "date":
        return date_matches(predicted_answer, ground_truth_answer, lower_limit, upper_limit)
    if output_type in ("decimal", "integer"):
        return numeric_matches(predicted_answer, ground_truth_answer, lower_limit, upper_limit)
    if output_type is None:
        # output_type not supplied - infer it, checking date first: is_number() would
        # also match the "09" inside a date string like "09/11/2014", so a successful
        # full-date parse must win over a partial numeric match.
        if parse_date(ground_truth_answer) is not None:
            return date_matches(predicted_answer, ground_truth_answer, lower_limit, upper_limit)
        if is_number(ground_truth_answer):
            return numeric_matches(predicted_answer, ground_truth_answer, lower_limit, upper_limit)
    return string_matches(predicted_answer, ground_truth_answer)
