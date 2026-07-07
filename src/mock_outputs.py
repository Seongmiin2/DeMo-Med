"""Generate fake (mock) model outputs for all four methods, without calling any LLM API.

This exists purely to exercise src/eval_demomed.py before real API keys are wired up.
Each method gets a different, deliberately-imperfect accuracy rate so the evaluation
script has something non-trivial to measure (correct / wrong-number / non-numeric /
abstain cases all show up).

Usage:
    python src/mock_outputs.py --input data/pilot/pilot_20.csv
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from common.console import configure_utf8_stdout  # noqa: E402
from common.jsonl_utils import write_jsonl  # noqa: E402
from common.numeric_utils import extract_number  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "pilot" / "pilot_20.csv"
MOCK_DIR = PROJECT_ROOT / "outputs" / "mock"

# method -> (P(correct), P(wrong numeric), P(non-numeric text), P(abstain))
# Only "demo_med" ever abstains, since abstaining is specific to its verification step.
METHOD_PROFILES = {
    "llm_only": (0.50, 0.35, 0.15, 0.00),
    "cot": (0.65, 0.25, 0.10, 0.00),
    "open_book": (0.75, 0.15, 0.10, 0.00),
    "demo_med": (0.80, 0.05, 0.00, 0.15),
}

NON_NUMERIC_ANSWERS = [
    "Unable to determine from the note",
    "See explanation",
    "Not applicable",
]


def pick_outcome(rng: random.Random, profile) -> str:
    p_correct, p_wrong, p_non_numeric, p_abstain = profile
    roll = rng.random()
    if roll < p_correct:
        return "correct"
    if roll < p_correct + p_wrong:
        return "wrong_numeric"
    if roll < p_correct + p_wrong + p_non_numeric:
        return "non_numeric"
    return "abstain"


def make_answer(rng: random.Random, outcome: str, ground_truth_answer, unit):
    """Return (answer, unit, abstain, abstain_reason) for a given outcome."""
    if outcome == "correct":
        return str(ground_truth_answer), unit, False, ""

    if outcome == "wrong_numeric":
        gt_value = extract_number(ground_truth_answer)
        if gt_value is None:
            return rng.choice(NON_NUMERIC_ANSWERS), unit, False, ""
        perturbed = gt_value * rng.choice([0.5, 0.7, 1.3, 1.8, 2.0])
        return f"{perturbed:.2f}", unit, False, ""

    if outcome == "non_numeric":
        return rng.choice(NON_NUMERIC_ANSWERS), unit, False, ""

    # abstain
    return "", unit, True, rng.choice(
        ["missing required patient value", "unit mismatch between note and formula"]
    )


def build_record(method: str, row: pd.Series, rng: random.Random) -> dict:
    outcome = pick_outcome(rng, METHOD_PROFILES[method])
    unit = str(row.get("output_type") or "")
    answer, unit, abstain, abstain_reason = make_answer(
        rng, outcome, row.get("ground_truth_answer"), unit
    )

    raw_id = row.get("id")
    record = {
        "id": int(raw_id) if pd.notna(raw_id) else None,
        "calculator": row.get("calculator_name"),
        "answer": answer,
        "unit": unit,
    }

    if method == "llm_only":
        record["reason"] = "mock output, no reasoning generated"
    elif method == "cot":
        record.update(
            {
                "entities": {},
                "formula_or_rule": "mock formula",
                "calculation": "mock calculation",
            }
        )
    elif method == "open_book":
        record.update({"entities": {}, "formula_or_rule": "mock formula"})
    elif method == "demo_med":
        record.update(
            {
                "extracted_entities": {},
                "solution_plan": [],
                "verification": {
                    "missing_values": [abstain_reason] if abstain else [],
                    "unit_errors": [],
                    "condition_errors": [],
                    "arithmetic_check": "fail" if abstain else "pass",
                },
                "abstain": abstain,
                "abstain_reason": abstain_reason,
            }
        )

    return record


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"error: {args.input} not found. Run src/make_pilot.py first.")
        sys.exit(1)

    df = pd.read_csv(args.input)
    MOCK_DIR.mkdir(parents=True, exist_ok=True)

    for method in METHOD_PROFILES:
        rng = random.Random(f"{args.seed}-{method}")
        records = [build_record(method, row, rng) for _, row in df.iterrows()]
        out_path = MOCK_DIR / f"{method}.jsonl"
        write_jsonl(out_path, records)
        print(f"wrote {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
