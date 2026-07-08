"""Build a pilot set from data/raw/test.csv: a fixed set of 10 calculators
(5 equation-based + 5 rule/scoring-based), sampled evenly with a fixed seed.

Replaces the old "trim from however many calculators exist down to N" sampler -
with 55 calculators in the test split, "pick up to N total" and "N per calculator"
were never the same thing. Picking the 10 calculators up front (and stratifying by
type, mixing a known-easy and known-hard item in each half) makes pilot_20/pilot_100
comparable to each other and to any follow-up run.

Usage:
    python src/make_pilot.py --n-per-calculator 2    # -> pilot_20.csv
    python src/make_pilot.py --n-per-calculator 10   # -> pilot_100.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from common.column_normalizer import normalize_columns  # noqa: E402
from common.console import configure_utf8_stdout  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "test.csv"
PILOT_DIR = PROJECT_ROOT / "data" / "pilot"

# 5 equation-based + 5 rule/scoring-based, each half mixing a known-easy and a
# known-hard calculator. Exact calculator_name strings, verified against test.csv.
PILOT_CALCULATORS = [
    # equation-based
    "Creatinine Clearance (Cockcroft-Gault Equation)",  # easy, canonical
    "Body Mass Index (BMI)",                             # easy, 2-variable
    "Mean Arterial Pressure (MAP)",                       # easy, 2-variable
    "Ideal Body Weight",                                  # easy-medium, sex-conditioned
    "QTc Fridericia Calculator",                          # hard: cube root, broke every method in the earlier pilot
    # rule/scoring-based
    "PERC Rule for Pulmonary Embolism",                   # easy, binary checklist
    "SIRS Criteria",                                      # easy, 4-criteria checklist
    "CURB-65 Score for Pneumonia Severity",               # medium, 5-criteria
    "HEART Score for Major Cardiac Events",               # medium, graded (0/1/2) criteria
    "Sequential Organ Failure Assessment (SOFA) Score",   # hard: 6 organ systems
]


def sample_stratified(df: pd.DataFrame, n_per_calculator: int, seed: int) -> pd.DataFrame:
    """Sample exactly n_per_calculator rows from each of PILOT_CALCULATORS."""
    chosen_parts = []
    for calculator in PILOT_CALCULATORS:
        group = df[df["calculator_name"] == calculator]
        if len(group) < n_per_calculator:
            raise ValueError(
                f"'{calculator}' only has {len(group)} rows in the input, "
                f"need {n_per_calculator}"
            )
        chosen_parts.append(group.sample(n=n_per_calculator, random_state=seed))

    chosen = pd.concat(chosen_parts)
    return chosen.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle order


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-per-calculator", type=int, default=2, help="rows sampled per calculator (2 -> pilot_20, 10 -> pilot_100)"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"error: {args.input} not found. Run src/load_data.py first.")
        sys.exit(1)

    df = pd.read_csv(args.input)
    if "calculator_name" not in df.columns:
        df = normalize_columns(df)

    pilot = sample_stratified(df, args.n_per_calculator, args.seed)

    total = args.n_per_calculator * len(PILOT_CALCULATORS)
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PILOT_DIR / f"pilot_{total}.csv"
    pilot.to_csv(out_path, index=False)

    print(f"sampled {len(pilot)} rows ({args.n_per_calculator} x {len(PILOT_CALCULATORS)} calculators)")
    print("calculator_name distribution in pilot:")
    print(pilot["calculator_name"].value_counts().to_string())
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
