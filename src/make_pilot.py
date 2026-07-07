"""Build a small pilot set from data/raw/test.csv, sampled evenly across calculator_name.

Usage:
    python src/make_pilot.py --n 20
    python src/make_pilot.py --n 100
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


def sample_evenly(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample up to n rows total, spread as evenly as possible across calculator_name."""
    calculators = df["calculator_name"].dropna().unique().tolist()
    if not calculators:
        return df.sample(n=min(n, len(df)), random_state=seed)

    per_calculator = max(1, n // len(calculators))

    chosen_parts = []
    for calculator in calculators:
        group = df[df["calculator_name"] == calculator]
        take = min(per_calculator, len(group))
        chosen_parts.append(group.sample(n=take, random_state=seed))
    chosen = pd.concat(chosen_parts)

    # Top up (or trim) to land as close to n as possible.
    if len(chosen) < n:
        remaining = df.drop(chosen.index)
        top_up = remaining.sample(n=min(n - len(chosen), len(remaining)), random_state=seed)
        chosen = pd.concat([chosen, top_up])
    elif len(chosen) > n:
        chosen = chosen.sample(n=n, random_state=seed)

    return chosen.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle order


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="target pilot size")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"error: {args.input} not found. Run src/load_data.py first.")
        sys.exit(1)

    df = pd.read_csv(args.input)
    if "calculator_name" not in df.columns:
        df = normalize_columns(df)

    pilot = sample_evenly(df, args.n, args.seed)

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PILOT_DIR / f"pilot_{args.n}.csv"
    pilot.to_csv(out_path, index=False)

    print(f"sampled {len(pilot)} / requested {args.n} rows")
    print("calculator_name distribution in pilot:")
    print(pilot["calculator_name"].value_counts().to_string())
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
