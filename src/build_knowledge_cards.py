"""Scaffold blank knowledge-card JSON files for calculators that appear in a pilot
set but don't have a card yet. Fill in the generated TODO fields by hand.

Usage:
    python src/build_knowledge_cards.py --pilot data/pilot/pilot_20.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from common.console import configure_utf8_stdout  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CARDS_DIR = PROJECT_ROOT / "knowledge_cards"
DEFAULT_PILOT = PROJECT_ROOT / "data" / "pilot" / "pilot_20.csv"


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def blank_card(calculator_name: str) -> dict:
    return {
        "calculator_name": calculator_name,
        "task_type": "TODO: equation-based | rule-based | scoring",
        "formula_or_rule": "TODO",
        "required_entities": [],
        "unit_rules": {},
        "condition_rules": {},
        "output_format": {"value": "number", "unit": "TODO"},
    }


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    args = parser.parse_args()

    if not args.pilot.exists():
        print(f"error: {args.pilot} not found. Run src/make_pilot.py first.")
        sys.exit(1)

    df = pd.read_csv(args.pilot)
    calculators = sorted(df["calculator_name"].dropna().unique())

    existing = {p.stem for p in CARDS_DIR.glob("*.json")}

    created = []
    for calculator_name in calculators:
        slug = slugify(calculator_name)
        if slug in existing:
            continue
        out_path = CARDS_DIR / f"{slug}.json"
        out_path.write_text(json.dumps(blank_card(calculator_name), indent=2), encoding="utf-8")
        created.append(out_path)

    print(f"calculators in pilot: {len(calculators)}")
    if created:
        print("created blank cards (fill in the TODO fields by hand):")
        for path in created:
            print(f"  - {path}")
    else:
        print("no new cards needed, all calculators already have a card.")


if __name__ == "__main__":
    main()
