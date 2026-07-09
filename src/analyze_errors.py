"""List every incorrect / abstained / parse-failed prediction per method+provider
for manual error review.

Adds blank `formula_arithmetic_error_type` and `labeler` columns for later manual
double-coding (Formula vs Arithmetic vs Unit/Condition error) - entity extraction
errors are already auto-scored in eval_demomed.py's entity_error_rate, but telling
a formula mistake apart from an arithmetic slip needs a human to actually read the
case, which is out of scope for this script.

Usage:
    python src/analyze_errors.py --gold data/pilot/pilot_20.csv --pred_dir outputs/real
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from common.console import configure_utf8_stdout  # noqa: E402
from common.jsonl_utils import read_jsonl  # noqa: E402
from eval_demomed import DEFAULT_GOLD, DEFAULT_PRED_DIR, find_prediction_files, is_correct  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--pred_dir", type=Path, default=DEFAULT_PRED_DIR)
    args = parser.parse_args()

    gold_df = pd.read_csv(args.gold)
    gold_by_id = {int(row["id"]): row.to_dict() for _, row in gold_df.iterrows()}

    files = find_prediction_files(args.pred_dir)
    if not files:
        print(f"error: no {{method}}_{{provider}}.jsonl files found under {args.pred_dir}")
        sys.exit(1)

    error_rows = []
    for (method, provider), path in sorted(files.items()):
        for pred in read_jsonl(path):
            gold = gold_by_id.get(pred.get("id"))
            if gold is None:
                continue
            if is_correct(pred, gold):
                continue

            error_rows.append(
                {
                    "method": method,
                    "provider": provider,
                    "id": pred.get("id"),
                    "calculator_name": gold.get("calculator_name"),
                    "predicted_answer": pred.get("answer"),
                    "ground_truth_answer": gold.get("ground_truth_answer"),
                    "lower_limit": gold.get("lower_limit"),
                    "upper_limit": gold.get("upper_limit"),
                    "parse_failure": pred.get("parse_failure", False),
                    "abstain": pred.get("abstain", False),
                    "abstain_reason": pred.get("abstain_reason", ""),
                    "formula_arithmetic_error_type": "",  # fill in manually: formula | arithmetic | unit | condition | other
                    "labeler": "",
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "demomed_error_cases.csv"
    pd.DataFrame(error_rows).to_csv(out_path, index=False)
    print(f"{len(error_rows)} error cases -> {out_path}")


if __name__ == "__main__":
    main()
