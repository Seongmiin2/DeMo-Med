"""List every incorrect / abstained prediction per method for manual error review.

Usage:
    python src/analyze_errors.py --gold data/pilot/pilot_20.csv --pred_dir outputs/mock
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from common.console import configure_utf8_stdout  # noqa: E402
from common.jsonl_utils import read_jsonl  # noqa: E402
from common.numeric_utils import answer_is_correct  # noqa: E402
from eval_demomed import DEFAULT_GOLD, DEFAULT_PRED_DIR, METHODS  # noqa: E402

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

    error_rows = []
    for method in METHODS:
        predictions = read_jsonl(args.pred_dir / f"{method}.jsonl")
        for pred in predictions:
            gold = gold_by_id.get(pred.get("id"))
            if gold is None:
                continue

            is_correct = (not pred.get("abstain")) and answer_is_correct(
                pred.get("answer"),
                gold.get("ground_truth_answer"),
                gold.get("lower_limit"),
                gold.get("upper_limit"),
            )
            if is_correct:
                continue

            error_rows.append(
                {
                    "method": method,
                    "id": pred.get("id"),
                    "calculator_name": gold.get("calculator_name"),
                    "predicted_answer": pred.get("answer"),
                    "ground_truth_answer": gold.get("ground_truth_answer"),
                    "lower_limit": gold.get("lower_limit"),
                    "upper_limit": gold.get("upper_limit"),
                    "abstain": pred.get("abstain", False),
                    "abstain_reason": pred.get("abstain_reason", ""),
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "demomed_error_cases.csv"
    pd.DataFrame(error_rows).to_csv(out_path, index=False)
    print(f"{len(error_rows)} error cases -> {out_path}")


if __name__ == "__main__":
    main()
