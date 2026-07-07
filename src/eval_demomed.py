"""Score predictions from each method against gold answers.

Usage:
    python src/eval_demomed.py --gold data/pilot/pilot_20.csv --pred_dir outputs/mock
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from common.console import configure_utf8_stdout  # noqa: E402
from common.jsonl_utils import read_jsonl  # noqa: E402
from common.numeric_utils import answer_is_correct, is_number  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD = PROJECT_ROOT / "data" / "pilot" / "pilot_20.csv"
DEFAULT_PRED_DIR = PROJECT_ROOT / "outputs" / "mock"
RESULTS_DIR = PROJECT_ROOT / "results"

METHODS = ["llm_only", "cot", "open_book", "demo_med"]


def evaluate_method(method: str, pred_dir: Path, gold_by_id: dict) -> dict:
    pred_path = pred_dir / f"{method}.jsonl"
    predictions = read_jsonl(pred_path)

    n = 0
    correct = 0
    no_numeric_output = 0
    abstained = 0

    for pred in predictions:
        gold = gold_by_id.get(pred.get("id"))
        if gold is None:
            continue
        n += 1

        if pred.get("abstain"):
            abstained += 1

        gt_answer = gold.get("ground_truth_answer")
        if is_number(gt_answer) and not is_number(pred.get("answer")):
            no_numeric_output += 1

        if not pred.get("abstain") and answer_is_correct(
            pred.get("answer"), gt_answer, gold.get("lower_limit"), gold.get("upper_limit")
        ):
            correct += 1

    def rate(x):
        return round(x / n, 4) if n else 0.0

    return {
        "method": method,
        "N": n,
        "answer_accuracy": rate(correct),
        "no_numeric_output_rate": rate(no_numeric_output),
        "abstain_rate": rate(abstained),
    }


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--pred_dir", type=Path, default=DEFAULT_PRED_DIR)
    args = parser.parse_args()

    if not args.gold.exists():
        print(f"error: {args.gold} not found.")
        sys.exit(1)
    if not args.pred_dir.exists():
        print(f"error: {args.pred_dir} not found.")
        sys.exit(1)

    gold_df = pd.read_csv(args.gold)
    gold_by_id = {int(row["id"]): row.to_dict() for _, row in gold_df.iterrows()}

    rows = [evaluate_method(method, args.pred_dir, gold_by_id) for method in METHODS]
    summary = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "demomed_summary.csv"
    summary.to_csv(out_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
