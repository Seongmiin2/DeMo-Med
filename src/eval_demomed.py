"""Score predictions from each method/provider against gold answers.

Reads every outputs/real/{method}_{provider}.jsonl file present (any provider,
not just one), scores each independently, and additionally runs McNemar's test
on the three condition-pair comparisons that isolate a specific confound:
open_book vs open_book_abstain (pure abstain effect), open_book_abstain vs
demo_med_single (does verification add anything beyond abstain?), and
demo_med_single vs demo_med_multi (does real decomposition + code execution
beat one structured prompt?) - computed separately per provider, since a
paired comparison only makes sense within the same set of model responses.

Usage:
    python src/eval_demomed.py --gold data/pilot/pilot_20.csv --pred_dir outputs/real
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

sys.path.append(str(Path(__file__).resolve().parent))
from common.console import configure_utf8_stdout  # noqa: E402
from common.jsonl_utils import read_jsonl  # noqa: E402
from common.numeric_utils import answer_is_correct, entities_match, is_number  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD = PROJECT_ROOT / "data" / "pilot" / "pilot_20.csv"
DEFAULT_PRED_DIR = PROJECT_ROOT / "outputs" / "real"
RESULTS_DIR = PROJECT_ROOT / "results"

METHODS = ["llm_only", "cot", "open_book", "open_book_abstain", "demo_med_single", "demo_med_multi"]

MCNEMAR_PAIRS = [
    ("open_book", "open_book_abstain"),
    ("open_book_abstain", "demo_med_single"),
    ("demo_med_single", "demo_med_multi"),
]


def find_prediction_files(pred_dir: Path) -> dict:
    """{(method, provider): Path} for every outputs/real/{method}_{provider}.jsonl found.

    Matches the *longest* method name that prefixes the filename stem first -
    "open_book" is itself a prefix of "open_book_abstain", so a naive per-method
    glob would double-match "open_book_abstain_openai.jsonl" as method="open_book",
    provider="abstain_openai".
    """
    found = {}
    methods_longest_first = sorted(METHODS, key=len, reverse=True)
    for path in pred_dir.glob("*.jsonl"):
        stem = path.stem
        for method in methods_longest_first:
            prefix = method + "_"
            if stem.startswith(prefix):
                provider = stem[len(prefix):]
                found[(method, provider)] = path
                break
    return found


def get_entities(pred: dict):
    entities = pred.get("entities") or pred.get("extracted_entities")
    return entities if isinstance(entities, dict) else None


def is_correct(pred: dict, gold: dict) -> bool:
    if pred.get("abstain") or pred.get("parse_failure"):
        return False
    return answer_is_correct(
        pred.get("answer"), gold.get("ground_truth_answer"), gold.get("lower_limit"), gold.get("upper_limit"), gold.get("output_type")
    )


def evaluate_file(method: str, provider: str, path: Path, gold_by_id: dict) -> dict:
    predictions = read_jsonl(path)

    n = 0
    correct = 0
    no_numeric_output = 0
    abstained = 0
    parse_failures = 0
    entity_errors = 0
    entity_checkable = 0

    for pred in predictions:
        gold = gold_by_id.get(pred.get("id"))
        if gold is None:
            continue
        n += 1

        if pred.get("parse_failure"):
            parse_failures += 1
            continue  # can't score an answer, unit, or entities that were never parsed

        if pred.get("abstain"):
            abstained += 1

        output_type = gold.get("output_type")
        if output_type in ("decimal", "integer") and not is_number(pred.get("answer")):
            no_numeric_output += 1

        if is_correct(pred, gold):
            correct += 1

        entities = get_entities(pred)
        if entities is not None:
            entity_checkable += 1
            if not entities_match(entities, gold.get("relevant_entities")):
                entity_errors += 1

    def rate(x, denom=n):
        return round(x / denom, 4) if denom else 0.0

    return {
        "method": method,
        "provider": provider,
        "N": n,
        "answer_accuracy": rate(correct),
        "no_numeric_output_rate": rate(no_numeric_output),
        "abstain_rate": rate(abstained),
        "parse_failure_rate": rate(parse_failures),
        "entity_error_rate": rate(entity_errors, entity_checkable) if entity_checkable else None,
    }


def mcnemar_compare(method_a: str, method_b: str, provider: str, files: dict, gold_by_id: dict):
    """Paired comparison of two methods on the same provider's predictions for the same items."""
    if (method_a, provider) not in files or (method_b, provider) not in files:
        return None

    preds_a = {p["id"]: p for p in read_jsonl(files[(method_a, provider)])}
    preds_b = {p["id"]: p for p in read_jsonl(files[(method_b, provider)])}
    shared_ids = set(preds_a) & set(preds_b) & set(gold_by_id)

    a_right_b_wrong = a_wrong_b_right = both_right = both_wrong = 0
    for item_id in shared_ids:
        gold = gold_by_id[item_id]
        a_ok = is_correct(preds_a[item_id], gold)
        b_ok = is_correct(preds_b[item_id], gold)
        if a_ok and b_ok:
            both_right += 1
        elif a_ok and not b_ok:
            a_right_b_wrong += 1
        elif b_ok and not a_ok:
            a_wrong_b_right += 1
        else:
            both_wrong += 1

    discordant = a_right_b_wrong + a_wrong_b_right
    p_value = binomtest(min(a_right_b_wrong, a_wrong_b_right), discordant, 0.5).pvalue if discordant else 1.0

    return {
        "provider": provider,
        "method_a": method_a,
        "method_b": method_b,
        "n": len(shared_ids),
        "a_correct_b_wrong": a_right_b_wrong,
        "a_wrong_b_correct": a_wrong_b_right,
        "both_correct": both_right,
        "both_wrong": both_wrong,
        "p_value": round(p_value, 4),
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

    files = find_prediction_files(args.pred_dir)
    if not files:
        print(f"error: no {{method}}_{{provider}}.jsonl files found under {args.pred_dir}")
        sys.exit(1)

    rows = [evaluate_file(method, provider, path, gold_by_id) for (method, provider), path in sorted(files.items())]
    summary = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "demomed_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nsaved -> {summary_path}")

    providers = sorted({provider for _, provider in files})
    mcnemar_rows = [
        result
        for provider in providers
        for method_a, method_b in MCNEMAR_PAIRS
        if (result := mcnemar_compare(method_a, method_b, provider, files, gold_by_id)) is not None
    ]
    if mcnemar_rows:
        mcnemar_df = pd.DataFrame(mcnemar_rows)
        mcnemar_path = RESULTS_DIR / "mcnemar_pairs.csv"
        mcnemar_df.to_csv(mcnemar_path, index=False)
        print("\n" + mcnemar_df.to_string(index=False))
        print(f"\nsaved -> {mcnemar_path}")


if __name__ == "__main__":
    main()
