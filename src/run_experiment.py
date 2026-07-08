"""Run the four prompting methods against a pilot set using a real LLM API.

This is phase 2 (after the mock pipeline works end-to-end). By default it runs
in --dry-run mode and only prints the prompts it *would* send, so it never
calls a paid API unless you explicitly pass --live.

Usage:
    python src/run_experiment.py --input data/pilot/pilot_20.csv --dry-run
    python src/run_experiment.py --input data/pilot/pilot_20.csv --live --method demo_med
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from common.console import configure_utf8_stdout  # noqa: E402
from common.jsonl_utils import write_jsonl  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
CARDS_DIR = PROJECT_ROOT / "knowledge_cards"
REAL_DIR = PROJECT_ROOT / "outputs" / "real"

METHODS = ["llm_only", "cot", "open_book", "demo_med"]
NEEDS_KNOWLEDGE_CARD = {"open_book", "demo_med"}
DEFAULT_MODEL = "gpt-4o"
SYSTEM_PROMPT = (
    "You are a careful clinical calculation assistant. Follow the instructions "
    "exactly and respond with a single JSON object only, no prose outside the JSON."
)


def load_prompt_template(method: str) -> str:
    return (PROMPTS_DIR / f"{method}.txt").read_text(encoding="utf-8")


def load_knowledge_card(calculator_name: str) -> str:
    """Look up a knowledge card by slugified calculator name; empty string if missing."""
    import re

    slug = re.sub(r"[^a-z0-9]+", "_", calculator_name.lower()).strip("_")
    card_path = CARDS_DIR / f"{slug}.json"
    if not card_path.exists():
        return "(no knowledge card available for this calculator)"
    return card_path.read_text(encoding="utf-8")


def build_prompt(method: str, row: pd.Series) -> str:
    # Templates embed literal JSON examples (e.g. "{ \"answer\": ... }"), so plain
    # str.format() would choke on those braces. Do targeted placeholder replacement instead.
    template = load_prompt_template(method)
    knowledge_card = ""
    if method in NEEDS_KNOWLEDGE_CARD:
        knowledge_card = load_knowledge_card(row["calculator_name"])
    prompt = template.replace("{patient_note}", str(row.get("patient_note", "")))
    prompt = prompt.replace("{question}", str(row.get("question", "")))
    prompt = prompt.replace("{knowledge_card}", knowledge_card)
    return prompt


_client = None


def get_client():
    """Lazily create the OpenAI client so --dry-run never needs an API key installed/set."""
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI()  # reads OPENAI_API_KEY from the environment (set via .env)
    return _client


def call_llm(prompt: str, model: str, max_retries: int = 3) -> str:
    """Send `prompt` to the OpenAI chat completions API and return its raw text response."""
    client = get_client()
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001 - retry on any transient API error
            last_error = exc
            if attempt < max_retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"call_llm failed after {max_retries} attempts") from last_error


def parse_json_response(raw_text: str) -> dict:
    """Best-effort JSON parse of a model response (models sometimes wrap JSON in prose)."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start, end = raw_text.find("{"), raw_text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw_text[start : end + 1])
        raise


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "pilot" / "pilot_20.csv")
    parser.add_argument("--method", choices=METHODS, default=None, help="run a single method only")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--live", action="store_true", help="actually call the LLM API")
    parser.add_argument("--dry-run", dest="live", action="store_false")
    parser.set_defaults(live=False)
    args = parser.parse_args()

    load_dotenv()

    if not args.input.exists():
        print(f"error: {args.input} not found. Run src/make_pilot.py first.")
        sys.exit(1)

    df = pd.read_csv(args.input)
    methods = [args.method] if args.method else METHODS

    for method in methods:
        records = []
        rows = list(df.iterrows())
        iterator = tqdm(rows, desc=method) if args.live else rows
        for _, row in iterator:
            prompt = build_prompt(method, row)

            if not args.live:
                print(f"\n===== [{method}] id={row.get('id')} (dry-run, not sent) =====")
                print(prompt)
                continue

            row_id = int(row["id"])
            try:
                raw_text = call_llm(prompt, model=args.model)
                parsed = parse_json_response(raw_text)
                parsed["id"] = row_id
            except Exception as exc:  # noqa: BLE001 - one bad row shouldn't kill the whole run
                print(f"  [{method}] id={row_id} FAILED: {exc}")
                parsed = {"id": row_id, "answer": "", "unit": "", "error": str(exc)}
            records.append(parsed)

        if args.live:
            out_path = REAL_DIR / f"{method}.jsonl"
            write_jsonl(out_path, records)
            print(f"wrote {len(records)} records -> {out_path}")

    if not args.live:
        print("\n(dry-run only, nothing was sent or saved. Re-run with --live once call_llm() is implemented.)")


if __name__ == "__main__":
    main()
