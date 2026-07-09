"""Run the 6 prompting methods against a pilot set using a real LLM API.

By default this runs in --dry-run mode and only prints the prompts it *would*
send, so it never calls a paid API unless you explicitly pass --live.

Conditions:
    llm_only, cot, open_book, open_book_abstain, demo_med_single, demo_med_multi

demo_med_multi is the real multi-call decomposition: an M2 (extract) call, an
M3 (verify) call, then M4 - not an LLM call, `formula_executor.py` computes the
answer directly from the verified entities. The other 5 conditions are single
LLM calls.

Usage:
    python src/run_experiment.py --input data/pilot/pilot_20.csv --dry-run
    python src/run_experiment.py --input data/pilot/pilot_20.csv --live --provider openai
    python src/run_experiment.py --input data/pilot/pilot_20.csv --live --provider together --method demo_med_multi
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
import formula_executor  # noqa: E402
from common.console import configure_utf8_stdout  # noqa: E402
from common.jsonl_utils import write_jsonl  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
CARDS_DIR = PROJECT_ROOT / "knowledge_cards"
REAL_DIR = PROJECT_ROOT / "outputs" / "real"

METHODS = ["llm_only", "cot", "open_book", "open_book_abstain", "demo_med_single", "demo_med_multi"]
NEEDS_KNOWLEDGE_CARD = {"open_book", "open_book_abstain", "demo_med_single", "demo_med_multi"}
# demo_med_single reuses the original single-call demo_med.txt prompt verbatim.
PROMPT_FILE_OVERRIDES = {"demo_med_single": "demo_med"}

PROVIDERS = {
    "openai": {
        "base_url": None,  # default OpenAI endpoint
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "supports_json_response_format": True,
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "default_model": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "supports_json_response_format": False,  # not guaranteed for every hosted model
    },
}

SYSTEM_PROMPT = (
    "You are a careful clinical calculation assistant. Follow the instructions "
    "exactly and respond with a single JSON object only, no prose outside the JSON."
)


class JSONParseFailure(Exception):
    """Raised when the model's response can't be parsed as JSON even after retries."""


def load_prompt_template(name: str) -> str:
    filename = PROMPT_FILE_OVERRIDES.get(name, name)
    return (PROMPTS_DIR / f"{filename}.txt").read_text(encoding="utf-8")


def load_knowledge_card(calculator_name: str) -> str:
    """Look up a knowledge card by slugified calculator name; empty string if missing."""
    slug = re.sub(r"[^a-z0-9]+", "_", calculator_name.lower()).strip("_")
    card_path = CARDS_DIR / f"{slug}.json"
    if not card_path.exists():
        return "(no knowledge card available for this calculator)"
    return card_path.read_text(encoding="utf-8")


def fill_placeholders(template: str, values: dict) -> str:
    # Templates embed literal JSON examples (e.g. "{ \"answer\": ... }"), so plain
    # str.format() would choke on those braces. Do targeted placeholder replacement instead.
    result = template
    for key, value in values.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def build_single_call_prompt(method: str, row: pd.Series) -> str:
    template = load_prompt_template(method)
    knowledge_card = load_knowledge_card(row["calculator_name"]) if method in NEEDS_KNOWLEDGE_CARD else ""
    return fill_placeholders(
        template,
        {"patient_note": row.get("patient_note", ""), "question": row.get("question", ""), "knowledge_card": knowledge_card},
    )


def build_extract_prompt(row: pd.Series) -> str:
    template = load_prompt_template("demo_med_extract")
    return fill_placeholders(
        template,
        {
            "patient_note": row.get("patient_note", ""),
            "question": row.get("question", ""),
            "knowledge_card": load_knowledge_card(row["calculator_name"]),
        },
    )


def build_verify_prompt(row: pd.Series, extracted_entities: dict) -> str:
    template = load_prompt_template("demo_med_verify")
    return fill_placeholders(
        template,
        {
            "knowledge_card": load_knowledge_card(row["calculator_name"]),
            "extracted_entities": json.dumps(extracted_entities),
        },
    )


_clients = {}


def get_client(provider: str):
    """Lazily create one OpenAI-SDK client per provider, so --dry-run never needs an API key."""
    if provider not in _clients:
        from openai import OpenAI

        config = PROVIDERS[provider]
        import os

        api_key = os.environ.get(config["api_key_env"])
        _clients[provider] = OpenAI(api_key=api_key, base_url=config["base_url"])
    return _clients[provider]


def call_llm(prompt: str, provider: str, model: str, max_retries: int = 3) -> str:
    """Send `prompt` to the given provider's chat completions API, return the raw text response."""
    client = get_client(provider)
    config = PROVIDERS[provider]
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            kwargs = {}
            if config["supports_json_response_format"]:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                **kwargs,
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


def call_llm_and_parse(prompt: str, provider: str, model: str, max_parse_retries: int = 2) -> dict:
    """call_llm() + parse_json_response(), retrying the whole generation (not just the
    parse) if the model's output isn't valid JSON. Raises JSONParseFailure if it never
    parses - the caller records this distinctly from a wrong-but-parsed answer."""
    last_error = None
    for attempt in range(1, max_parse_retries + 1):
        raw_text = call_llm(prompt, provider, model)
        try:
            return parse_json_response(raw_text)
        except json.JSONDecodeError as exc:
            last_error = exc
    raise JSONParseFailure(f"could not parse JSON after {max_parse_retries} attempts") from last_error


def run_demo_med_multi(row: pd.Series, provider: str, model: str) -> dict:
    """M2 (extract) -> M3 (verify) -> M4 (formula_executor, not an LLM call)."""
    calculator_name = row["calculator_name"]

    extract_prompt = build_extract_prompt(row)
    extracted = call_llm_and_parse(extract_prompt, provider, model)
    entities = extracted.get("entities", {})

    verify_prompt = build_verify_prompt(row, entities)
    verification = call_llm_and_parse(verify_prompt, provider, model)

    record = {
        "calculator": calculator_name,
        "extracted_entities": verification.get("corrected_entities") or entities,
        "verification": {
            "missing_values": verification.get("missing_values", []),
            "unit_errors": verification.get("unit_errors", []),
            "condition_errors": verification.get("condition_errors", []),
        },
    }

    if verification.get("verification_result") != "pass":
        record.update(
            answer="",
            unit="",
            abstain=True,
            abstain_reason=verification.get("abstain_reason") or "verification failed",
        )
        return record

    try:
        value, unit = formula_executor.execute(calculator_name, record["extracted_entities"])
        record.update(answer=str(value), unit=unit, abstain=False, abstain_reason="")
    except Exception as exc:  # noqa: BLE001 - a bad/missing entity means abstain, not a guess
        record.update(answer="", unit="", abstain=True, abstain_reason=f"formula execution failed: {exc}")

    return record


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "pilot" / "pilot_20.csv")
    parser.add_argument("--method", choices=METHODS, default=None, help="run a single method only")
    parser.add_argument("--provider", choices=list(PROVIDERS), default="openai")
    parser.add_argument("--model", default=None, help="defaults to the provider's default_model")
    parser.add_argument("--live", action="store_true", help="actually call the LLM API")
    parser.add_argument("--dry-run", dest="live", action="store_false")
    parser.set_defaults(live=False)
    args = parser.parse_args()

    load_dotenv()
    model = args.model or PROVIDERS[args.provider]["default_model"]

    if not args.input.exists():
        print(f"error: {args.input} not found. Run src/make_pilot.py first.")
        sys.exit(1)

    df = pd.read_csv(args.input)
    methods = [args.method] if args.method else METHODS

    for method in methods:
        records = []
        rows = list(df.iterrows())
        iterator = tqdm(rows, desc=f"{method} [{args.provider}]") if args.live else rows
        for _, row in iterator:
            if not args.live:
                if method == "demo_med_multi":
                    print(f"\n===== [demo_med_multi] id={row.get('id')} M2 prompt (dry-run, not sent) =====")
                    print(build_extract_prompt(row))
                else:
                    print(f"\n===== [{method}] id={row.get('id')} (dry-run, not sent) =====")
                    print(build_single_call_prompt(method, row))
                continue

            row_id = int(row["id"])
            try:
                if method == "demo_med_multi":
                    parsed = run_demo_med_multi(row, args.provider, model)
                else:
                    prompt = build_single_call_prompt(method, row)
                    parsed = call_llm_and_parse(prompt, args.provider, model)
                parsed["id"] = row_id
            except JSONParseFailure as exc:
                print(f"  [{method}] id={row_id} PARSE FAILURE: {exc}")
                parsed = {"id": row_id, "answer": "", "unit": "", "parse_failure": True}
            except Exception as exc:  # noqa: BLE001 - one bad row shouldn't kill the whole run
                print(f"  [{method}] id={row_id} FAILED: {exc}")
                parsed = {"id": row_id, "answer": "", "unit": "", "error": str(exc)}
            records.append(parsed)

        if args.live:
            out_path = REAL_DIR / f"{method}_{args.provider}.jsonl"
            write_jsonl(out_path, records)
            print(f"wrote {len(records)} records -> {out_path}")

    if not args.live:
        print("\n(dry-run only, nothing was sent or saved. Re-run with --live once ready.)")


if __name__ == "__main__":
    main()
