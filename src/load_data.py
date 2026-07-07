"""Load the MedCalc-Bench dataset from Hugging Face and save it as normalized CSVs.

Tries, in order:
    1. nsk7153/MedCalc-Bench-Verified
    2. ncbi/MedCalc-Bench-v1.2
If both fail (no internet, gated repo, package missing, ...), prints manual
download instructions instead of crashing.

Usage:
    python src/load_data.py
"""

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent))
from common.column_normalizer import normalize_columns  # noqa: E402
from common.console import configure_utf8_stdout  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATASET_CANDIDATES = [
    "nsk7153/MedCalc-Bench-Verified",
    "ncbi/MedCalc-Bench-v1.2",
]

MANUAL_DOWNLOAD_MESSAGE = """
Could not load any MedCalc-Bench dataset automatically.

Things to check:
  1. Are you logged in to Hugging Face? Run: huggingface-cli login
  2. Is the dataset gated? Visit the dataset page on huggingface.co and
     accept any usage terms, then set HF_TOKEN in your .env file.
  3. Try downloading manually from GitHub instead:
       https://github.com/ncbi-nlp/MedCalc-Bench
     and place train/test CSV files under data/raw/ yourself
     (as data/raw/train.csv and data/raw/test.csv).

Datasets tried:
""" + "\n".join(f"  - {name}" for name in DATASET_CANDIDATES)


def load_first_available_dataset():
    """Try each dataset name in order, return (name, DatasetDict) for the first that loads."""
    from datasets import load_dataset

    last_error = None
    for name in DATASET_CANDIDATES:
        try:
            print(f"Trying to load '{name}' from Hugging Face...")
            dataset = load_dataset(name)
            print(f"Loaded '{name}'.")
            return name, dataset
        except Exception as exc:  # noqa: BLE001 - we want to try the next candidate on any failure
            print(f"  failed: {exc}")
            last_error = exc
    raise RuntimeError("All dataset candidates failed") from last_error


def describe_split(name: str, df: pd.DataFrame) -> None:
    print(f"\n--- split: {name} ---")
    print(f"rows: {len(df)}")
    print(f"columns: {list(df.columns)}")
    print("missing values per column:")
    print(df.isna().sum().to_string())
    if "calculator_name" in df.columns:
        print("calculator_name distribution:")
        print(df["calculator_name"].value_counts().to_string())


def main() -> None:
    configure_utf8_stdout()
    load_dotenv()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        dataset_name, dataset = load_first_available_dataset()
    except Exception:
        print(MANUAL_DOWNLOAD_MESSAGE)
        sys.exit(1)

    print(f"\nUsing dataset: {dataset_name}")
    print(f"Available splits: {list(dataset.keys())}")

    for split_name, split_data in dataset.items():
        df = split_data.to_pandas()
        df = normalize_columns(df)
        describe_split(split_name, df)

        # Map to the conventional train.csv / test.csv names this project expects,
        # keep any other split names as-is (e.g. "validation").
        out_name = {"train": "train.csv", "test": "test.csv"}.get(split_name, f"{split_name}.csv")
        out_path = RAW_DIR / out_name
        df.to_csv(out_path, index=False)
        print(f"saved -> {out_path}")

    if not (RAW_DIR / "test.csv").exists():
        print(
            "\nwarning: no 'test' split was found/saved. "
            "src/make_pilot.py expects data/raw/test.csv to exist."
        )


if __name__ == "__main__":
    main()
