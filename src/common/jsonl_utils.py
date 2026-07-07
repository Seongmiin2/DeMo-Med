"""Small helpers for reading and writing .jsonl files (one JSON object per line)."""

import json
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(path) -> list[dict]:
    """Read a .jsonl file into a list of dicts. Returns [] if the file is empty/missing."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def iter_jsonl(path) -> Iterator[dict]:
    """Same as read_jsonl but lazy, for large files."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, records: Iterable[dict]) -> None:
    """Write an iterable of dicts to path, one JSON object per line. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def append_jsonl(path, record: dict) -> None:
    """Append a single record to a .jsonl file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")
