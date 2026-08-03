"""Streaming JSONL utilities shared by preprocessing and evaluation."""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TextIO


class JsonlError(ValueError):
    """Raised when a JSONL record cannot be decoded."""


def iter_jsonl_paths(path: str | Path) -> list[Path]:
    candidate = Path(path)
    if candidate.is_file() and _is_jsonl(candidate):
        return [candidate]
    if candidate.is_dir():
        return sorted(item for item in candidate.iterdir() if item.is_file() and _is_jsonl(item))
    raise FileNotFoundError(f"JSONL path not found: {candidate}")


def open_text(path: str | Path) -> TextIO:
    candidate = Path(path)
    if candidate.name.endswith(".gz"):
        return gzip.open(candidate, "rt", encoding="utf-8")
    return candidate.open("r", encoding="utf-8")


def stream_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    for file_path in iter_jsonl_paths(path):
        with open_text(file_path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise JsonlError(f"{file_path.name}:{line_number}: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise JsonlError(
                        f"{file_path.name}:{line_number}: record must be a JSON object"
                    )
                yield record


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(stream_jsonl(path))


def write_jsonl_atomic(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{destination}.tmp")
    count = 0
    try:
        opener = (
            gzip.open
            if destination.name.endswith(".gz")
            else lambda target, mode, encoding, newline: target.open(
                mode, encoding=encoding, newline=newline
            )
        )
        with opener(temporary, "wt", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
                count += 1
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def _is_jsonl(path: Path) -> bool:
    return path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz")
