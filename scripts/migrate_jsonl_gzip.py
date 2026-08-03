"""Convert JSONL datasets to gzip while verifying decompressed content."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GzipMigration:
    source: Path
    target: Path
    records: int
    source_sha256: str
    decompressed_sha256: str


def _gzip_target(source: Path) -> Path:
    if source.suffix != ".jsonl":
        raise ValueError(f"source must end in .jsonl: {source}")
    return source.with_name(f"{source.name}.gz")


def gzip_jsonl(
    source: str | Path,
    target: str | Path | None = None,
    *,
    remove_source: bool = False,
) -> GzipMigration:
    """Create a verified .jsonl.gz copy, preserving every decompressed byte."""
    source_path = Path(source)
    target_path = Path(target) if target is not None else _gzip_target(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if target_path.exists():
        raise FileExistsError(target_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{target_path}.tmp")
    source_digest = hashlib.sha256()
    record_count = 0
    try:
        with (
            source_path.open("rb") as input_handle,
            gzip.open(temporary, "wb") as output_handle,
        ):
            while chunk := input_handle.read(1024 * 1024):
                source_digest.update(chunk)
                record_count += chunk.count(b"\n")
                output_handle.write(chunk)

        decompressed_digest = hashlib.sha256()
        with gzip.open(temporary, "rb") as input_handle:
            while chunk := input_handle.read(1024 * 1024):
                decompressed_digest.update(chunk)
        if source_digest.digest() != decompressed_digest.digest():
            raise RuntimeError(f"gzip verification failed: {source_path}")

        os.replace(temporary, target_path)
        if remove_source:
            source_path.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return GzipMigration(
        source=source_path,
        target=target_path,
        records=record_count,
        source_sha256=source_digest.hexdigest(),
        decompressed_sha256=decompressed_digest.hexdigest(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="gzip JSONL files and verify their decompressed content"
    )
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--remove-source", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = args.paths or sorted(args.data_root.rglob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no .jsonl files found under {args.data_root}")
    for source in paths:
        result = gzip_jsonl(source, remove_source=args.remove_source)
        print(
            f"{result.source} -> {result.target} "
            f"records={result.records} sha256={result.source_sha256}"
        )


if __name__ == "__main__":
    main()