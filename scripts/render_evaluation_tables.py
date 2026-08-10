"""Render canonical Hard-pool result tables from report summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation_tables import render_hard_tables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="render canonical FCSR evaluation tables")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = render_hard_tables(args.reports_dir, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
