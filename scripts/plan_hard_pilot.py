"""Generate DeepSeek Skill selections and JSON plans for public pilot tasks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from agent.hard_pilot import PublicPilotTask
from agent.planning import DeepSeekPlanningClient, plan_task
from data_io import stream_jsonl, write_jsonl_atomic


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--body-char-limit", type=int, default=1600)
    return parser.parse_args(argv)


def main(argv=None, *, client=None) -> int:
    args = parse_args(argv)
    if args.limit < 1:
        raise ValueError("limit must be positive")
    if client is None:
        project_root = Path(__file__).resolve().parents[1]
        load_dotenv(project_root / ".env", override=False)
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        client = DeepSeekPlanningClient(
            api_key=api_key,
            base_url=os.environ.get(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            ),
        )
    tasks = [
        PublicPilotTask.model_validate(record)
        for record in stream_jsonl(args.tasks)
    ][: args.limit]
    results = [
        plan_task(
            task,
            client=client,
            model=args.model,
            body_char_limit=args.body_char_limit,
        )
        for task in tasks
    ]
    write_jsonl_atomic(
        args.output,
        (result.model_dump(mode="json") for result in results),
    )
    print(f"Generated {len(results)} validated plans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
