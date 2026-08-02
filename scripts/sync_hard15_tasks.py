"""Download leakage-safe public context for the fixed SkillsBench Hard-15."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.task_catalog import load_pilot_catalog
from agent.task_packages import default_snapshot_downloader, sync_task_packages


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "data" / "agent" / "hard15" / "task_catalog.json",
    )
    parser.add_argument(
        "--packages-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "agent" / "hard15" / "packages",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "agent" / "hard15" / "task_manifest.json",
    )
    parser.add_argument("--revision")
    return parser.parse_args(argv)


def main(argv=None, *, downloader=default_snapshot_downloader) -> int:
    args = parse_args(argv)
    catalog = load_pilot_catalog(args.catalog)
    manifest = sync_task_packages(
        catalog,
        args.packages_dir,
        downloader=downloader,
        revision=args.revision,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    ready_count = sum(task.planning_ready for task in manifest.tasks)
    print(f"Planning-ready task packages: {ready_count}/{len(manifest.tasks)}")
    print(f"Manifest: {args.manifest}")
    return 0 if manifest.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())

