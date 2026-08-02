import json
import unittest
from collections import Counter
from pathlib import Path

from agent.task_catalog import load_pilot_catalog


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "agent" / "hard15" / "task_catalog.json"
IDS_PATH = PROJECT_ROOT / "data" / "agent" / "hard15" / "task_ids.txt"


class Hard15CatalogTests(unittest.TestCase):
    def test_catalog_has_fixed_strata_unique_ids_and_matching_text_list(self) -> None:
        catalog = load_pilot_catalog(CATALOG_PATH)

        self.assertEqual(len(catalog.tasks), 15)
        self.assertEqual(len({task.task_id for task in catalog.tasks}), 15)
        self.assertEqual(
            Counter(task.stratum for task in catalog.tasks),
            Counter(
                {
                    "single_full": 3,
                    "single_missing": 2,
                    "multi_full": 5,
                    "multi_missing": 5,
                }
            ),
        )
        listed_ids = IDS_PATH.read_text(encoding="utf-8").splitlines()
        self.assertEqual(listed_ids, [task.task_id for task in catalog.tasks])

    def test_catalog_is_pinned_and_uses_default_task_paths(self) -> None:
        payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        catalog = load_pilot_catalog(CATALOG_PATH)

        self.assertEqual(payload["skillsbench_version"], "v1.1")
        self.assertEqual(
            catalog.github_commit,
            "b63b7b2850226b6aa4fb5929a8c1ac7bc4d9a6af",
        )
        for task in catalog.tasks:
            self.assertEqual(task.source_task_id, task.task_id)
            self.assertEqual(task.source_path, f"tasks/{task.task_id}")
            self.assertGreater(task.estimated_context_bytes, 0)


if __name__ == "__main__":
    unittest.main()
