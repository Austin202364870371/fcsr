import json
import tempfile
import unittest
from pathlib import Path

from preprocessing import (
    collect_benchmark_skill_ids,
    detect_language_group,
    normalize_category,
    stratified_sample,
)


def skill(skill_id: str, category: str = "", language: str = "en") -> dict:
    descriptions = {
        "en": "Build and validate a software artifact.",
        "zh": "构建并验证一个软件产物。",
        "ja": "ソフトウェア成果物を構築して検証します。",
    }
    return {
        "skill_id": skill_id,
        "name": skill_id.rsplit("/", 1)[-1],
        "description": descriptions[language],
        "body": f"# {skill_id}\n\n{descriptions[language]}",
        "category": category,
        "language": language,
    }


class SamplingTests(unittest.TestCase):
    def test_collects_all_benchmark_labeled_skill_ids(self) -> None:
        records = [
            {
                "query_id": "q1",
                "gt_skill_ids": ["gt/a"],
                "core_gt_ids": ["gt/b"],
                "auxiliary_gt_ids": ["gt/c"],
                "relevance": {"degraded/d": 1},
            }
        ]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "queries.jsonl"
            path.write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            excluded = collect_benchmark_skill_ids(path)

        self.assertEqual(excluded, {"gt/a", "gt/b", "gt/c", "degraded/d"})

    def test_category_and_language_fallbacks(self) -> None:
        self.assertEqual(normalize_category(skill("dev/a", "Engineering")), "engineering")
        self.assertEqual(normalize_category(skill("design/b")), "design")
        self.assertEqual(normalize_category({**skill("plain"), "category": ""}), "other")
        self.assertEqual(detect_language_group(skill("zh/a", language="zh")), "zh")
        self.assertEqual(
            detect_language_group({**skill("x/a"), "language": "", "description": "中文说明"}),
            "han",
        )

    def test_sampling_is_deterministic_exact_and_benchmark_safe(self) -> None:
        records = [
            skill("gold/a", "gold"),
            *(skill(f"dev/{index}", "development", "en") for index in range(6)),
            *(skill(f"data/{index}", "data", "zh") for index in range(5)),
            *(skill(f"design/{index}", "", "ja") for index in range(4)),
        ]

        first = stratified_sample(records, {"gold/a"}, sample_size=8, seed=42)
        second = stratified_sample(records, {"gold/a"}, sample_size=8, seed=42)

        self.assertEqual(first.skill_ids, second.skill_ids)
        self.assertEqual(len(first.skill_ids), 8)
        self.assertEqual(len(set(first.skill_ids)), 8)
        self.assertNotIn("gold/a", first.skill_ids)
        selected_strata = {
            (normalize_category(item), detect_language_group(item))
            for item in first.records
        }
        self.assertEqual(len(selected_strata), 3)

    def test_deduplicates_identical_content(self) -> None:
        first = skill("dev/a")
        duplicate = {**first, "skill_id": "dev/b"}
        result = stratified_sample([first, duplicate, skill("data/c", "data")], set(), 2, 7)

        selected = set(result.skill_ids)
        self.assertEqual(len(selected & {"dev/a", "dev/b"}), 1)


if __name__ == "__main__":
    unittest.main()
