import unittest

from preprocessing import filter_identity_and_overlap, mine_local_negatives


TOPICS = (
    "schema migration",
    "image compression",
    "network tracing",
    "invoice reconciliation",
    "accessibility auditing",
    "cache invalidation",
    "dependency analysis",
    "release automation",
    "log aggregation",
    "database backup",
    "api pagination",
    "test generation",
    "color contrast",
    "layout typography",
    "component theming",
    "interaction prototyping",
    "data visualization",
    "icon composition",
    "responsive spacing",
    "design review",
    "motion timing",
    "content hierarchy",
)


def make_skill(index: int, category: str = "development") -> dict:
    topic = TOPICS[index % len(TOPICS)]
    return {
        "skill_id": f"{category}/skill-{index}",
        "name": f"skill {index}",
        "description": f"Handle {topic} using a specialized workflow.",
        "body": (
            f"Inspect inputs for {topic}. Apply the domain-specific procedure, "
            f"check its result, and produce artifact {index}."
        ),
        "category": category,
    }


class NegativeMiningTests(unittest.TestCase):
    def test_filter_records_exact_and_high_overlap_reasons(self) -> None:
        positive = make_skill(0)
        exact = {**make_skill(90), "body": positive["body"]}
        overlap = {
            **make_skill(91),
            "body": positive["body"] + " A tiny additional suffix.",
        }

        distinct = {
            **make_skill(92),
            "body": "Generate watercolor thumbnails and export transparent PNG assets.",
        }
        result = filter_identity_and_overlap(
            positive,
            [positive, exact, overlap, distinct],
            threshold=0.75,
        )

        reasons = {item["skill_id"]: item["reason"] for item in result.removed}
        self.assertEqual(reasons[positive["skill_id"]], "same_skill_id")
        self.assertEqual(reasons[exact["skill_id"]], "same_body")
        self.assertEqual(reasons[overlap["skill_id"]], "high_body_trigram_overlap")
        self.assertEqual([item["skill_id"] for item in result.kept], ["development/skill-92"])

    def test_local_mining_is_deterministic_unique_and_respects_source_caps(self) -> None:
        pool = [make_skill(index) for index in range(12)]
        pool.extend(make_skill(index, "design") for index in range(12, 22))
        query = {
            "query_id": "syn::development/skill-0",
            "query": "Validate structured records and produce a checked report.",
            "positive_skill_id": "development/skill-0",
        }

        first = list(mine_local_negatives([query], pool, seed=42))[0]
        second = list(mine_local_negatives([query], pool, seed=42))[0]

        self.assertEqual(first, second)
        candidates = first["negative_candidates"]
        ids = [item["skill_id"] for item in candidates]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn(query["positive_skill_id"], ids)
        counts = {
            source: sum(item["source"] == source for item in candidates)
            for source in ("bm25", "same_category", "random")
        }
        self.assertLessEqual(counts["bm25"], 3)
        self.assertLessEqual(counts["same_category"], 2)
        self.assertLessEqual(counts["random"], 1)
        self.assertEqual(len(candidates), 6)

    def test_unicode_tokenization_supports_cjk_query_matching(self) -> None:
        pool = [
            {
                "skill_id": "data/chinese",
                "name": "数据验证",
                "description": "验证表格数据并生成报告",
                "body": "读取输入文件，检查缺失值。",
                "category": "data",
            },
            make_skill(1, "data"),
            make_skill(2, "data"),
            make_skill(3, "other"),
            make_skill(4, "other"),
            make_skill(5, "other"),
            make_skill(6, "other"),
        ]
        query = {
            "query_id": "q",
            "query": "需要验证表格数据并输出报告",
            "positive_skill_id": "data/skill-1",
        }

        result = list(mine_local_negatives([query], pool, seed=7))[0]

        self.assertEqual(result["negative_candidates"][0]["skill_id"], "data/chinese")
        self.assertEqual(result["negative_candidates"][0]["source"], "bm25")


    def test_local_mining_reports_stages_and_each_processed_query(self) -> None:
        pool = [make_skill(index) for index in range(12)]
        queries = [
            {
                "query_id": f"q-{index}",
                "query": "Validate structured records and produce a checked report.",
                "positive_skill_id": f"development/skill-{index}",
            }
            for index in range(2)
        ]
        stages = []
        updates = []

        results = list(
            mine_local_negatives(
                queries,
                pool,
                seed=42,
                stage=stages.append,
                progress=updates.append,
            )
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(
            stages,
            ["loading_skills", "building_bm25", "mining_queries"],
        )
        self.assertEqual(updates, [1, 2])


if __name__ == "__main__":
    unittest.main()