import unittest

from agent.models import SkillBundle, SkillCandidate
from agent.selectors import FirstRankedSelector


class SelectorTests(unittest.TestCase):
    def test_first_ranked_selector_is_deterministic(self) -> None:
        skill = SkillCandidate(
            skill_id="s1",
            name="s1",
            description="x",
            rank=1,
            score=1.0,
            tool_name="echo",
        )
        selector = FirstRankedSelector(
            argument_builder=lambda task, selected: {"text": task}
        )

        selection = selector.select(
            "hello",
            SkillBundle(strategy="flat", skills=[skill]),
        )

        self.assertEqual(selection.skill_id, "s1")
        self.assertEqual(selection.tool_name, "echo")
        self.assertEqual(selection.arguments, {"text": "hello"})

    def test_empty_bundle_has_no_selection(self) -> None:
        selection = FirstRankedSelector().select(
            "hello",
            SkillBundle(strategy="flat", skills=[]),
        )

        self.assertIsNone(selection)

    def test_argument_builder_must_return_object(self) -> None:
        skill = SkillCandidate(
            skill_id="s1",
            name="s1",
            rank=1,
            score=1.0,
            tool_name="echo",
        )
        selector = FirstRankedSelector(
            argument_builder=lambda task, selected: [task]
        )

        with self.assertRaisesRegex(ValueError, "arguments"):
            selector.select(
                "hello",
                SkillBundle(strategy="flat", skills=[skill]),
            )


if __name__ == "__main__":
    unittest.main()
