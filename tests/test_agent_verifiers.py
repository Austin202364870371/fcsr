import unittest

from agent.verifiers import VerifierRegistry


class VerifierRegistryTests(unittest.TestCase):
    def test_exact_verifier_compares_structured_values(self) -> None:
        registry = VerifierRegistry.with_defaults()

        result = registry.verify(
            "exact",
            actual={"x": 1},
            expected={"x": 1},
        )

        self.assertTrue(result.passed)

    def test_contains_keys_requires_all_expected_keys(self) -> None:
        registry = VerifierRegistry.with_defaults()

        result = registry.verify(
            "contains_keys",
            actual={"x": 1},
            expected=["x", "y"],
        )

        self.assertFalse(result.passed)
        self.assertIn("y", result.details)

    def test_unknown_verifier_fails_closed(self) -> None:
        result = VerifierRegistry().verify(
            "missing",
            actual=1,
            expected=1,
        )

        self.assertFalse(result.passed)
        self.assertIn("unknown verifier", result.details)


if __name__ == "__main__":
    unittest.main()
