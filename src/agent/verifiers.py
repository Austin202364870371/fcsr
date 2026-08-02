"""Deterministic verification for executable agent tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent.models import VerificationResult


VerifierFunction = Callable[[Any, Any], tuple[bool, str]]


class VerifierRegistry:
    """Resolve verifier identifiers without falling back to model judgment."""

    def __init__(self) -> None:
        self._functions: dict[str, VerifierFunction] = {}

    @classmethod
    def with_defaults(cls) -> "VerifierRegistry":
        registry = cls()
        registry.register("exact", _verify_exact)
        registry.register("contains_keys", _verify_contains_keys)
        return registry

    def register(self, verifier_id: str, function: VerifierFunction) -> None:
        if not isinstance(verifier_id, str) or not verifier_id:
            raise ValueError("verifier_id must be a non-empty string")
        if verifier_id in self._functions:
            raise ValueError(f"verifier already registered: {verifier_id}")
        self._functions[verifier_id] = function

    def verify(
        self,
        verifier_id: str,
        actual: Any,
        expected: Any,
    ) -> VerificationResult:
        function = self._functions.get(verifier_id)
        if function is None:
            return VerificationResult(
                passed=False,
                verifier_id=verifier_id,
                details=f"unknown verifier: {verifier_id}",
            )
        try:
            passed, details = function(actual, expected)
        except Exception as error:
            return VerificationResult(
                passed=False,
                verifier_id=verifier_id,
                details=f"verifier error: {type(error).__name__}: {error}",
            )
        return VerificationResult(
            passed=passed,
            verifier_id=verifier_id,
            details=details,
        )


def _verify_exact(actual: Any, expected: Any) -> tuple[bool, str]:
    passed = actual == expected
    details = "values match" if passed else f"expected {expected!r}, got {actual!r}"
    return passed, details


def _verify_contains_keys(actual: Any, expected: Any) -> tuple[bool, str]:
    if not isinstance(actual, Mapping):
        return False, "actual value is not a mapping"
    if not isinstance(expected, Sequence) or isinstance(expected, (str, bytes)):
        return False, "expected value must be a sequence of keys"
    missing = [key for key in expected if key not in actual]
    if missing:
        return False, f"missing keys: {missing}"
    return True, "all expected keys are present"
