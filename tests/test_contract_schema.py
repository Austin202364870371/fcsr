import unittest

from contract_schema import (
    ContractValidationError,
    compute_source_hash,
    validate_contract,
)


SKILL = {
    "skill_id": "design-principles/affordances",
    "name": "affordances",
    "description": (
        "Las affordances son propiedades de un objeto que sugieren cómo puede ser usado."
    ),
    "body": """# Affordances (Prestaciones)

## Aplicacion en Diseno

- Aspecto 3D o sombreado sugiere \"presionable\"
- Estados hover que invitan al click

## Anti-patterns

- Links que no parecen links

## Metricas

- Discoverability Score: porcentaje de usuarios que encuentran funciones
""",
}


def evidence_for(source_field: str, quote: str, evidence_id: str) -> dict:
    source = SKILL[source_field]
    start = source.index(quote)
    return {
        "evidence_id": evidence_id,
        "source_field": source_field,
        "quote": quote,
        "start_char": start,
        "end_char": start + len(quote),
    }


def valid_contract() -> dict:
    return {
        "schema_version": "contract_v2",
        "skill_id": SKILL["skill_id"],
        "source_hash": compute_source_hash(SKILL),
        "source_languages": ["es"],
        "canonical_language": "en",
        "capability": {
            "summary": (
                "Apply affordance principles so interface elements communicate how "
                "they can be used."
            ),
            "evidence_ids": ["ev_capability"],
        },
        "operations": [
            {
                "action": "design",
                "target": "buttons",
                "outcome": "make buttons appear pressable",
                "qualifiers": ["use three-dimensional appearance or shadow"],
                "evidence_ids": ["ev_operation"],
            }
        ],
        "inputs": [],
        "outputs": [],
        "preconditions": [],
        "constraints": [
            {
                "statement": "Links should look interactive.",
                "evidence_ids": ["ev_constraint"],
            }
        ],
        "dependencies": [],
        "exclusions": [],
        "quality_criteria": [
            {
                "statement": "Measure how many users discover available functions.",
                "evidence_ids": ["ev_quality"],
            }
        ],
        "evidence": [
            evidence_for(
                "description",
                "Las affordances son propiedades de un objeto que sugieren cómo puede ser usado.",
                "ev_capability",
            ),
            evidence_for(
                "body",
                'Aspecto 3D o sombreado sugiere "presionable"',
                "ev_operation",
            ),
            evidence_for(
                "body",
                "Links que no parecen links",
                "ev_constraint",
            ),
            evidence_for(
                "body",
                "Discoverability Score: porcentaje de usuarios que encuentran funciones",
                "ev_quality",
            ),
        ],
        "extraction": {
            "method": "llm",
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "prompt_version": "contract_v2_prompt_001",
            "temperature": 0.0,
            "status": "validated",
            "attempts": 1,
            "warnings": [],
        },
    }


class ContractSchemaTests(unittest.TestCase):
    def test_accepts_multilingual_evidence_backed_contract(self) -> None:
        contract = validate_contract(valid_contract(), SKILL)

        self.assertEqual(contract["canonical_language"], "en")
        self.assertEqual(contract["source_languages"], ["es"])
        self.assertEqual(contract["operations"][0]["action"], "design")
        self.assertEqual(
            contract["quality_criteria"][0]["evidence_ids"],
            ["ev_quality"],
        )

    def test_rejects_source_hash_that_does_not_match_skill(self) -> None:
        contract = valid_contract()
        contract["source_hash"] = "a" * 64

        with self.assertRaisesRegex(ContractValidationError, "source_hash"):
            validate_contract(contract, SKILL)

    def test_rejects_incorrect_evidence_offsets(self) -> None:
        contract = valid_contract()
        contract["evidence"][0]["start_char"] += 1
        contract["evidence"][0]["end_char"] += 1

        with self.assertRaisesRegex(ContractValidationError, "offset"):
            validate_contract(contract, SKILL)

    def test_rejects_semantic_item_with_unknown_evidence_reference(self) -> None:
        contract = valid_contract()
        contract["operations"][0]["evidence_ids"] = ["ev_missing"]

        with self.assertRaisesRegex(ContractValidationError, "ev_missing"):
            validate_contract(contract, SKILL)

    def test_rejects_non_llm_extraction_method(self) -> None:
        contract = valid_contract()
        contract["extraction"]["method"] = "rules"

        with self.assertRaisesRegex(ContractValidationError, "method"):
            validate_contract(contract, SKILL)


if __name__ == "__main__":
    unittest.main()
