from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

from easyicu.research_agent import cohort_schema, schema
from easyicu.research_agent.planning import cohort_contract, robustness_contract


def _imports(module) -> set[str]:
    tree = ast.parse(Path(inspect.getsourcefile(module)).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_robustness_contract_legacy_exports_are_identical() -> None:
    legacy = importlib.import_module("easyicu.research_agent.robustness_panel")
    for name in (
        "MIN_AXIS_COUNTS",
        "RobustnessPlanError",
        "RobustnessSpec",
        "validate_robustness_specs",
    ):
        assert getattr(legacy, name) is getattr(robustness_contract, name)


def test_schema_uses_the_pure_robustness_contract() -> None:
    assert schema.RobustnessPlanError is robustness_contract.RobustnessPlanError
    assert schema.RobustnessSpec is robustness_contract.RobustnessSpec
    assert (
        schema.validate_robustness_specs
        is robustness_contract.validate_robustness_specs
    )


def test_robustness_contract_has_no_runtime_or_evidence_dependency() -> None:
    imported = _imports(robustness_contract)
    forbidden = {
        "easyicu.research_agent.evidence",
        "easyicu.research_agent.authority.lock_contract",
        "easyicu.research_agent.robustness_panel",
        "easyicu.research_agent.runtime_artifacts",
    }
    assert not imported & forbidden
    assert all(not name.endswith("pipeline") for name in imported)


def test_robustness_spec_round_trip_preserves_payload() -> None:
    spec = robustness_contract.RobustnessSpec(
        spec_id="complete_case",
        axis="missing",
        description="Complete-case sensitivity",
        missing_override={"strategy": "complete_case"},
    )
    restored = robustness_contract.RobustnessSpec.from_dict(spec.to_dict())
    assert restored == spec


def test_cohort_contract_legacy_exports_are_identical() -> None:
    for name in (
        "ALLOWED_CTAS_AGGREGATIONS",
        "Aggregation",
        "CohortDefinition",
        "CohortSchemaError",
        "ConceptPredicate",
        "PatternRegistry",
        "PredicateOp",
        "TimeAnchor",
        "TimeWindow",
        "UNIVERSAL_ANCHORS",
        "clear_cohort_concept_ids",
        "coerce_cohort_definition",
        "cohort_definition_sha",
        "concept_id_exists",
        "default_pattern_registry",
        "ensure_cohort_definition",
        "expand_named_cohort",
        "known_concept_ids",
        "register_cohort_concept_ids",
        "register_pattern",
        "register_patterns_from_file",
        "reset_pattern_registry",
        "validate_cohort_definition",
        "validate_concept_predicate",
    ):
        assert getattr(cohort_schema, name) is getattr(cohort_contract, name)


def test_schema_uses_the_pure_cohort_contract() -> None:
    assert schema.CohortDefinition is cohort_contract.CohortDefinition
    assert schema.CohortSchemaError is cohort_contract.CohortSchemaError


def test_cohort_contract_has_no_runtime_or_evidence_dependency() -> None:
    imported = _imports(cohort_contract)
    forbidden_suffixes = (
        "cohort_schema",
        "evidence",
        "lock_authority",
        "pipeline",
        "runtime_artifacts",
    )
    assert not any(name.endswith(forbidden_suffixes) for name in imported)


def test_cohort_contract_owns_one_process_registry() -> None:
    assert (
        cohort_schema.default_pattern_registry()
        is cohort_contract.default_pattern_registry()
    )
    cohort_contract.clear_cohort_concept_ids()
    try:
        cohort_schema.register_cohort_concept_ids(["materialized_signal"])
        assert cohort_contract.concept_id_exists("materialized_signal")
    finally:
        cohort_contract.clear_cohort_concept_ids()


def test_cohort_contract_resolves_packaged_concept_dictionary() -> None:
    assert cohort_contract._CONCEPT_DICT_PATH.is_file()
    assert cohort_contract.known_concept_ids()


def test_cohort_contract_round_trip_and_digest_match_legacy_path() -> None:
    definition = cohort_contract.CohortDefinition(
        name="adult",
        inclusion=(
            cohort_contract.ConceptPredicate(
                concept_id="age",
                time_window=cohort_contract.TimeWindow("icu_admit", 0, 24),
                aggregation="max",
                op=">=",
                value=18,
            ),
        ),
    )
    restored = cohort_contract.CohortDefinition.from_dict(definition.to_dict())
    assert restored == definition
    assert cohort_schema.cohort_definition_sha(definition) == (
        cohort_contract.cohort_definition_sha(definition)
    )
