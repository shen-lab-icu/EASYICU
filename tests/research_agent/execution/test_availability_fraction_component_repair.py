"""Regression coverage for audited availability denominators and render scaling."""

from __future__ import annotations

import ast

from easyicu.research_agent.agents.core import _REPLANNER_GUIDE
from easyicu.research_agent.audits.validators import LLMConceptAuditor
from easyicu.research_agent.repair_registry import (
    RepairClass,
    automatic_repair_allowed,
    repair_metadata_for,
)
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair
from easyicu.research_agent.schema import CohortDescriptor, ResearchContext


_FINDING = (
    "Availability fractions are rendered without reconciling n_nonmissing "
    "and missing_n to n; the script explicitly declines this check while "
    "presenting the resulting percentages."
)


def _candidate_code() -> str:
    return """
import pandas as pd

continuous = pd.DataFrame({
    "n": [3, 1],
    "n_nonmissing": [3, 1],
    "missing_n": [1, 3],
})
missing_count = continuous["missing_n"]
availability_fraction = (
    continuous["n_nonmissing"] / continuous["n"]
).to_numpy(dtype=float)
summary = {
    "count_validation": {
        "availability_denominator": "n",
        "missing_n_reconciliation": (
            "not imposed because n_nonmissing and missing_n are not "
            "required to sum to n"
        ),
    },
    "availability_definition": "n_nonmissing / n",
}
""".lstrip()


def test_audited_availability_uses_both_count_components() -> None:
    repaired, names = deterministic_concept_audit_repair(
        _candidate_code(),
        [_FINDING],
    )

    assert names == ["availability_fraction_component_denominator_v1"]
    assert 'continuous["n_nonmissing"] + continuous["missing_n"]' in repaired
    assert "n_nonmissing / (n_nonmissing + missing_n)" in repaired
    assert "availability denominator reconstructed exactly" in repaired
    ast.parse(repaired)

    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    assert namespace["availability_fraction"].tolist() == [0.75, 0.25]


def test_availability_component_repair_is_idempotent() -> None:
    once, first_names = deterministic_concept_audit_repair(
        _candidate_code(),
        [_FINDING],
    )
    twice, second_names = deterministic_concept_audit_repair(once, [_FINDING])

    assert first_names == ["availability_fraction_component_denominator_v1"]
    assert second_names == []
    assert twice == once


def test_availability_component_repair_requires_audit_authority() -> None:
    code = _candidate_code()
    repaired, names = deterministic_concept_audit_repair(
        code,
        ["The heatmap title could be shorter."],
    )

    assert names == []
    assert repaired == code


def test_availability_component_repair_requires_missing_component_use() -> None:
    code = """
availability_fraction = frame["n_nonmissing"] / frame["n"]
""".lstrip()
    repaired, names = deterministic_concept_audit_repair(code, [_FINDING])

    assert names == []
    assert repaired == code


def test_availability_component_repair_refuses_ambiguous_divisions() -> None:
    code = """
missing = frame["missing_n"]
first = frame["n_nonmissing"] / frame["n"]
second = frame["n_nonmissing"] / frame["n"]
""".lstrip()
    repaired, names = deterministic_concept_audit_repair(code, [_FINDING])

    assert names == []
    assert repaired == code


def test_availability_component_repair_has_exact_structural_authority() -> None:
    repair_id = "availability_fraction_component_denominator_v1"
    metadata = repair_metadata_for(repair_id)

    assert metadata.classification_source == "exact"
    assert metadata.repair_class is RepairClass.STRUCTURAL
    assert automatic_repair_allowed(repair_id)


def _auditor_prompt() -> str:
    context = ResearchContext(
        research_question="q",
        cohort=CohortDescriptor(cohort_name="c", database="synthetic", n_stays=1),
        variables=[],
    )
    auditor = LLMConceptAuditor.__new__(LLMConceptAuditor)
    return auditor._prompt(context=context, script_text="pass\n", step=None)


def test_concept_auditor_distinguishes_display_from_analytical_scaling() -> None:
    prompt = _auditor_prompt()

    assert "local rendering-only normalization" in prompt
    assert "upstream analytical preprocessing" in prompt
    assert "Do not call that display transform a recomputation" in prompt
    assert "transformed values are not reused as a scientific artifact" in prompt


def test_concept_auditor_does_not_invent_generic_n_semantics() -> None:
    prompt = _auditor_prompt()

    assert "Do not assume that a generically named `n` field is the total" in prompt
    assert "n_nonmissing / (n_nonmissing + missing_n)" in prompt
    assert "do not demand that it equal an otherwise undefined `n`" in prompt


def test_replanner_must_not_publish_a_self_contradictory_render_contract() -> None:
    assert "Keep rendering contracts internally consistent" in _REPLANNER_GUIDE
    assert "Do not simultaneously request the display transform" in _REPLANNER_GUIDE


def test_new_shared_rules_remain_case_neutral() -> None:
    prompt = _auditor_prompt()
    contract = prompt.split("Do not assume that a generically named", 1)[1].split(
        "If a generic outcome column", 1
    )[0]
    for forbidden in (
        "canonical9",
        "mimic",
        "miiv",
        "sepsis",
        "phenotype_profiles",
        "cluster_profile",
    ):
        assert forbidden not in contract.lower(), forbidden
