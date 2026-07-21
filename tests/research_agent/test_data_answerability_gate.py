from __future__ import annotations

import pytest

from easyicu.research_agent.authority.source_status import (
    SourceStatusContract,
    source_status_contract_digest,
)
from easyicu.research_agent.gates.data_answerability import (
    primary_exposure_answerability_findings,
)
from easyicu.research_agent.literature import (
    HypothesisBlueprintAgent,
    LiteratureBundle,
)
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    MissingnessProfile,
    ResearchContext,
)


def _context(
    *,
    domain: dict[str, object],
    missing_n: int,
    missingness_semantics: str | None = None,
    source_status_contract: dict[str, object] | None = None,
) -> ResearchContext:
    provenance: dict[str, object] = {}
    if source_status_contract is not None:
        provenance["source_status_contracts"] = {"exposure_x": source_status_contract}
    return ResearchContext(
        research_question="Estimate the association of exposure with outcome.",
        cohort=CohortDescriptor(
            cohort_name="answerability",
            database="synthetic",
            n_patients=100,
            n_stays=100,
            provenance=provenance,
        ),
        variables=[
            ConceptDescriptor(
                name="exposure_x",
                role="intervention",
                dtype="float64",
                observed_domain=domain,
                missingness_semantics=missingness_semantics,
                missingness=MissingnessProfile(
                    fraction_missing=missing_n / 100,
                    n_missing=missing_n,
                    n_total=100,
                ),
            ),
            ConceptDescriptor(name="outcome_y", role="outcome", dtype="int64"),
        ],
        primary_exposure="exposure_x",
        target_outcome="outcome_y",
    )


def test_single_observed_level_with_unknown_missing_semantics_blocks_before_planner():
    context = _context(
        domain={
            "n_unique": 1,
            "is_constant": True,
            "is_binary": True,
            "min": 1.0,
            "max": 1.0,
        },
        missing_n=84,
    )

    findings = primary_exposure_answerability_findings(context)

    assert len(findings) == 1
    assert (
        findings[0].detail["kind"] == "scientifically_infeasible_requires_data_contract"
    )
    assert findings[0].detail["required_action"] == (
        "supply_host_owned_source_absence_contract"
    )

    blueprint = HypothesisBlueprintAgent().run(
        context=context,
        literature=LiteratureBundle(
            research_question=context.research_question, citations=[]
        ),
    )
    assert blueprint.feasibility_status == "blocked"
    assert any(
        "only one observed level" in note for note in blueprint.domain_gate_notes
    )


def test_two_observed_levels_remain_answerable():
    context = _context(
        domain={
            "n_unique": 2,
            "is_constant": False,
            "is_binary": True,
            "min": 0.0,
            "max": 1.0,
        },
        missing_n=10,
    )

    assert primary_exposure_answerability_findings(context) == []


def test_free_text_missingness_semantics_cannot_authorize_absence():
    context = _context(
        domain={
            "n_unique": 1,
            "is_constant": True,
            "is_binary": True,
            "min": 1.0,
            "max": 1.0,
        },
        missing_n=84,
        missingness_semantics=(
            "Missing means verified event absence under complete source coverage."
        ),
    )

    findings = primary_exposure_answerability_findings(context)

    assert len(findings) == 1
    assert findings[0].detail["kind"] == (
        "scientifically_infeasible_requires_data_contract"
    )
    assert findings[0].detail["missingness_semantics_present"] is True


def _verified_absence_contract(**overrides: object) -> dict[str, object]:
    contract: dict[str, object] = {
        "schema_version": "easyicu.source_status_contract/1",
        "variable": "exposure_x",
        "n_total": 100,
        "counts": {
            "observed": 16,
            "verified_absent": 84,
            "unmeasured": 0,
            "source_missing": 0,
            "contradictory": 0,
        },
        "source_coverage": "complete",
        "verified_absent_value": 0,
        "authority_kind": "event_reconciliation",
        "authority_evidence_sha256": "a" * 64,
        "source_columns": ["exposure_x_n", "exposure_x_measured", "exposure_x"],
        "row_status_artifact_sha256": "b" * 64,
        "row_status_column": "exposure_x_source_status",
        "row_identity_sha256": "c" * 64,
    }
    contract.update(overrides)
    return contract


def test_verified_absence_contract_still_requires_host_materialization():
    context = _context(
        domain={
            "n_unique": 1,
            "is_constant": True,
            "is_binary": True,
            "min": 1.0,
            "max": 1.0,
        },
        missing_n=84,
        source_status_contract=_verified_absence_contract(),
    )

    findings = primary_exposure_answerability_findings(context)

    assert len(findings) == 1
    assert findings[0].detail["kind"] == "source_status_contract_not_materialized"
    assert findings[0].detail["verified_absent_n"] == 84
    assert findings[0].detail["required_action"] == (
        "host_materialize_verified_absence_into_bound_exposure"
    )


def test_source_status_contract_must_reconcile_with_locked_context():
    raw = _verified_absence_contract(
        counts={
            "observed": 15,
            "verified_absent": 85,
            "unmeasured": 0,
            "source_missing": 0,
            "contradictory": 0,
        }
    )
    context = _context(
        domain={"n_unique": 1, "is_constant": True, "min": 1.0, "max": 1.0},
        missing_n=84,
        source_status_contract=raw,
    )

    findings = primary_exposure_answerability_findings(context)

    assert len(findings) == 1
    assert findings[0].detail["kind"] == "source_status_contract_binding_mismatch"
    assert findings[0].detail["issues"] == [
        "observed count disagrees with variable nonmissing_n",
        "non-observed source-status counts disagree with missing_n",
    ]


def test_nonobserved_source_states_require_row_level_authority():
    raw = _verified_absence_contract()
    raw.pop("row_status_artifact_sha256")

    with pytest.raises(ValueError, match="non-observed rows require"):
        SourceStatusContract.model_validate(raw)


def test_source_status_contract_digest_is_canonical():
    first = SourceStatusContract.model_validate(_verified_absence_contract())
    reordered = SourceStatusContract.model_validate(
        dict(reversed(list(_verified_absence_contract().items())))
    )

    assert source_status_contract_digest(first) == source_status_contract_digest(
        reordered
    )


def test_complete_constant_exposure_is_scientifically_infeasible():
    context = _context(
        domain={"n_unique": 1, "is_constant": True, "min": 0.0, "max": 0.0},
        missing_n=0,
    )

    findings = primary_exposure_answerability_findings(context)

    assert len(findings) == 1
    assert (
        findings[0].detail["kind"] == "scientifically_infeasible_no_exposure_contrast"
    )
