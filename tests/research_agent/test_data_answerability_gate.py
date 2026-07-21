from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.source_status import (
    SourceStatusContract,
    source_status_contract_digest,
)
from easyicu.research_agent.gates.data_answerability import (
    analysis_answerability_findings,
    primary_exposure_answerability_findings,
    target_outcome_answerability_findings,
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
    outcome_domain: dict[str, object] | None = None,
    outcome_missing_n: int = 0,
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
            ConceptDescriptor(
                name="outcome_y",
                role="outcome",
                dtype="int64",
                observed_domain=outcome_domain,
                missingness=(
                    MissingnessProfile(
                        fraction_missing=outcome_missing_n / 100,
                        n_missing=outcome_missing_n,
                        n_total=100,
                    )
                    if outcome_domain is not None
                    else None
                ),
            ),
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


def test_single_observed_outcome_level_blocks_before_planner() -> None:
    context = _context(
        domain={
            "n_unique": 2,
            "is_constant": False,
            "is_binary": True,
            "min": 0.0,
            "max": 1.0,
        },
        missing_n=0,
        outcome_domain={
            "n_unique": 1,
            "is_constant": True,
            "is_binary": True,
            "min": 0.0,
            "max": 0.0,
        },
    )

    findings = target_outcome_answerability_findings(context)

    assert len(findings) == 1
    assert findings[0].detail["kind"] == (
        "scientifically_infeasible_no_outcome_contrast"
    )
    blueprint = HypothesisBlueprintAgent().run(
        context=context,
        literature=LiteratureBundle(
            research_question=context.research_question,
            citations=[],
        ),
    )
    assert blueprint.feasibility_status == "blocked"


def test_missing_outcomes_cannot_be_treated_as_the_absent_event_level() -> None:
    context = _context(
        domain={
            "n_unique": 2,
            "is_constant": False,
            "is_binary": True,
            "min": 0.0,
            "max": 1.0,
        },
        missing_n=0,
        outcome_domain={
            "n_unique": 1,
            "is_constant": True,
            "is_binary": True,
            "min": 1.0,
            "max": 1.0,
        },
        outcome_missing_n=40,
    )

    findings = analysis_answerability_findings(context)

    assert len(findings) == 1
    assert findings[0].detail["missing_n"] == 40
    assert findings[0].detail["required_action"] == (
        "revise_question_cohort_or_outcome"
    )


def test_two_observed_outcome_levels_remain_answerable() -> None:
    context = _context(
        domain={
            "n_unique": 2,
            "is_constant": False,
            "is_binary": True,
            "min": 0.0,
            "max": 1.0,
        },
        missing_n=0,
        outcome_domain={
            "n_unique": 2,
            "is_constant": False,
            "is_binary": True,
            "min": 0.0,
            "max": 1.0,
        },
    )

    assert analysis_answerability_findings(context) == []


def test_constant_outcome_aborts_pipeline_before_any_provider_call(
    ra,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort = tmp_path / "constant_outcome.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "exposure_x": [0, 1, 0, 1],
            "outcome_y": [0, 0, 0, 0],
        }
    ).to_parquet(cohort, index=False)
    llm = ra.MockLLMClient()
    provider_calls = 0

    def forbidden_provider_call(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("answerability failure must precede the provider")

    monkeypatch.setattr(llm, "complete", forbidden_provider_call)
    skill = ra.ClinicalSkill(
        key="constant_outcome_answerability",
        name="Constant outcome answerability",
        description="Verify pre-Planner outcome feasibility.",
        research_question_template="Is exposure_x associated with outcome_y?",
        target_outcome="outcome_y",
        primary_predictor="exposure_x",
        expected_variables=["exposure_x", "outcome_y"],
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "work",
        llm=llm,
        enable_literature=False,
    )

    result = pipeline.run(
        cohort=cohort,
        cohort_name="constant_outcome",
        database="synthetic",
        primary_exposure="exposure_x",
        skill=skill,
    )

    assert provider_calls == 0
    assert result.plan_path == ""
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["notes"] == "aborted: data_answerability_failed"


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
