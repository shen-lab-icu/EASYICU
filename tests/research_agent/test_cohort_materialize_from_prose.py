"""Execute-phase 真强制: translate the agent's prose 纳排 into typed predicates
and materialise the filtered analysis cohort.

E1 run12 showed the framework's cohort enforcement never engaged: the replanner
grew a ``01_cohort_definition`` step but left ``plan.cohort`` empty, the
materialiser no-op'd, and the primary regression ran on the full universe while
the step re-applied 纳排 in its own code. ``5c9537b`` made that auditable; this
closes the loop — when the executing plan carries a cohort step with an empty
structured cohort, ``run_execute_phase`` extracts the criteria the agent stated
in prose, materialises ``cohort_analysis.parquet``, and re-points the runner so
downstream steps read the filtered cohort.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _is_replan(user: str) -> bool:
    upper = user.upper()
    return "PROBE SUMMARY:" in upper and "CURRENT PLAN:" in upper


def _is_extraction(user: str) -> bool:
    return (
        "COHORT-DEFINITION STEP PROSE" in user and "AVAILABLE PER-STAY COLUMNS" in user
    )


def test_plan_phase_materialized_cohort_is_adopted_without_coder(tmp_path: Path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.cohort.schema import (
        CohortDefinition,
        ConceptPredicate,
        TimeWindow,
        materialize_locked_analysis_cohort,
    )
    from easyicu.research_agent.execution.cohort_adoption import (
        adopt_existing_host_cohort_materialization,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    universe_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"age": [10, 20, 30]}).to_parquet(universe_path, index=False)
    definition = CohortDefinition(
        name="adult",
        inclusion=(
            ConceptPredicate(
                "age",
                TimeWindow("icu_admit", 0, 24),
                "max",
                ">=",
                18,
            ),
        ),
    )
    plan = AnalysisPlan(
        research_question="Test host cohort adoption.",
        cohort=definition,
        steps=[
            AnalysisStep(
                step_id="01_cohort",
                intent="Define and report the locked analysis cohort.",
                expected_outputs=[
                    "artifact:analysis_cohort",
                    "table:cohort_flow",
                ],
            )
        ],
    )
    result = materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=plan,
        universe_path=universe_path,
    )
    evidence = EvidenceStore(tmp_path)
    records: list[dict] = []
    preexecuted: set[str] = set()
    findings = []

    adopt_existing_host_cohort_materialization(
        plan=plan,
        run_dir=tmp_path,
        cohort_path=result["path"],
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="test",
        gate_stamp={
            "deterministic_gate_schema_version": "test",
            "deterministic_gate_engine_code_sha256": "a" * 64,
            "deterministic_gate_fingerprint": "b" * 64,
        },
        per_step_records=records,
        preexecuted_step_ids=preexecuted,
        findings=findings,
    )

    assert findings == []
    assert preexecuted == {"01_cohort"}
    assert len(records) == 1
    assert records[0]["generation_mode"] == "deterministic_cohort_materializer"
    assert records[0]["plan_scientific_signature"]
    assert records[0]["evidence_ids"] == [
        "analysis_cohort_execute_repair",
        "cohort_flow_execute_repair",
    ]
    assert not any(key.startswith("step_provider_call_") for key in records[0])


@pytest.mark.parametrize(
    ("development_sample_size", "cohort_products"),
    [
        (None, ["table:analysis_cohort"]),
        (100, ["table:analysis_cohort"]),
        (None, ["artifact:analysis_cohort", "table:cohort_flow"]),
    ],
)
def test_prose_cohort_is_extracted_materialised_and_enforced(
    ra,
    synthetic_cohort,
    tmp_path: Path,
    development_sample_size: int | None,
    cohort_products: list[str],
):
    from easyicu.research_agent.providers.mocks import (
        MockLLMClient,
        PatternScriptedMockLLMClient,
        _mock_plan_json,
    )
    from easyicu.research_agent.providers.llm import LLMRouter
    from easyicu.research_agent.planning.robustness_contract import RobustnessSpec
    from easyicu.research_agent.reporting.article_contract import (
        augment_plan_for_article_contract,
        build_article_analysis_contract,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    question = "Is admission SOFA-2 associated with ICU mortality?"
    context = ra.build_research_context(
        research_question=question,
        cohort=synthetic_cohort,
        cohort_name="prose_cohort",
        database="synthetic",
        target_outcome="death",
    )
    base_plan = AnalysisPlan.model_validate_json(_mock_plan_json(context))
    complete_plan, _findings = augment_plan_for_article_contract(
        plan=base_plan,
        contract=build_article_analysis_contract(
            context,
            analysis_type=base_plan.analysis_type,
        ),
    )
    planned = complete_plan.model_copy(
        update={
            "robustness_specs": [
                RobustnessSpec(
                    spec_id="complete_case_covariates",
                    axis="missing",
                    description=(
                        "Repeat the association analysis among complete cases."
                    ),
                    missing_override={"strategy": "complete_case"},
                )
            ],
        }
    )
    executing_plan = planned.model_copy(
        update={
            "steps": [
                AnalysisStep(
                    step_id="01_cohort_definition",
                    intent=(
                        "Define the adult ICU analysis cohort: include age >= 18 "
                        "and ICU LoS >= 1 day; report attrition."
                    ),
                    expected_outputs=cohort_products,
                    method="cohort_definition",
                ),
                *planned.steps,
            ]
        }
    )
    extraction = json.dumps(
        {
            "inclusion": [
                {"concept_id": "age", "op": ">=", "value": 18},
                {"concept_id": "los_icu", "op": ">=", "value": 1},
            ],
            "exclusion": [],
        }
    )
    planner = PatternScriptedMockLLMClient(
        [
            ("ICU-AWARE RESEARCH PLAN", [planned.model_dump_json()] * 8),
            ("PROBE SUMMARY:", [executing_plan.model_dump_json()] * 8),
            ("COHORT-DEFINITION STEP PROSE", [extraction] * 2),
        ],
        default=planned.model_dump_json(),
    )

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=LLMRouter(default=MockLLMClient(), planner=planner),
        runner_kind="subprocess",
        runner_kwargs={"allow_unsafe_host_fallback": True},
        development_sample_size=development_sample_size,
    )
    result = pipeline.run(
        question=question,
        cohort=synthetic_cohort,
        cohort_name="prose_cohort",
        database="synthetic",
        target_outcome="death",
    )

    run_dir = Path(result.workdir)

    # The filtered analysis cohort was materialised and is smaller than the
    # 800-row universe (age >= 18 keeps all, but los_icu >= 1 drops some).
    analysis_cohort = run_dir / "cohort_analysis.parquet"
    assert analysis_cohort.exists(), "cohort_analysis.parquet was not materialised"
    n_cohort = len(pd.read_parquet(analysis_cohort))
    assert 0 < n_cohort < 800, n_cohort
    if development_sample_size is not None:
        sampled = run_dir / "cohort_analysis_development_sample.parquet"
        assert sampled.exists()
        assert len(pd.read_parquet(sampled)) == development_sample_size
        sample_manifest = json.loads(
            (run_dir / "development_execution_sample.json").read_text(encoding="utf-8")
        )
        assert sample_manifest["paper_authority"] is False
        assert sample_manifest["parent"]["rows"] == n_cohort
        assert sample_manifest["sample"]["rows"] == development_sample_size

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    findings = manifest["findings"]

    materialised = [
        f
        for f in findings
        if f.get("validator") == "cohort_materializer"
        and (f.get("detail") or {}).get("stage") == "execute_repair"
    ]
    assert materialised, "no execute_repair cohort_materializer finding"
    assert materialised[0]["detail"]["n_analysis_cohort"] == n_cohort

    # 真强制, not just auditable: the contract error must NOT fire once the
    # cohort is materialised.
    contract_errors = [
        f
        for f in findings
        if f.get("validator") == "cohort_contract" and f.get("severity") == "error"
    ]
    assert not contract_errors, "cohort_contract error fired despite materialisation"

    # The locked cohort on disk now reflects the real definition, not the empty
    # placeholder.
    locked = json.loads((run_dir / "cohort_locked.json").read_text(encoding="utf-8"))
    assert locked["cohort"]["inclusion"], "cohort_locked.json still has empty 纳排"

    core_step_ids = {
        "01_cohort_definition",
        "01_table_one",
        "02_outcome_incidence",
        "03_missingness_audit",
        "03_missingness_audit_figure",
        "04_primary_association",
        "04_primary_association_figure",
    }
    core_statuses = {
        record.get("step_id"): record.get("status")
        for record in manifest.get("per_step_records", [])
        if record.get("step_id") in core_step_ids
    }
    assert core_step_ids <= set(core_statuses), core_statuses
    assert set(core_statuses.values()) == {"ok"}, core_statuses
    if "table:cohort_flow" in cohort_products:
        flow = pd.read_csv(run_dir / "cohort_analysis_flow.csv")
        assert int(flow.iloc[0]["n_before"]) == 800
        assert int(flow.iloc[-1]["n_remaining"]) == n_cohort
        cohort_step = next(
            record
            for record in manifest["per_step_records"]
            if record["step_id"] == "01_cohort_definition"
        )
        assert cohort_step["generation_mode"] == "deterministic_cohort_materializer"
        assert cohort_step["evidence_ids"] == [
            "analysis_cohort_execute_repair",
            "cohort_flow_execute_repair",
        ]
        assert cohort_step["step_provider_call_categories"] == [
            "cohort_definition_translation"
        ]


def test_failed_prose_materialization_does_not_mutate_executing_plan():
    from easyicu.research_agent.execution.cohort_adoption import (
        commit_staged_cohort_plan,
        stage_candidate_cohort_plan,
    )
    from easyicu.research_agent.planning.cohort_contract import (
        CohortDefinition,
        ConceptPredicate,
        TimeWindow,
    )
    from easyicu.research_agent.schema import AnalysisPlan

    live = AnalysisPlan(research_question="q", steps=[])
    definition = CohortDefinition(
        name="adult",
        inclusion=[
            ConceptPredicate(
                concept_id="age",
                time_window=TimeWindow("icu_admit", 0, 24),
                aggregation="max",
                op=">=",
                value=18,
            )
        ],
    )
    staged = stage_candidate_cohort_plan(live, definition)

    class AuthorityState:
        def __init__(self):
            self.rebound: list[AnalysisPlan] = []

        def rebind_cohort(self, *, plan, context):
            del context
            self.rebound.append(plan)

    authority_state = AuthorityState()
    committed = commit_staged_cohort_plan(
        live,
        staged,
        materialization_status="error",
        authority_state=authority_state,
        context=None,
    )

    assert committed is False
    assert live.cohort is None
    assert staged.cohort.inclusion
    assert authority_state.rebound == []

    class RejectingAuthority:
        def rebind_cohort(self, *, plan, context):
            del plan, context
            raise RuntimeError("authority mismatch")

    with pytest.raises(RuntimeError, match="authority mismatch"):
        commit_staged_cohort_plan(
            live,
            staged,
            materialization_status="applied",
            authority_state=RejectingAuthority(),
            context=None,
        )
    assert live.cohort is None
