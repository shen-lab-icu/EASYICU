"""Shared synthetic fixtures for provider and resume contracts."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.concept_dict_audit import write_concept_dict_fingerprint
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.authority.run_input import build_environment_identity, build_scientific_identity, seal_run_input_capsule
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, TimeWindow


def _write_capsule_resume_fixture(ra, tmp_path: Path):
    """Create a pre-execution run whose input identity is already sealed."""

    run_id = "run_capsule_identity"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "exposure_a": [0, 1, 0, 1],
            "outcome": [0, 1, 0, 1],
        }
    )
    cohort_path = run_dir / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)
    window = TimeWindow(
        name="first_24h",
        anchor="icu_admission",
        start_hours=0,
        end_hours=24,
        rationale="Prespecified baseline window.",
    )
    run_kwargs = {
        "question": "Is exposure A associated with the outcome?",
        "cohort": cohort,
        "cohort_name": "capsule_cohort",
        "database": "miiv",
        "target_outcome": "outcome",
        "primary_exposure": "exposure_a",
        "inclusion_criteria": ["adult ICU stays"],
        "exclusion_criteria": ["missing outcome"],
        "id_columns": ["stay_id"],
        "outcome_columns": ["outcome"],
        "time_windows": [window],
        "concept_descriptions": {"exposure_a": "Prespecified binary exposure."},
        "notes": "Prespecified context note.",
        "resume_run_id": run_id,
        "stop_after_analysis": True,
    }
    context = build_research_context(
        research_question=run_kwargs["question"],
        cohort=cohort_path,
        cohort_name=run_kwargs["cohort_name"],
        database=run_kwargs["database"],
        target_outcome=run_kwargs["target_outcome"],
        primary_exposure=run_kwargs["primary_exposure"],
        inclusion_criteria=run_kwargs["inclusion_criteria"],
        exclusion_criteria=run_kwargs["exclusion_criteria"],
        id_columns=run_kwargs["id_columns"],
        outcome_columns=run_kwargs["outcome_columns"],
        time_windows=run_kwargs["time_windows"],
        concept_descriptions=run_kwargs["concept_descriptions"],
        notes=run_kwargs["notes"],
    )
    context_path = run_dir / "research_context.json"
    context_path.write_text(context.model_dump_json(indent=2), encoding="utf-8")
    plan = AnalysisPlan(
        research_question=run_kwargs["question"],
        steps=[
            AnalysisStep(
                step_id="01_summary",
                intent="Summarize the prespecified cohort.",
                inputs=["exposure_a", "outcome"],
                expected_outputs=["table:summary"],
                method="descriptive_statistics",
            )
        ],
    )
    plan_path = run_dir / "analysis_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    evidence = EvidenceStore(run_dir)
    fingerprint_path = run_dir / "concept_dict_fingerprint.json"
    fingerprint = write_concept_dict_fingerprint(fingerprint_path)
    evidence.register_file(kind="log", description="Frozen dictionary fingerprint.",
                           source_path=fingerprint_path, evidence_id="concept_dict_fingerprint")
    evidence.register_file(
        kind="log",
        description="Frozen research context.",
        source_path=context_path,
        evidence_id="research_context",
        producer="pipeline",
        generation_mode="system",
    )
    evidence.register_file(
        kind="log",
        description="Frozen analysis plan.",
        source_path=plan_path,
        evidence_id="analysis_plan",
        producer="planner",
        generation_mode="llm",
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=ra.MockLLMClient(),
        enable_literature=False,
        enable_memory=False,
        enable_latex=False,
    )
    identity = build_scientific_identity(
        cohort=cohort,
        question=run_kwargs["question"],
        cohort_name=run_kwargs["cohort_name"],
        database=run_kwargs["database"],
        target_outcome=run_kwargs["target_outcome"],
        primary_exposure=run_kwargs["primary_exposure"],
        cross_database_validation=None,
        inclusion_criteria=run_kwargs["inclusion_criteria"],
        exclusion_criteria=run_kwargs["exclusion_criteria"],
        id_columns=run_kwargs["id_columns"],
        time_columns=None,
        outcome_columns=run_kwargs["outcome_columns"],
        time_windows=run_kwargs["time_windows"],
        concept_descriptions=run_kwargs["concept_descriptions"],
        user_preferences=None,
        notes=run_kwargs["notes"],
        skill_key=None,
        experiment_spec=None,
        source_files=None,
        disable_icu_context=False,
    )
    seal_run_input_capsule(
        run_dir=run_dir,
        evidence=evidence,
        scientific_identity=identity,
        initial_environment=build_environment_identity(llm_signature="mock"),
        context_path=context_path,
        cohort_path=cohort_path,
        experiment_spec_path=None,
    )
    (run_dir / "manifest_partial.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu.research_manifest_partial/1",
                "run_id": run_id,
                "concept_dict_fingerprint": fingerprint.to_dict(),
                "context_path": context_path.name,
                "plan_path": plan_path.name,
                "per_step_records": [],
                "evidence": [
                    record.model_dump(mode="json") for record in evidence.records()
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return pipeline, run_dir, run_kwargs

