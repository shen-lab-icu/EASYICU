"""The keys a v0.1.0 run context carries, with synthetic values.

Taken from v0.1.0's agent_pipeline_runs._write_projection and checked against
the key set of a run written by a v0.1.0-era build (S1480 compatibility).
"""

V010_RUN_CONTEXT = {
    "run_id": "run-v010", "study_id": "study-v010", "scientific_configuration_sha256": "a" * 64,
    "mode": "research_agent_pipeline", "run_type": "full", "engine": "easyicu.research_agent.pipeline",
    "question": "Synthetic question.", "source": {"label": "MIMIC-IV Demo", "database": "miiv", "reader_label": "MIMIC-IV Demo"},
    "summary": {"selected_concepts": ["age"], "materialized_concept_count": 1, "coverage_sufficient": True},
    "local_first": {"uploads": 0}, "plan_revision_source_run_id": "",
}
