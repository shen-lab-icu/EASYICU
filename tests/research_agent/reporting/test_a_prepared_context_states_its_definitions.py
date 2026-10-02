"""A context prepared from a materialized extract states its definitions.

The Methods owner quotes the recorded definitions and windows of the selected
exposure and outcome from the frozen research context.  It accepted only the
version-1 schema and treated anything else as a legacy untyped artifact.  A
context prepared from a materialized extract (``easyicu.research_context/3``,
every real-data run) therefore never yielded a definition: Methods kept no
host-quoted exposure window or outcome definition, and the Writer's own
numeric sentences for them were deleted.  Every typed version now yields the
facts; an unknown version still yields none, and a typed context that does not
validate still fails closed.

A synthetic two-stay export; no study's values.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceEnforcementMode, EvidenceStore
from easyicu.research_agent.authority.manuscript_method_facts import MethodFactAuthorityError
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.research_context.typed import RESEARCH_CONTEXT_V3_SCHEMA_VERSION
from tests.support.typed_trajectory import TRAJECTORY_QUESTION, typed_trajectory_bundle


def _prepared_run(tmp_path, edit=None):
    paths, _cohort, _trajectory = typed_trajectory_bundle(tmp_path)
    context = build_research_context(
        research_question=TRAJECTORY_QUESTION,
        cohort=paths["parquet"],
        cohort_name="typed_capsule",
        database="miiv",
        target_outcome="death",
        primary_exposure="lact_max",
        id_columns=("stay_id",),
        outcome_columns=("death",),
    )
    payload = json.loads(context.model_dump_json())
    if edit is not None:
        edit(payload)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    context_path = run_dir / "research_context.json"
    context_path.write_text(json.dumps(payload), encoding="utf-8")
    store = EvidenceStore(run_dir, enforcement_mode=EvidenceEnforcementMode.STRICT)
    store.register_file(
        kind="log", description="Frozen research context.", source_path=context_path,
        evidence_id="research_context", producer="pipeline", generation_mode="system",
    )
    return store, payload


def test_a_prepared_context_quotes_its_exposure_and_outcome(tmp_path):
    store, payload = _prepared_run(tmp_path)
    assert payload["schema_version"] == RESEARCH_CONTEXT_V3_SCHEMA_VERSION

    facts = store.manuscript_method_facts()

    roles = {fact.text.split(":", 1)[0] for fact in facts}
    assert "Recorded source definition for the selected exposure" in roles
    assert "Recorded source definition for the primary outcome" in roles
    assert all(fact.evidence_id == "research_context" for fact in facts)

    scaffold = "## Methods\n\n### Variables\n\n" + "\n\n".join(fact.scaffold for fact in facts)
    safe, removed = store.enforce_evidence_bound_scaffold(scaffold, per_step_records=[])
    assert not removed
    bound = store.bind_manuscript(safe, per_step_records=[])
    _, _, untraced = bind_numeric_values(bound, evidence=store, per_step_records=[])
    assert not untraced


def test_an_unknown_context_version_states_nothing(tmp_path):
    store, _ = _prepared_run(tmp_path, edit=lambda payload: payload.update(schema_version="easyicu.research_context/9"))

    assert store.manuscript_method_facts() == ()


def test_a_typed_context_that_does_not_validate_fails_closed(tmp_path):
    store, _ = _prepared_run(tmp_path, edit=lambda payload: payload.pop("materialized_inputs"))

    with pytest.raises(MethodFactAuthorityError):
        store.manuscript_method_facts()
