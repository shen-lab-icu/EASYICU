from __future__ import annotations

import hashlib
import json

import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.authority.manuscript_method_facts import (
    MethodFactAuthorityError,
    load_manuscript_method_facts,
)
from easyicu.research_agent.reporting.manuscript_method_facts import (
    missing_bound_method_facts,
    place_manuscript_method_facts,
)
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.schema import ResearchContext


def _context():
    return ResearchContext.model_validate(
        {
            "research_question": "Describe an early clinical status and hospital outcome.",
            "cohort": {"cohort_name": "test", "database": "miiv", "n_stays": 17},
            "primary_exposure": "status",
            "target_outcome": "death",
            "variables": [
                {
                    "name": "status",
                    "dtype": "int64",
                    "description": "A score increase of at least 2 points",
                    "analysis_window": "icu_admission[0,24]h",
                    "analysis_window_role": "outer_observation_window",
                    "clinical_definition": {
                        "contract_id": "example",
                        "definition": "Example",
                        "version": "1",
                        "source_id": "example-source",
                        "status": "source_bound_golden",
                        "validation_status": "independent_clinical_review_pending",
                        "canonical_definition": True,
                        "definition_time_anchor": "event_onset",
                        "database_conformance": {"miiv": "mapping_only"},
                    },
                },
                {
                    "name": "death",
                    "dtype": "bool",
                    "description": "In-hospital mortality",
                },
            ],
        }
    )


def _source(tmp_path, *, context=None, producer="pipeline", generation_mode="system"):
    source = tmp_path / "input.json"
    context = context if context is not None else _context()
    source.write_text(context.model_dump_json(), encoding="utf-8")
    store = EvidenceStore(tmp_path, enforcement_mode=EvidenceEnforcementMode.STRICT)
    record = store.register_file(
        kind="log",
        source_path=source,
        description="Typed source context",
        evidence_id="research_context",
        producer=producer,
        generation_mode=generation_mode,
    )
    return store, record


def test_exact_method_facts_preserve_source_and_separate_time_and_validation(tmp_path):
    store, record = _source(tmp_path)
    original = (tmp_path / record.relative_path).read_bytes()
    facts = store.manuscript_method_facts()
    assert len(facts) == 6
    assert "at least 2 points" in facts[0].scaffold
    assert "outer observation window" in facts[1].scaffold
    assert "0 to 24 hours relative to ICU admission" in facts[1].scaffold
    assert "event onset" not in facts[1].scaffold
    assert "clinical definition time anchor" in facts[2].scaffold
    assert "event onset" in facts[2].scaffold
    assert "independent clinical review pending" in facts[3].scaffold
    assert "mapping only" in facts[4].scaffold
    assert "In-hospital mortality" in facts[5].scaffold
    assert all(f.source_sha256 == hashlib.sha256(original).hexdigest() for f in facts)
    assert (tmp_path / record.relative_path).read_bytes() == original


def test_exact_registered_definition_passes_strict_not_a_free_result_assertion(
    tmp_path,
):
    store, _ = _source(tmp_path)
    facts = store.manuscript_method_facts()
    scaffold = "## Methods\n\n### Variables\n\n" + facts[0].scaffold
    safe, removed = store.enforce_evidence_bound_scaffold(scaffold)
    assert not removed
    assert facts[0].scaffold in safe
    bound = store.bind_manuscript(safe, per_step_records=[])
    assert "evidence/research_context__input.json" in bound
    _, _, untraced = bind_numeric_values(bound, evidence=store, per_step_records=[])
    assert not untraced
    with pytest.raises(EvidenceEnforcementError):
        store.enforce_evidence_bound_scaffold("## Results\n\n" + facts[0].scaffold)
    with pytest.raises(EvidenceEnforcementError):
        store.enforce_evidence_bound_scaffold(scaffold.replace("2 points", "3 points"))
    with pytest.raises(EvidenceEnforcementError):
        store.enforce_evidence_bound_scaffold(
            scaffold + " The exposure reduced mortality."
        )
    with pytest.raises(EvidenceEnforcementError):
        store.enforce_evidence_bound_scaffold(
            scaffold.replace("research_context}", "foreign}")
        )


def test_report_only_projection_restores_same_source_facts_without_mutation(tmp_path, monkeypatch):
    from easyicu.research_agent.reporting.writer_only_migration import _claim_policy_projection

    store, _ = _source(tmp_path)
    facts = store.manuscript_method_facts()
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(EvidenceStore, "__init__", lambda *a, **kw: pytest.fail("mutable store"))
    projected, errors = _claim_policy_projection(tmp_path, "## Methods\n\n### Variables\n")
    assert not errors
    assert all(fact.scaffold in projected for fact in facts)
    altered = projected.replace("2 points", "3 points")
    repaired, errors = _claim_policy_projection(tmp_path, altered)
    assert "methods" in errors
    assert "3 points" not in repaired
    assert facts[0].scaffold in repaired
    assert before == {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}


def test_exact_method_metadata_does_not_disable_numeric_binding(tmp_path):
    payload = _context().model_dump(mode="json")
    payload["variables"][0]["description"] = "A score increase of at least 300 points"
    store, _ = _source(tmp_path, context=ResearchContext.model_validate(payload))
    text = (
        "## Methods\n\n### Variables\n\n" + store.manuscript_method_facts()[0].scaffold
    )
    safe, _ = store.enforce_evidence_bound_scaffold(text)
    with pytest.raises(EvidenceEnforcementError, match="numeric value"):
        bind_numeric_values(
            store.bind_manuscript(safe), evidence=store, per_step_records=[]
        )


@pytest.mark.parametrize(
    "formatting",
    [
        lambda text: text,
        lambda text: "**" + text + "**",
        lambda text: "- " + text,
        lambda text: text.lower(),
    ],
)
def test_validation_status_cannot_be_forged_even_without_comparative_or_numeric_words(
    tmp_path, formatting
):
    store, _ = _source(tmp_path)
    status = store.manuscript_method_facts()[3].scaffold
    false_status = status.replace(
        "independent clinical review pending", "independent clinical review complete"
    )
    with pytest.raises(EvidenceEnforcementError):
        store.enforce_evidence_bound_scaffold(
            "## Methods\n\n### Variables\n\n" + formatting(false_status)
        )


@pytest.mark.parametrize(
    "field,value", [("producer", "coder"), ("generation_mode", "llm")]
)
def test_foreign_context_owner_cannot_issue_method_facts(tmp_path, field, value):
    store, _ = _source(tmp_path, **{field: value})
    with pytest.raises(MethodFactAuthorityError, match="context owner"):
        store.manuscript_method_facts()


@pytest.mark.parametrize("change", ["tamper", "delete", "symlink"])
def test_method_context_source_must_be_current_and_immutable(tmp_path, change):
    store, record = _source(tmp_path)
    path = tmp_path / record.relative_path
    if change == "tamper":
        path.write_text("{}", encoding="utf-8")
    else:
        path.unlink()
        if change == "symlink":
            path.symlink_to(tmp_path / "input.json")
    with pytest.raises(MethodFactAuthorityError, match="missing or has drifted"):
        load_manuscript_method_facts(root=tmp_path, records=[record])


def test_no_context_or_untyped_legacy_context_grants_no_authority(tmp_path):
    assert not load_manuscript_method_facts(root=tmp_path, records=[])
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps({"primary_exposure": "made-up"}), encoding="utf-8")
    store = EvidenceStore(tmp_path)
    store.register_file(
        kind="log",
        source_path=source,
        description="legacy",
        evidence_id="research_context",
        producer="pipeline",
        generation_mode="system",
    )
    assert not store.manuscript_method_facts()


def test_placement_is_idempotent_scoped_and_does_not_rewrite_prose(tmp_path):
    store, _ = _source(tmp_path)
    facts = store.manuscript_method_facts()
    original = "## Methods\n\n### Variables\n\nAge was recorded.\n\n### Statistical analysis\n\nNo inference.\n\n## Results\n\nPreserved results.\n"
    actual, fields = place_manuscript_method_facts(original, facts)
    assert len(fields) == len(facts)
    assert "\n\nAge was recorded." in actual
    assert (
        actual.split("### Statistical analysis")[1]
        == original.split("### Statistical analysis")[1]
    )
    assert place_manuscript_method_facts(actual, facts) == (actual, ())
    no_variables = original.replace("### Variables", "### Other")
    assert place_manuscript_method_facts(no_variables, facts) == (no_variables, ())


def test_writer_authority_syntax_inside_metadata_is_not_executable(tmp_path):
    payload = _context().model_dump(mode="json")
    payload["variables"][0]["description"] = "Status {claim:fake.result}"
    store, _ = _source(tmp_path, context=ResearchContext.model_validate(payload))
    with pytest.raises(MethodFactAuthorityError, match="unsupported markup"):
        store.manuscript_method_facts()


def test_current_context_is_loaded_not_mutable_unregistered_root_file(tmp_path):
    store, _ = _source(tmp_path)
    (tmp_path / "research_context.json").write_text("{}", encoding="utf-8")
    assert len(store.manuscript_method_facts([])) == 6


def test_post_filter_source_fact_coverage_is_not_replaced_by_a_generic_paragraph(
    tmp_path,
):
    from easyicu.research_agent.reporting.manuscript_post import (
        drop_untraceable_numeric_sentences,
    )

    payload = _context().model_dump(mode="json")
    payload["variables"][0]["description"] = "An increase of 300 points"
    store, _ = _source(tmp_path, context=ResearchContext.model_validate(payload))
    facts = store.manuscript_method_facts([])
    text, _ = place_manuscript_method_facts(
        "## Methods\n\n### Variables\n\nAge was obtained from the source.\n",
        facts,
    )
    bound = store.bind_manuscript(text, per_step_records=[])
    assert not missing_bound_method_facts(bound, facts, store.bind_manuscript)
    cleaned, removed = drop_untraceable_numeric_sentences(
        bound, evidence=store, per_step_records=[]
    )
    assert removed
    assert missing_bound_method_facts(cleaned, facts, store.bind_manuscript) == (
        "variables[0].description",
    )


def test_numeric_deletion_records_dependent_context_without_dropping_the_next_paragraph(
    tmp_path,
):
    from easyicu.research_agent.reporting.manuscript_post import (
        drop_untraceable_numeric_sentences,
    )

    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    text = "## Methods\n\n### Variables\n\nThe value was 333. It represented a score.\n\nAge was recorded."
    cleaned, removed = drop_untraceable_numeric_sentences(
        text, evidence=store, per_step_records=[]
    )
    assert "333" not in cleaned
    assert "It represented" not in cleaned
    assert "Age was recorded." in cleaned
    assert removed[0]["dependent_context_drops"] == ["It represented a score."]


def test_write_phase_owner_projects_then_audits_the_same_current_source(tmp_path):
    from easyicu.research_agent.reporting.manuscript_method_facts import (
        project_source_method_facts,
        audit_bound_source_method_facts,
    )
    from easyicu.research_agent.reporting.readiness import _MANUSCRIPT_ERROR_VALIDATORS

    store, record = _source(tmp_path)
    scaffold, finding = project_source_method_facts(
        "## Methods\n\n### Variables\n\nAge was recorded.\n",
        evidence=store,
        per_step_records=[],
    )
    assert finding.detail["source_sha256"] == record.sha256
    assert finding.severity == "info"
    safe, _ = store.enforce_evidence_bound_scaffold(scaffold)
    bound = store.bind_manuscript(safe, per_step_records=[])
    assert (
        audit_bound_source_method_facts(bound, evidence=store, per_step_records=[])
        is None
    )
    broken = bound.replace("In-hospital mortality", "ICU mortality")
    failure = audit_bound_source_method_facts(
        broken, evidence=store, per_step_records=[]
    )
    assert failure.severity == "error"
    assert failure.validator in _MANUSCRIPT_ERROR_VALIDATORS
    assert failure.detail["source_fields"] == ["variables[1].description"]
