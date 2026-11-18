"""Final binding failures feed only affected sections back through the gates."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.agents.reporting import WriterAgent
from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementError, EvidenceEnforcementMode, EvidenceStore, sha256_of_file,
)
from easyicu.research_agent.authority.manuscript_claim_policy import filter_evidence_bound_scaffold
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.reporting.administrative_authority import render_manuscript_administrative_sections
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.reporting.manuscript_quality import audit_manuscript_quality
from easyicu.research_agent.reporting.manuscript_sections import completed_section_repair_candidate
from easyicu.research_agent.reporting import write_phase
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, CritiqueReport, ResearchContext, ValidationFinding

from .test_manuscript_definition_numerics import _grouping, _register_context
from .test_manuscript_quality import _valid_manuscript


@pytest.mark.parametrize("still_broken", [False, True])
def test_final_quality_failure_revalidates_within_the_bound_without_analysis(
    tmp_path, monkeypatch, still_broken,
):
    good = _valid_manuscript()
    bad = good.replace("## Results", "## Missing results", 1)
    drafts, bindings, publications = [], [], []
    evidence = EvidenceStore(tmp_path)
    plan = AnalysisPlan(research_question="ICU cohort description", analysis_type="association_study", steps=[])
    for name in ("_activate_publication_figure", "_activate_publication_inputs", "_write_reproducibility_artifacts"):
        monkeypatch.setattr(write_phase, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(write_phase, "execution_gate_status", lambda **kwargs: {"execution_complete": True})

    def draft(*args, **kwargs):
        drafts.append(kwargs.get("section_repair"))
        return write_phase._DraftStageResult((), (), None, None, good)

    def bind(*args, **kwargs):
        bindings.append(kwargs["scaffold"])
        text = bad if len(bindings) == 1 or still_broken else good
        return write_phase._BindingStageResult(
            text, tmp_path / "bound.md",
            CritiqueReport(status="blocked" if text == bad else "pass", reviewer="test"),
        )

    monkeypatch.setattr(write_phase, "_draft_manuscript", draft)
    monkeypatch.setattr(write_phase, "_bind_and_review_manuscript", bind)
    monkeypatch.setattr(write_phase, "_publish_and_audit_manuscript", lambda *args, **kwargs: publications.append(kwargs["bound"]))
    result = write_phase.run_write_phase(
        SimpleNamespace(_user_extension_activation=SimpleNamespace(receipt={"activation_sha256": "a" * 64, "skills": [], "mcp_servers": []})),
        plan_result=SimpleNamespace(
            context=None, agent_context=None, evidence=evidence, findings=[],
            role_resolver=lambda _: object(), prompt_version="test", resume_state=None,
            repro_envelope=None,
        ),
        execute_result=SimpleNamespace(plan=plan, runtime_state=object(), per_step_records=[]),
        run_dir=tmp_path, run_id="run", stop_after_analysis=False,
        manuscript_title=None, manuscript_authors=None, run_language="en",
        emit_progress=lambda *args, **kwargs: None,
    )
    assert len(drafts) == len(bindings) == (4 if still_broken else 2)
    assert drafts[0] is None
    assert drafts[1][0] == good  # Repair source remains raw, never rendered citations.
    assert "results" in drafts[1][1]
    assert publications == [bad if still_broken else good]
    assert result.manuscript_critique.status == ("blocked" if still_broken else "pass")


def test_repaired_quality_artifacts_preserve_old_versions_and_bind_new_source(tmp_path):
    store = EvidenceStore(tmp_path)
    first = _valid_manuscript().replace("## Results", "## Missing results", 1)
    findings = []
    write_phase._persist_manuscript_quality_artifacts(
        bound=first, bound_evidence_id="bound_old", run_dir=tmp_path,
        evidence=store, findings=findings,
    )
    old = {r.evidence_id: (r.relative_path, r.sha256) for r in store.records()}
    errors = write_phase._persist_manuscript_quality_artifacts(
        bound=_valid_manuscript(), bound_evidence_id="bound_new", run_dir=tmp_path,
        evidence=store, findings=findings,
    )
    assert not errors
    for path, digest in old.values():
        assert sha256_of_file(Path(tmp_path) / path) == digest
    latest = [r for r in store.records() if r.metadata.get("source_evidence_id") == "bound_new"]
    assert len(latest) == 1
    assert latest[0].evidence_id not in old


def test_drafting_reviewer_sees_current_errors_and_preserves_recovered_history(tmp_path):
    import json

    evidence = EvidenceStore(tmp_path)
    plan = AnalysisPlan(research_question="Describe the cohort", steps=[
        AnalysisStep(step_id="model", intent="Summarise the cohort", expected_outputs=["table:result"]),
    ])
    findings = [
        ValidationFinding(validator="recovered_runtime", severity="error", message="Recovered old runtime error",
                          detail={"step_id": "model", "attempt_id": "old"}),
        ValidationFinding(validator="current_runtime", severity="error", message="Unresolved current runtime error",
                          detail={"step_id": "model", "attempt_id": "current"}),
        ValidationFinding(validator="reviewer_round", severity="error", message="Prior derived reject"),
    ]
    records = [
        {"step_id": "model", "attempt_id": "old", "status": "error"},
        {"step_id": "model", "attempt_id": "current", "status": "ok"},
    ]
    write_phase._run_drafting_reviewer_round(
        SimpleNamespace(), plan=plan, per_step_records=records,
        evidence=evidence, findings=findings, bound=_valid_manuscript(),
        repro_envelope=None, run_dir=tmp_path, context=None,
    )
    review = json.dumps(json.loads((tmp_path / "reviewer_report.json").read_text()))
    assert "current_runtime" in review
    assert "recovered_runtime" not in review
    assert "reviewer_round" not in review
    assert len([f for f in findings if f.validator.endswith("_runtime")]) == 2
    active, historical, _ = write_phase.current_validation_findings(
        plan=plan, per_step_records=records, findings=findings, evidence=evidence,
        run_dir=tmp_path, manuscript_text=_valid_manuscript(), context=None,
    )
    assert any(f.message == "Prior derived reject" for f in historical)
    assert any(f.validator == "current_runtime" for f in active)

    # Another manuscript version must link its own report, retaining version 1.
    first_json = [r for r in evidence.records() if r.evidence_id.startswith("reviewer_report_json")][0]
    write_phase._run_drafting_reviewer_round(
        SimpleNamespace(), plan=plan, per_step_records=records,
        evidence=evidence, findings=findings, bound=_valid_manuscript() + "\n",
        repro_envelope=None, run_dir=tmp_path, context=None,
    )
    latest_id = findings[-1].evidence_ids[-1]
    assert latest_id != first_json.evidence_id
    assert evidence.get(latest_id).sha256 == sha256_of_file(tmp_path / "reviewer_report.json")
    assert sha256_of_file(tmp_path / first_json.relative_path) == first_json.sha256


GOOD = """# Potassium and renal readmission in a retrospective ICU cohort

**Keywords:** potassium, kidney, readmission, cohort, prediction

## Abstract

**Background:** Renal readmission is a clinically relevant endpoint.

**Methods:** We assessed the supplied potassium representation.

**Results:** The cohort comprised 120 ICU stays {evidence:renal_summary}.

**Conclusions:** These estimates describe predictive discrimination within the study population.

## Introduction

Potassium measurements can inform the design of prognostic research.

## Methods

### Study design and cohort
We studied the supplied ICU cohort {evidence:renal_summary}.

### Variables
Serum potassium was represented by its supplied maximum value {evidence:renal_summary}.

### Statistical analysis
The executed analysis assessed discrimination {evidence:renal_summary}.

### Software and reproducibility
Analyses used registered software artifacts {evidence:renal_summary}.

## Results

### Cohort characteristics
The cohort comprised 120 ICU stays {evidence:renal_summary}.

### Model performance
The AUROC was 0.74 {evidence:renal_summary}.

### Secondary analyses
The outcome count was 15 {evidence:renal_summary}.

## Discussion

Interpretation is limited to the studied population.

## Limitations

Independent validation will require a separate cohort.

## Conclusion

The evaluation addresses discrimination for renal readmission within the study population.
""" + "\n\n" + render_manuscript_administrative_sections(None)


def _section(text, name):
    return text.split(f"## {name}\n", 1)[1].split("\n## ", 1)[0].strip()


def _exercise(tmp_path, monkeypatch, bad, responses, *, changed_bindings=False, initial_findings=(), baseline_mentions=None, definition_labels=None, grouping=None):
    context = ResearchContext.model_validate({
        "research_question": "Assess potassium-based renal readmission prediction.",
        "cohort": {"cohort_name": "synthetic", "database": "miiv", "n_stays": 120},
        "primary_exposure": "potassium", "target_outcome": "readmission",
        "variables": [{"name": "potassium", "dtype": "float64", "description": "Serum potassium"},
                      {"name": "readmission", "dtype": "bool", "description": "Renal readmission"}],
    })
    plan = AnalysisPlan(research_question=context.research_question, analysis_type="prediction_model", steps=[
        AnalysisStep(step_id="renal_model", intent="Assess discrimination", method="descriptive", expected_outputs=["table:performance"], planned_analysis_role="primary"),
        AnalysisStep(step_id="renal_events", intent="Describe endpoint", method="descriptive", expected_outputs=["table:events"], planned_analysis_role="secondary"),
    ])
    if baseline_mentions is not None:
        monkeypatch.setattr(write_phase, "baseline_reporting_mentions", lambda *_args: baseline_mentions)
    store = EvidenceStore(tmp_path, enforcement_mode=EvidenceEnforcementMode.STRICT)
    if grouping:
        context = context.model_copy(update={"cohort": context.cohort.model_copy(update={
            "provenance": {"exposure_group_labels": [grouping]},
        })})
        _register_context(store, context)
    if definition_labels or grouping:
        plan.display_labels = definition_labels or {}
        store.register_json(kind="log", description="Registered synthetic plan", payload=plan.model_dump(mode="json"),
                            filename="plan.json", evidence_id="analysis_plan", producer="planner", generation_mode="system")
    summary = {"cohort_n": 120, "another_count": 121, "auroc": 0.74, "events": 15}
    store.register_json(kind="statistic", description="Synthetic summary", payload=summary,
                        filename="summary.json", evidence_id="renal_summary", produced_by_step="renal_model")
    store.register_json(kind="statistic", description="Different synthetic owner", payload=summary,
                        filename="other.json", evidence_id="other_summary", produced_by_step="other_model")
    for step_id, evidence_id in (("renal_model", "renal_summary"), ("other_model", "other_summary")):
        store.register_step_summary_numerics(step_id=step_id, evidence_id=evidence_id, summary=summary)
    records = [{"step_id": "renal_model", "status": "ok", "step_summary": summary,
                "step_summary_evidence_id": "renal_summary", "evidence_ids": ["renal_summary"]},
               {"step_id": "other_model", "status": "ok", "step_summary": summary,
                "step_summary_evidence_id": "other_summary", "evidence_ids": ["other_summary"]}]
    client = ScriptedMockLLMClient(responses, repeat_last=True)
    writer = WriterAgent(client)
    drafts, bindings, findings = [], [], []

    def draft(*_args, section_repair, **_kwargs):
        drafts.append(section_repair)
        if section_repair is None:
            text = bad
            findings.extend(initial_findings)
        else:
            try:
                text, _ = writer.repair_sections(section_repair[0], section_errors=section_repair[1],
                    context=context, analysis_plan=plan, evidence_ids=("renal_summary", "other_summary"),
                    evidence_digest="Registered synthetic cohort and discrimination summaries.")
            except Exception as exc:
                candidate = completed_section_repair_candidate(exc, expected_section_keys=tuple(section_repair[1]))
                if candidate is None:
                    raise
                text = candidate[0]
        return write_phase._DraftStageResult((), ("renal_summary",), None, None, text)

    def bind(*_args, scaffold, **_kwargs):
        bindings.append(scaffold)
        # Keep the real sentence-level evidence grammar and STRICT value binder.
        filtered = filter_evidence_bound_scaffold(scaffold, resolve_claim=lambda _: None,
                    resolve_evidence=lambda ref: store.get(ref) is not None)
        # Some nonnumeric interpretation prose is intentionally outside this
        # harness's authority. It is not used to construct numeric evidence.
        assert "The cohort comprised" in filtered.scaffold
        bound = store.bind_manuscript(scaffold, per_step_records=records)
        if definition_labels or grouping:
            from easyicu.research_agent.reporting.manuscript_post import drop_untraceable_numeric_sentences
            bound, removed = drop_untraceable_numeric_sentences(bound, evidence=store, per_step_records=records)
            if removed:
                findings.append(ValidationFinding(validator="manuscript_numeric_auditor", severity="warning",
                                message="Filtered numeric sentences", detail={"removed_sentences": removed}))
        bound, values, unknown = bind_numeric_values(bound, evidence=store, per_step_records=records,
                                                     enforcement_mode=EvidenceEnforcementMode.STRICT)
        if changed_bindings and len(bindings) > 1:
            for claim in values.values():
                if claim.source_field == "cohort_n":
                    claim.source_field = "another_count"
        audit = audit_manuscript_quality(bound, analysis_plan=plan, expected_baseline_mentions=baseline_mentions)
        return write_phase._BindingStageResult(bound, tmp_path / "bound.md",
                    CritiqueReport(status="pass" if audit.status == "pass" else "blocked", reviewer="scripted"),
                    numeric_binding_map=values, untraced_numerics=tuple(unknown))

    monkeypatch.setattr(write_phase, "_draft_manuscript", draft)
    monkeypatch.setattr(write_phase, "_bind_and_review_manuscript", bind)
    result = write_phase._draft_bind_and_repair_manuscript(
        None, context=context, agent_context=context, evidence=store, findings=findings, literature=None,
        per_step_records=records, plan_result=SimpleNamespace(resume_state=None),
        execute_result=SimpleNamespace(plan=plan), critic=None, role_resolver=None,
        prompt_version="test", runtime_state=None, run_dir=tmp_path, run_id="synthetic",
        run_language="en", writer_probe_mode=False, writer_probe_failed_steps=(),
        emit_progress=lambda *_args, **_kwargs: None,
    )
    return result, client, drafts, bindings, findings


def test_final_audit_repairs_only_the_failing_section_and_records_attempt(tmp_path, monkeypatch):
    bad = GOOD.replace("### Model performance\nThe AUROC was 0.74 {evidence:renal_summary}.",
                       "### Model performance\nSee Figure 1 {evidence:renal_summary}.")
    result, client, drafts, bindings, _ = _exercise(tmp_path, monkeypatch, bad, ["## Results\n\n" + _section(GOOD, "Results")])
    assert result[1].manuscript_critique.status == "pass"
    assert len(drafts) == len(bindings) == 2 and len(client.calls) == 1
    assert tuple(drafts[1][1]) == ("results",)
    assert _section(result[0].scaffold, "Methods") == _section(GOOD, "Methods")
    feedback = client.calls[0][0][-1].content
    assert "MANUSCRIPT_RESULT_SUBSECTION_CALLOUT_ONLY" in feedback
    assert "The cohort comprised {manuscript_value:" in client.calls[0][0][1].content
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert receipt["status"] == "pass"
    assert receipt["max_attempts"] == 3 and len(receipt["attempts"]) == 1
    assert receipt["attempts"][0]["numeric_checks"] == "pass"
    assert receipt["attempts"][0]["evidence_checks"] == "pass"


@pytest.mark.parametrize("keep_citation", [True, False])
def test_repeated_sentence_joins_callout_repair_and_must_keep_its_citation(
    tmp_path, monkeypatch, keep_citation,
):
    repeated = "The available records capture readmissions within the hospital network during follow-up"
    citation = "{evidence:renal_summary}"
    sentence = f"{repeated} {citation}."
    bad = GOOD.replace(
        "The AUROC was 0.74 {evidence:renal_summary}.", "See Figure 1 {evidence:renal_summary}.",
    )
    bad = bad.replace("## Discussion\n\n", f"## Discussion\n\n{sentence}\n\n")
    bad = bad.replace("## Conclusion\n\n", f"## Conclusion\n\n{sentence}\n\n")
    conclusion = "## Conclusion\n\n" + _section(GOOD, "Conclusion")
    rephrased = f"The findings therefore apply to readmissions recorded by this network {citation}."
    if keep_citation:
        conclusion += "\n\n" + rephrased
    responses = ["## Results\n\n" + _section(GOOD, "Results"), conclusion]
    if keep_citation:
        (draft, binding), client, drafts, bindings, _ = _exercise(tmp_path, monkeypatch, bad, responses)
        assert len(drafts) == len(bindings) == 2 and len(client.calls) == 2
        assert binding.manuscript_critique.status == "pass"
        assert _section(draft.scaffold, "Discussion") == _section(bad, "Discussion")
        assert rephrased in _section(draft.scaffold, "Conclusion")
        feedback = client.calls[1][0][-1].content
        assert "MANUSCRIPT_REPEATED_SENTENCE" in feedback and repeated in feedback
        assert "once, then refer back to it or interpret it in new words" in feedback
        assert f"{citation}: 1 occurrence(s)" in feedback
    else:
        with pytest.raises(EvidenceEnforcementError, match="existing evidence or claim binding") as failure:
            _exercise(tmp_path, monkeypatch, bad, responses)
        assert failure.value.detail == {
            "check": "section_tokens", "section": "conclusion", "missing_tokens": {citation: 1},
        }
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert receipt["status"] == ("pass" if keep_citation else "rejected")
    assert receipt["max_attempts"] == 3 and len(receipt["attempts"]) == 1
    assert receipt["attempts"][0]["audit_codes"] == [
        "MANUSCRIPT_REPEATED_SENTENCE", "MANUSCRIPT_RESULT_SUBSECTION_CALLOUT_ONLY",
    ]


@pytest.mark.parametrize(("severity", "status"), [("warning", "pass"), ("error", "exhausted")])
def test_previously_unlisted_editorial_error_gets_the_same_bounded_repair(tmp_path, monkeypatch, severity, status):
    from dataclasses import replace

    from easyicu.research_agent.reporting import manuscript_quality

    code = "MANUSCRIPT_UNSUPPORTED_EDITORIAL_FINDING"
    finding = manuscript_quality.ManuscriptQualityFinding(
        code=code, severity=severity, section="Discussion", message="A synthetic audit finding.",
    )

    def audit_with_finding(*args, **kwargs):
        audit = audit_manuscript_quality(*args, **kwargs)
        return replace(audit, findings=(*audit.findings, finding))

    monkeypatch.setattr(manuscript_quality, "audit_manuscript_quality", audit_with_finding)
    bad = GOOD.replace(
        "The AUROC was 0.74 {evidence:renal_summary}.", "See Figure 1 {evidence:renal_summary}.",
    )
    _, client, drafts, bindings, _ = _exercise(
        tmp_path, monkeypatch, bad, [
            "## Results\n\n" + _section(GOOD, "Results"),
            "## Discussion\n\n" + _section(GOOD, "Discussion"),
        ] + ["## Discussion\n\n" + _section(GOOD, "Discussion")] * 2,
    )
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert receipt["status"] == status
    if severity == "warning":
        assert len(drafts) == len(bindings) == 2 and len(client.calls) == 1
        assert len(receipt["attempts"]) == 1
        assert receipt["attempts"][0]["audit_codes"] == ["MANUSCRIPT_RESULT_SUBSECTION_CALLOUT_ONLY"]
    else:
        assert len(drafts) == len(bindings) == 4 and len(client.calls) == 4
        assert len(receipt["attempts"]) == 3
        assert receipt["remaining_codes"] == [code]


def test_repeated_sentence_warning_alone_does_not_spend_a_writer_attempt(tmp_path, monkeypatch):
    sentence = "The available records capture readmissions within the hospital network during follow-up."
    bad = GOOD.replace("## Discussion\n\n", f"## Discussion\n\n{sentence}\n\n")
    bad = bad.replace("## Conclusion\n\n", f"## Conclusion\n\n{sentence}\n\n")
    _, client, drafts, _, _ = _exercise(tmp_path, monkeypatch, bad, ["unused"])
    assert drafts == [None] and client.calls == []
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert receipt["status"] == "not_needed" and receipt["attempts"] == []


def test_final_audit_repair_is_bounded_and_retains_typed_codes(tmp_path, monkeypatch):
    bad = GOOD.replace("### Model performance\nThe AUROC was 0.74 {evidence:renal_summary}.",
                       "### Model performance\nSee Figure 1 {evidence:renal_summary}.")
    _, client, drafts, _, _ = _exercise(tmp_path, monkeypatch, bad, ["## Results\n\n" + _section(bad, "Results")])
    assert len(client.calls) == 3 and len(drafts) == 4
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert receipt["status"] == "exhausted"
    assert "MANUSCRIPT_RESULT_SUBSECTION_CALLOUT_ONLY" in receipt["remaining_codes"]


@pytest.mark.parametrize("change", ["number", "display", "evidence", "binding"])
def test_section_repair_cannot_change_an_existing_bound_fact(tmp_path, monkeypatch, change):
    bad = GOOD.replace("The AUROC was 0.74 {evidence:renal_summary}.", "See Figure 1 {evidence:renal_summary}.")
    replacement = "## Results\n\n" + _section(GOOD, "Results")
    if change == "number":
        replacement = replacement.replace("120", "121")  # Both numbers exist in the evidence.
    if change == "display":
        replacement = replacement.replace("120", "120.0")
    if change == "evidence":
        replacement = replacement.replace("120 ICU stays {evidence:renal_summary}", "120 ICU stays {evidence:other_summary}")
    with pytest.raises(EvidenceEnforcementError, match="MANUSCRIPT_REPAIR_AUTHORITY_CHANGED"):
        _exercise(tmp_path, monkeypatch, bad, [replacement], changed_bindings=change == "binding")
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert receipt["status"] == "rejected"
    assert len(receipt["attempts"]) == 1


def test_an_unregistered_number_fails_closed_before_any_section_repair(tmp_path, monkeypatch):
    bad = GOOD.replace("120 ICU stays", "98765 ICU stays")
    with pytest.raises(EvidenceEnforcementError):
        _exercise(tmp_path, monkeypatch, bad, ["## Results\n\n" + _section(GOOD, "Results")])


def test_numeric_failure_never_enters_editorial_repair(tmp_path, monkeypatch):
    bad = GOOD.replace("The AUROC was 0.74 {evidence:renal_summary}.", "See Figure 1 {evidence:renal_summary}.")
    finding = ValidationFinding(validator="manuscript_numeric_auditor", severity="error",
                                message="A numeric claim is fabricated.", detail={"code": "FABRICATED_NUMBER"})
    with pytest.raises(EvidenceEnforcementError, match="MANUSCRIPT_AUDIT_NON_REPAIRABLE"):
        _exercise(tmp_path, monkeypatch, bad, ["unused"], initial_findings=(finding,))
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert receipt["status"] == "non_repairable" and receipt["attempts"] == []
    assert receipt["remaining_codes"] == [
        "MANUSCRIPT_AUDIT_NON_REPAIRABLE", "MANUSCRIPT_RESULT_SUBSECTION_CALLOUT_ONLY",
    ]


@pytest.mark.parametrize("owner, before, after, code, baseline", [
    ("Abstract", "**Results:** The cohort comprised 120 ICU stays {evidence:renal_summary}.",
     "**Results:**", "MANUSCRIPT_ABSTRACT_LABEL_MISSING_OR_EMPTY", None),
    ("Abstract", "**Conclusions:** These estimates describe predictive discrimination within the study population.",
     "**Conclusions:** Independent validation is required.", "MANUSCRIPT_CONCLUSION_WITHOUT_INTERPRETATION", None),
    ("Conclusion", "The evaluation addresses discrimination for renal readmission within the study population.",
     "Independent validation is required.", "MANUSCRIPT_CONCLUSION_WITHOUT_INTERPRETATION", None),
    ("Methods", "Serum potassium was represented by its supplied maximum value",
     "The predictor was represented by its supplied maximum value", "MANUSCRIPT_BASELINE_METHODS_INCOMPLETE",
     {"potassium": ("Serum potassium",)}),
    ("Results", "The outcome count was 15 {evidence:renal_summary}.",
     "", "MANUSCRIPT_SUBSECTION_MISSING_OR_EMPTY", None),
    ("Results", "The outcome count was 15 {evidence:renal_summary}.",
     "The outcome count was", "MANUSCRIPT_SECTION_TRUNCATED", None),
])
def test_each_final_structural_finding_returns_to_its_writer_owner(
    tmp_path, monkeypatch, owner, before, after, code, baseline,
):
    bad = GOOD.replace(before, after)
    result, client, drafts, _, _ = _exercise(
        tmp_path, monkeypatch, bad, [f"## {owner}\n\n" + _section(GOOD, owner)],
        baseline_mentions=baseline,
    )
    assert result[1].manuscript_critique.status == "pass"
    assert len(client.calls) == 1 and list(drafts[1][1]) == [owner.lower()]
    assert code in client.calls[0][0][-1].content


@pytest.mark.parametrize("validator", ["manuscript_numeric_auditor", "evidence_bound_writer"])
def test_non_repairable_binding_error_does_not_spend_a_model_attempt(tmp_path, monkeypatch, validator):
    from easyicu.research_agent.reporting.manuscript_audit_repair import run_manuscript_audit_repair

    calls = []
    findings = []
    def draft(repair):
        calls.append(repair)
        findings.append(ValidationFinding(validator=validator, severity="error", message="Binding failed."))
        return SimpleNamespace(scaffold=GOOD, writer_error_message=None)
    with pytest.raises(EvidenceEnforcementError, match="MANUSCRIPT_AUDIT_NON_REPAIRABLE"):
        run_manuscript_audit_repair(
            draft=draft, bind=lambda _: SimpleNamespace(bound=GOOD), findings=findings,
            section_errors=lambda *_: {"results": ("MANUSCRIPT_SECTION_TRUNCATED",)},
            run_dir=None, evidence=None,
        )
    assert calls == [None]


@pytest.mark.parametrize("validator, detail", [
    ("manuscript_numeric_auditor", {"removed_sentences": [{"sentence": "Fabricated number."}]}),
    ("evidence_bound_writer", {"missing_evidence_ids": ["foreign_source"]}),
])
def test_filtering_a_binding_failure_cannot_make_it_editorially_repairable(tmp_path, validator, detail):
    from easyicu.research_agent.reporting.manuscript_audit_repair import run_manuscript_audit_repair

    calls, findings = [], []
    def draft(repair):
        calls.append(repair)
        findings.append(ValidationFinding(validator=validator, severity="warning", message="Filtered.", detail=detail))
        return SimpleNamespace(scaffold=GOOD, writer_error_message=None)
    with pytest.raises(EvidenceEnforcementError, match="MANUSCRIPT_AUDIT_NON_REPAIRABLE"):
        run_manuscript_audit_repair(
            draft=draft, bind=lambda _: SimpleNamespace(bound=GOOD), findings=findings,
            section_errors=lambda *_: {"results": ("MANUSCRIPT_SECTION_TRUNCATED",)},
            run_dir=None, evidence=None,
        )
    assert calls == [None]



@pytest.mark.parametrize("validator, detail", [
    ("manuscript_numeric_auditor", {"removed_sentences": [{"sentence": "Unsupported number."}]}),
    ("evidence_bound_writer", {"missing_evidence_ids": ["foreign_source"]}),
])
def test_a_routine_filter_warning_without_audit_findings_keeps_the_draft(tmp_path, validator, detail):
    from easyicu.research_agent.reporting.manuscript_audit_repair import run_manuscript_audit_repair

    calls, findings = [], []
    def draft(repair):
        calls.append(repair)
        findings.append(ValidationFinding(validator=validator, severity="warning", message="Filtered.", detail=detail))
        return SimpleNamespace(scaffold=GOOD, writer_error_message=None)
    result = run_manuscript_audit_repair(
        draft=draft, bind=lambda _: SimpleNamespace(bound=GOOD), findings=findings,
        section_errors=lambda *_: {}, run_dir=None, evidence=None,
    )
    assert calls == [None] and result[0].scaffold == GOOD and len(findings) == 1


def test_incomplete_repair_exhausts_with_typed_codes_and_a_receipt(tmp_path, monkeypatch):
    bad = GOOD.replace("The AUROC was 0.74 {evidence:renal_summary}.", "See Figure 1 {evidence:renal_summary}.")
    with pytest.raises(EvidenceEnforcementError, match="MANUSCRIPT_SUBSECTION_MISSING_OR_EMPTY"):
        _exercise(tmp_path, monkeypatch, bad, ["## Results\nIncomplete response."])
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert receipt["status"] == "exhausted" and len(receipt["attempts"]) == 3
    assert "MANUSCRIPT_RESULT_SUBSECTION_CALLOUT_ONLY" in receipt["remaining_codes"]
    assert "MANUSCRIPT_SUBSECTION_MISSING_OR_EMPTY" in receipt["remaining_codes"]



def test_repair_model_can_keep_opaque_values_without_receiving_or_changing_them(tmp_path, monkeypatch):
    from easyicu.research_agent.reporting.manuscript_sections import _masked_section_values

    bad = GOOD.replace("The AUROC was 0.74 {evidence:renal_summary}.", "See Figure 1 {evidence:renal_summary}.")
    masked, _ = _masked_section_values("## Results\n\n" + _section(bad, "Results"))
    # A model receives the existing values as markers; it can repair the text
    # and add the missing metric from its independently authorized digest.
    response = masked.replace("### Model performance\n", "### Model performance\nThe AUROC was 0.74 {evidence:renal_summary}.\n")
    result, client, _, _, _ = _exercise(tmp_path, monkeypatch, bad, [response])
    assert result[1].manuscript_critique.status == "pass"
    assert "120 ICU stays" in result[0].scaffold and "15 {evidence:renal_summary}" in result[0].scaffold
    assert "{manuscript_value:" not in result[0].scaffold
    current = client.calls[0][0][1].content.split("CURRENT SECTION", 1)[1].split("CITATION RULE", 1)[0]
    assert "120" not in current and "15 " not in current


@pytest.mark.parametrize("before, after, owner, code", [
    ("The outcome count was 15 {evidence:renal_summary}.", "The outcome count was", "Results", "MANUSCRIPT_SECTION_TRUNCATED"),
    ("The outcome count was 15 {evidence:renal_summary}.", "", "Results", "MANUSCRIPT_SUBSECTION_MISSING_OR_EMPTY"),
    ("The AUROC was 0.74 {evidence:renal_summary}.", "See Figure 1 {evidence:renal_summary}.", "Results", "MANUSCRIPT_RESULT_SUBSECTION_CALLOUT_ONLY"),
    ("The evaluation addresses discrimination for renal readmission within the study population.", "Independent validation is required.", "Conclusion", "MANUSCRIPT_CONCLUSION_WITHOUT_INTERPRETATION"),
])
def test_structural_repair_remains_available_with_registered_numeric_labels(tmp_path, monkeypatch, before, after, owner, code):
    bad = GOOD.replace(before, after)
    label = "patients aged 65 years or older"
    bad = bad.replace("### Variables\n", "### Variables\nWe studied " + label + " {evidence:renal_summary}.\n\n")
    response = f"## {owner}\n\n" + _section(GOOD, owner)
    result, client, _, _, findings = _exercise(tmp_path, monkeypatch, bad, [response],
                                              definition_labels={"age=older": label})
    assert result[1].manuscript_critique.status == "pass" and len(client.calls) == 1
    assert code in client.calls[0][0][-1].content
    assert not any((f.detail or {}).get("removed_sentences") for f in findings)
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert receipt["status"] == "pass" and receipt["attempts"][0]["numeric_checks"] == "pass"


@pytest.mark.parametrize("binding_change", [None, "drop", "replace"])
def test_adjustment_repair_keeps_methods_tokens_verbatim(tmp_path, monkeypatch, binding_change):
    bad = GOOD.replace("The executed analysis assessed discrimination", "The adjustment set comprised age and severity")
    bad = bad.replace("The AUROC was", "After adjustment for age and sex, the AUROC was")
    response = "## Methods\n\n" + _section(bad, "Methods").replace("age and severity", "age and sex")
    if binding_change:
        response = response.replace("{evidence:renal_summary}", "" if binding_change == "drop" else "{evidence:other_summary}", 1)
        with pytest.raises(EvidenceEnforcementError, match="existing evidence or claim binding"):
            _exercise(tmp_path, monkeypatch, bad, [response])
        receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
        assert receipt["attempts"][0]["authority_failure"]["check"] == "section_tokens"
        assert receipt["attempts"][0]["authority_failure"]["section"] == "methods"
        assert receipt["attempts"][0]["numeric_checks"] == "pending"
    else:
        result, client, _, _, _ = _exercise(tmp_path, monkeypatch, bad, [response])
        assert result[1].manuscript_critique.status == "pass"
        feedback = client.calls[0][0][-1].content
        assert "MANUSCRIPT_ADJUSTMENT_SET_CONFLICT" in feedback
        assert "TOKENS TO KEEP VERBATIM" in feedback
        assert "{evidence:renal_summary}" in feedback and "4 occurrence(s)" in feedback


def test_adjustment_repair_still_refuses_an_unaffected_section_change():
    from easyicu.research_agent.reporting.manuscript_audit_repair import _require_preserved_sections

    changed = GOOD.replace("Interpretation is limited", "Interpretation remains limited")
    with pytest.raises(EvidenceEnforcementError, match="unaffected section") as failure:
        _require_preserved_sections(GOOD, changed, ("methods",))
    assert failure.value.detail == {"check": "unaffected_section", "section": "discussion"}


def test_a_repair_candidate_with_new_filter_removals_is_still_rejected(tmp_path, monkeypatch):
    bad = GOOD.replace("The AUROC was 0.74 {evidence:renal_summary}.", "See Figure 1 {evidence:renal_summary}.")
    response = "## Results\n\n" + _section(GOOD, "Results") + "\n\nThere were 991 deaths {evidence:renal_summary}."
    with pytest.raises(EvidenceEnforcementError, match="MANUSCRIPT_AUDIT_NON_REPAIRABLE"):
        _exercise(tmp_path, monkeypatch, bad, [response], definition_labels={"age=older": "patients aged 65 years or older"})
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert receipt["status"] == "rejected" and len(receipt["attempts"]) == 1


def test_adjustment_repair_cannot_drop_a_definition_written_in_scientific_notation(tmp_path, monkeypatch):
    label = "potassium at least 1 × 10⁻³ mmol/L"
    bad = GOOD.replace("The executed analysis assessed discrimination",
                       "The adjustment set comprised age and severity, using " + label)
    bad = bad.replace("The AUROC was", "After adjustment for age and sex, the AUROC was")
    response = "## Methods\n\n" + _section(bad, "Methods").replace("age and severity, using " + label, "age and sex")
    with pytest.raises(EvidenceEnforcementError, match="existing bound number or source changed"):
        _exercise(tmp_path, monkeypatch, bad, [response], definition_labels={"potassium=low": label})


def test_cited_paraphrases_do_not_block_repair_of_a_callout_only_subsection(tmp_path, monkeypatch):
    definition = ("Serum potassium was represented by two ordered categories: under 4.0 mmol/L and 4.0 mmol/L or higher "
                  "{evidence:research_context}.")
    contrast = ("The contrast was serum potassium of 4.0 mmol/L or higher versus below 4.0 mmol/L, "
                "with the under-4.0 group as the reference {evidence:research_context} {evidence:renal_summary}.")
    bad = GOOD.replace("The AUROC was 0.74 {evidence:renal_summary}.", "See Figure 1 {evidence:renal_summary}.")
    bad = bad.replace("## Introduction\n\n", "## Introduction\n\n" + "For clinical context, " + definition[0].lower() + definition[1:] + "\n\n")
    bad = bad.replace("### Variables\n", "### Variables\n" + definition + "\n\n" + contrast + "\n\n")
    bad = bad.replace("## Limitations\n\n", "## Limitations\n\n" + "In this setting, " + definition[0].lower() + definition[1:] + "\n\n")
    result, client, drafts, bindings, findings = _exercise(
        tmp_path, monkeypatch, bad, ["## Results\n\n" + _section(GOOD, "Results")],
        grouping=_grouping("potassium", 4, "mmol/L"),
    )
    assert result[1].manuscript_critique.status == "pass"
    assert len(client.calls) == 1 and len(drafts) == len(bindings) == 2
    assert "MANUSCRIPT_RESULT_SUBSECTION_CALLOUT_ONLY" in client.calls[0][0][-1].content
    assert not any((finding.detail or {}).get("removed_sentences") for finding in findings)
    assert result[1].untraced_numerics == ()
    assert result[1].bound.count("4.0[^claim_") == 9
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert receipt["status"] == "pass" and len(receipt["attempts"]) == 1
    assert receipt["attempts"][0]["numeric_checks"] == "pass"


def test_final_audit_retries_rejected_candidate_with_typed_feedback(tmp_path, monkeypatch):
    bad = GOOD.replace("The AUROC was 0.74 {evidence:renal_summary}.", "See Figure 1 {evidence:renal_summary}.")
    results = "## Results\n\n" + _section(GOOD, "Results")
    missing_heading = results.replace("### Secondary analyses", "### Extra detail")

    (draft, binding), client, _, _, _ = _exercise(
        tmp_path, monkeypatch, bad, [missing_heading, results],
    )

    assert binding.manuscript_critique.status == "pass"
    assert _section(draft.scaffold, "Results") == _section(GOOD, "Results")
    assert "MANUSCRIPT_SUBSECTION_MISSING_OR_EMPTY" in client.calls[1][0][-1].content
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert [row["status"] for row in receipt["attempts"]] == ["rejected", "pass"]


def test_title_contract_retry_feedback_stays_with_its_owner():
    from easyicu.research_agent.reporting.manuscript_audit_repair import run_manuscript_audit_repair
    from easyicu.research_agent.reporting.manuscript_sections import ManuscriptSectionContractError

    before = "# Ward transfers\n\n## Results\n\nTransfers were recorded.\n"
    after = "# A cohort study of ward transfers\n\n**Keywords:** transfer, cohort\n\n## Results\n\nWard transfers were recorded.\n"
    requests = []

    def draft(repair):
        if repair is None:
            return SimpleNamespace(scaffold=before, writer_error_message=None)
        _, request = repair
        requests.append(request)
        if len(requests) == 1:
            raise ManuscriptSectionContractError(
                section_name="Title and Keywords", missing_subsections=("Keywords",),
            )
        return SimpleNamespace(scaffold=after, writer_error_message=None)

    result, _ = run_manuscript_audit_repair(
        draft=draft, bind=lambda item: SimpleNamespace(bound=item.scaffold),
        section_errors=lambda item, _: (
            {"title": ("MANUSCRIPT_KEYWORDS_MISSING: Name the cohort topic.",),
             "results": ("MANUSCRIPT_SECTION_TRUNCATED: Finish the transfer summary.",)}
            if item.scaffold == before else {}
        ),
        findings=[], run_dir=None, evidence=None,
    )
    assert result.scaffold == after
    initial, retry = requests
    assert "MANUSCRIPT_SUBSECTION_MISSING_OR_EMPTY: Keywords" in retry["title"]
    assert retry["results"] == initial["results"]


def test_an_unsupported_dictionary_definition_never_reaches_editorial_repair():
    from easyicu.research_agent.reporting.manuscript_audit_repair import run_manuscript_audit_repair

    semantics = SimpleNamespace(kind="positive_only_event")
    context = SimpleNamespace(variables=[SimpleNamespace(observation_semantics=semantics)])
    scaffold = GOOD.replace("## Introduction\n", "## Introduction\n\nWe used the diagnosis status from intake records.\n")
    drafted = []

    def draft(request):
        drafted.append(request)
        return SimpleNamespace(scaffold=scaffold, writer_error_message=None)

    with pytest.raises(EvidenceEnforcementError, match="MANUSCRIPT_DEFINITION_NOT_FROM_DICTIONARY"):
        run_manuscript_audit_repair(
            draft=draft, bind=lambda _: SimpleNamespace(bound=scaffold),
            section_errors=lambda *_: {}, findings=[], run_dir=None, evidence=None, context=context,
        )
    assert drafted == [None]
