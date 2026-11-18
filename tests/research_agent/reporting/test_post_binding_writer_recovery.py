"""Final binding failures feed only affected sections back through the gates."""

import json
import re
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
from easyicu.research_agent.reporting.manuscript_audit_repair import (
    run_manuscript_audit_repair, splice_manuscript_sections,
)
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


def _strict_pipeline():
    return SimpleNamespace(
        _evidence_enforcement_mode=EvidenceEnforcementMode.STRICT,
        _max_numeric_claims_per_step=0, _enable_nature_writing_skill=False,
        _user_extension_activation=SimpleNamespace(writing_advisory=None),
        _writer_digest_widened=False, _writer_digest_secondary_cap_per_step=0,
    )


def _renal_context():
    return ResearchContext.model_validate({
        "research_question": "Assess potassium-based renal readmission prediction.",
        "cohort": {"cohort_name": "synthetic", "database": "miiv", "n_stays": 120},
        "primary_exposure": "potassium", "target_outcome": "readmission",
        "variables": [{"name": "potassium", "dtype": "float64", "description": "Serum potassium"},
                      {"name": "readmission", "dtype": "bool", "description": "Renal readmission"}],
    })


def test_splice_inserts_missing_results_after_title_and_abstract():
    before = "# Study title\n\n## Abstract\n\nAbstract prose.\n\n## Discussion\n\nDiscussion prose.\n"
    results = "## Results\n\nResults prose.\n\n"
    after, keys = splice_manuscript_sections(before, results, ("results",))
    assert keys == ["results"]
    assert after == before.replace("## Discussion", results + "## Discussion")


def _uncovered(outcome, subsection=None):
    return ValidationFinding(
        validator="writer_question_coverage", severity="error", message="A question part is uncovered.",
        detail={
            "reason_code": "writer_question_part_uncovered", "step_id": "distribution",
            "product_id": "table:distribution", "outcome": outcome,
            "repair_facts": [] if subsection is None else [{
                "claim_placeholder": f"{{claim:distribution.report_fact_{outcome}}}",
                "subsection": subsection, "text": "A registered description.",
            }],
        },
    )


@pytest.mark.parametrize("use_host_fallback", [False, True])
def test_repair_receipt_lists_unresolved_coverage(tmp_path, use_host_fallback):
    store = EvidenceStore(tmp_path)
    unrepairable = _uncovered("exposure_distribution")
    repairable = _uncovered("outcome_distribution", "Model performance")
    token = repairable.detail["repair_facts"][0]["claim_placeholder"]
    bad = GOOD.replace("Interpretation is limited to the studied population.", "")

    def draft(request):
        scaffold = GOOD if use_host_fallback or request else bad
        return write_phase._DraftStageResult((), (), None, None, scaffold)

    def bind(item):
        coverage = [unrepairable]
        if use_host_fallback and token not in item.scaffold:
            coverage.append(repairable)
        return SimpleNamespace(bound=item.scaffold, question_coverage_findings=coverage)

    run_manuscript_audit_repair(
        draft=draft, bind=bind,
        section_errors=lambda item, _: {"discussion": ("MANUSCRIPT_SECTION_EMPTY: Add interpretation.",)}
        if item.scaffold == bad else {},
        findings=[], run_dir=tmp_path, evidence=store,
    )
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert receipt["status"] == "non_repairable"
    assert receipt["unresolved_question_parts"] == [unrepairable.detail]
    if use_host_fallback:
        assert receipt["host_fallback"]["status"] == "pass"


def test_host_fallback_rejects_partial_question_coverage(tmp_path):
    present = _uncovered("outcome_distribution", "Model performance")
    absent = _uncovered("exposure_distribution", "Unavailable heading")
    token = present.detail["repair_facts"][0]["claim_placeholder"]

    def bind(item):
        return SimpleNamespace(
            bound=item.scaffold, question_coverage_findings=[absent] if token in item.scaffold else [present, absent],
        )

    with pytest.raises(EvidenceEnforcementError, match="host fallback failed"):
        run_manuscript_audit_repair(
            draft=lambda _: write_phase._DraftStageResult((), (), None, None, GOOD),
            bind=bind, section_errors=lambda *_: {}, findings=[],
            run_dir=tmp_path, evidence=EvidenceStore(tmp_path),
        )
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert len(receipt["attempts"]) == 3
    assert receipt["host_fallback"]["status"] == "rejected"
    assert receipt["status"] == "exhausted"
    assert receipt["unresolved_question_parts"] == [present.detail, absent.detail]


@pytest.mark.parametrize("exit_path", ["skip", "writer_exception", "section_tokens", "exhausted"])
def test_repairable_coverage_preserves_the_termination_status(tmp_path, exit_path):
    coverage = _uncovered("ward_transfer_distribution", "Ward transfers")
    before = "## Results\n\nWard transfers were tabulated {evidence:ward_summary}.\n"

    def draft(request):
        scaffold, error = before, None
        if request:
            if exit_path == "writer_exception":
                raise RuntimeError("Scripted response failed")
            if exit_path == "section_tokens":
                scaffold = "## Results\n\nTransfer counts were tabulated.\n"
            if exit_path == "exhausted":
                error = "Scripted response did not produce a section"
        return write_phase._DraftStageResult((), (), None, error, scaffold)

    def run():
        return run_manuscript_audit_repair(
            draft=draft, bind=lambda item: SimpleNamespace(bound=item.scaffold, question_coverage_findings=[coverage]),
            section_errors=lambda *_: {}, findings=[], run_dir=tmp_path, evidence=EvidenceStore(tmp_path),
            skip_repair=exit_path == "skip",
        )

    if exit_path == "skip":
        run()
        expected_status = "not_needed"
    elif exit_path == "writer_exception":
        with pytest.raises(RuntimeError, match="Scripted response failed"):
            run()
        expected_status = "rejected"
    elif exit_path == "section_tokens":
        with pytest.raises(EvidenceEnforcementError, match="MANUSCRIPT_REPAIR_AUTHORITY_CHANGED"):
            run()
        expected_status = "rejected"
    else:
        with pytest.raises(EvidenceEnforcementError, match="MANUSCRIPT_AUDIT_REPAIR_FAILED"):
            run()
        expected_status = "exhausted"
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert receipt["status"] == expected_status
    assert receipt["unresolved_question_parts"] == [coverage.detail]
    if exit_path == "exhausted":
        assert "writer_question_part_uncovered" in receipt["remaining_codes"]


def test_headingless_results_repair_ignores_appended_discussion():
    client = ScriptedMockLLMClient([
        _section(GOOD, "Results") + "\n\n## Discussion\n\nThis extra section is ignored.",
        "## Discussion\n\n" + _section(GOOD, "Discussion"),
    ])
    plan = AnalysisPlan(research_question="Predict readmission.", analysis_type="prediction_model", steps=[])
    try:
        candidate, keys = WriterAgent(client).repair_sections(
            GOOD, context=_renal_context(), analysis_plan=plan, evidence_ids=("renal_summary",),
            section_errors={"results": ("Repair Results.",), "discussion": ("Repair Discussion.",)},
        )
    except Exception as exc:
        completed = completed_section_repair_candidate(exc, expected_section_keys=("results", "discussion"))
        if completed is None:
            raise
        candidate, keys = completed
    assert keys == ("results", "discussion")
    assert candidate.count("## Discussion") == 1
    assert "This extra section is ignored" not in candidate
    assert _section(candidate, "Results") == _section(GOOD, "Results")


def test_real_draft_reports_spliced_section_when_writer_loses_token(tmp_path):
    context = _renal_context()
    client = ScriptedMockLLMClient([
        "## Results\n\n" + _section(GOOD, "Results").replace("{evidence:renal_summary}", "", 1),
    ])
    with pytest.raises(EvidenceEnforcementError, match="existing evidence or claim binding") as stop:
        write_phase._draft_manuscript(
            _strict_pipeline(), context=context, agent_context=context, evidence=EvidenceStore(tmp_path),
            findings=[], literature=None, per_step_records=[], resume_state=None,
            prompt_version="test", role_resolver=lambda _: client,
            runtime_state=SimpleNamespace(semantics=None), run_dir=tmp_path, run_id="synthetic",
            execute_result=SimpleNamespace(plan=AnalysisPlan(
                research_question="Predict readmission.", analysis_type="prediction_model", steps=[],
            )), run_language="en", emit_progress=lambda *_args, **_kwargs: None,
            section_repair=(GOOD, {"results": ("Rewrite the results.",)}),
        )
    assert len(client.calls) == 1
    assert stop.value.detail["check"] == "section_tokens"
    assert stop.value.detail["spliced_section_keys"] == ["results"]


@pytest.fixture
def strict_writer_run(tmp_path, monkeypatch):
    from easyicu.research_agent.audits.envelope_consumers import RegisteredOutputEnvelopeConsumer

    context = _renal_context()
    plan = AnalysisPlan(research_question=context.research_question, analysis_type="prediction_model", steps=[
        AnalysisStep(step_id="renal_model", intent="Assess discrimination", method="descriptive",
                     expected_outputs=["table:performance"], planned_analysis_role="primary"),
        AnalysisStep(step_id="renal_events", intent="Describe the endpoint", method="descriptive",
                     expected_outputs=["table:events"], planned_analysis_role="secondary"),
    ])
    store = EvidenceStore(tmp_path, enforcement_mode=EvidenceEnforcementMode.STRICT)
    summary = {"cohort_n": 120, "auroc": 0.74, "events": 15}
    store.register_json(kind="statistic", description="Synthetic summary", payload=summary,
                        filename="summary.json", evidence_id="renal_summary", aliases=["renal_result"],
                        produced_by_step="renal_model")
    records = [{"step_id": "renal_model", "status": "ok", "step_summary": summary,
                "step_summary_evidence_id": "renal_summary", "evidence_ids": ["renal_summary"]}]
    monkeypatch.setattr(RegisteredOutputEnvelopeConsumer, "authoritative_writer_records",
                        lambda self, records, **kwargs: records)
    findings = []
    client = ScriptedMockLLMClient([], repeat_last=True)

    def run(scaffold, responses):
        client.responses.extend(responses)
        if scaffold is not None:
            write_phase._store_raw_scaffold(tmp_path, scaffold, evidence=store, prompt_version="test")
        return write_phase._draft_bind_and_repair_manuscript(
            _strict_pipeline(), context=context, agent_context=context, evidence=store,
            findings=findings, literature=None, per_step_records=records,
            plan_result=SimpleNamespace(resume_state={"per_step_records": records}),
            execute_result=SimpleNamespace(plan=plan), critic=SimpleNamespace(
                review_manuscript=lambda **_: CritiqueReport(reviewer="scripted", status="pass"),
            ), role_resolver=lambda _: client, prompt_version="test", runtime_state=SimpleNamespace(semantics=None),
            run_dir=tmp_path, run_id="synthetic", run_language="en", writer_probe_mode=False,
            writer_probe_failed_steps=(), emit_progress=lambda *_args, **_kwargs: None,
        )

    return run, client, findings, store


_OUTCOME = "The outcome count was 15 {evidence:renal_summary}."
_UNBOUND = "The outcome count was 991 {evidence:renal_summary}."
_STRICT_GOOD = GOOD.replace(
    "These estimates describe predictive discrimination within the study population.",
    "This study describes baseline characteristics.",
).replace(
    "The evaluation addresses discrimination for renal readmission within the study population.",
    "The prespecified analysis was performed.",
)


@pytest.mark.parametrize("reinsert", [False, True])
def test_strict_numeric_removal_uses_typed_feedback_and_shared_budget(tmp_path, strict_writer_run, reinsert):
    from easyicu.research_agent.reporting.manuscript_audit_repair import NUMERIC_REMOVAL

    run, client, findings, _ = strict_writer_run
    initial = _STRICT_GOOD.replace(_OUTCOME, _UNBOUND)
    response = "## Results\n\n" + _section(initial if reinsert else _STRICT_GOOD, "Results")
    if reinsert:
        with pytest.raises(EvidenceEnforcementError, match="MANUSCRIPT_BINDING_FILTER_REPAIR_EXHAUSTED"):
            run(initial, [response])
        expected_calls, expected_removals = 3, 4
    else:
        draft, binding = run(initial, [response])
        assert binding.manuscript_critique.status == "pass"
        assert "991" not in draft.scaffold
        assert "991" not in binding.bound
        assert _section(binding.bound, "Audit") == "1 sentence(s) were removed for unbindable values."
        assert binding.bound_path.read_text() == binding.bound
        expected_calls = expected_removals = 1

    assert len(client.calls) == expected_calls
    for messages, *_ in client.calls:
        prompt = "\n".join(message.content for message in messages)
        assert "991" not in prompt
        assert "The outcome count was {manuscript_value:" not in prompt
        assert NUMERIC_REMOVAL in prompt and "number_1" in prompt
        assert "Rewrite the section without that claim, or cite only bound values." in prompt
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert receipt["max_attempts"] == 3
    assert receipt["status"] == ("exhausted" if reinsert else "pass")
    assert len(receipt["attempts"]) == expected_calls
    assert "991" not in json.dumps([(row["feedback"], row["findings"]) for row in receipt["attempts"]])
    assert sum((f.detail or {}).get("removed_sentence_count", 0) for f in findings) == expected_removals
    first = receipt["attempts"][0]
    assert {"code": NUMERIC_REMOVAL, "path": "results", "facts": {"number_ids": ["number_1"]}} in first["feedback"]
    if reinsert:
        assert all(row["numeric_checks"] == "filtered" and row["status"] == "rejected"
                   for row in receipt["attempts"])


def test_structural_and_filter_failures_spend_the_same_three_attempts(tmp_path, strict_writer_run):
    from easyicu.research_agent.reporting.manuscript_audit_repair import NUMERIC_REMOVAL

    run, client, _, _ = strict_writer_run
    initial = _STRICT_GOOD.replace(_OUTCOME, "")
    incomplete = "## Results\n\n" + _section(initial, "Results")
    reinsert = "## Results\n\n" + _section(_STRICT_GOOD.replace(_OUTCOME, _UNBOUND), "Results")
    with pytest.raises(EvidenceEnforcementError, match="MANUSCRIPT_BINDING_FILTER_REPAIR_EXHAUSTED"):
        run(initial, [incomplete, reinsert])
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    first, second, third = receipt["attempts"]
    assert len(client.calls) == 3
    assert first["numeric_checks"] == "pending"
    assert second["numeric_checks"] == third["numeric_checks"] == "filtered"
    assert NUMERIC_REMOVAL not in first["audit_codes"]
    assert NUMERIC_REMOVAL in client.calls[2][0][-1].content


def test_binding_filter_removes_prose_citing_a_registered_alias(strict_writer_run):
    run, client, _, _ = strict_writer_run
    initial = _STRICT_GOOD.replace(_OUTCOME, _UNBOUND.replace("renal_summary", "renal_result"))
    response = "## Results\n\n" + _section(_STRICT_GOOD, "Results")
    _, binding = run(initial, [response])
    assert binding.manuscript_critique.status == "pass"
    assert len(client.calls) == 1
    assert "The outcome count was" not in client.calls[0][0][1].content



@pytest.mark.parametrize(("citation", "removed_count"), [("retired_enrollment", 1), ("renal_summary", 1)])
def test_comparison_counts_as_one_binding_filter_removal(tmp_path, strict_writer_run, citation, removed_count):
    run, client, _, _ = strict_writer_run
    opener = "Potassium measurements can inform the design of prognostic research."
    prefix = "{evidence:renal_summary} " if citation == "renal_summary" else ""
    comparison = f"Enrollment was 812 {prefix}vs. 913 across wards {{evidence:{citation}}}."
    initial = _STRICT_GOOD.replace(opener, f"{comparison} {opener}")
    response = f"## Introduction\n\n{opener} Enrollment records informed the sampling protocol."
    _, binding = run(initial, [response])
    assert len(client.calls) == 1
    assert "Enrollment was" not in "\n".join(message.content for message in client.calls[0][0])
    assert _section(binding.bound, "Audit") == f"{removed_count} sentence(s) were removed for unbindable values."
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert receipt["status"] == "pass"


@pytest.mark.parametrize("rejected", [
    "Diaz et al. catalogued 867 ward transfers.",
    "The sampling diagram in Fig. 8 covers 867 admissions.",
])
def test_binding_filter_boundary_keeps_abbreviated_rejections_intact(rejected):
    from easyicu.research_agent.reporting.manuscript_audit_repair import sentence_spans

    kept = "The intake records were incomplete."
    text = f"{rejected} {kept}"
    result = filter_evidence_bound_scaffold(text, resolve_claim=lambda _: None, resolve_evidence=lambda _: False)
    assert result.filtered_sentences == (rejected,)
    assert result.scaffold == kept + "\n"
    assert tuple(text[span.start:span.end].strip() for span in sentence_spans(text)) == (rejected, kept)


@pytest.mark.parametrize("before, rejected, after", [
    (
        "",
        "### Recruitment window of 867 days {evidence:renal_result}",
        "Enrollment windows were prespecified.",
    ),
    (
        "| Field | Source |\n| --- | --- |\n| Referral | Intake forms |\n",
        "| Recruitment window | 867 days {evidence:renal_result} |",
        "| Selection | Consecutive |\nEnrollment windows were prespecified.",
    ),
])
def test_numeric_filter_removes_rejected_markdown_from_the_repair_source(
    tmp_path, strict_writer_run, before, rejected, after,
):
    run, client, _, _ = strict_writer_run
    opener = "Potassium measurements can inform the design of prognostic research."
    initial = _STRICT_GOOD.replace(opener, f"{opener}\n\n{before}{rejected}\n{after}")
    response = f"## Introduction\n\n{opener}\n\n{before}{after}"
    _, binding = run(initial, [response])
    assert binding.manuscript_critique.status == "pass"
    assert len(client.calls) == 1
    prompt = "\n".join(message.content for message in client.calls[0][0])
    assert "Recruitment window" not in prompt
    assert before + after in prompt
    numeric_filtered = (tmp_path / "manuscript_scaffold_numeric_filtered.md").read_text()
    assert before + after in numeric_filtered
    assert _section(binding.bound, "Audit") == "1 sentence(s) were removed for unbindable values."


@pytest.mark.parametrize(("before", "rejected", "after"), [
    (
        "| Measure | Value |\r\n| --- | --- |\r\n"
        "| Discrimination | {manuscript_value:auroc} |\r\n",
        "| Review interval | 867 days {evidence:renal_result} |\r\n",
        "| Sampling | Consecutive |\r\nThe registry retained complete intake forms.\r\n",
    ),
    (
        "",
        "### Review interval of 867 days {evidence:renal_result}\n",
        "The discrimination estimate was {manuscript_value:auroc}.  \n",
    ),
    (
        "### Intake forms\n",
        "The log recorded 867 referrals\n",
        "| Sampling | Consecutive |\nThe registry retained complete intake forms.\n",
    ),
])
def test_numeric_removal_preserves_adjacent_markdown_and_prose(strict_writer_run, before, rejected, after):
    from easyicu.research_agent.reporting.manuscript_audit_repair import (
        numeric_removal_feedback, scaffold_without_removed_sentences,
    )
    from easyicu.research_agent.reporting.manuscript_post import drop_untraceable_numeric_sentences

    *_, store = strict_writer_run
    prefix = "## Results\n\n"
    suffix = "## Discussion\n\nThe archives were incomplete.\n"
    scaffold = prefix + before + rejected + after + suffix
    citation = "{evidence:renal_result}"
    bound = scaffold.replace(citation, store.bind_manuscript(citation))
    spans = []
    filtered, removed = drop_untraceable_numeric_sentences(bound, evidence=store, removal_spans=spans)
    expected = prefix + before + after + suffix
    assert filtered == expected
    assert len(removed) == 1

    _, drops, feedback = numeric_removal_feedback(bound, spans)
    assert len(drops) == len(feedback) == 1
    assert feedback[0].path == "results"
    assert scaffold_without_removed_sentences(scaffold, drops) == expected


@pytest.mark.parametrize("markup", [
    "{evidence:renal_result}",
    "[renal_result](evidence/ref.md)",
    "[^claim_7]",
    "<!-- interim. still annotation -->",
    "[@local_reference]",
])
def test_numeric_removal_uses_the_same_sentence_as_audit(strict_writer_run, markup):
    from easyicu.research_agent.reporting.manuscript_audit_repair import (
        numeric_removal_feedback, scaffold_without_removed_sentences, sentence_spans,
    )
    from easyicu.research_agent.reporting.manuscript_post import drop_untraceable_numeric_sentences

    *_, store = strict_writer_run
    rejected = f"Mora et al. {markup} documented 867 referrals."
    kept = "The archives were incomplete."
    bound = f"## Introduction\n\n{rejected} {kept}\n"
    spans = []
    filtered, removed = drop_untraceable_numeric_sentences(bound, evidence=store, removal_spans=spans)
    assert rejected == removed[0]["sentence"]
    assert filtered.strip() == f"## Introduction\n\n{kept}"
    assert tuple(bound[span.start:span.end].strip() for span in sentence_spans(bound)) == (rejected, kept)
    _, drops, feedback = numeric_removal_feedback(bound, spans)
    assert len(drops) == len(feedback) == 1
    assert scaffold_without_removed_sentences(bound, drops).strip() == filtered.strip()


@pytest.mark.parametrize("separator", [" ", "\n", "\r\n  "])
def test_evidence_filter_keeps_a_cited_comparison_across_wraps(separator):
    comparison = f"The intake logs recorded 812 vs.{separator}913 referrals {{evidence:intake_registry}}."
    scaffold = f"## Introduction\n\n{comparison}\n"
    result = filter_evidence_bound_scaffold(
        scaffold, resolve_claim=lambda _: None, resolve_evidence={"intake_registry"}.__contains__,
    )
    assert result.filtered_sentences == ()
    assert result.scaffold == scaffold


@pytest.mark.parametrize("separator", [" ", "\n"])
def test_evidence_filter_keeps_citation_after_abbreviated_author(separator):
    sentence = f"867 referrals were logged by Mora et al.{separator}{{evidence:intake_registry}} at intake."
    scaffold = f"## Introduction\n\n{sentence}\n"
    result = filter_evidence_bound_scaffold(
        scaffold, resolve_claim=lambda _: None, resolve_evidence={"intake_registry"}.__contains__,
    )
    assert result.filtered_sentences == ()
    assert result.scaffold == scaffold


def test_rejected_second_level_heading_preserves_the_section_owner(strict_writer_run):
    from easyicu.research_agent.reporting.manuscript_audit_repair import (
        numeric_removal_feedback, scaffold_without_removed_sentences,
    )
    from easyicu.research_agent.reporting.manuscript_post import drop_untraceable_numeric_sentences

    *_, store = strict_writer_run
    scaffold = (
        "## Methods\n\nThe catalogue used intake records.\n\n"
        "## Findings: 867 referrals {evidence:renal_result}\n"
        "The registry retained complete intake forms.\n\n"
        "## Discussion\n\nThe archives were incomplete.\n"
    )
    bound = store.bind_manuscript(scaffold)
    spans = []
    filtered, _ = drop_untraceable_numeric_sentences(bound, evidence=store, removal_spans=spans)
    _, drops, feedback = numeric_removal_feedback(bound, spans)
    assert [item.path for item in feedback] == ["results"]
    expected = scaffold.replace("## Findings: 867 referrals {evidence:renal_result}", "## Findings")
    assert filtered == store.bind_manuscript(expected)
    assert scaffold_without_removed_sentences(scaffold, drops) == expected


def test_rejected_second_level_heading_without_a_section_owner_stops(strict_writer_run):
    from easyicu.research_agent.reporting.manuscript_audit_repair import LOSS_STOP
    from easyicu.research_agent.reporting.manuscript_post import drop_untraceable_numeric_sentences

    *_, store = strict_writer_run
    manuscript = "## Intake catalogue of 867 referrals\nThe registry used complete forms.\n"
    with pytest.raises(EvidenceEnforcementError, match=LOSS_STOP) as stop:
        drop_untraceable_numeric_sentences(manuscript, evidence=store)
    assert stop.value.detail["code"] == LOSS_STOP


def test_numeric_filter_maps_rewritten_opener_back_to_the_repair_source(strict_writer_run):
    run, client, _, _ = strict_writer_run
    opener = "Interpretation is limited to the studied population."
    initial = _STRICT_GOOD.replace(opener, (
        "The validation sample comprised 711 admissions {evidence:retired_enrollment}.\n\n"
        "The screening total was therefore 867 {evidence:renal_result}.\n\n"
        "Sampling was prespecified. "
        f"{opener}"
    ))
    response = f"## Discussion\n\n{opener} Sampling was prespecified."
    _, binding = run(initial, [response])
    assert binding.manuscript_critique.status == "pass"
    assert len(client.calls) == 1
    prompt = "\n".join(message.content for message in client.calls[0][0])
    assert "screening total" not in prompt
    assert _section(binding.bound, "Audit") == "2 sentence(s) were removed for unbindable values."


@pytest.mark.parametrize("heading", ["结果", "Findings"])
def test_binding_filter_routes_recognized_results_headings_to_the_writer(strict_writer_run, heading):
    run, client, _, _ = strict_writer_run
    initial = _STRICT_GOOD.replace("## Results\n", f"## {heading}\n").replace(
        _OUTCOME, "The event count was 867 {evidence:renal_summary}."
    )
    response = "## Results\n\n" + _section(_STRICT_GOOD, "Results")
    draft, binding = run(initial, [response])
    assert binding.manuscript_critique.status == "pass"
    assert len(client.calls) == 1
    assert "The AUROC was {manuscript_value:" in client.calls[0][0][1].content
    assert f"## {heading}\n" not in draft.scaffold
    assert "event count" not in draft.scaffold
    assert _section(binding.bound, "Audit") == "1 sentence(s) were removed for unbindable values."



def test_claim_token_with_extra_numeric_prose_stops_before_expansion(tmp_path, ra):
    from ..authority.test_scientific_claim_authority import _store_with_association_claim

    store = _store_with_association_claim(ra, tmp_path)
    claim = store.scientific_claims()[0]
    prose = f"{claim.placeholder} and the ward-transfer tally included 867 admissions."
    with pytest.raises(EvidenceEnforcementError, match="lacks deterministic") as stop:
        store.enforce_evidence_bound_scaffold(f"## Discussion\n\n{prose}")
    assert stop.value.detail["unsupported_scientific_claim_sentences"] == [prose]


def test_binding_repair_revalidates_sections_outside_the_rewrite(tmp_path, strict_writer_run, monkeypatch):
    from dataclasses import replace

    run, client, _, _ = strict_writer_run
    initial = _STRICT_GOOD.replace(_OUTCOME, _UNBOUND)
    response = "## Results\n\n" + _section(_STRICT_GOOD, "Results")
    real_bind = write_phase._bind_and_review_manuscript
    calls = []

    def bind(*args, **kwargs):
        binding = real_bind(*args, **kwargs)
        calls.append(binding)
        if len(calls) == 2:
            binding = replace(binding, bound=binding.bound.replace(
                "Independent validation will require a separate cohort.", "",
            ))
        return binding

    monkeypatch.setattr(write_phase, "_bind_and_review_manuscript", bind)
    _, binding = run(initial, [response])
    assert len(calls) == 3 and len(client.calls) == 2
    assert _section(binding.bound, "Limitations") == _section(_STRICT_GOOD, "Limitations")
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    rejected, accepted = receipt["attempts"]
    assert rejected["status"] == "rejected" and accepted["status"] == "pass"
    assert any(f["code"] == "MANUSCRIPT_REPAIR_REGRESSION"
               and f["path"] == "limitations" for f in rejected["findings"])


def test_stale_placeholder_on_a_bound_sentence_records_zero_removed_sentences(tmp_path, strict_writer_run):
    from easyicu.research_agent.reporting.manuscript_audit_repair import EVIDENCE_REMOVAL

    run, client, _, _ = strict_writer_run
    initial = _STRICT_GOOD.replace(_OUTCOME, _OUTCOME.replace(".", " {evidence:retired_summary}."))
    response = "## Results\n\n" + _section(_STRICT_GOOD, "Results")
    _, binding = run(initial, [response])
    assert len(client.calls) == 1
    assert binding.manuscript_critique.status == "pass"
    assert _section(binding.bound, "Audit") == "0 sentence(s) were removed for unbindable values."
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert receipt["attempts"][0]["feedback"] == [{
        "code": EVIDENCE_REMOVAL, "path": "results",
        "facts": {"evidence_id": "retired_summary", "sentence_removed": False},
    }]
    assert "retired_summary" not in (tmp_path / "manuscript_scaffold.md").read_text()


def test_resume_retains_the_stale_evidence_loss_guard(tmp_path, strict_writer_run):
    from easyicu.research_agent.reporting.manuscript_audit_repair import LOSS_STOP

    run, client, _, _ = strict_writer_run
    initial = _STRICT_GOOD.replace(
        "The cohort comprised 120 ICU stays {evidence:renal_summary}.\n\n### Model performance",
        "The cohort comprised 992 ICU stays {evidence:retired_summary}.\n\n### Model performance",
    ).replace(_OUTCOME, "The outcome count was 991 {evidence:retired_summary}.")
    for scaffold in (initial, None):
        with pytest.raises(EvidenceEnforcementError, match=LOSS_STOP) as stop:
            run(scaffold, [])
        assert stop.value.detail["losses"][0]["facts"]["sentences_removed"] == 2
        assert "retired_summary" in (tmp_path / "manuscript_scaffold.md").read_text()
    assert client.calls == []


@pytest.mark.parametrize("keep_other_section", [False, True])
def test_dh4_shape_repairs_numeric_and_stale_evidence_removals_with_repeated_sentence(
    tmp_path, strict_writer_run, keep_other_section,
):
    from easyicu.research_agent.reporting.manuscript_audit_repair import EVIDENCE_REMOVAL, NUMERIC_REMOVAL

    run, client, findings, _ = strict_writer_run
    repeated = "Potassium measurements can inform the design of prognostic research."
    stale = "The outcome count was 777 {evidence:retired_summary}."
    initial = _STRICT_GOOD.replace(_OUTCOME, _UNBOUND).replace(
        "## Discussion\n\n", f"## Discussion\n\n{repeated}\n\n",
    ).replace(
        "The AUROC was 0.74 {evidence:renal_summary}.",
        "The AUROC was 0.74 {evidence:renal_summary}.\n\n" + stale,
    )
    results = "## Results\n\n" + _section(_STRICT_GOOD, "Results")
    if not keep_other_section:
        results += "\n\n## Limitations\n\n"
    draft, binding = run(initial, [results, "## Discussion\n\n" + _section(_STRICT_GOOD, "Discussion")])
    assert binding.manuscript_critique.status == "pass"
    assert len(client.calls) == 2
    assert _section(draft.scaffold, "Limitations") == _section(GOOD, "Limitations")
    assert _section(binding.bound, "Audit") == "2 sentence(s) were removed for unbindable values."
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    attempt, = receipt["attempts"]
    assert set(attempt["section_keys"]) == {"results", "discussion"}
    assert {"MANUSCRIPT_SUBSECTION_MISSING_OR_EMPTY", "MANUSCRIPT_REPEATED_SENTENCE",
            EVIDENCE_REMOVAL, NUMERIC_REMOVAL} <= set(attempt["audit_codes"])
    assert any(item["facts"].get("evidence_id") == "retired_summary" for item in attempt["feedback"])
    assert any((f.detail or {}).get("reason_code") == EVIDENCE_REMOVAL for f in findings)
    assert "777" not in "\n".join(message.content for call in client.calls for message in call[0])


@pytest.mark.parametrize("headline", [False, True])
def test_binding_filter_loss_stops_before_writer(tmp_path, strict_writer_run, monkeypatch, headline):
    from easyicu.research_agent.reporting.manuscript_audit_repair import LOSS_STOP
    from easyicu.research_agent.reporting.report_fact import DescriptiveReportFact

    run, client, _, store = strict_writer_run
    if headline:
        source = store.get("renal_summary")
        fact = DescriptiveReportFact(
            subsection="Model performance", text="The AUROC was 0.74", evidence_id=source.evidence_id,
            source_sha256=source.sha256, source_fields=("auroc",), required_result_sections=("Results",),
            step_id="renal_model", product_id="table:performance",
        )
        monkeypatch.setattr(write_phase, "compile_primary_counts_only_report_facts", lambda *args, **kwargs: (fact,))
        initial = _STRICT_GOOD.replace(
            "The AUROC was 0.74 {evidence:renal_summary}.",
            "The AUROC was 0.74 and calibration slope was 991 {evidence:renal_summary}.",
        )
    else:
        initial = _STRICT_GOOD.replace(_OUTCOME, _UNBOUND).replace(
            "The cohort comprised 120 ICU stays {evidence:renal_summary}.\n\n### Model performance",
            "The cohort comprised 992 ICU stays {evidence:renal_summary}.\n\n### Model performance",
        )
    with pytest.raises(EvidenceEnforcementError, match=LOSS_STOP) as stop:
        run(initial, [])
    assert client.calls == []
    loss, = stop.value.detail["losses"]
    assert loss["path"] == "results"
    assert loss["facts"]["headline_result_removed"] is headline
    assert loss["facts"]["more_than_half_removed"] is not headline
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert receipt["status"] == "non_repairable" and receipt["attempts"] == []


def _exercise(tmp_path, monkeypatch, bad, responses, *, changed_bindings=False, initial_findings=(), baseline_mentions=None, definition_labels=None, grouping=None, full_draft=None):
    context = _renal_context()
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
        elif full_draft is not None:
            text = full_draft
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
    assert receipt["status"] == ("rejected" if change == "evidence" else "exhausted")
    assert len(receipt["attempts"]) == (1 if change == "evidence" else 3)
    assert all(row["status"] == "rejected" for row in receipt["attempts"])


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


@pytest.mark.parametrize("drop_token", [False, True])
def test_final_repair_splices_results_into_the_original_scaffold(tmp_path, monkeypatch, drop_token):
    original = GOOD.replace(
        "Interpretation is limited to the studied population.",
        "Interpretation is limited to the studied population.  \n\nOriginal follow-up.",
    ).replace("The AUROC was 0.74", "See Figure 1")
    candidate = GOOD.replace(
        "Interpretation is limited to the studied population.",
        "There were 991 deaths {evidence:unverified}.",
    )
    if drop_token:
        candidate = candidate.replace(
            "The AUROC was 0.74 {evidence:renal_summary}.", "The AUROC was 0.74.",
        )
        with pytest.raises(EvidenceEnforcementError, match="existing evidence or claim binding"):
            _exercise(tmp_path, monkeypatch, original, [], full_draft=candidate)
    else:
        (draft, binding), _, _, bindings, _ = _exercise(tmp_path, monkeypatch, original, [], full_draft=candidate)
        assert binding.manuscript_critique.status == "pass"
        assert draft.scaffold == bindings[-1]
        assert draft.scaffold.split("## Discussion", 1)[1] == original.split("## Discussion", 1)[1]
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert receipt["attempts"][0]["spliced_section_keys"] == ["results"]
    if drop_token:
        assert receipt["attempts"][0]["authority_failure"]["check"] == "section_tokens"


@pytest.mark.parametrize("recovers", [False, True])
def test_a_repair_candidate_with_new_filter_removals_is_still_rejected(tmp_path, monkeypatch, recovers):
    bad = GOOD.replace("The AUROC was 0.74 {evidence:renal_summary}.", "See Figure 1 {evidence:renal_summary}.")
    good = "## Results\n\n" + _section(GOOD, "Results")
    unsupported = good + "\n\nThere were 991 deaths {evidence:renal_summary}."
    options = {"definition_labels": {"age=older": "patients aged 65 years or older"}}
    if recovers:
        (_, binding), client, _, _, _ = _exercise(tmp_path, monkeypatch, bad, [unsupported, good], **options)
        assert binding.manuscript_critique.status == "pass"
        assert "991" not in binding.bound
        assert "MANUSCRIPT_AUDIT_NON_REPAIRABLE" in client.calls[1][0][-1].content
    else:
        with pytest.raises(EvidenceEnforcementError, match="MANUSCRIPT_AUDIT_NON_REPAIRABLE"):
            _exercise(tmp_path, monkeypatch, bad, [unsupported], **options)
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert receipt["status"] == ("pass" if recovers else "exhausted")
    assert [row["status"] for row in receipt["attempts"]] == (
        ["rejected", "pass"] if recovers else ["rejected"] * 3
    )


def test_adjustment_repair_cannot_drop_a_definition_written_in_scientific_notation(tmp_path, monkeypatch):
    label = "potassium at least 1 × 10⁻³ mmol/L"
    bad = GOOD.replace("The executed analysis assessed discrimination",
                       "The adjustment set comprised age and severity, using " + label)
    bad = bad.replace("The AUROC was", "After adjustment for age and sex, the AUROC was")
    response = "## Methods\n\n" + _section(bad, "Methods").replace("age and severity, using " + label, "age and sex")
    with pytest.raises(EvidenceEnforcementError, match="MANUSCRIPT_REPAIR_AUTHORITY_CHANGED"):
        _exercise(tmp_path, monkeypatch, bad, [response], definition_labels={"potassium=low": label})
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert all("existing bound number or source changed" in row["findings"][0]["facts"]["message"]
               for row in receipt["attempts"])


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


@pytest.mark.parametrize(("writer_action", "exposure", "outcome", "group"), [
    ("add", "sepsis", "mortality", "Sepsis"),
    ("omit", "sepsis", "mortality", "Sepsis"),
    ("no_fact", "sepsis", "mortality", "Sepsis"),
    ("add", "aki", "prolonged_length_of_stay", "Acute kidney injury"),
])
def test_final_repair_covers_the_distribution_question(
    tmp_path, monkeypatch, writer_action, exposure, outcome, group,
):
    from ..authority.test_scientific_claim_authority import _descriptive_distribution_summary
    from easyicu.research_agent.audits.envelope_consumers import RegisteredOutputEnvelopeConsumer

    source_id = "distribution_summary"
    product = "table:exposure_outcome_distribution"
    summary = _descriptive_distribution_summary()
    summary.update(status="ok", adjusted_effect=None, exposure=exposure, outcome=outcome,
                   interval_method="patient_cluster_robust_wald")
    summary["descriptive_estimates"]["exposure_prevalence"] = [] if writer_action == "no_fact" else [
        {
            **{k: v for k, v in level.items() if k != "events"},
            "n": 100, "denominator": 200, "estimate_pct": 50.0,
            "ci_low_pct": 46.080072, "ci_high_pct": 53.919928,
        }
        for level in summary["descriptive_estimates"]["outcome_absolute_risks"]
    ]
    store = EvidenceStore(tmp_path, enforcement_mode=EvidenceEnforcementMode.STRICT)
    store.register_json(kind="statistic", description="Synthetic clustered distribution", payload=summary,
                        filename="distribution.json", evidence_id=source_id, produced_by_step="distribution",
                        generation_mode="deterministic_standard")
    store.register_step_summary_numerics(step_id="distribution", evidence_id=source_id, summary=summary)
    store.register_step_summary_scientific_claims(step_id="distribution", evidence_id=source_id, summary=summary)
    records = [{"step_id": "distribution", "status": "ok", "step_summary": summary,
                "step_summary_evidence_id": source_id, "evidence_ids": [source_id]}]
    monkeypatch.setattr(RegisteredOutputEnvelopeConsumer, "authoritative_writer_records",
                        lambda self, records, **kwargs: records)
    context = ResearchContext.model_validate({
        "research_question": "Describe the synthetic distribution.",
        "cohort": {"cohort_name": "synthetic", "database": "miiv", "n_stays": 200},
        "primary_exposure": exposure, "target_outcome": outcome,
        "variables": [{"name": exposure, "dtype": "bool", "description": group},
                      {"name": outcome, "dtype": "bool", "description": outcome.replace("_", " ")}],
    })
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_epidemiology",
        display_labels={f"{exposure}=0": f"No {group.lower()}", f"{exposure}=1": group,
                        outcome: outcome.replace("_", " ")},
        steps=[AnalysisStep(
            step_id="distribution", intent="Describe the exposure and outcome.",
            planned_analysis_role="primary", expected_outputs=[product], inputs=[exposure, outcome],
            exposure_outcome_distribution_spec={
                "exposure": exposure, "exposure_levels": [0, 1],
                "outcome": outcome, "outcome_levels": [0, 1], "outcome_positive_value": 1,
                "level_match_policy": "exact_typed", "denominator_policy": "all_declared_rows",
                "missing_outcome_policy": "fail_closed", "confidence_level": 0.95,
            },
        )],
    )
    claims = store.scientific_claims()
    tokens = "\n\n".join(claim.placeholder for claim in claims)
    initial = (GOOD.replace("The cohort comprised 120 ICU stays", "The analysis cohort comprised 200 ICU stays")
               .replace("renal_summary", source_id)
               .replace("These estimates describe predictive discrimination within the study population.",
                        "\n\n" + claims[-1].placeholder)
               .replace("### Model performance\nThe AUROC was 0.74 {evidence:distribution_summary}.",
                        "### Descriptive results\n" + tokens)
               .replace("\n### Secondary analyses\nThe outcome count was 15 {evidence:distribution_summary}.\n", "")
               .replace("The evaluation addresses discrimination for renal readmission within the study population.",
                        claims[-1].placeholder))
    try:
        store.enforce_evidence_bound_scaffold(initial, per_step_records=records)
    except EvidenceEnforcementError as exc:
        pytest.fail(str(exc.detail))
    client = ScriptedMockLLMClient([], repeat_last=True)
    writer = WriterAgent(client)
    drafts = []

    def draft(*_args, section_repair, **_kwargs):
        drafts.append(section_repair)
        scaffold = initial
        if section_repair:
            response = "## Results\n\n" + _section(initial, "Results")
            if writer_action == "add":
                feedback = " ".join(section_repair[1]["results"])
                token = re.search(r"\{claim:distribution\.[^}]+\}", feedback).group()
                response = response.replace("### Descriptive results", "### Descriptive results\n\n" + token)
            client.responses.append(response)
            try:
                scaffold, _ = writer.repair_sections(
                    section_repair[0], section_errors=section_repair[1], context=context,
                    evidence_ids=(source_id,), analysis_plan=plan,
                )
            except Exception as exc:
                candidate = completed_section_repair_candidate(exc, expected_section_keys=tuple(section_repair[1]))
                if candidate is None:
                    raise
                scaffold = candidate[0]
        return write_phase._DraftStageResult((), (source_id,), None, None, scaffold)

    real_draft = write_phase._draft_manuscript
    monkeypatch.setattr(write_phase, "_draft_manuscript", draft)
    monkeypatch.setattr(write_phase, "_persist_reader_artifacts", lambda **kwargs: None)
    findings = [ValidationFinding(
        validator="writer_question_coverage", severity="error", message="Prior draft coverage failure.",
        detail={"step_id": "distribution", "reason_code": "writer_question_part_uncovered"},
    )]
    critique_pass = CritiqueReport(reviewer="scripted", status="pass")
    draft, binding = write_phase._draft_bind_and_repair_manuscript(
        SimpleNamespace(_evidence_enforcement_mode=EvidenceEnforcementMode.STRICT),
        context=context, agent_context=context, evidence=store, findings=findings, literature=None,
        per_step_records=records, plan_result=SimpleNamespace(resume_state=None),
        execute_result=SimpleNamespace(plan=plan), critic=SimpleNamespace(review_manuscript=lambda **_: critique_pass),
        role_resolver=None, prompt_version="test", runtime_state=None, run_dir=tmp_path,
        run_id="synthetic", run_language="en", writer_probe_mode=False, writer_probe_failed_steps=(),
        emit_progress=lambda *_args, **_kwargs: None,
    )
    audit = json.loads((tmp_path / "manuscript_quality_audit.json").read_text())
    assert audit["status"] == "pass"
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    active, _, _ = write_phase.current_validation_findings(
        plan=plan, per_step_records=records, findings=findings, evidence=store,
        run_dir=tmp_path, manuscript_text=binding.bound, context=context,
    )
    coverage = [f for f in active if f.validator == "writer_question_coverage"]
    if writer_action == "no_fact":
        assert len(drafts) == 1
        assert binding.manuscript_critique.status == "blocked"
        assert any("writer_question_coverage" in c for c in binding.manuscript_critique.concerns)
        assert all("quality audit" not in c for c in binding.manuscript_critique.concerns)
        assert [f.detail["outcome"] for f in coverage] == ["exposure_distribution"]
        assert coverage[0].detail["reason_code"] == "writer_question_part_uncovered"
        assert receipt["status"] == "non_repairable"
        assert receipt["remaining_codes"] == ["writer_question_part_uncovered"]
        return

    assert len(client.calls) == (1 if writer_action == "add" else 3)
    assert binding.manuscript_critique.status == "pass" and coverage == []
    assert not any(f.validator in {"writer_question_coverage", "critic_agent"} for f in active)
    assert binding.numeric_binding_map and not binding.untraced_numerics
    assert "50.00%" in binding.bound
    assert group in _section(binding.bound, "Results")
    assert "{claim:" in draft.scaffold and "{claim:" not in binding.bound
    assert receipt["status"] == "pass"
    first = receipt["attempts"][0]
    assert first["audit_codes"] == ["writer_question_part_uncovered"]
    feedback = first["feedback"][0]["facts"]
    assert (feedback["step_id"], feedback["product_id"], feedback["outcome"]) == (
        "distribution", product, "exposure_distribution",
    )
    fact, = feedback["repair_facts"]
    assert fact["claim_placeholder"] in draft.scaffold
    assert group in fact["text"]
    prompt = client.calls[0][0][-1].content
    assert all(part in prompt for part in ("distribution", product, "exposure_distribution", group, fact["claim_placeholder"]))
    assert (tmp_path / "manuscript_scaffold.md").read_text() == draft.scaffold
    if writer_action == "omit":
        fallback = receipt["host_fallback"]
        assert fallback["status"] == "pass"
        assert fallback["numeric_checks"] == fallback["evidence_checks"] == "pass"
        assert all(row["status"] == "unresolved" for row in receipt["attempts"])
        assert draft.scaffold.count(fact["claim_placeholder"]) == 1
        resumed_findings = []
        resumed_draft = real_draft(
            _strict_pipeline(), context=context, agent_context=context, evidence=store,
            findings=resumed_findings, literature=None, per_step_records=records,
            resume_state={"per_step_records": records}, prompt_version="test",
            role_resolver=lambda _: client, runtime_state=SimpleNamespace(semantics=None),
            execute_result=SimpleNamespace(plan=plan), run_dir=tmp_path, run_id="synthetic",
            run_language="en", emit_progress=lambda *_args, **_kwargs: None,
        )
        assert len(client.calls) == 3
        assert fact["claim_placeholder"] in resumed_draft.scaffold
        assert any(f.validator == "writer_resume" for f in resumed_findings)
        resumed_binding = write_phase._bind_and_review_manuscript(
            _strict_pipeline(), critic=SimpleNamespace(review_manuscript=lambda **_: critique_pass),
            evidence=store, findings=resumed_findings, literature=None, per_step_records=records,
            current_evidence_names=resumed_draft.current_evidence_names, scaffold=resumed_draft.scaffold,
            writer_error_message=resumed_draft.writer_error_message, writer_probe_mode=False,
            writer_probe_failed_steps=(), run_dir=tmp_path, reader_display_labels=plan.display_labels,
            manuscript_language="en", context=context, plan=plan, reader_tables=resumed_draft.reader_tables,
        )
        assert resumed_binding.manuscript_critique.status == "pass"
        assert not resumed_binding.question_coverage_findings
    else:
        assert "host_fallback" not in receipt


@pytest.mark.parametrize("keep_protected_tokens", [False, True])
def test_methods_adjustment_repair_checks_both_protected_sources(tmp_path, monkeypatch, keep_protected_tokens):
    repeated = "The hospital archive supplied the index forms for this retrospective assessment."
    original = GOOD.replace("## Discussion\n\n", f"## Discussion\n\n{repeated}\n\n").replace(
        "## Conclusion\n\n", f"## Conclusion\n\n{repeated}\n\n",
    ).replace(
        "The executed analysis assessed discrimination",
        "The adjustment set comprised age and severity",
    ).replace(
        "Analyses used registered software artifacts {evidence:renal_summary}.",
        "Analyses used registered software artifacts {evidence:other_summary}.",
    ).replace("The AUROC was", "After adjustment for age and sex, the AUROC was")
    methods = "## Methods\n\n" + _section(original, "Methods").replace("age and severity", "age and sex")
    conclusion = "## Conclusion\n\n" + _section(GOOD, "Conclusion")
    if keep_protected_tokens:
        (_, binding), _, _, _, _ = _exercise(tmp_path, monkeypatch, original, [methods, conclusion])
        assert binding.manuscript_critique.status == "pass"
    else:
        methods = methods.replace("sex {evidence:renal_summary}", "sex").replace(" {evidence:other_summary}", "")
        with pytest.raises(EvidenceEnforcementError, match="existing evidence or claim binding") as stop:
            _exercise(tmp_path, monkeypatch, original, [methods, conclusion])
        assert stop.value.detail["missing_tokens"] == {
            "{evidence:renal_summary}": 1, "{evidence:other_summary}": 1,
        }
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    attempt, = receipt["attempts"]
    assert "methods" in attempt["section_keys"]
    assert {"MANUSCRIPT_ADJUSTMENT_SET_CONFLICT", "MANUSCRIPT_REPEATED_SENTENCE"} <= set(attempt["audit_codes"])
    if keep_protected_tokens:
        assert attempt["status"] == "pass"
    else:
        assert attempt["authority_failure"]["check"] == "section_tokens"
        assert attempt["authority_failure"]["section"] == "methods"
        assert attempt["numeric_checks"] == "pending"


def test_callout_repair_keeps_methods_bytes_despite_whole_draft_rendering(tmp_path, monkeypatch):
    from easyicu.research_agent.reporting.manuscript_audit_repair import _section_blocks
    from easyicu.research_agent.reporting.manuscript_quality import repair_reader_internal_phrases

    repeated = "The hospital archive supplied the index forms for this retrospective assessment."
    original = GOOD.replace("## Discussion\n\n", f"## Discussion\n\n{repeated}\n\n").replace(
        "## Conclusion\n\n", f"## Conclusion\n\n{repeated}\n\n",
    ).replace(
        "The AUROC was 0.74 {evidence:renal_summary}.", "See Figure 1 {evidence:renal_summary}.",
    ).replace("The executed analysis assessed discrimination", "The executed analysis used intake forms")
    candidate = original.replace("See Figure 1 {evidence:renal_summary}.",
                                 "The AUROC was 0.74 {evidence:renal_summary}.")
    candidate = candidate.replace(f"## Conclusion\n\n{repeated}\n\n", "## Conclusion\n\n")
    candidate, _ = repair_reader_internal_phrases(
        candidate, reader_display_labels={"intake forms": "abstracted intake forms"},
    )
    candidate = candidate.replace("abstracted intake forms {evidence:renal_summary}", "abstracted intake forms")
    methods = dict(_section_blocks(original))["methods"]
    assert dict(_section_blocks(candidate))["methods"] != methods
    (draft, binding), _, _, _, _ = _exercise(tmp_path, monkeypatch, original, [], full_draft=candidate)
    assert binding.manuscript_critique.status == "pass"
    assert dict(_section_blocks(draft.scaffold))["methods"] == methods
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    attempt, = receipt["attempts"]
    assert attempt["status"] == "pass"
    assert "methods" not in attempt["section_keys"]
    assert {"MANUSCRIPT_REPEATED_SENTENCE", "MANUSCRIPT_RESULT_SUBSECTION_CALLOUT_ONLY"} <= set(attempt["audit_codes"])


@pytest.mark.parametrize("citation", ["[^entry_a]", "{evidence:renal_result}", "[intake notes](references/intake.md)"])
def test_numeric_filter_removes_the_dependent_sentence_citation(strict_writer_run, citation):
    from easyicu.research_agent.reporting.manuscript_audit_repair import (
        numeric_removal_feedback, scaffold_without_removed_sentences,
    )
    from easyicu.research_agent.reporting.manuscript_post import drop_untraceable_numeric_sentences

    *_, store = strict_writer_run
    prefix = "## Results\n\n"
    source = prefix + f"The warehouse logged 967 crates. It used the dispatch register.{citation}\n"
    ranges = []
    filtered, _ = drop_untraceable_numeric_sentences(source, evidence=store, removal_spans=ranges)
    _, drops, _ = numeric_removal_feedback(source, ranges)
    assert filtered == scaffold_without_removed_sentences(source, drops) == prefix


@pytest.mark.parametrize("fence", ["```python\ncutoff = settings.default.\n```\n", "~~~~text\nfloor = settings.minimum.\n~~~~\n"])
def test_numeric_filter_preserves_the_whole_fence_before_prose(strict_writer_run, fence):
    from easyicu.research_agent.reporting.manuscript_post import drop_untraceable_numeric_sentences

    *_, store = strict_writer_run
    prefix = "## Methods\n\n" + fence
    suffix = "\n## Results\n"
    source = prefix + "The archive held 967 ledgers {evidence:renal_result}.\n" + suffix
    filtered, _ = drop_untraceable_numeric_sentences(source, evidence=store)
    assert filtered == prefix + suffix


@pytest.mark.parametrize(("before", "rejected", "after"), [
    (
        "- Source: dispatch register\n",
        "- Delay: 967 days {evidence:renal_result}\n",
        "- Custodian: shipping office\n",
    ),
    (
        "> Source: dispatch register\n",
        "> Delay: 967 days {evidence:renal_result}\n",
        "The shipping office kept the paper copy.\n",
    ),
])
def test_numeric_filter_preserves_siblings_of_a_markdown_block(strict_writer_run, before, rejected, after):
    from easyicu.research_agent.reporting.manuscript_post import drop_untraceable_numeric_sentences

    *_, store = strict_writer_run
    filtered, _ = drop_untraceable_numeric_sentences(before + rejected + after, evidence=store)
    assert filtered == before + after


@pytest.mark.parametrize("second", ["It arrived after dispatch.", "The second warehouse logged 968 crates."])
def test_adjacent_numeric_deletions_consume_whole_line(strict_writer_run, second):
    from easyicu.research_agent.reporting.manuscript_audit_repair import (
        numeric_removal_feedback, scaffold_without_removed_sentences,
    )
    from easyicu.research_agent.reporting.manuscript_post import drop_untraceable_numeric_sentences

    *_, store = strict_writer_run
    prefix = "## Results\n\nThe dispatch register was open.\n\n"
    suffix = "The paper records were archived.\n"
    source = prefix + f"The warehouse logged 967 crates. \t{second}\n" + suffix
    ranges = []
    filtered, _ = drop_untraceable_numeric_sentences(source, evidence=store, removal_spans=ranges)
    _, drops, _ = numeric_removal_feedback(source, ranges)
    assert filtered == scaffold_without_removed_sentences(source, drops) == prefix + suffix


@pytest.mark.parametrize("first", ["**Background:** The warehouse logged 967 vs.", "The warehouse logged 967 vs.",
                                  "The warehouses are compared in Fig."])
def test_claim_policy_keeps_wrapped_numeric_sentence_with_its_citation(first):
    continuation = "968 crates {evidence:dispatch}." if first.startswith("**") else "37. {evidence:dispatch}"
    comparison = f"{first}\n{continuation}"
    source = f"## Introduction\n\n{comparison}\n"
    result = filter_evidence_bound_scaffold(
        source, resolve_claim=lambda _: None, resolve_evidence={"dispatch"}.__contains__,
    )
    assert result.filtered_sentences == ()
    assert result.scaffold == source


@pytest.mark.parametrize("body", [
    "- Cabinet setup\n  ```python\n  source = cabinet.catalog\n\n  export = source\n  ```",
    "- The curator cataloged the notebooks.\n"
    "  - The labels were kept.\n"
    "  - The covers were scanned.",
])
def test_claim_policy_preserves_list_child_block_lines(body):
    source = f"## Introduction\n\n{body}\n\n## Results\nSee Table 1 {{evidence:catalog_summary}}.\n"
    result = filter_evidence_bound_scaffold(
        source, resolve_claim=lambda _: None, resolve_evidence={"catalog_summary"}.__contains__,
    )
    assert result.filtered_sentences == ()
    assert result.scaffold == source


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("citation", ["[^entry_a]", "{evidence:renal_result}", "[ledger](reports/dispatch.md)"])
@pytest.mark.parametrize(("before", "rejected", "after"), [
    ("", "The dépôt held 967 crates. It used an alternate source.<cite>\n",
     "The records are archived.\n"),
    ("| Facility | Notes |\n", "The dépôt held 967 crates.\tThe annex held 968 crates.<cite>\n",
     "| Source | Dispatch register |\n"),
    ("- Dispatch ledger\n", "- The dépôt held 967 crates<cite>\n",
     "- Paper copy\n"),
    ("", "````text\nThe dépôt held 967 crates<cite>\n````\n",
     "## Archive\nThe paper copy remains.\n"),
    ("", "<div>\nThe dépôt held 967 crates<cite>\n</div>\n",
     "## Archive\nThe paper copy remains.\n"),
    ("> Dispatch ledger\n", "> The dépôt held 967 crates<cite>\n",
     "The paper copy is archived.\n"),
    ("````text\ndispatch.\n```\narchive.\n````\n", "The dépôt held 967 crates<cite>.\n",
     "## Archive\nThe paper copy remains.\n"),
    (
        "- Index export\n  ```python\n  source = catalog.entries\n\n  export = source\n  ```\n"
        "\n## Results\n\n",
        "The cabinet contained 941 specimen labels<cite>.\n",
        "\n## Archive\nThe paper index is available.\n",
    ),
    (
        "- Cabinet contents\n  - Original labels\n",
        "  - The cabinet contained 941 specimen labels<cite>\n",
        "  - Scanned catalog\n",
    ),
    ("<div>\nDispatch ledger.\n</div>\n", "The dépôt held 967 crates<cite>\n",
     "## Archive\nThe paper copy remains.\n"),
    ("", "The dépôt held 967 crates. The annex held 968 crates. It had no spare crates.<cite>\n",
     "The paper copy is archived.\n"),
    ("", "The dépôt held\n967 crates.\n<cite>\n", "The paper copy is archived.\n"),
    ("Facility | Notes\n--- | ---\n", "Dépôt | 967 crates<cite>\n", "Annex | Paper copy\n"),
    ("", "The dépôt held 967 crates.\t The annex held 968 crates.<cite>  ",
     "The paper copy is archived.\n"),
    ("The clerk archived a copy.  ", "The dépôt held 967 crates<cite>.",
     "  The paper copy is archived.\n"),
])
def test_numeric_and_repair_edits_preserve_identical_source_bytes(
    strict_writer_run, newline, citation, before, rejected, after,
):
    from easyicu.research_agent.reporting.manuscript_audit_repair import (
        numeric_removal_feedback, scaffold_without_removed_sentences,
    )
    from easyicu.research_agent.reporting.manuscript_post import drop_untraceable_numeric_sentences

    *_, store = strict_writer_run
    prefix = ("## Introduction\n\nDépôt dispatch logs.\n\n" + before).replace("\n", newline)
    rejected = rejected.replace("<cite>", citation).replace("\n", newline)
    suffix = after.replace("\n", newline)
    source = prefix + rejected + suffix
    bound = source.replace(citation, store.bind_manuscript(citation).strip()) if citation.startswith("{evidence:") else source
    removed_end = len(bound) - len(suffix)
    ranges = []
    filtered, _ = drop_untraceable_numeric_sentences(bound, evidence=store, removal_spans=ranges)
    _, drops, _ = numeric_removal_feedback(bound, ranges)
    assert [(start, end) for start, end, _ in ranges] == [(len(prefix), removed_end)]
    assert filtered.encode("utf-8") == (prefix + suffix).encode("utf-8")
    assert scaffold_without_removed_sentences(source, drops).encode("utf-8") == (prefix + suffix).encode("utf-8")


def test_numeric_filter_stops_when_a_rejected_number_has_no_closed_fence(strict_writer_run):
    from easyicu.research_agent.reporting.manuscript_audit_repair import LOSS_STOP
    from easyicu.research_agent.reporting.manuscript_post import drop_untraceable_numeric_sentences

    *_, store = strict_writer_run
    source = "## Introduction\n\n````text\nThe depot recorded 967 deliveries.\n```\n## Archive\n"
    with pytest.raises(EvidenceEnforcementError, match=LOSS_STOP) as stop:
        drop_untraceable_numeric_sentences(source, evidence=store)
    assert stop.value.detail["code"] == LOSS_STOP


def test_numeric_filter_stops_at_an_unmatched_list_fence(strict_writer_run):
    from easyicu.research_agent.reporting.manuscript_audit_repair import LOSS_STOP
    from easyicu.research_agent.reporting.manuscript_post import drop_untraceable_numeric_sentences

    *_, store = strict_writer_run
    source = (
        "- Index export\n  ````text\n  The cabinet contained 941 specimen labels.\n\n  ```\n\n"
        "## Appendix\n\n````text\nThe archive is available.\n````\n"
    )
    ranges = []
    with pytest.raises(EvidenceEnforcementError, match=LOSS_STOP) as stop:
        drop_untraceable_numeric_sentences(source, evidence=store, removal_spans=ranges)
    assert stop.value.detail["code"] == LOSS_STOP
    assert ranges == []


@pytest.mark.parametrize("placeholder", ["{EVIDENCE:dispatch}", "{CLAIM:dispatch.count}"])
def test_strict_authority_checks_include_leading_annotations(tmp_path, placeholder):
    source = f"<!-- {placeholder} --> The shipping office kept the dispatch ledger."
    store = EvidenceStore(tmp_path, enforcement_mode=EvidenceEnforcementMode.STRICT)
    with pytest.raises(EvidenceEnforcementError, match="authority placeholders are malformed"):
        store.bind_manuscript(source)
    result = filter_evidence_bound_scaffold(source, resolve_claim=lambda _: None)
    assert result.filtered_sentences == (source,)
