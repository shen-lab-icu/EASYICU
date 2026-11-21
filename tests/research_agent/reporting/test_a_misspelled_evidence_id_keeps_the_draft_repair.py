"""A misspelled evidence ID defers only its own sentence, never the draft's repair.

150a demo r13 dH3/dA6: each Writer draft cited one evidence ID with a digit
too many or too few. Treating that as a reason to defer the whole STRICT check
skipped the bounded repair pass for every other rejected sentence, so the final
binder failed closed and the run ended diagnostic_only with a 34-word scaffold.
"""

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementError, EvidenceEnforcementMode, EvidenceStore,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.reporting import write_phase
from easyicu.research_agent.reporting.administrative_authority import render_manuscript_administrative_sections
from easyicu.research_agent.reporting.manuscript_audit_repair import (
    EVIDENCE_REMOVAL, UnregisteredEvidence, enforce_outside_stale_evidence, sentence_key,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, CritiqueReport, ResearchContext

DRAFT = """# Potassium and renal readmission in a retrospective ICU cohort

**Keywords:** potassium, kidney, readmission, cohort, prediction

## Abstract

**Background:** Renal readmission is a clinically relevant endpoint.

**Methods:** We assessed the supplied potassium representation.

**Results:** The cohort comprised 120 ICU stays {evidence:renal_summary}.

**Conclusions:** This study describes baseline characteristics.

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

The prespecified analysis was performed.
""" + "\n\n" + render_manuscript_administrative_sections(None)

MISSPELLED = "The outcome count was 15 {evidence:renal_summary} {evidence:renal_summaryy}."
UNCITED = "Renal readmission occurred in 15 of 120 stays."


def _section(text, name):
    return text.split(f"## {name}\n", 1)[1].split("\n## ", 1)[0].strip()


@pytest.fixture
def strict_run(tmp_path, monkeypatch):
    from easyicu.research_agent.audits.envelope_consumers import RegisteredOutputEnvelopeConsumer

    context = ResearchContext.model_validate({
        "research_question": "Assess potassium-based renal readmission prediction.",
        "cohort": {"cohort_name": "synthetic", "database": "miiv", "n_stays": 120},
        "primary_exposure": "potassium", "target_outcome": "readmission",
        "variables": [{"name": "potassium", "dtype": "float64", "description": "Serum potassium"},
                      {"name": "readmission", "dtype": "bool", "description": "Renal readmission"}],
    })
    plan = AnalysisPlan(research_question=context.research_question, analysis_type="prediction_model", steps=[
        AnalysisStep(step_id="renal_model", intent="Assess discrimination", method="descriptive",
                     expected_outputs=["table:performance"], planned_analysis_role="primary"),
    ])
    store = EvidenceStore(tmp_path, enforcement_mode=EvidenceEnforcementMode.STRICT)
    summary = {"cohort_n": 120, "auroc": 0.74, "events": 15}
    store.register_json(kind="statistic", description="Synthetic summary", payload=summary,
                        filename="summary.json", evidence_id="renal_summary", produced_by_step="renal_model")
    records = [{"step_id": "renal_model", "status": "ok", "step_summary": summary,
                "step_summary_evidence_id": "renal_summary", "evidence_ids": ["renal_summary"]}]
    monkeypatch.setattr(RegisteredOutputEnvelopeConsumer, "authoritative_writer_records",
                        lambda self, records, **kwargs: records)

    def no_decision(*_args, **_kwargs):
        raise ValueError("scripted: no valid repair decision")

    # The bounded pass falls back to dropping exactly the sentences STRICT names.
    monkeypatch.setattr(write_phase, "decide_writer_evidence_repairs", no_decision)
    findings = []
    client = ScriptedMockLLMClient([], repeat_last=True)
    pipeline = SimpleNamespace(
        _evidence_enforcement_mode=EvidenceEnforcementMode.STRICT,
        _max_numeric_claims_per_step=0, _enable_nature_writing_skill=False,
        _user_extension_activation=SimpleNamespace(writing_advisory=None),
        _writer_digest_widened=False, _writer_digest_secondary_cap_per_step=0,
    )

    def run(scaffold, responses):
        client.responses.extend(responses)
        write_phase._store_raw_scaffold(tmp_path, scaffold, evidence=store, prompt_version="test")
        return write_phase._draft_bind_and_repair_manuscript(
            pipeline, context=context, agent_context=context, evidence=store,
            findings=findings, literature=None, per_step_records=records,
            plan_result=SimpleNamespace(resume_state={"per_step_records": records}),
            execute_result=SimpleNamespace(plan=plan), critic=SimpleNamespace(
                review_manuscript=lambda **_: CritiqueReport(reviewer="scripted", status="pass"),
            ), role_resolver=lambda _: client, prompt_version="test", runtime_state=SimpleNamespace(semantics=None),
            run_dir=tmp_path, run_id="synthetic", run_language="en", writer_probe_mode=False,
            writer_probe_failed_steps=(), emit_progress=lambda *_args, **_kwargs: None,
        )

    return run, findings


def test_a_misspelled_id_does_not_skip_the_repair_of_other_rejected_prose(tmp_path, strict_run):
    run, findings = strict_run
    draft = DRAFT.replace("The outcome count was 15 {evidence:renal_summary}.", MISSPELLED).replace(
        "Interpretation is limited to the studied population.",
        f"{UNCITED} Interpretation is limited to the studied population.",
    )

    _, binding = run(draft, ["## Results\n\n" + _section(DRAFT, "Results")])

    assert binding.manuscript_critique.status == "pass"
    assert UNCITED not in binding.bound and "renal_summaryy" not in binding.bound
    repair, = [f for f in findings if f.validator == "evidence_bound_writer"
               and "deterministic drop fallback" in f.message]
    assert [item["sentence"] for item in repair.detail["evidence_repairs"]] == [UNCITED]
    receipt = json.loads((tmp_path / "manuscript_audit_repair.json").read_text())
    assert {"code": EVIDENCE_REMOVAL, "path": "results",
            "facts": {"evidence_id": "renal_summaryy", "sentence_removed": False}} in receipt["attempts"][0]["feedback"]


def test_results_only_a_misspelled_id_supported_still_stop_with_the_loss_reason(tmp_path, strict_run):
    from easyicu.research_agent.reporting.manuscript_audit_repair import LOSS_STOP

    run, findings = strict_run
    draft = DRAFT.replace(
        "The cohort comprised 120 ICU stays {evidence:renal_summary}.\n\n### Model performance",
        "The cohort comprised 992 ICU stays {evidence:renal_summaryy}.\n\n### Model performance",
    ).replace(
        "The outcome count was 15 {evidence:renal_summary}.", "The outcome count was 991 {evidence:renal_summaryy}.",
    ).replace(
        "Interpretation is limited to the studied population.",
        f"{UNCITED} Interpretation is limited to the studied population.",
    )

    with pytest.raises(EvidenceEnforcementError, match=LOSS_STOP) as stop:
        run(draft, [])

    # The repairable sentence was repaired; the misspelled-ID results were left
    # for the binder, whose loss guard stops the run instead of passing it.
    assert stop.value.detail["losses"][0]["facts"]["sentences_removed"] == 2
    repair, = [f for f in findings if f.validator == "evidence_bound_writer"
               and "deterministic drop fallback" in f.message]
    assert [item["sentence"] for item in repair.detail["evidence_repairs"]] == [UNCITED]


def test_strict_outside_the_stale_sentence_is_unchanged() -> None:
    stale_sentence = "Readmission occurred in 15 stays."
    other = "Readmission occurred in 9 stays."
    scaffold = f"## Results\n\n{stale_sentence} {other}\n"
    stale = (UnregisteredEvidence("results", sentence_key(stale_sentence), "renal_summaryy"),)

    def strict(text):
        rejected = [s for s in (stale_sentence, other) if s in text]
        if rejected:
            raise EvidenceEnforcementError("STRICT", detail={
                "removed_sentences": rejected, "unsupported_scientific_claim_sentences": [],
            })
        return text, []

    check = enforce_outside_stale_evidence(stale, strict)
    with pytest.raises(EvidenceEnforcementError) as refused:
        check(scaffold)
    assert refused.value.detail["removed_sentences"] == [other]
    assert check(scaffold.replace(f" {other}", "")) == (scaffold.replace(f" {other}", ""), [])
    assert enforce_outside_stale_evidence((), strict) is strict
