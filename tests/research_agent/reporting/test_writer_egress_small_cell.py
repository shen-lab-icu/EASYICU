"""A small count and what recovers it never reach the model through the Writer.

The Writer v2 digest withholds a small cell (n_events=3) together with the
total, rate and complement that would recover it.  These tests capture every
request the Writer sends for the Results and Abstract sections (draft,
structural retry, quality repair, section repair) and the evidence-repair
classifier, and assert that none of them carries the raw values.  The human
manuscript is not touched here.  Fixtures are synthetic; the client records
and answers locally, so no provider is called.
"""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path

import pytest

from easyicu.research_agent.agents.core import WriterAgent
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient
from easyicu.research_agent.reporting.manuscript_sections import (
    ManuscriptReaderQualityContractError,
    manuscript_section_specs,
)
from easyicu.research_agent.reporting.writer_evidence import _render_writer_evidence_digest_v2
from easyicu.research_agent.reporting.writer_evidence_repair import decide_writer_evidence_repairs
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)

_SMALL = {"n_events": 3, "n_total": 120, "n_non_events": 117, "rate": 0.025, "rate_pct": 2.5,
          "survival_proportion": 0.975, "survival_pct": 97.5}
_LARGE = {"n_events": 30, "n_total": 120, "n_non_events": 90, "rate": 0.25}
_SECTIONS = ("Results", "Abstract")


def _plan() -> AnalysisPlan:
    step = AnalysisStep(
        step_id="fixture", intent="Describe the prespecified outcome.", inputs=["death"],
        expected_outputs=["table:outcome"], method="descriptive_summary",
    )
    return AnalysisPlan(
        research_question="Describe the outcome.", steps=[step],
        cohort={"name": "primary", "selection_mode": "all_input_rows", "inclusion": [], "exclusion": []},
    )


def _step(criterion: str, parameters: dict, before: int, excluded: int, missing: int) -> dict:
    return {"criterion": criterion, "parameters": parameters, "n_before": before,
            "n_excluded": excluded, "n_remaining": before - excluded, "n_excluded_missing": missing}


def _export_report(*steps: dict, source_total: int = 1000) -> dict:
    left = steps[-1]["n_remaining"]
    return {
        "mode": "adult_first", "count_unit": "icu_stay", "source_total": source_total,
        "selected": left, "selected_before_cap": left, "selected_before_concept_prefilter": left,
        "demographic_steps": list(steps), "concept_matches": None, "selected_before_icd": left,
        "max_patients_applied": False, "applied_filters": ["demographics"],
        "icd": {"enabled": False, "include_tokens": [], "exclude_tokens": []},
    }


def _context(report: dict | None = None) -> ResearchContext:
    selection: dict = {"basis": "export_contract", "host_applied": []}
    if report is not None:
        selection["export_report"] = report
    return ResearchContext(
        research_question="Describe in-hospital mortality.",
        cohort=CohortDescriptor(
            cohort_name="synthetic_study", database="miiv", n_stays=500,
            id_columns=["stay_id"], outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(name="stay_id", dtype="object", role=VariableRole.ID),
            ConceptDescriptor(name="death", dtype="int64", role=VariableRole.OUTCOME),
        ],
        target_outcome="death",
        user_preferences=UserPreferences(data_constraints=json.dumps({"source_selection": selection})),
    )


def _digest(tmp_path: Path, group: dict) -> str:
    evidence = EvidenceStore(tmp_path)
    source = tmp_path / "summary.json"
    source.write_text("{}", encoding="utf-8")
    evidence.register_file(
        kind="statistic", description="Summary for fixture.", source_path=source,
        produced_by_step="fixture", evidence_id="summary", producer="runner",
    )
    for name, value in group.items():
        evidence.register_numeric_claim(
            value=str(value), canonical=float(value), evidence_id="summary", step_id="fixture",
            source_field=f"reportable_descriptive_results.groups[0].{name}",
        )
    return _render_writer_evidence_digest_v2(
        [{"step_id": "fixture", "status": "ok", "step_summary": {}, "evidence_ids": ["summary"]}],
        evidence=evidence, run_dir=tmp_path,
    )


def _manuscript(*, complete: bool = True) -> str:
    parts = []
    for spec in manuscript_section_specs(_plan()):
        if spec.key == "title":
            parts.append("# A study title")
            continue
        body = "The section states what the study did."
        subs = "".join(f"\n\n### {name}\n\n{body}" for name in spec.required_subsections)
        parts.append(f"## {spec.section_name}\n\n{body}" + (subs if complete else ""))
    return "\n\n".join(parts)


def _section_text(spec) -> str:
    body = "The section states what the study did."
    subs = "".join(f"\n\n### {sub}\n\n{body}" for sub in spec.required_subsections)
    return ("# A study title" if spec.key == "title" else f"## {spec.section_name}\n\n{body}") + subs


def _client(*, first_incomplete: bool = False) -> PatternScriptedMockLLMClient:
    """A recording client that answers each section request locally."""
    rules = []
    for spec in manuscript_section_specs(_plan()):
        full = _section_text(spec)
        answers = [full.split("\n\n###")[0] if first_incomplete else full] + [full] * 5
        rules.append((f"Write ONLY the **{spec.section_name}** section", answers))
    return PatternScriptedMockLLMClient(rules, default="x")


def _texts(llm: PatternScriptedMockLLMClient) -> list[str]:
    return ["\n".join(str(m.content or "") for m in messages) for messages, _ in llm.calls]


def _egress_prompts(digest: str, context: ResearchContext | None = None) -> dict[str, list[str]]:
    """Every request text the Writer sends, for each entry point that drafts or revises."""
    common = {
        "context": context or _context(), "analysis_plan": _plan(), "evidence_ids": ["summary"],
        "evidence_digest": digest, "literature_digest": None, "reader_display_labels": {},
    }
    captured: dict[str, list[str]] = {}

    def run(label: str, action, *, first_incomplete: bool = False) -> None:
        llm = _client(first_incomplete=first_incomplete)
        # The stub prose fails the reader-quality gate; its bounded repair
        # rounds are requests too, so they are captured before the raise.
        with contextlib.suppress(ManuscriptReaderQualityContractError):
            action(WriterAgent(llm))
        captured[label] = _texts(llm)

    run("draft", lambda w: w.run(**common))
    # The structural retry: the first answer of every section lacks its subsections.
    run("structural_retry", lambda w: w.run(**common), first_incomplete=True)
    run("section_repair", lambda w: w.repair_sections(
        _manuscript(), section_errors={"results": ("A rejected sentence.",), "abstract": ("Another.",)},
        **common,
    ))
    run("quality_repair", lambda w: w.repair_existing(_manuscript(complete=False), **common))
    # The bounded evidence-repair classifier sees the same digest.
    llm = _client()
    with pytest.raises(Exception):
        decide_writer_evidence_repairs(
            llm, evidence_ids=["summary"], evidence_digest=digest,
            missing_sentences=["The study observed an outcome."],
        )
    captured["evidence_repair"] = _texts(llm)
    return captured


def _section_prompts(prompts: list[str], section: str) -> list[str]:
    return [p for p in prompts if f"Write ONLY the **{section}** section" in p]


_LEAKS = (r"(?<![\d.])3(?![\d.]|\d)", r"2\.5(?!\d)", r"0\.025", r"(?<![\d.])117(?![\d.])", r"97\.5", r"0\.975")


def test_no_request_for_results_or_abstract_carries_a_small_count_or_what_recovers_it(tmp_path):
    captured = _egress_prompts(_digest(tmp_path, _SMALL))

    for label in ("draft", "section_repair", "quality_repair", "structural_retry"):
        for section in _SECTIONS:
            found = _section_prompts(captured[label], section)
            assert found, (label, section)
            for prompt in found:
                # Only the digest and the research context carry facts.
                facts = prompt.split("MACHINE EVIDENCE DIGEST:", 1)[1]
                assert "n_events=suppressed" in facts, (label, section)
                for leak in _LEAKS:
                    assert not re.search(leak, facts), (label, section, leak)
    for label, prompts in captured.items():
        for prompt in prompts:
            for token in ("0.025", "97.5", "0.975", "n_non_events=117", "n_events=3"):
                assert token not in prompt, (label, token)


def test_the_evidence_repair_classifier_digest_is_suppressed_too(tmp_path):
    captured = _egress_prompts(_digest(tmp_path, _SMALL))

    assert captured["evidence_repair"]
    for prompt in captured["evidence_repair"]:
        assert "n_events=suppressed" in prompt
        for leak in _LEAKS:
            assert not re.search(leak, prompt.split("MACHINE EVIDENCE DIGEST:", 1)[1]), leak


def test_a_count_above_the_threshold_reaches_the_model_as_written(tmp_path):
    captured = _egress_prompts(_digest(tmp_path, _LARGE))

    for label in ("draft", "section_repair", "quality_repair", "structural_retry"):
        for section in _SECTIONS:
            for prompt in _section_prompts(captured[label], section):
                facts = prompt.split("MACHINE EVIDENCE DIGEST:", 1)[1]
                assert "n_events=30" in facts and "n_total=120" in facts
                assert "n_non_events=90" in facts and "rate=0.25" in facts
                assert "suppressed" not in facts.split("RESEARCH CONTEXT:", 1)[0]


def _population_prompts(report: dict) -> list[str]:
    """Results and Abstract requests for a recorded export selection."""
    context = _context(report)
    context.cohort.n_stays = report["demographic_steps"][-1]["n_remaining"]
    captured = _egress_prompts("## EXECUTED METHOD BOUNDARY\n- none", context)
    return [
        prompt for label in ("draft", "section_repair", "quality_repair") for section in _SECTIONS
        for prompt in _section_prompts(captured[label], section)
    ]


def _counts_line(prompt: str) -> str:
    return prompt.split("Selection counts", 1)[1].split("\n", 1)[0]


def test_a_small_selection_count_never_reaches_a_section_request():
    # 7 stays excluded by the second criterion; 4 of the first step's 120 lacked a value.
    report = _export_report(_step("age", {"age_min": 18}, 1000, 120, 4), _step("los", {"los_min": 24}, 880, 7, 0))
    prompts = _population_prompts(report)

    assert prompts
    for prompt in prompts:
        line = _counts_line(prompt)
        assert "suppressed" in line
        for count in ("1,000", "120", "880", "873", "7 excluded", "4 of them"):
            assert count not in line, count


def test_a_small_missing_count_alone_is_withheld_beside_exact_chain_counts():
    report = _export_report(_step("age", {"age_min": 18}, 1000, 120, 4), _step("los", {"los_min": 24}, 880, 80, 0))
    for prompt in _population_prompts(report):
        line = _counts_line(prompt)
        assert "suppressed of them for want of a value" in line
        assert "the source database held 1,000" in line and "120 excluded" in line
        assert "4 of them" not in line


def test_selection_counts_above_the_threshold_reach_the_request_as_written():
    report = _export_report(_step("age", {"age_min": 18}, 1000, 120, 14), _step("los", {"los_min": 24}, 880, 80, 0))
    for prompt in _population_prompts(report):
        line = _counts_line(prompt)
        assert "suppressed" not in line
        assert "120 excluded (14 of them" in line and "the analysis input held 800" in line
