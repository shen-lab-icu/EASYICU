"""A manuscript states the population its analysis selected, not the question's.

A research question names a population; it selects no one.  Rows enter an
analysis through the criteria the source export applied (a concept-derived
population included) and the plan's cohort predicates.  Every Writer section
receives that population, and the host cites a population statement on the
Writer's behalf only when the plan selected its rows by predicate: otherwise
every owner such a statement could cite records the export or the question,
not a selection.  Fixtures are synthetic; the populations they name vary so
that no rule keys on one condition.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from easyicu.research_agent.agents.core import WriterAgent
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient
from easyicu.research_agent.reporting import manuscript_sections, write_phase
from easyicu.research_agent.reporting.manuscript_post import (
    _repair_common_writer_citation_omissions,
)
from easyicu.research_agent.reporting.manuscript_sections import (
    MANUSCRIPT_SECTION_SPECS,
    ManuscriptReaderQualityContractError,
    manuscript_section_specs,
    render_manuscript_sections,
    repair_existing_manuscript_sections,
    repair_named_manuscript_sections,
)
from easyicu.research_agent.reporting.population_selection import (
    POPULATION_STATEMENT_NOT_HOST_CITED,
    POPULATION_SUBSECTION,
    analyzed_population,
    writer_population_block,
)
from easyicu.research_agent.reporting.writer_evidence import _render_writer_evidence_digest
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)


def _predicate(concept_id: str, op: str, value: Any, aggregation: str = "max") -> dict:
    return {
        "concept_id": concept_id,
        "time_window": {"anchor": "icu_admission", "start_offset_hours": 0, "end_offset_hours": 24},
        "aggregation": aggregation,
        "op": op,
        "value": value,
    }


_ALL_ROWS = {"name": "primary", "selection_mode": "all_input_rows", "inclusion": [], "exclusion": []}
_SELECTED = {
    "name": "primary",
    "selection_mode": "predicate_filtered",
    "inclusion": [_predicate("sep3", "==", True)],
    "exclusion": [_predicate("age", "<", 18, "first")],
}
_MODEL_SENTENCE = (
    "The primary association was estimated with logistic regression because the "
    "outcome was binary."
)


def _plan(cohort: dict | None) -> AnalysisPlan:
    step = AnalysisStep(
        step_id="01_primary",
        intent="Describe the prespecified outcome.",
        inputs=["death"],
        expected_outputs=["table:outcome"],
        method="descriptive_summary",
    )
    return AnalysisPlan(research_question="Describe the outcome.", steps=[step], cohort=cohort)


def _context(
    question: str = "Describe in-hospital mortality.",
    *,
    inclusion: tuple[str, ...] = (),
    exclusion: tuple[str, ...] = (),
    constraints: dict | None = None,
) -> ResearchContext:
    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(
            cohort_name="web_study",
            database="miiv",
            n_stays=100,
            inclusion_criteria=list(inclusion),
            exclusion_criteria=list(exclusion),
            id_columns=["stay_id"],
            outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(name="stay_id", dtype="object", role=VariableRole.ID),
            ConceptDescriptor(name="death", dtype="int64", role=VariableRole.OUTCOME),
        ],
        target_outcome="death",
        user_preferences=UserPreferences(
            data_constraints=json.dumps(constraints) if constraints is not None else None
        ),
    )


def _store(tmp_path: Path, *evidence_ids: str) -> EvidenceStore:
    store = EvidenceStore(tmp_path)
    for evidence_id in evidence_ids:
        path = tmp_path / f"{evidence_id}.json"
        path.write_text("{}", encoding="utf-8")
        store.register_file(
            kind="statistic",
            description=evidence_id,
            source_path=path,
            evidence_id=evidence_id,
            producer="test",
        )
    return store


# The record -----------------------------------------------------------


def test_a_plan_keeping_every_row_of_an_uncontracted_export_analyzes_every_icu_stay() -> None:
    population = analyzed_population(plan=_plan(_ALL_ROWS), context=_context())

    assert population is not None
    # The field names the kernel's population builder shares.
    assert population.record() == {
        "selection_mode": "all_input_rows",
        "inclusion_predicates": [],
        "exclusion_predicates": [],
        "applied_contracts": {"inclusion": [], "exclusion": []},
        "concept_population": None,
        "source_scope": "all_icu_stays_of_source_export",
    }


def test_the_criteria_an_export_applied_are_its_population() -> None:
    context = _context(
        inclusion=("age range: 18 to *", " age range: 18 to * ", ""),
        exclusion=("each patient's later ICU stays",),
    )

    population = analyzed_population(plan=_plan(_ALL_ROWS), context=context)

    assert population is not None
    assert population.source_scope == "all_input_rows_of_contracted_export"
    assert population.record()["applied_contracts"] == {
        "inclusion": ["age range: 18 to *"],
        "exclusion": ["each patient's later ICU stays"],
    }
    block = writer_population_block(population)
    assert "every input row of the source export" in block
    assert "inclusion: age range: 18 to *" in block
    assert "exclusion: each patient's later ICU stays" in block


@pytest.mark.parametrize(
    "wording",
    [
        {"label": "Adults with septic shock", "review": "Adult ICU stays with septic shock"},
        {"review": "Older adults with acute kidney injury", "exclusion_statement": "Excluding dialysis"},
    ],
)
def test_the_studys_own_wording_is_not_a_criterion_the_export_applied(
    wording: dict[str, str],
) -> None:
    # A context written before the Web caller declared only typed criteria
    # filed the study's wording as criteria, and the same words in
    # data_constraints.cohort.
    context = _context(
        inclusion=tuple(wording[key] for key in ("label", "review") if key in wording),
        exclusion=(wording["exclusion_statement"],) if "exclusion_statement" in wording else (),
        constraints={"cohort": wording},
    )

    population = analyzed_population(plan=_plan(_ALL_ROWS), context=context)

    assert population is not None
    assert population.source_scope == "all_icu_stays_of_source_export"
    assert population.record()["applied_contracts"] == {"inclusion": [], "exclusion": []}
    block = writer_population_block(population)
    assert "- Criteria the source export applied before analysis: none." in block
    for words in wording.values():
        assert words not in block


@pytest.mark.parametrize(
    ("definition", "hours"), [("sepsis3", 24.0), ("aki", 48.0), ("ventilation", 12.0)]
)
def test_a_concept_derived_export_is_not_called_every_icu_stay(
    definition: str, hours: float
) -> None:
    """Extraction admitted only stays positive for the concept; no criterion lists it."""

    context = _context(
        constraints={"concept_cohort_window": {"definition": definition, "window_end_hours": hours}}
    )

    population = analyzed_population(plan=_plan(_ALL_ROWS), context=context)

    assert population is not None
    assert population.source_scope == "all_input_rows_of_contracted_export"
    assert population.record()["concept_population"] == {
        "definition": definition,
        "window_end_hours": hours,
    }
    block = writer_population_block(population)
    assert "every ICU stay" not in block
    assert f"concept-derived population {definition}" in block
    assert f"at or before {hours:g} h after ICU admission" in block


def test_a_plan_stating_predicates_selects_the_rows_they_admit() -> None:
    plan = _plan(_SELECTED)

    population = analyzed_population(plan=plan, context=_context(inclusion=("age range: 18 to *",)))

    assert population is not None
    assert population.source_scope == "predicate_selected"
    record = population.record()
    assert record["inclusion_predicates"] == [plan.cohort.inclusion[0].to_dict()]
    assert record["exclusion_predicates"] == [plan.cohort.exclusion[0].to_dict()]
    block = writer_population_block(population)
    assert "rows that meet the plan's cohort predicates" in block
    assert "sep3 == True (max over 0 to 24 h from icu_admission)" in block
    assert "age < 18 (first over 0 to 24 h from icu_admission)" in block
    assert "inclusion: age range: 18 to *" in block


def test_a_predicate_filtered_cohort_without_predicates_keeps_every_row() -> None:
    cohort = {"name": "primary", "selection_mode": "predicate_filtered", "inclusion": [], "exclusion": []}

    population = analyzed_population(plan=_plan(cohort), context=_context())

    assert population is not None
    assert population.selection_mode == "predicate_filtered"
    assert population.source_scope == "all_icu_stays_of_source_export"


@pytest.mark.parametrize("case", ["no plan", "no cohort", "no context", "unreadable concept record"])
def test_without_typed_owners_the_host_states_no_population(case: str) -> None:
    plan = None if case == "no plan" else _plan(None if case == "no cohort" else _ALL_ROWS)
    context = (
        None
        if case == "no context"
        else _context(
            constraints=(
                {"concept_cohort_window": {"definition": "sepsis3"}}
                if case == "unreadable concept record"
                else None
            )
        )
    )

    population = analyzed_population(plan=plan, context=context)

    assert population is None
    block = writer_population_block(population)
    assert "not stated by the host" in block
    assert "any criterion or cohort wording in RESEARCH CONTEXT, select no one" in block


# The Writer -----------------------------------------------------------


_QUESTIONS = [
    "Among older adults with acute kidney injury, do first-day creatinine "
    "trajectories form distinct classes?",
    "In patients receiving vasopressors, is early lactate associated with death?",
    "After cardiac surgery, which organ-support patterns precede ICU death?",
]


def _common(question: str) -> dict[str, Any]:
    return {
        "context": _context(question),
        "analysis_plan": _plan(_ALL_ROWS),
        "evidence_ids": ["research_context"],
        "evidence_digest": "## EXECUTED METHOD BOUNDARY\n- none",
        "literature_digest": None,
        "reader_display_labels": {},
        "language": "en",
    }


def _section_text(spec: Any, *, complete: bool = True) -> str:
    if spec.key == "title":
        return "# A study title"
    body = "The section states what the study did."
    subsections = "".join(f"\n\n### {name}\n\n{body}" for name in spec.required_subsections)
    return f"## {spec.section_name}\n\n{body}" + (subsections if complete else "")


def _section_requests(
    entry: str, common: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    """Every section request one Writer entry makes.

    Each request is answered with a complete section, except the draft's first
    Methods answer, so the draft's structural retry is among the requests.
    """

    specs = manuscript_section_specs(common["analysis_plan"])
    by_name = {spec.section_name: spec for spec in specs}
    methods = next(spec for spec in specs if spec.key == "methods")
    requests: list[dict[str, Any]] = []

    def call_section(**kwargs: Any) -> str:
        requests.append(kwargs)
        spec = by_name[kwargs["section_name"]]
        first_methods = [request["section_name"] for request in requests].count(methods.section_name) == 1
        return _section_text(spec, complete=not (entry == "draft" and spec is methods and first_methods))

    manuscript = "\n\n".join(_section_text(spec) for spec in specs)
    with contextlib.suppress(ManuscriptReaderQualityContractError):
        if entry == "draft":
            render_manuscript_sections(call_section=call_section, common=common)
        elif entry == "named repair":
            repair_named_manuscript_sections(
                manuscript,
                section_errors={"methods": ("A rejected sentence.",)},
                call_section=call_section,
                common=common,
            )
        else:
            pending = [[(methods, "- An offending excerpt.")]]
            monkeypatch.setattr(
                manuscript_sections,
                "_quality_repair_specs",
                lambda *args, **kwargs: pending.pop() if pending else [],
            )
            repair_existing_manuscript_sections(manuscript, call_section=call_section, common=common)
    return requests


@pytest.mark.parametrize("entry", ["draft", "named repair", "quality repair"])
def test_every_section_request_leads_with_the_analyzed_population(
    entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = _common(_QUESTIONS[0])
    block = writer_population_block(
        analyzed_population(plan=common["analysis_plan"], context=common["context"])
    )

    requests = _section_requests(entry, common, monkeypatch)

    assert requests
    assert all(request["instruction"].startswith(block) for request in requests)
    names = {request["section_name"] for request in requests}
    if entry == "draft":
        assert names >= {spec.section_name for spec in manuscript_section_specs(common["analysis_plan"])}
        assert any("STRUCTURAL CONTRACT REPAIR" in request["instruction"] for request in requests)
    else:
        assert names == {"Methods"}


@pytest.mark.parametrize("question", _QUESTIONS)
def test_every_writer_section_states_the_analyzed_population(
    question: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = _common(question)
    block = writer_population_block(
        analyzed_population(plan=common["analysis_plan"], context=common["context"])
    )
    llm = PatternScriptedMockLLMClient([], default="## Section\n\nText.")
    writer = WriterAgent(llm)

    for request in _section_requests("draft", common, monkeypatch):
        writer._call_section(**request)
        prompt = llm.calls[-1][0][1].content
        assert prompt.count(block) == 1
        assert "every ICU stay in the source export" in prompt
        assert "any criterion or cohort wording in RESEARCH CONTEXT, select no one" in prompt
        assert "around the ANALYZED POPULATION above" in prompt
        assert "this research question's population" not in prompt
        # The question still reaches the Writer, as the question.
        assert question in prompt


def test_the_population_sections_ask_for_the_analyzed_population() -> None:
    instructions = {spec.key: spec.instruction for spec in MANUSCRIPT_SECTION_SPECS}

    for key in ("title", "abstract", "methods"):
        assert "ANALYZED POPULATION" in instructions[key]
    assert "inclusion/exclusion criteria" not in instructions["methods"]
    assert "State no inclusion or exclusion criterion it does not list." in instructions["methods"]


# The host's citation repair ---------------------------------------------


_POPULATION_SENTENCES = [
    "Adult patients with acute kidney injury were included.",
    "The cohort comprised patients receiving vasopressors.",
    "Patients undergoing cardiac surgery met the inclusion criteria.",
    "An exclusion criterion removed stays shorter than one day.",
    # The exposure rule would cite it first.
    "Adults with acute kidney injury were included, and the exposure was "
    "derived from creatinine.",
]


def _methods(sentence: str) -> str:
    """A sentence where the manuscript states its population, then a model sentence."""

    return (
        "## Methods\n### Study design and cohort\n"
        f"{sentence}\n"
        f"### Statistical analysis\n{_MODEL_SENTENCE}\n"
    )


@pytest.mark.parametrize("sentence", _POPULATION_SENTENCES)
@pytest.mark.parametrize("scope", ["every ICU stay", "contracted export", "no typed population"])
def test_the_host_does_not_cite_a_population_the_plan_did_not_select(
    tmp_path: Path, sentence: str, scope: str
) -> None:
    store = _store(
        tmp_path,
        "01_define_cohort_and_derive",
        "04_primary_adjusted_association_model",
        "research_context",
    )
    context = _context(inclusion=("age range: 18 to *",) if scope == "contracted export" else ())
    population = (
        None
        if scope == "no typed population"
        else analyzed_population(plan=_plan(_ALL_ROWS), context=context)
    )

    repaired, repairs = _repair_common_writer_citation_omissions(
        _methods(sentence), evidence=store, population=population
    )

    lines = repaired.splitlines()
    assert lines[2] == sentence
    assert "{evidence:04_primary_adjusted_association_model}" in lines[4]
    assert {"reason_code": POPULATION_STATEMENT_NOT_HOST_CITED, "sentence": sentence} in repairs


def test_the_abstracts_methods_paragraph_states_the_population_too(tmp_path: Path) -> None:
    store = _store(tmp_path, "01_define_cohort_and_derive")
    line = "**Methods:** Adult patients with acute kidney injury were included."

    repaired, repairs = _repair_common_writer_citation_omissions(
        f"## Abstract\n{line}\n",
        evidence=store,
        population=analyzed_population(plan=_plan(_ALL_ROWS), context=_context()),
    )

    assert repaired.splitlines()[1] == line
    assert [item.get("reason_code") for item in repairs] == [POPULATION_STATEMENT_NOT_HOST_CITED]


@pytest.mark.parametrize(
    ("heading", "sentence"),
    [
        # A model's terms, not the population.
        ("## Methods\n### Variables", "No alternative exposure was included in the adjustment set."),
        # Another study's population.
        ("## Discussion", "Prior cohort studies were restricted to patients receiving vasopressors."),
        ("## Introduction", "Earlier work reported its inclusion criteria for adult patients."),
    ],
)
def test_the_same_words_elsewhere_keep_the_hosts_citation(
    tmp_path: Path, heading: str, sentence: str
) -> None:
    store = _store(tmp_path, "01_define_cohort_and_derive", "research_context")

    repaired, repairs = _repair_common_writer_citation_omissions(
        f"{heading}\n{sentence}\n",
        evidence=store,
        population=analyzed_population(plan=_plan(_ALL_ROWS), context=_context()),
    )

    assert "{evidence:" in repaired.splitlines()[-1]
    assert all("evidence_id" in item for item in repairs) and repairs


def test_rows_a_plan_selected_by_predicate_keep_their_population_citation(tmp_path: Path) -> None:
    store = _store(tmp_path, "01_define_cohort_and_derive")
    population = analyzed_population(plan=_plan(_SELECTED), context=_context())
    sentence = "Adult patients with sepsis were included."

    repaired, repairs = _repair_common_writer_citation_omissions(
        f"## Methods\n### Study design and cohort\n{sentence}\n",
        evidence=store,
        population=population,
    )

    assert repaired.splitlines()[2] == (
        "Adult patients with sepsis were included {evidence:01_define_cohort_and_derive}."
    )
    assert [item.get("evidence_id") for item in repairs] == ["01_define_cohort_and_derive"]


def test_the_population_subsection_is_the_one_the_writer_contract_requires() -> None:
    (methods,) = [spec for spec in MANUSCRIPT_SECTION_SPECS if spec.key == "methods"]

    assert methods.required_subsections[0] == POPULATION_SUBSECTION


def test_a_population_statement_keeps_the_citation_the_writer_gave_it(tmp_path: Path) -> None:
    store = _store(tmp_path, "01_define_cohort_and_derive", "research_context")
    sentence = "Adults with acute kidney injury were included {evidence:research_context}."

    repaired, repairs = _repair_common_writer_citation_omissions(
        f"## Methods\n### Study design and cohort\n{sentence}\n",
        evidence=store,
        population=analyzed_population(plan=_plan(_ALL_ROWS), context=_context()),
    )

    assert repaired.splitlines()[2] == sentence
    assert repairs == []


# The write phase --------------------------------------------------------


def test_the_writer_digest_records_the_population_it_states(tmp_path: Path) -> None:
    context = _context(
        constraints={"concept_cohort_window": {"definition": "aki", "window_end_hours": 48}}
    )
    population = analyzed_population(plan=_plan(_ALL_ROWS), context=context)
    assert population is not None

    digest = _render_writer_evidence_digest(context=context, run_dir=tmp_path, population=population)

    lines = digest.splitlines()
    run_context = json.loads(lines[lines.index("RUN_CONTEXT") + 1])
    assert run_context["population_selection"] == population.record()
    assert run_context["research_question"] == context.research_question
    without = _render_writer_evidence_digest(context=context, run_dir=tmp_path)
    assert "population_selection" not in without
    assert _render_writer_evidence_digest(context=context, run_dir=tmp_path, population=None) == without


def test_the_write_phase_reports_appended_citations_and_uncited_population_statements() -> None:
    population = analyzed_population(plan=_plan(_ALL_ROWS), context=_context())
    appended = {"evidence_id": "04_model", "sentence": "The model was a logistic regression."}
    declined = {
        "reason_code": POPULATION_STATEMENT_NOT_HOST_CITED,
        "sentence": "Adults with acute kidney injury were included.",
    }

    citation_finding, population_finding = write_phase._citation_repair_findings(
        [appended, declined], population=population
    )

    assert citation_finding.detail == {"citation_repairs": [appended]}
    assert population_finding.detail == {
        "reason_code": POPULATION_STATEMENT_NOT_HOST_CITED,
        "source_scope": "all_icu_stays_of_source_export",
        "sentences": ["Adults with acute kidney injury were included."],
    }
    assert write_phase._citation_repair_findings([], population=None) == []


def _calls(function: Any, name: str) -> list[ast.Call]:
    tree = ast.parse(inspect.getsource(function).lstrip())
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


def test_the_draft_stage_hands_one_population_to_the_digest_and_the_citation_repair() -> None:
    draft = write_phase._draft_manuscript

    (built,) = _calls(draft, "analyzed_population")
    assert {item.arg: ast.unparse(item.value) for item in built.keywords} == {
        "plan": "execute_result.plan",
        "context": "context",
    }
    for name in (
        "_repair_common_writer_citation_omissions",
        "_render_writer_evidence_digest",
        "_render_writer_evidence_digest_v2",
    ):
        (call,) = _calls(draft, name)
        keywords = {item.arg: ast.unparse(item.value) for item in call.keywords}
        assert keywords["population"] == "population"
    (reported,) = _calls(draft, "_citation_repair_findings")
    assert {item.arg: ast.unparse(item.value) for item in reported.keywords} == {
        "population": "population"
    }
