"""A grouped exposure's rules are stated in the Methods, reproduced by the host.

A grouped exposure's column holds level codes, and a level's meaning is the
rule the host forms it with.  The context states each grouping's levels with
their derivation parameters, and the method-fact owner reproduces from them
one exact Methods line per level ("Recorded exposure grouping for the grouped
exposure: <grouping> level <code>: <rule>"), placed under Variables by the
host and filtered like every other method fact: a look-alike line the Writer
wrote is removed.

Such a line states numbers -- a threshold of 180 mg/dL, a window of 24 h --
that the numeric binder would read as untraced results.  Each fact states the
numbers it reproduced from typed parameters (``typed_values``), and the
binder binds those numbers, and only those, on the fact's own line placed
exactly in its section, footnoting each to the fact's source field, the
context's evidence and its digest.  A fact written from free text states no
typed number, so a description's "300 points" still needs a result claim.

A grouping that names no rule -- one built before groupings carried their
parameters, or one whose line would read as markup -- gives no line.
Parameters that do not form the levels stated beside them cannot be
reproduced and stop the facts.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.authority.manuscript_method_facts import (
    MethodFactAuthorityError,
)
from easyicu.research_agent.contracts.exposure_group_rules import (
    EXPOSURE_GROUP_LABELS_KEY,
    grouping_rule_values,
)
from easyicu.research_agent.reporting.manuscript_method_facts import (
    method_fact_binding,
    place_manuscript_method_facts,
    released_fact_lines,
)
from easyicu.research_agent.reporting.manuscript_post import (
    bind_numeric_values,
    drop_untraceable_numeric_sentences,
)
from easyicu.research_agent.schema import ResearchContext

_PREFIX = "Recorded exposure grouping for the grouped exposure: "
_GLUCOSE = "Glucose over 0-24 h after ICU admission"


def _parameters(**changes: Any) -> dict[str, Any]:
    """A day-one glucose grouping shaped like a real descriptive study's:
    low below 70 mg/dL, high above 180 mg/dL, the rest, and no value."""

    parameters: dict[str, Any] = {
        "concept": "glu",
        "window": {"start_hours": 0, "end_hours": 24},
        "scale": "nominal",
        "groups": [
            {
                "id": "g1",
                "rule": {"summary": "min", "op": "<", "value": 70, "unit": "mg/dL"},
            },
            {
                "id": "g2",
                "rule": {"summary": "max", "op": ">", "value": 180, "unit": "mg/dL"},
            },
            {"id": "g3", "rule": "otherwise"},
        ],
        "unmeasured": "own_group",
        "source_columns": {"min": "glu_min", "max": "glu_max"},
    }
    parameters.update(changes)
    return parameters


def _grouping(variable: str = "glu_group_x1", **changes: Any) -> dict[str, Any]:
    grouping: dict[str, Any] = {
        "variable": variable,
        "concept": "glu",
        "scale": "nominal",
        "levels": [
            {"code": 1, "group": "g1", "label": "低血糖"},
            {"code": 2, "group": "g2", "label": "仅高血糖"},
            {"code": 3, "group": "g3", "label": "血糖正常"},
            {"code": 4, "group": "gU", "label": "未测"},
        ],
        "parameters": _parameters(),
    }
    grouping.update(changes)
    return grouping


def _descriptive(*groupings: Any, description: str = "In-hospital mortality"):
    """A descriptive study's context: no selected exposure, a grouped one."""

    return ResearchContext.model_validate(
        {
            "research_question": "Describe hospital death across day-one glucose groups.",
            "cohort": {
                "cohort_name": "synthetic",
                "database": "eicu",
                "n_stays": 40,
                "provenance": {EXPOSURE_GROUP_LABELS_KEY: list(groupings)},
            },
            "primary_exposure": None,
            "target_outcome": "death",
            "variables": [
                {"name": "glu_group_x1", "dtype": "int64"},
                {"name": "death", "dtype": "bool", "description": description},
            ],
        }
    )


def _store(tmp_path, context: ResearchContext) -> EvidenceStore:
    source = tmp_path / "input.json"
    source.write_text(context.model_dump_json(), encoding="utf-8")
    store = EvidenceStore(tmp_path, enforcement_mode=EvidenceEnforcementMode.STRICT)
    store.register_file(
        kind="log",
        source_path=source,
        description="Typed source context",
        evidence_id="research_context",
        producer="pipeline",
        generation_mode="system",
    )
    return store


def _grouping_facts(store: EvidenceStore) -> list:
    return [
        fact
        for fact in store.manuscript_method_facts()
        if fact.text.startswith(_PREFIX)
    ]


def _bound_draft(store: EvidenceStore) -> str:
    """The host's draft: its facts placed, filtered and bound."""

    draft, _placed = place_manuscript_method_facts(
        "## Methods\n\n### Variables\n\nThe study grouped its stays.\n",
        store.manuscript_method_facts(),
    )
    safe, removed = store.enforce_evidence_bound_scaffold(draft)
    assert not removed
    return store.bind_manuscript(safe, per_step_records=[])


def _bind(store: EvidenceStore, text: str, **binding: Any):
    return bind_numeric_values(text, evidence=store, per_step_records=[], **binding)


def test_each_level_of_a_grouped_exposure_is_stated_with_its_rule(tmp_path) -> None:
    store = _store(tmp_path, _descriptive(_grouping()))

    facts = _grouping_facts(store)

    assert [fact.text for fact in facts] == [
        f"{_PREFIX}Glucose level 1: {_GLUCOSE}: minimum < 70 mg/dL",
        f"{_PREFIX}Glucose level 2: {_GLUCOSE}: maximum > 180 mg/dL, "
        "and none of minimum < 70 mg/dL",
        f"{_PREFIX}Glucose level 3: {_GLUCOSE}: "
        "none of minimum < 70 mg/dL; maximum > 180 mg/dL",
        f"{_PREFIX}Glucose level 4: {_GLUCOSE}: no value recorded",
    ]
    assert [fact.source_field for fact in facts] == [
        f"cohort.provenance.{EXPOSURE_GROUP_LABELS_KEY}[0].levels[{index}]"
        for index in range(4)
    ]
    assert [fact.typed_values for fact in facts] == [
        (0.0, 24.0, 70.0),
        (0.0, 24.0, 180.0, 70.0),
        (0.0, 24.0, 70.0, 180.0),
        (0.0, 24.0),
    ]
    assert {fact.section for fact in facts} == {"variables"}


def test_the_numbers_a_rule_states_are_read_from_its_parameters() -> None:
    values = grouping_rule_values(
        _parameters(window={"start_hours": 0, "end_hours": 168.5})
    )

    assert values == {
        "g1": (0.0, 168.5, 70.0),
        "g2": (0.0, 168.5, 180.0, 70.0),
        "g3": (0.0, 168.5, 70.0, 180.0),
        "gU": (0.0, 168.5),
    }
    assert grouping_rule_values(_parameters(window=None))["g1"] == (70.0,)


def test_the_hosts_lines_bind_their_numbers_to_the_grouping(tmp_path) -> None:
    store = _store(tmp_path, _descriptive(_grouping()))
    bound = _bound_draft(store)

    result, bindings, untraced = _bind(store, bound, **method_fact_binding(store, []))

    assert untraced == []
    assert sorted(
        (claim.value, claim.evidence_id, claim.source_field)
        for claim in bindings.values()
    ) == [
        (
            "180",
            "research_context",
            f"cohort.provenance.{EXPOSURE_GROUP_LABELS_KEY}[0].levels[{level}]",
        )
        for level in (1, 2)
    ]
    digest = store.get("research_context").sha256
    assert (
        f"field=cohort.provenance.{EXPOSURE_GROUP_LABELS_KEY}[0].levels[1]; "
        f"evidence=research_context; sha256={digest}"
    ) in result


def test_without_the_facts_the_binder_reads_the_threshold_as_a_result(
    tmp_path,
) -> None:
    store = _store(tmp_path, _descriptive(_grouping()))
    bound = _bound_draft(store)

    with pytest.raises(EvidenceEnforcementError, match="numeric value"):
        _bind(store, bound)


def test_only_exact_typed_fact_lines_are_found_in_their_section(tmp_path) -> None:
    store = _store(
        tmp_path, _descriptive(_grouping(), description="A score of 300 points")
    )
    bound = _bound_draft(store)
    binding = method_fact_binding(store, [])
    found = released_fact_lines(bound, binding["method_facts"], binding["bind_fact"])
    assert [fact.source_field for _start, _end, fact in found] == [
        f"cohort.provenance.{EXPOSURE_GROUP_LABELS_KEY}[0].levels[{index}]"
        for index in range(4)
    ]
    assert all(
        bound[start:end] == binding["bind_fact"](fact.scaffold)
        for start, end, fact in found
    )


@pytest.mark.parametrize("bad_code", [True, 1.0])
def test_grouping_facts_reject_non_integer_level_codes(tmp_path, bad_code) -> None:
    grouping = _grouping()
    grouping["levels"][0]["code"] = bad_code
    store = _store(tmp_path, _descriptive(grouping))
    with pytest.raises(MethodFactAuthorityError):
        store.manuscript_method_facts()


def test_grouping_facts_reject_a_different_declared_scale(tmp_path) -> None:
    store = _store(tmp_path, _descriptive(_grouping(scale="ordinal")))
    with pytest.raises(MethodFactAuthorityError):
        store.manuscript_method_facts()


def test_a_line_the_writer_rewrote_in_the_facts_shape_binds_nothing(
    tmp_path,
) -> None:
    store = _store(tmp_path, _descriptive(_grouping()))
    bound = _bound_draft(store)
    (level_two,) = [
        line for line in bound.splitlines() if "level 2:" in line and _PREFIX in line
    ]

    for rewritten in (
        level_two.replace("180 mg/dL", "180 mg/dl"),
        level_two.replace("180 mg/dL", "180 mg/dL (n = 250)"),
    ):
        with pytest.raises(EvidenceEnforcementError, match="numeric value"):
            _bind(
                store,
                bound.replace(level_two, rewritten),
                **method_fact_binding(store, []),
            )


def test_only_the_numbers_a_fact_states_from_its_parameters_bind(tmp_path) -> None:
    store = _store(tmp_path, _descriptive(_grouping()))
    bound = _bound_draft(store)
    facts = store.manuscript_method_facts()
    missing = [
        dataclasses.replace(fact, typed_values=(0.0, 24.0, 70.0))
        if fact.source_field.endswith("levels[1]")
        else fact
        for fact in facts
    ]

    with pytest.raises(EvidenceEnforcementError, match="numeric value") as untraced:
        _bind(
            store,
            bound,
            method_facts=missing,
            bind_fact=method_fact_binding(store, [])["bind_fact"],
        )
    assert untraced.value.detail["untraced"] == ["180"]


def test_a_fact_line_outside_its_section_binds_nothing(tmp_path) -> None:
    store = _store(tmp_path, _descriptive(_grouping()))
    bound = _bound_draft(store)
    (level_two,) = [
        line for line in bound.splitlines() if "level 2:" in line and _PREFIX in line
    ]
    moved = bound.replace(level_two + "\n", "") + "\n## Results\n\n" + level_two + "\n"

    with pytest.raises(EvidenceEnforcementError, match="numeric value"):
        _bind(store, moved, **method_fact_binding(store, []))


def test_an_extra_number_on_an_exact_fact_line_is_still_untraced(tmp_path) -> None:
    grouping = _grouping()
    grouping["parameters"]["groups"][1]["rule"]["unit"] = "mg/dL (300 points)"
    store = _store(tmp_path, _descriptive(grouping))
    bound = _bound_draft(store)
    with pytest.raises(EvidenceEnforcementError) as rejected:
        _bind(store, bound, **method_fact_binding(store, []))
    assert set(rejected.value.detail["untraced"]) == {"300"}


def test_decimal_window_and_threshold_bind_to_each_own_level(tmp_path) -> None:
    grouping = _grouping()
    grouping["parameters"]["window"]["end_hours"] = 168.5
    grouping["parameters"]["groups"][1]["rule"]["value"] = 180.25
    store = _store(tmp_path, _descriptive(grouping))
    bound, bindings, untraced = _bind(
        store, _bound_draft(store), **method_fact_binding(store, [])
    )
    assert not untraced
    assert [(claim.value, claim.canonical) for claim in bindings.values()].count(
        ("168.5", 168.5)
    ) == 4
    assert [(claim.value, claim.canonical) for claim in bindings.values()].count(
        ("180.25", 180.25)
    ) == 2
    assert "sha256=" in bound


def test_a_fact_from_free_text_still_needs_a_result_claim(tmp_path) -> None:
    store = _store(
        tmp_path,
        _descriptive(description="A score increase of at least 300 points"),
    )
    bound = _bound_draft(store)

    with pytest.raises(EvidenceEnforcementError, match="numeric value") as untraced:
        _bind(store, bound, **method_fact_binding(store, []))
    assert untraced.value.detail["untraced"] == ["300"]


def test_the_sentence_filter_keeps_the_hosts_lines(tmp_path) -> None:
    store = _store(tmp_path, _descriptive(_grouping()))
    bound = _bound_draft(store).replace(
        "The study grouped its stays.",
        "Mortality was 412 deaths. The study grouped its stays.",
    )

    kept, removed = drop_untraceable_numeric_sentences(
        bound, evidence=store, per_step_records=[], **method_fact_binding(store, [])
    )

    assert [row["untraced"] for row in removed] == [["412"]]
    assert all(fact.text in kept for fact in _grouping_facts(store))
    # Bound alone, without the facts, the host's lines would go too.
    _kept, removed_alone = drop_untraceable_numeric_sentences(
        bound, evidence=store, per_step_records=[]
    )
    assert len(removed_alone) == 3


def test_a_line_the_writer_wrote_in_their_place_is_removed(tmp_path) -> None:
    store = _store(tmp_path, _descriptive(_grouping()))
    invented = (
        f"{_PREFIX}Glucose level 1: severe hypoglycaemia {{evidence:research_context}}."
    )

    with pytest.raises(EvidenceEnforcementError):
        store.enforce_evidence_bound_scaffold(
            "## Methods\n\n### Variables\n\n" + invented
        )


@pytest.mark.parametrize(
    "grouping",
    [
        pytest.param(
            {k: v for k, v in _grouping().items() if k != "parameters"},
            id="built_before_groupings_carried_parameters",
        ),
        pytest.param(
            _grouping(
                parameters=_parameters(
                    groups=[
                        {
                            "id": "g1",
                            "rule": {
                                "summary": "min",
                                "op": "<",
                                "value": 70,
                                "unit": "mg_dL",
                            },
                        },
                        {
                            "id": "g2",
                            "rule": {"summary": "max", "op": ">", "value": 180},
                        },
                        {"id": "g3", "rule": "otherwise"},
                    ]
                )
            ),
            id="a_unit_read_as_markup",
        ),
    ],
)
def test_a_grouping_that_names_no_plain_rule_gives_no_line(
    tmp_path, grouping: dict[str, Any]
) -> None:
    other = _grouping("glu_group_x2", parameters=_parameters(window=None))
    store = _store(tmp_path, _descriptive(grouping, other))

    facts = _grouping_facts(store)

    # The other grouping is stated; the one with no plain rule is not.
    assert [fact.source_field for fact in facts] == [
        f"cohort.provenance.{EXPOSURE_GROUP_LABELS_KEY}[1].levels[{index}]"
        for index in range(4)
    ]


def test_a_context_with_no_grouping_states_none(tmp_path) -> None:
    store = _store(tmp_path, _descriptive())

    facts = store.manuscript_method_facts()

    assert [fact.text for fact in facts] == [
        "Recorded source definition for the primary outcome: “In-hospital mortality”"
    ]
    assert [fact.typed_values for fact in facts] == [()]


@pytest.mark.parametrize(
    "grouping",
    [
        pytest.param(_grouping(concept="lact"), id="another_concept"),
        pytest.param(
            _grouping(
                levels=[
                    {"code": 1, "group": "g2", "label": "仅高血糖"},
                    {"code": 2, "group": "g1", "label": "低血糖"},
                    {"code": 3, "group": "g3", "label": "血糖正常"},
                    {"code": 4, "group": "gU", "label": "未测"},
                ]
            ),
            id="levels_in_another_order",
        ),
        pytest.param(
            _grouping(parameters=_parameters(unmeasured="exclude")),
            id="another_level_count",
        ),
        pytest.param(
            _grouping(parameters=_parameters(scale="sideways")),
            id="parameters_that_form_no_grouping",
        ),
        pytest.param(_grouping(variable=7), id="no_column"),
    ],
)
def test_parameters_that_do_not_form_the_stated_levels_stop_the_facts(
    tmp_path, grouping: dict[str, Any]
) -> None:
    store = _store(tmp_path, _descriptive(grouping))

    with pytest.raises(MethodFactAuthorityError):
        store.manuscript_method_facts()


def test_the_write_phase_binds_the_hosts_lines(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from easyicu.research_agent.reporting import write_phase
    from easyicu.research_agent.schema import CritiqueReport

    store = _store(tmp_path, _descriptive(_grouping()))
    # No step result: the counts-only report facts are another owner's.
    monkeypatch.setattr(
        write_phase, "compile_primary_counts_only_report_facts", lambda *a, **k: []
    )
    scaffold, _placed = place_manuscript_method_facts(
        "# Draft\n\n## Abstract\n\n**Background:** Context for the analysis is "
        "described here.\n\n**Methods:** The prespecified analysis was performed."
        "\n\n**Results:**\n\n**Conclusions:**\nIndependent validation is required."
        "\n\n## Methods\n\n### Variables\n\nThe study grouped its stays.\n\n"
        "## Results\n\n## Discussion\n\nIndependent validation is required.\n\n"
        "## Conclusion\n\nIndependent validation is required.",
        store.manuscript_method_facts(),
    )
    findings: list = []

    output = write_phase._bind_and_review_manuscript(
        SimpleNamespace(_evidence_enforcement_mode=EvidenceEnforcementMode.STRICT),
        critic=SimpleNamespace(
            review_manuscript=lambda **kwargs: CritiqueReport(
                reviewer="test fixture", status="blocked"
            )
        ),
        evidence=store,
        findings=findings,
        literature=None,
        per_step_records=[],
        current_evidence_names=["research_context"],
        scaffold=scaffold,
        writer_error_message=None,
        writer_probe_mode=False,
        writer_probe_failed_steps=(),
        run_dir=tmp_path,
        reader_display_labels={},
        manuscript_language="en",
    )

    unmarked = re.sub(r"\[\^claim_\d+\]", "", output.bound)
    assert all(fact.text in unmarked for fact in _grouping_facts(store))
    assert "180[^claim_1] mg/dL" in output.bound
    assert (
        f"field=cohort.provenance.{EXPOSURE_GROUP_LABELS_KEY}[0].levels[1]; evidence=research_context; sha256="
        in output.bound
    )
    assert not [
        finding
        for finding in findings
        if finding.validator == "manuscript_numeric_auditor"
        or (finding.detail or {}).get("reason_code")
        == "writer_source_method_facts_missing"
    ]


def test_report_only_binding_uses_verified_grouping_facts_without_mutation(
    tmp_path, monkeypatch
) -> None:
    from easyicu.research_agent.literature import LiteratureBundle
    from easyicu.research_agent.reporting.registered_report_inputs import (
        bind_registered_report_numbers,
    )
    from easyicu.research_agent.schema import AnalysisPlan

    context = _descriptive(_grouping())
    store = _store(tmp_path, context)
    (tmp_path / "research_context.json").write_text(context.model_dump_json())
    for name, evidence_id, payload in (
        (
            "analysis_plan.json",
            "analysis_plan",
            AnalysisPlan(
                research_question=context.research_question, steps=[]
            ).model_dump(),
        ),
        (
            "preplan_literature_bundle.json",
            "preplan_literature_bundle",
            LiteratureBundle(
                research_question=context.research_question, citations=[]
            ).model_dump(),
        ),
    ):
        source = tmp_path / name
        source.write_text(json.dumps(payload))
        store.register_file(
            kind="log",
            source_path=source,
            description="Synthetic registered input",
            evidence_id=evidence_id,
            producer="pipeline",
            generation_mode="system",
        )
    (tmp_path / "manifest.json").write_text(json.dumps({"per_step_records": []}))
    scaffold, _placed = place_manuscript_method_facts(
        "## Methods\n\n### Variables\n\nThe study grouped its stays.\n",
        store.manuscript_method_facts(),
    )
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        EvidenceStore, "__init__", lambda *a, **kw: pytest.fail("mutable store opened")
    )

    bound, count = bind_registered_report_numbers(tmp_path, scaffold)

    assert count == 2 and "180[^claim_1] mg/dL" in bound
    assert (
        "field=cohort.provenance.exposure_group_labels[0].levels[1]; evidence=research_context; sha256="
        in bound
    )
    with pytest.raises(EvidenceEnforcementError):
        bind_registered_report_numbers(
            tmp_path, scaffold.replace("180 mg/dL", "181 mg/dL")
        )
    assert before == {
        str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    }
