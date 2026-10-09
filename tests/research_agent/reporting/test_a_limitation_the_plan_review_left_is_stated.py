"""A limitation the approved plan review left to the study is stated in Limitations.

The scientific review of the approved plan is bound into the approval by its
digest.  A major finding the review routes to ``study_authority_change`` is a
design limitation the study kept, so the host places its fixed sentence at the
top of ``## Limitations``, after the source facts, and the bound manuscript
fails closed without it.  Each sentence passes the unchanged claim policy, the
binder and the numeric binder.  The sentences and the review name the same
codes: every code the review owner can raise as such a finding has a sentence,
and every sentence names a code the owner raises.  A review that changed after
the approval, or a finding with no sentence, stops the manuscript with a typed
reason.  Synthetic reviews only.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

import pytest

import easyicu.research_agent as research_agent
from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.planning.scientific_review import (
    PlanScientificFinding,
    PlanScientificReview,
)
from easyicu.research_agent.reporting.manuscript_method_facts import (
    place_manuscript_method_facts,
)
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.reporting.plan_review_limitations import (
    PLAN_REVIEW_LIMITATION_SENTENCES,
    PLAN_REVIEW_LIMITATIONS_MISSING,
    PLAN_REVIEW_LIMITATION_UNMAPPED,
    PLAN_REVIEW_UNVERIFIED,
    audit_bound_plan_review_limitations,
    writer_plan_review_limitations,
)

REVIEW_ID = "scientific_plan_review"
#: The Conclusion carries no sentence: one without evidence would fail the
#: strict policy for its own reason.
LIMITATIONS = "## Limitations\n\nThe study is observational.\n\n## Conclusion\n"


def _finding(code: str, severity: str = "major", **fields) -> PlanScientificFinding:
    return PlanScientificFinding(
        code=code,
        severity=severity,
        dimension="icu_clinical_design",
        message="A finding.",
        remediation="Revise the study.",
        **fields,
    )


def _store(tmp_path: Path, findings: list[PlanScientificFinding]) -> EvidenceStore:
    review = PlanScientificReview(
        status="analysis_only",
        approval_allowed=True,
        top_journal_candidate=False,
        score=70,
        dimension_scores={},
        findings=findings,
        context_sha256="a" * 64,
        plan_sha256="b" * 64,
        literature_sha256="c" * 64,
        figure_strategy_sha256="d" * 64,
        generated_at="2026-10-09T00:00:00Z",
    )
    source = tmp_path / "scientific_plan_review.json"
    source.write_text(review.model_dump_json(indent=2), encoding="utf-8")
    store = EvidenceStore(tmp_path, enforcement_mode=EvidenceEnforcementMode.STRICT)
    store.register_file(
        kind="log",
        source_path=source,
        description="Pre-execution scientific review of the exact plan.",
        evidence_id=REVIEW_ID,
        producer="plan_scientific_review",
        generation_mode="deterministic_skill",
    )
    return store


def _kept(code: str) -> PlanScientificFinding:
    return _finding(code, remediation_route="study_authority_change")


def _all_kept() -> list[PlanScientificFinding]:
    return [_kept(code) for code in PLAN_REVIEW_LIMITATION_SENTENCES]


# -- which findings the manuscript states ------------------------------------------


def test_only_a_major_finding_left_to_the_study_is_stated(tmp_path) -> None:
    store = _store(
        tmp_path,
        [
            _kept("POPULATION_CRITERION_NOT_APPLIED"),
            # A second criterion the data cannot apply: one sentence.
            _kept("POPULATION_CRITERION_NOT_APPLIED"),
            # The researcher's authorization routes it to the study.
            _finding(
                "ADJUSTMENT_RATIONALE_OR_TIMING_UNBOUND",
                requires_user_authorization=True,
            ),
            # The Planner revises it, or it is minor: no sentence.
            _finding(
                "UNADJUSTED_ASSOCIATION_NOT_ARTICLE_GRADE",
                remediation_route="agent_plan_revision",
            ),
            _finding(
                "POPULATION_SCOPE_AMENDMENT_DECLARED",
                severity="minor",
                remediation_route="study_authority_change",
            ),
        ],
    )

    limitations, failure = writer_plan_review_limitations(store)

    assert failure is None
    assert [item.code for item in limitations] == [
        "POPULATION_CRITERION_NOT_APPLIED",
        "ADJUSTMENT_RATIONALE_OR_TIMING_UNBOUND",
    ]
    assert limitations[0].scaffold == (
        PLAN_REVIEW_LIMITATION_SENTENCES["POPULATION_CRITERION_NOT_APPLIED"]
        + " {evidence:scientific_plan_review}."
    )
    assert limitations[0].source_field == (
        "scientific_plan_review.findings.POPULATION_CRITERION_NOT_APPLIED"
    )


def test_a_run_without_a_plan_review_states_none(tmp_path) -> None:
    store = EvidenceStore(tmp_path, enforcement_mode=EvidenceEnforcementMode.STRICT)

    assert writer_plan_review_limitations(store) == ((), None)
    assert (
        audit_bound_plan_review_limitations(
            LIMITATIONS, evidence=store, per_step_records=[]
        )
        is None
    )


def test_a_finding_without_a_sentence_stops_the_manuscript(tmp_path) -> None:
    store = _store(tmp_path, [_kept("A_LIMITATION_NO_SENTENCE_STATES")])

    limitations, failure = writer_plan_review_limitations(store)

    assert limitations == ()
    assert failure is not None and failure.severity == "error"
    assert failure.validator == "evidence_bound_writer"
    assert failure.detail["reason_code"] == PLAN_REVIEW_LIMITATION_UNMAPPED
    assert failure.detail["finding_codes"] == ["A_LIMITATION_NO_SENTENCE_STATES"]
    # Reported once, where the limitations are placed.
    assert (
        audit_bound_plan_review_limitations(
            LIMITATIONS, evidence=store, per_step_records=[]
        )
        is None
    )


def test_a_review_changed_after_the_approval_is_not_read(tmp_path) -> None:
    store = _store(tmp_path, _all_kept())
    record = store.get(REVIEW_ID)
    path = tmp_path / record.relative_path
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    limitations, failure = writer_plan_review_limitations(store)

    assert limitations == ()
    assert failure is not None
    assert failure.detail["reason_code"] == PLAN_REVIEW_UNVERIFIED


# -- placement, the unchanged policy and the audit --------------------------------


class _SourceFact:
    """A source fact of the Limitations section, as a host owner places it."""

    section = "limitations"
    scaffold = "Target trial emulation assumptions: stated {evidence:research_context}."
    source_field = "step.executed_method_design.assumptions"
    source_sha256 = "e" * 64


def test_the_limitations_follow_the_source_facts_at_the_top(tmp_path) -> None:
    limitations, _ = writer_plan_review_limitations(_store(tmp_path, _all_kept()))

    placed, fields = place_manuscript_method_facts(
        LIMITATIONS, (_SourceFact(), *limitations)
    )

    body = placed.split("## Limitations")[1].split("## Conclusion")[0]
    lines = [line for line in body.splitlines() if line]
    assert lines == [
        _SourceFact.scaffold,
        *(item.scaffold for item in limitations),
        "The study is observational.",
    ]
    assert fields[1:] == tuple(item.source_field for item in limitations)
    # Placing again adds nothing.
    assert place_manuscript_method_facts(placed, (_SourceFact(), *limitations)) == (
        placed,
        (),
    )


def test_every_sentence_survives_the_policy_and_the_binders(tmp_path) -> None:
    store = _store(tmp_path, _all_kept())
    limitations, _ = writer_plan_review_limitations(store)
    placed, _ = place_manuscript_method_facts(LIMITATIONS, limitations)

    safe, removed = store.enforce_evidence_bound_scaffold(placed, per_step_records=[])
    assert not removed
    bound = store.bind_manuscript(safe, per_step_records=[])
    _, _, untraced = bind_numeric_values(bound, evidence=store, per_step_records=[])

    assert not untraced
    assert len(limitations) == len(PLAN_REVIEW_LIMITATION_SENTENCES)
    assert (
        audit_bound_plan_review_limitations(bound, evidence=store, per_step_records=[])
        is None
    )


def test_a_dropped_limitation_fails_the_manuscript(tmp_path) -> None:
    store = _store(tmp_path, _all_kept())
    limitations, _ = writer_plan_review_limitations(store)
    placed, _ = place_manuscript_method_facts(LIMITATIONS, limitations[1:])
    bound = store.bind_manuscript(placed, per_step_records=[])

    finding = audit_bound_plan_review_limitations(
        bound, evidence=store, per_step_records=[]
    )

    assert finding is not None and finding.severity == "error"
    assert finding.detail["reason_code"] == PLAN_REVIEW_LIMITATIONS_MISSING
    assert finding.detail["source_fields"] == [limitations[0].source_field]


def test_the_report_only_projection_states_them_too(tmp_path) -> None:
    from easyicu.research_agent.reporting.writer_only_migration import (
        _claim_policy_projection,
    )

    _store(tmp_path, _all_kept())

    projected, errors = _claim_policy_projection(tmp_path, LIMITATIONS)

    assert not errors
    for text in PLAN_REVIEW_LIMITATION_SENTENCES.values():
        assert text in projected


# -- the sentences and the review owner name the same codes -----------------------


def _keyword_branches(call: ast.Call) -> list[dict[str, ast.expr]]:
    """Each consistent reading of the call's keywords under its conditions."""

    keywords = {item.arg: item.value for item in call.keywords if item.arg}
    tests: dict[str, ast.expr] = {}

    def collect(node: ast.expr) -> None:
        if isinstance(node, ast.IfExp):
            tests.setdefault(ast.dump(node.test), node.test)
            collect(node.body)
            collect(node.orelse)

    for value in keywords.values():
        collect(value)
    branches = []
    for truth in itertools.product((True, False), repeat=len(tests)):
        chosen = dict(zip(tests, truth))

        def resolve(node: ast.expr) -> ast.expr:
            while isinstance(node, ast.IfExp):
                node = node.body if chosen[ast.dump(node.test)] else node.orelse
            return node

        branches.append({name: resolve(value) for name, value in keywords.items()})
    return branches


def _literal(node: ast.expr | None):
    return node.value if isinstance(node, ast.Constant) else ...


def _codes_left_to_the_study() -> tuple[set[str], list[str]]:
    """Codes the review owner can raise as a major finding routed to the study."""

    package = Path(research_agent.__file__).parent
    codes: set[str] = set()
    unresolved: list[str] = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in ast.walk(tree):
            if not (
                isinstance(call, ast.Call)
                and getattr(call.func, "id", getattr(call.func, "attr", None))
                == "PlanScientificFinding"
            ):
                continue
            where = f"{path.relative_to(package)}:{call.lineno}"
            if any(item.arg is None for item in call.keywords) or call.args:
                unresolved.append(where)
                continue
            for branch in _keyword_branches(call):
                severity = _literal(branch.get("severity"))
                route = _literal(
                    branch.get("remediation_route", ast.Constant("unclassified"))
                )
                authorized = _literal(
                    branch.get("requires_user_authorization", ast.Constant(False))
                )
                if severity not in ("major", ...):
                    continue
                if route not in ("study_authority_change", ...) and authorized is False:
                    continue
                code = _literal(branch.get("code"))
                if isinstance(code, str):
                    codes.add(code)
                else:
                    unresolved.append(where)
    return codes, unresolved


def _later_changes_to_severity_or_route() -> list[str]:
    """``model_copy`` updates of a finding's severity, route or authorization."""

    source = Path(research_agent.__file__).parent / "planning" / "scientific_review.py"
    changes = []
    for call in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if not (
            isinstance(call, ast.Call)
            and getattr(call.func, "attr", None) == "model_copy"
        ):
            continue
        for item in call.keywords:
            if item.arg != "update" or not isinstance(item.value, ast.Dict):
                continue
            for key, value in zip(item.value.keys, item.value.values):
                name = _literal(key)
                # The review's routing pass may only resolve the route its owner
                # assigns (``remediation_route_for_finding``).
                if (
                    name == "remediation_route"
                    and isinstance(value, ast.Call)
                    and (
                        getattr(value.func, "id", None)
                        == "remediation_route_for_finding"
                    )
                ):
                    continue
                if name in (
                    "severity",
                    "remediation_route",
                    "requires_user_authorization",
                ):
                    changes.append(f"scientific_review.py:{call.lineno} {name}")
    return changes


def test_the_sentences_and_the_review_name_the_same_codes() -> None:
    assert _later_changes_to_severity_or_route() == []
    codes, unresolved = _codes_left_to_the_study()

    assert unresolved == []
    assert codes == set(PLAN_REVIEW_LIMITATION_SENTENCES)
