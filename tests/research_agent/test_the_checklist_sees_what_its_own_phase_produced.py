"""The coverage report was handed a view of the run taken before it started.

``run_write_phase`` freezes one digest-verified evidence view near its top and
hands it to every analysis-facing writer consumer.  The freeze is deliberate and
its reason is good: the store is append-only across resume, so a reader walking
``evidence.records`` can pick up an old failed figure or statistic.

The reporting-guideline checklist is built ~750 lines later, at the END of the
same function, and was given that same frozen view.  Between the two, the write
phase registers the artefacts the checklist asks about.  So it reported STROBE
items as ``open`` -- "Awaiting: manuscript_scaffold_bound", "Awaiting:
multiple_testing_report, causal_audit_report" -- for artefacts the run had
already produced and bound.

Verified on canary29's E1 run by running the host's own resolver
(``_available_aliases`` + ``_alias_satisfied``) against that run's own
``evidence/evidence_index.json``: ``manuscript_scaffold_bound``,
``multiple_testing_report`` and ``causal_audit_report`` all resolve.  The
checklist emitted by the same run says it is awaiting all three.

That cost E1 its paper verdict.  It executed 13/13 steps with zero failures and
scored plan 1.0, code 1.0, evidence-binding 1.0 -- and stopped at
``analysis_only`` on reporting completeness 0.833, "3/22 STROBE item(s) open".

Two of those three are this defect.  The checklist is a coverage report over what
the run produced, not an analysis-facing consumer, so it takes its own snapshot
immediately before it is built; every earlier consumer keeps the frozen view.

The third is NOT this defect and is not fixed here: item 15 wants any of
``outcome_rate``, ``outcome_incidence``, ``outcome_events``,
``mortality_by_exposure``, ``event_counts``, and the run publishes that same fact
under ``exposure_outcome_distribution`` / ``prevalence_mortality``.  A separate
producer/reader naming gap, recorded rather than absorbed here.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from easyicu.research_agent.reporting import write_phase
from easyicu.research_agent.reporting.reporting_checklist import (
    _alias_satisfied,
    _available_aliases,
    build_strobe_checklist,
)


class _Rec:
    """The two fields ``_available_aliases`` reads off an evidence record."""

    def __init__(self, evidence_id: str) -> None:
        self.evidence_id = evidence_id
        self.relative_path = f"evidence/{evidence_id}__{evidence_id}.json"
        self.metadata: dict = {}
        self.description = ""
        self.kind = "log"


_WRITE_PHASE_ARTEFACTS = ("manuscript_scaffold_bound", "causal_audit_report")


def _statuses(*, include_write_phase_artefacts: bool) -> dict[str, str]:
    ids = ["cohort_flow", "table_one", "analysis_plan", "research_context"]
    if include_write_phase_artefacts:
        ids += list(_WRITE_PHASE_ARTEFACTS)
    report = build_strobe_checklist(
        evidence_records=[_Rec(name) for name in ids],
        bound_manuscript="",
        task_kind="association_study",
    )
    return {item.item_id: item.status for item in report.items}


def test_the_items_close_once_the_phases_own_artefacts_are_visible() -> None:
    """The property that was false.

    1b is the abstract, satisfied by the bound scaffold.  12e is sensitivity
    analyses, whose alias row is a list of ALTERNATIVES, so the causal audit
    alone satisfies it -- which matters, because the multiple-testing report is
    produced in a later phase and no snapshot inside this one can see it.
    """

    after = _statuses(include_write_phase_artefacts=True)
    assert after["1b"] == "addressed"
    assert after["12e"] == "addressed"


def test_the_defect_itself_is_pinned() -> None:
    """Without them the same two items are open -- what the run reported."""

    before = _statuses(include_write_phase_artefacts=False)
    assert before["1b"] == "open"
    assert before["12e"] == "open"


def test_the_item_this_does_not_fix_stays_open() -> None:
    """Honest scope.

    Item 15 asks for outcome counts under five names the run does not use. It
    is a real gap and a different one; if it started passing here that would
    mean this test had stopped measuring what it claims.
    """

    assert _statuses(include_write_phase_artefacts=True)["15"] == "open"


# --- the wiring ---------------------------------------------------------------


def _phase_function_tree(name: str) -> ast.FunctionDef:
    tree = ast.parse(inspect.getsource(write_phase))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is gone")


def _publication_stage_tree() -> ast.FunctionDef:
    return _phase_function_tree("_publish_and_audit_manuscript")


def _checklist_evidence_argument() -> str:
    fn = _publication_stage_tree()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != "build_strobe_checklist":
            continue
        for kw in node.keywords:
            if kw.arg == "evidence_records":
                return ast.unparse(kw.value)
    raise AssertionError("the STROBE checklist no longer takes evidence records")


def _assignment_line(target_name: str) -> int:
    fn = _publication_stage_tree()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == target_name for t in node.targets
        ):
            return node.lineno
    raise AssertionError(f"nothing assigns {target_name!r} in run_write_phase")


def _registration_line(evidence_id: str) -> int:
    fn = _publication_stage_tree()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if (
                kw.arg == "evidence_id"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == evidence_id
            ):
                return node.lineno
    raise AssertionError(f"{evidence_id!r} is no longer registered in this phase")


def test_the_checklist_snapshot_is_taken_after_the_phase_registers_its_own() -> None:
    """The wiring, not just the resolver.

    Ordering IS the defect: the same call with the same arguments is right or
    wrong depending only on when the view it reads was taken. Anchored on the
    causal-audit registration because that is the artefact inside this phase
    that item 12e depends on.
    """

    argument = _checklist_evidence_argument()
    assert argument.isidentifier(), (
        "the checklist is handed an expression rather than a named snapshot, so "
        f"when it was taken cannot be read: {argument!r}"
    )
    snapshot_line = _assignment_line(argument)
    registration_line = _registration_line("causal_audit_report")
    assert snapshot_line > registration_line, (
        "the checklist reads a view frozen before this phase registered the "
        f"artefacts it asks about (snapshot line {snapshot_line}, causal-audit "
        f"registration line {registration_line})"
    )


def test_the_snapshot_is_the_same_digest_verified_kind() -> None:
    """Not a looser read.

    The freeze exists to keep old, failed or superseded evidence out of the
    manuscript. A fresher view must still be the current/digest-verified one,
    or this would trade a stale report for an unverified one.
    """

    fn = _publication_stage_tree()
    argument = _checklist_evidence_argument()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == argument for t in node.targets
        ):
            assert "current_verified_records" in ast.unparse(node.value), (
                "the checklist snapshot is not the digest-verified view: "
                + ast.unparse(node.value)
            )
            return
    raise AssertionError(f"nothing assigns {argument!r}")


def test_the_analysis_facing_consumers_keep_the_frozen_view() -> None:
    """The freeze must not be widened for everyone.

    Its stated purpose is that a manuscript reader cannot pick up an old failed
    figure or statistic across resume. Re-snapshotting for a coverage report is
    safe; re-snapshotting for the writer is exactly what the freeze forbids.
    """

    draft = _phase_function_tree("_draft_manuscript")
    frozen_users = [
        node
        for node in ast.walk(draft)
        if isinstance(node, ast.Name) and node.id == "current_verified_evidence_records"
    ]
    verified_snapshots = [
        node
        for node in ast.walk(draft)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "current_verified_records"
    ]
    # The draft stage takes exactly one verified snapshot and hands that same
    # named value to its analysis-facing readers and immutable stage result.
    assert len(verified_snapshots) == 1
    assert len(frozen_users) >= 3, (
        "the frozen analysis-facing view has lost its readers; the freeze was "
        f"widened rather than a second snapshot added ({len(frozen_users)} uses)"
    )


# --- the recorded corpus ------------------------------------------------------

_CORPUS = Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


@pytest.mark.skipif(
    not _CORPUS.exists(), reason="recorded runs are not on this machine"
)
def test_recorded_checklists_await_artefacts_their_own_run_already_had() -> None:
    """Real bytes: the defect, measured where it happened.

    Recorded runs predate the fix, so items awaiting a satisfied alias are
    expected here; what is asserted is that every such alias is one THIS phase
    registers. An item awaiting something produced in a different phase would
    not be fixed by re-snapshotting inside this one and must not pass silently.
    """

    unexplained = []
    stale = 0
    for checklist_path in sorted(
        _CORPUS.glob("batch_*/*/aware/run_*/reporting_checklist_*.json")
    ):
        index_path = checklist_path.parent / "evidence" / "evidence_index.json"
        if not index_path.is_file():
            continue
        try:
            report = json.loads(checklist_path.read_text())
            records = json.loads(index_path.read_text())
        except (OSError, ValueError):
            continue
        aliases = _available_aliases([_Rec(str(r.get("evidence_id"))) for r in records])
        for item in report.get("items") or []:
            rationale = str(item.get("rationale") or "")
            if not rationale.startswith("Awaiting:"):
                continue
            wanted = [
                part.strip()
                for part in rationale[len("Awaiting:") :].split(",")
                if part.strip()
            ]
            satisfied = [name for name in wanted if _alias_satisfied(name, aliases)]
            if not satisfied:
                continue
            stale += 1
            outside = [name for name in satisfied if name not in _WRITE_PHASE_ARTEFACTS]
            if outside and not any(
                name in _WRITE_PHASE_ARTEFACTS for name in satisfied
            ):
                unexplained.append(
                    (checklist_path.parent.name, item.get("item_id"), outside)
                )

    if not stale:
        pytest.skip("no recorded checklist awaited an artefact the run already had")
    assert not unexplained, (
        "recorded items await a satisfied alias this phase does not register, so "
        f"re-snapshotting here would not close them: {unexplained[:5]}"
    )
