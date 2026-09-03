"""The last open STROBE item asks for a table the host produces under its own name.

STROBE item 15 -- "Report numbers of outcome events or summary measures over
time" -- listed five artefact names: ``outcome_rate``, ``outcome_incidence``,
``outcome_events``, ``mortality_by_exposure``, ``event_counts``.  Measured over
every recorded run that emitted a checklist, all five are absent from every one.

The fact itself is not absent.  ``exposure_outcome_distribution_executor`` is the
deterministic owner of exactly this product -- per exposure level: n, events,
rate and interval -- and registers it as ``table:exposure_outcome_distribution``.
On canary29's E1 run that step ran, passed, and bound its table; item 15 was
still reported open.

Producer and reader naming the same thing differently, one more time.  Fixed at
the reader here rather than the producer, because the producer's name is the
host's own typed product identity and the checklist row is a curated list of
artefacts that answer a reporting question -- the list was simply incomplete.

``reporting_checklist`` has no internal imports at all; it is a pure data table
plus pure helpers, and that is worth keeping.  So the coupling to the producer's
constant lives HERE, as a boundary contract test, instead of being introduced as
a module dependency from reporting into execution.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT,
)
from easyicu.research_agent.reporting.reporting_checklist import (
    _alias_satisfied,
    _available_aliases,
    build_strobe_checklist,
)


class _Rec:
    """A synthetic record, for cases where the test controls the ids."""

    def __init__(self, evidence_id: str) -> None:
        self.evidence_id = evidence_id
        self.relative_path = f"evidence/{evidence_id}__{evidence_id}.json"
        self.metadata: dict = {}
        self.description = ""
        self.kind = "table"


class _Whole:
    """A recorded evidence record, every field of it."""

    def __init__(self, record: dict) -> None:
        self.__dict__.update(record)


def _producer_product_name() -> str:
    """The bare name half of the owner's typed product identity."""

    kind, _, name = EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT.partition(":")
    assert kind == "table" and name, EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT
    return name


def _item_15(evidence_ids: list[str]):
    report = build_strobe_checklist(
        evidence_records=[_Rec(name) for name in evidence_ids],
        bound_manuscript="",
        task_kind="association_study",
    )
    return next(item for item in report.items if item.item_id == "15")


def test_the_outcome_count_item_credits_the_owners_own_table() -> None:
    """The property that was false."""

    item = _item_15(["cohort_flow", "table_one", _producer_product_name()])
    assert item.status == "addressed", item.rationale


def test_it_is_the_producers_constant_and_not_a_restated_string() -> None:
    """The boundary contract.

    If the owner ever renames its product, this fails here rather than silently
    reopening a reporting item on the next real run.
    """

    item = _item_15([])
    assert _producer_product_name() in item.required_evidence_aliases, (
        "the checklist row no longer names the product this owner registers: "
        f"{item.required_evidence_aliases}"
    )


def test_a_run_without_that_table_still_reports_the_item_open() -> None:
    """The item must keep its teeth.

    Crediting a product the run did not produce would turn a reporting gap into
    a silent pass, which is worse than the false alarm this replaces.
    """

    item = _item_15(["cohort_flow", "table_one", "analysis_plan"])
    assert item.status == "open"


def test_an_unrelated_table_does_not_credit_it() -> None:
    """Prefix matching is deliberate but must not reach sideways.

    ``_alias_satisfied`` credits descriptive suffixes of a required name. A
    different product that merely shares a word must not qualify.
    """

    item = _item_15(
        ["exposure_summary", "distribution_notes", "outcome_label_executability"]
    )
    assert item.status == "open"


# --- the recorded corpus ------------------------------------------------------

_CORPUS = Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


#: The commit that added ``exposure_outcome_distribution`` to STROBE item 15.
_ALIAS_COMMIT = "a2f4306"


def _ran_with_the_alias(run_dir: Path) -> bool:
    """Whether this run's recorded code contained the alias.

    Read from the run's own ``code_version.git_sha`` and answered by git, so a
    run cannot be mis-assigned by guessing from its timestamp.
    """

    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        sha = str(
            (json.loads(manifest.read_text()).get("code_version") or {}).get("git_sha")
            or ""
        )
    except (OSError, ValueError):
        return False
    if not sha:
        return False
    try:
        return (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", _ALIAS_COMMIT, sha],
                cwd=Path(__file__).resolve().parents[3],
                capture_output=True,
                timeout=20,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(
    not _CORPUS.exists(), reason="recorded runs are not on this machine"
)
def test_the_runs_that_reported_it_open_did_hold_that_table() -> None:
    """Real bytes: the fix has to reach the runs where this owner ran.

    Population is runs whose evidence holds the owner's table -- runs where the
    step actually executed.  It is NOT every run reporting item 15 open: one
    recorded run (2026-07-23) predates the deterministic owner entirely and has
    no such table, so nothing about renaming could reach it and asserting over
    it would fail on a different, older gap.

    Within the population, every run must currently report item 15 open, so the
    name is the whole gap.  A run that holds the table and still reports it open
    for some other reason would not be closed by this and must not pass here
    silently.

    DIRECTION FLIPPED 2026-08-01.  Until canary37 no recorded run had reached
    the writer stage at all, so every run holding the table reported the item
    ``open`` and this asserted exactly that -- the defect, as recorded.
    canary37 is the first run to get there, and it reports the item
    ``addressed``: the alias added in a2f4306 is credited in a real run.  So
    the assertion now says what must stay true rather than what used to be
    wrong, and this test fails again the moment a run holds the owner's table
    and is still told the item is open.
    """

    covered = 0
    without_the_owner = 0
    before_the_fix = 0
    still_open = []
    for checklist_path in sorted(
        _CORPUS.glob("batch_*/*/aware/run_*/reporting_checklist_strobe.json")
    ):
        index_path = checklist_path.parent / "evidence" / "evidence_index.json"
        if not index_path.is_file():
            continue
        try:
            report = json.loads(checklist_path.read_text())
            records = json.loads(index_path.read_text())
        except (OSError, ValueError):
            continue
        item = next(
            (i for i in report.get("items") or [] if i.get("item_id") == "15"), None
        )
        if item is None:
            continue
        # Whole records, not the thin stub the synthetic cases use:
        # ``_available_aliases`` derives names from metadata, description and
        # path as well as the id, and a stub carrying only the id reports a
        # smaller alias set than the run really had -- which reads as a defect
        # in the run instead of a gap in the fixture.
        aliases = _available_aliases([_Whole(r) for r in records])
        if not _alias_satisfied(_producer_product_name(), aliases):
            without_the_owner += 1
            continue
        # POPULATION = runs whose own code carried the alias. The corpus spans
        # both sides of the fix: run_20260801T130118_51bb78 executed at 171ac90
        # and reports the item open, which is the defect, not a regression.
        # Asking a pre-fix run to behave like a post-fix one is the
        # wrong-population mistake this suite has paid for before.
        if not _ran_with_the_alias(checklist_path.parent):
            before_the_fix += 1
            continue
        covered += 1
        if str(item.get("status")) == "open":
            still_open.append((checklist_path.parent.name, item.get("status")))

    # Asserted, not skipped. A silent skip is how a population filter bug turns
    # a watchdog off without anyone noticing, and this corpus HAS a covered run
    # (canary37). If that stops being true the right answer is a red test
    # saying so, not a green run saying nothing.
    assert covered, (
        "no recorded run both holds the owner's table and carried the alias "
        f"({without_the_owner} lack the table, {before_the_fix} predate the fix)"
    )
    assert not still_open, (
        "a recorded run holds the owner's table and STILL reports the "
        f"outcome-count item open, so the alias is not being credited: {still_open[:5]}"
    )
