"""A retry earns one fresh repair budget per identity it runs on, and no more."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from easyicu.research_agent.authority.provider_budget import (
    StepProviderCallBudget,
    provider_call_budget_receipt_path,
)
from easyicu.research_agent.contracts.retry_policy import retry_accounting_receipt
from easyicu.research_agent.execution import budget_epoch, retry_basis
from easyicu.research_agent.execution.budget_epoch import (
    EPOCH_FIELD,
    EPOCH_REPAIRS_FIELD,
    IDENTITY_FIELD,
    AttemptIdentity,
    epoch_ledger_path,
    epoch_receipt_path,
    load_epoch_ledger,
    select_budget_epoch,
)
from easyicu.research_agent.execution.retry_basis import load_failed_step_retry_basis
from easyicu.research_agent.execution.step_attempt_bootstrap import (
    prepare_step_attempt_bootstrap,
)
from easyicu.research_agent.repairs.coordination import StepRepairBudget
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

STEP = "s1"
A = AttemptIdentity("a" * 64, "p" * 64, "k" * 64, "sha256:" + "1" * 64)
B = replace(A, engine_code_sha256="b" * 64)


def _failed(**fields: Any) -> dict[str, Any]:
    return {
        "step_id": STEP,
        "status": "contract_failed",
        "attempt_sequence": 1,
        "step_llm_repair_attempts": 2,
        "step_llm_repair_budget": 2,
        "step_llm_repair_classes": ["runtime", "contract"],
        **fields,
    }


def _spent_in(epoch: int, identity: AttemptIdentity, *, cumulative: int, local: int) -> dict[str, Any]:
    return _failed(
        attempt_sequence=epoch + 1,
        step_llm_repair_attempts=cumulative,
        step_llm_repair_classes=["runtime"] * local,
        **{EPOCH_FIELD: epoch, IDENTITY_FIELD: identity.payload(), EPOCH_REPAIRS_FIELD: local},
    )


def _select(run_dir: Path, records, *, current=B, explicit=True, commit=True):
    return select_budget_epoch(
        run_dir=run_dir,
        step_id=STEP,
        records=records,
        latest_record=records[-1] if records else None,
        current_identity=current,
        explicit_rerun=explicit,
        attempt_id=f"run:{STEP}:9",
        reserved_final_category=None,
        commit=commit,
    )


@pytest.fixture
def capsule(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The identity the step's checkpoint-selected capsule ran on."""

    state: dict[str, Any] = {"identity": A}

    def read(*_args: Any, **_kwargs: Any):
        return state["identity"]

    monkeypatch.setattr(budget_epoch, "checkpoint_capsule_identity", read)
    monkeypatch.setattr(retry_basis, "checkpoint_capsule_identity", read)
    return state


def test_a_retry_under_changed_code_opens_one_fresh_epoch(tmp_path: Path, capsule) -> None:
    epoch = _select(tmp_path, [_failed()])

    assert (epoch.epoch, epoch.error) == (1, None)
    assert epoch.records == () and epoch.prior_record is None
    assert epoch.repair_count_offset == 2
    assert epoch.counter_field == EPOCH_REPAIRS_FIELD
    assert epoch.receipt_path == epoch_receipt_path(tmp_path, step_id=STEP, epoch=1)
    (grant,) = load_epoch_ledger(tmp_path, step_id=STEP)
    assert (grant.identity, grant.superseded_identity) == (B, A)
    assert grant.changed_components == ("engine_code_sha256",)
    assert grant.opened_by_attempt_id == f"run:{STEP}:9"


def test_an_unchanged_identity_spends_what_is_left(tmp_path: Path, capsule) -> None:
    epoch = _select(tmp_path, [_failed()], current=A)

    assert (epoch.epoch, epoch.opened, len(epoch.records)) == (0, None, 1)
    assert not epoch_ledger_path(tmp_path, step_id=STEP).exists()


def test_each_identity_is_granted_once(tmp_path: Path, capsule) -> None:
    first = _select(tmp_path, [_failed()])
    records = [_failed(), _spent_in(1, B, cumulative=4, local=2)]

    again = _select(tmp_path, records, current=B)
    rollback = _select(tmp_path, records, current=A)
    third = _select(tmp_path, records, current=replace(A, engine_code_sha256="c" * 64))

    assert first.epoch == 1
    # B already had its grant, and A spent the original budget.
    assert (again.epoch, again.opened, len(again.records)) == (1, None, 1)
    assert (rollback.epoch, rollback.opened) == (1, None)
    # A third identity is new: its grant starts after the four repairs spent.
    assert (third.epoch, third.repair_count_offset) == (2, 4)
    assert len(load_epoch_ledger(tmp_path, step_id=STEP)) == 2


def test_a_crash_after_the_grant_reuses_the_same_unspent_epoch(tmp_path: Path, capsule) -> None:
    _select(tmp_path, [_failed()])

    again = _select(tmp_path, [_failed()])

    assert (again.epoch, again.opened, again.records) == (1, None, ())
    assert again.repair_count_offset == 2
    assert len(load_epoch_ledger(tmp_path, step_id=STEP)) == 1


def test_a_read_only_selection_writes_nothing(tmp_path: Path, capsule) -> None:
    epoch = _select(tmp_path, [_failed()], commit=False)

    assert epoch.opened is not None and epoch.epoch == 1
    assert not epoch_ledger_path(tmp_path, step_id=STEP).exists()


def test_an_unprovable_original_identity_opens_nothing(tmp_path: Path, capsule) -> None:
    capsule["identity"] = None

    epoch = _select(tmp_path, [_failed()])

    assert (epoch.epoch, epoch.opened) == (0, None)


@pytest.mark.parametrize(
    "latest",
    [
        _failed(capsule_pending_repair_attempt_id=2),
        _failed(step_llm_repair_history_invalid=True),
        _failed(provider_call_budget_receipt_invalid=True),
        _failed(step_llm_repair_attempts="two"),
        _failed(status="ok"),
    ],
    ids=["repair_in_flight", "history_invalid", "receipt_invalid", "malformed_counter", "not_failed"],
)
def test_a_step_that_is_not_a_settled_failure_keeps_its_epoch(
    tmp_path: Path, capsule, latest: dict[str, Any]
) -> None:
    epoch = _select(tmp_path, [latest])

    assert (epoch.epoch, epoch.opened) == (0, None)


def test_only_an_explicit_rerun_opens_an_epoch(tmp_path: Path, capsule) -> None:
    epoch = _select(tmp_path, [_failed()], explicit=False)

    assert (epoch.epoch, epoch.opened) == (0, None)


def test_a_rebuilt_image_opens_an_epoch_but_an_unknown_one_does_not(
    tmp_path: Path, capsule
) -> None:
    rebuilt = _select(tmp_path / "rebuilt", [_failed()], current=replace(A, image_id="sha256:" + "2" * 64))
    unknown = _select(tmp_path / "unknown", [_failed()], current=replace(A, image_id=None))

    assert rebuilt.epoch == 1 and rebuilt.opened.changed_components == ("image_id",)
    assert (unknown.epoch, unknown.opened) == (0, None)


def test_a_damaged_ledger_an_orphan_receipt_or_an_unknown_tag_fails_closed(
    tmp_path: Path, capsule
) -> None:
    _select(tmp_path, [_failed()])
    ledger = epoch_ledger_path(tmp_path, step_id=STEP)
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["epochs"][0]["repair_count_offset"] = 0
    ledger.write_text(json.dumps(payload), encoding="utf-8")
    orphan = tmp_path / "orphan"
    receipt = epoch_receipt_path(orphan, step_id=STEP, epoch=1)
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}", encoding="utf-8")

    assert "digest" in _select(tmp_path, [_failed()]).error
    assert "no ledger entry" in _select(orphan, [_failed()]).error
    assert "unknown budget epoch" in _select(
        tmp_path / "tagged", [_failed(**{EPOCH_FIELD: 3})]
    ).error


def test_the_receipt_left_behind_is_bound_and_never_rewritten(tmp_path: Path, capsule) -> None:
    receipt = provider_call_budget_receipt_path(tmp_path, step_id=STEP)
    StepProviderCallBudget(5, step_id=STEP, receipt_path=receipt).consume("coder_generation")
    before = receipt.read_bytes()

    epoch = _select(tmp_path, [_failed()])

    (grant,) = load_epoch_ledger(tmp_path, step_id=STEP)
    assert grant.previous_receipt_sha256 == hashlib.sha256(before).hexdigest()
    assert epoch.receipt_path != receipt and receipt.read_bytes() == before


def test_a_call_in_flight_on_the_old_receipt_keeps_the_epoch(tmp_path: Path, capsule) -> None:
    receipt = provider_call_budget_receipt_path(tmp_path, step_id=STEP)
    provider = StepProviderCallBudget(5, step_id=STEP, receipt_path=receipt)
    assert StepRepairBudget(provider_budget=provider, step_record={}, max_llm_repairs=2).consume(
        "runtime"
    )

    epoch = _select(tmp_path, [_failed()])

    assert (epoch.epoch, epoch.opened) == (0, None)


def _bootstrap(tmp_path: Path, prior: dict[str, Any], *, explicit: bool, findings: list):
    step = AnalysisStep(
        step_id=STEP,
        intent="Summarize the locked cohort.",
        inputs=["stay_id"],
        expected_outputs=["table:summary"],
        method="descriptive_summary",
    )
    universe = tmp_path / "cohort_universe.parquet"
    cohort = tmp_path / "cohort_analysis.parquet"
    universe.write_bytes(b"universe")
    cohort.write_bytes(b"cohort")
    return prepare_step_attempt_bootstrap(
        resume_state={"step_attempt_history": [prior]},
        per_step_records=[],
        shared_lock=threading.Lock(),
        step=step,
        plan=AnalysisPlan(research_question="Summarize the cohort.", steps=[step]),
        run_id="run",
        run_dir=tmp_path,
        universe_path=universe,
        cohort_path=cohort,
        plan_scientific_signature=[{"step_id": STEP}],
        findings=findings,
        max_provider_calls=5,
        max_llm_repairs=2,
        reserve_concept_audit=False,
        allow_terminal_initial_generation_restart=explicit,
        explicit_rerun=explicit,
        current_identity=B,
    )


def test_an_explicit_retry_under_new_code_can_repair_again(tmp_path: Path, capsule) -> None:
    prior = _failed()
    findings: list = []

    result = _bootstrap(tmp_path, prior, explicit=True, findings=findings)

    repair = result.budget_runtime.repair_budget
    assert result.budget_runtime.integrity_error is None
    assert repair.llm_repair_attempts == 0 and repair.available("runtime")
    assert result.budget_runtime.receipt_path.name.endswith(".epoch-1.json")
    assert result.step_record[EPOCH_FIELD] == 1
    assert result.step_record[IDENTITY_FIELD] == B.payload()
    assert result.step_record["step_llm_repair_attempts"] == 2
    assert result.step_record[EPOCH_REPAIRS_FIELD] == 0
    # The capsule is still selected from the full history.
    assert result.prior_step_record is prior
    assert [finding.validator for finding in findings] == ["step_budget_epoch"]

    assert repair.consume("runtime")
    assert result.step_record["step_llm_repair_attempts"] == 3
    assert result.step_record[EPOCH_REPAIRS_FIELD] == 1


def test_a_resume_that_is_not_an_explicit_retry_stays_exhausted(tmp_path: Path, capsule) -> None:
    findings: list = []

    result = _bootstrap(tmp_path, _failed(), explicit=False, findings=findings)

    repair = result.budget_runtime.repair_budget
    assert repair.llm_repair_attempts == 2 and not repair.available("runtime")
    assert EPOCH_FIELD not in result.step_record and findings == []


def test_run_accounting_counts_repairs_across_epochs(tmp_path: Path, capsule) -> None:
    prior = {**_failed(), "attempt_id": "run:s1:1"}
    result = _bootstrap(tmp_path, prior, explicit=True, findings=[])
    assert result.budget_runtime.repair_budget.consume("runtime")

    receipt = retry_accounting_receipt([prior, result.step_record])

    # Two repairs under the original identity and one under the new one; an
    # epoch-local counter would have reported the maximum, two.
    assert receipt["logical_llm_repair_attempts"] == 3


def _run(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"per_step_records": records}), encoding="utf-8"
    )
    return run_dir


def test_the_retry_basis_reads_the_latest_epoch(tmp_path: Path, capsule) -> None:
    # The original budget is spent and says so; that flag must not follow the
    # step into the epoch opened for new code.
    original = _failed(step_llm_repair_budget_exhausted=True)
    run_dir = _run(tmp_path, [original])
    _select(run_dir, [original])

    opened = load_failed_step_retry_basis(run_dir, STEP)
    _run(tmp_path, [original, _spent_in(1, B, cumulative=4, local=2)])
    spent = load_failed_step_retry_basis(run_dir, STEP)

    assert (opened.budget_epoch, opened.repair_attempts) == (1, 0)
    assert opened.repair_budget_exhausted is False
    assert set(opened.used_identities) == {A, B}
    assert (spent.budget_epoch, spent.repair_attempts) == (1, 2)
    assert spent.repair_budget_exhausted is True
