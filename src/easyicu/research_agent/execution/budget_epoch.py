"""One bounded repair grant per attempt identity for a resumed step.

A step's LLM-repair and provider-call budgets are restored monotonically from
its attempt records and its durable provider receipt, so a retry of a failed
step can only replay what the failure left.  A retry earns a fresh budget only
when what the step runs on changed since then: the research-agent code, the
prompt pack, the execution kernel or the runner image.  Each such identity is
granted once.  The grant is appended to a ledger beside the step's receipt and
spends from a receipt of its own, so earlier receipts are never rewritten and
an identity that already had its grant (a rollback) earns nothing more.

Epoch 0 is the step's original receipt.  Epoch ``k`` spends from
``<stem>.epoch-<k>.json``.  Attempt records of an epoch ``k >= 1`` carry it;
an untagged record belongs to the epoch of the tagged record before it, or to
epoch 0.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..authority.provider_budget import (
    ProviderCallBudgetReceiptError,
    load_provider_call_budget_state,
    provider_call_budget_receipt_path,
)
from ..authority.run_input import engine_code_sha256
from ..authority.runtime_artifacts import RunArtifactAuthorityError, current_step_records
from ..authority.step_capsule import read_verified_content
from ..authority.step_runtime import (
    StepAuthorityRuntimeError,
    load_checkpoint_selected_step_capsule,
)
from ..canonical_json import canonical_sha256
from ..providers.prompts import prompt_pack_files

LEDGER_SCHEMA_VERSION = "easyicu.step_budget_epochs/1"
#: Fields the bootstrap writes on every attempt record of an epoch >= 1.
EPOCH_FIELD = "step_budget_epoch"
IDENTITY_FIELD = "step_attempt_identity"
#: The epoch-local logical-repair counter.  ``step_llm_repair_attempts`` stays
#: cumulative across epochs, as run-level retry accounting reads it.
EPOCH_REPAIRS_FIELD = "step_llm_repair_epoch_attempts"
CUMULATIVE_REPAIRS_FIELD = "step_llm_repair_attempts"

_IDENTITY_FIELDS = (
    "engine_code_sha256",
    "prompt_pack_sha256",
    "execution_kernel_identity_sha256",
    "image_id",
)


class BudgetEpochError(RuntimeError):
    """The step's epoch ledger, epoch receipts or epoch tags disagree."""


@dataclass(frozen=True)
class AttemptIdentity:
    """The code and runtime one attempt ran on; ``None`` means not recorded."""

    engine_code_sha256: Optional[str] = None
    prompt_pack_sha256: Optional[str] = None
    execution_kernel_identity_sha256: Optional[str] = None
    image_id: Optional[str] = None

    def changes_since(self, failed: "AttemptIdentity") -> tuple[str, ...]:
        """Components known on both sides that differ; unknown is unchanged."""

        return tuple(
            name
            for name in _IDENTITY_FIELDS
            if getattr(failed, name)
            and getattr(self, name)
            and getattr(failed, name) != getattr(self, name)
        )

    def payload(self) -> dict[str, Optional[str]]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Any) -> "AttemptIdentity":
        if not isinstance(payload, Mapping) or set(payload) != set(_IDENTITY_FIELDS):
            raise BudgetEpochError("an attempt identity names exactly its four components")
        values: dict[str, Optional[str]] = {}
        for name in _IDENTITY_FIELDS:
            value = payload[name]
            if value is not None and (not isinstance(value, str) or not value):
                raise BudgetEpochError(f"attempt identity {name} is not a digest or null")
            values[name] = value
        return cls(**values)


@lru_cache(maxsize=1)
def _current_code_identity() -> tuple[str, str, Optional[str]]:
    from .kernel_identity import ExecutionKernelIdentityError, build_execution_kernel_identity

    try:
        kernel: Optional[str] = build_execution_kernel_identity(
            Path(__file__).resolve().parents[2]
        ).identity_sha256
    except (ExecutionKernelIdentityError, OSError):
        kernel = None
    return (
        engine_code_sha256(),
        canonical_sha256(dict(prompt_pack_files())),
        kernel,
    )


def current_attempt_identity(*, image_id: Optional[str]) -> AttemptIdentity:
    """The identity a retry started by this process would run on.

    The code digests describe the code this process loaded, which is the code
    a retry it starts executes; the image identifier is the caller's fresh
    reading of the selected runner image.
    """

    engine, prompts, kernel = _current_code_identity()
    return AttemptIdentity(
        engine_code_sha256=engine,
        prompt_pack_sha256=prompts,
        execution_kernel_identity_sha256=kernel,
        image_id=image_id or None,
    )


def runtime_attempt_identity(runtime_bundle: Any) -> AttemptIdentity:
    """The identity an attempt of this run executes on.

    The kernel and image come from the run's validated runtime receipt, the
    same provenance a step capsule seals; without one (a runner that proves
    neither) they stay unknown and never count as a change.
    """

    provenance = runtime_bundle.get("provenance") if isinstance(runtime_bundle, Mapping) else None
    provenance = provenance if isinstance(provenance, Mapping) else {}
    engine, prompts, _source_kernel = _current_code_identity()
    return AttemptIdentity(
        engine_code_sha256=engine,
        prompt_pack_sha256=prompts,
        execution_kernel_identity_sha256=(
            str(provenance.get("execution_kernel_identity_sha256") or "") or None
        ),
        image_id=str(provenance.get("image_id") or "") or None,
    )


def earns_fresh_budget(
    current: AttemptIdentity, used: Sequence[AttemptIdentity]
) -> bool:
    """An identity earns a grant only when it differs from every used one."""

    return bool(used) and all(current.changes_since(identity) for identity in used)


def checkpoint_capsule_identity(run_dir: Path, step_id: str) -> Optional[AttemptIdentity]:
    """The identity the step's checkpoint-selected capsule ran on, if readable."""

    try:
        verified = load_checkpoint_selected_step_capsule(run_dir, step_id=step_id)
    except (StepAuthorityRuntimeError, RunArtifactAuthorityError):
        return None
    if verified is None:
        return None
    capsule = verified.capsule
    kernel = image = None
    if capsule.execution is not None:
        try:
            provenance = json.loads(
                read_verified_content(run_dir, capsule.execution.runtime_provenance)
            )
        except (ValueError, OSError, RuntimeError):
            provenance = None
        if isinstance(provenance, Mapping):
            kernel = str(provenance.get("execution_kernel_identity_sha256") or "") or None
            image = str(provenance.get("image_id") or "") or None
    return AttemptIdentity(
        engine_code_sha256=capsule.engine_code_sha256,
        prompt_pack_sha256=capsule.prompt_pack_sha256,
        execution_kernel_identity_sha256=kernel,
        image_id=image,
    )


_GRANT_FIELDS = frozenset(
    {
        "epoch",
        "identity",
        "superseded_identity",
        "changed_components",
        "receipt",
        "previous_receipt",
        "previous_receipt_sha256",
        "repair_count_offset",
        "opened_by_attempt_id",
    }
)


@dataclass(frozen=True)
class EpochGrant:
    """One fresh budget granted to one identity, as the ledger records it."""

    epoch: int
    identity: AttemptIdentity
    superseded_identity: AttemptIdentity
    changed_components: tuple[str, ...]
    receipt: str
    previous_receipt: str
    previous_receipt_sha256: Optional[str]
    #: Cumulative logical repairs spent before this epoch opened.
    repair_count_offset: int
    opened_by_attempt_id: str

    def payload(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "identity": self.identity.payload(),
            "superseded_identity": self.superseded_identity.payload(),
            "changed_components": list(self.changed_components),
            "receipt": self.receipt,
            "previous_receipt": self.previous_receipt,
            "previous_receipt_sha256": self.previous_receipt_sha256,
            "repair_count_offset": self.repair_count_offset,
            "opened_by_attempt_id": self.opened_by_attempt_id,
        }

    @classmethod
    def from_payload(cls, payload: Any, *, expected_epoch: int) -> "EpochGrant":
        if not isinstance(payload, Mapping) or set(payload) != _GRANT_FIELDS:
            raise BudgetEpochError("an epoch ledger entry has unexpected fields")
        offset = payload["repair_count_offset"]
        changed = payload["changed_components"]
        previous_sha = payload["previous_receipt_sha256"]
        if (
            payload["epoch"] != expected_epoch
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or not isinstance(changed, list)
            or not changed
            or any(name not in _IDENTITY_FIELDS for name in changed)
            or not all(
                isinstance(payload[name], str) and payload[name]
                for name in ("receipt", "previous_receipt", "opened_by_attempt_id")
            )
            or (previous_sha is not None and not isinstance(previous_sha, str))
        ):
            raise BudgetEpochError("an epoch ledger entry is malformed or out of sequence")
        return cls(
            epoch=expected_epoch,
            identity=AttemptIdentity.from_payload(payload["identity"]),
            superseded_identity=AttemptIdentity.from_payload(payload["superseded_identity"]),
            changed_components=tuple(changed),
            receipt=payload["receipt"],
            previous_receipt=payload["previous_receipt"],
            previous_receipt_sha256=previous_sha,
            repair_count_offset=offset,
            opened_by_attempt_id=payload["opened_by_attempt_id"],
        )


def epoch_receipt_path(run_dir: Path, *, step_id: str, epoch: int) -> Path:
    base = provider_call_budget_receipt_path(Path(run_dir), step_id=step_id)
    return base if epoch == 0 else base.with_name(f"{base.stem}.epoch-{epoch}.json")


def epoch_ledger_path(run_dir: Path, *, step_id: str) -> Path:
    base = provider_call_budget_receipt_path(Path(run_dir), step_id=step_id)
    return base.with_name(f"{base.stem}.epochs.json")


def load_epoch_ledger(run_dir: Path, *, step_id: str) -> tuple[EpochGrant, ...]:
    """The step's grants in order; without a ledger none were granted."""

    path = epoch_ledger_path(run_dir, step_id=step_id)
    if not path.exists():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BudgetEpochError(f"the epoch ledger is unreadable: {exc}") from exc
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"schema_version", "step_id", "epochs", "epochs_sha256"}
        or payload["schema_version"] != LEDGER_SCHEMA_VERSION
        or payload["step_id"] != step_id
        or not isinstance(payload["epochs"], list)
        or not payload["epochs"]
    ):
        raise BudgetEpochError("the epoch ledger does not describe this step")
    if canonical_sha256(payload["epochs"]) != payload["epochs_sha256"]:
        raise BudgetEpochError("the epoch ledger digest does not match its entries")
    return tuple(
        EpochGrant.from_payload(entry, expected_epoch=index)
        for index, entry in enumerate(payload["epochs"], start=1)
    )


def _write_epoch_ledger(
    run_dir: Path, *, step_id: str, grants: Sequence[EpochGrant]
) -> None:
    entries = [grant.payload() for grant in grants]
    payload = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "step_id": step_id,
        "epochs": entries,
        "epochs_sha256": canonical_sha256(entries),
    }
    path = epoch_ledger_path(run_dir, step_id=step_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _record_epochs(records: Sequence[Mapping[str, Any]], *, last_epoch: int) -> list[int]:
    """The epoch each record spends from, in order; an untagged one inherits."""

    current = 0
    assigned: list[int] = []
    for record in records:
        if EPOCH_FIELD in record:
            tag = record[EPOCH_FIELD]
            if isinstance(tag, bool) or not isinstance(tag, int) or not 1 <= tag <= last_epoch:
                raise BudgetEpochError(f"an attempt record names an unknown budget epoch: {tag!r}")
            if tag < current:
                raise BudgetEpochError("attempt records return to an earlier budget epoch")
            current = tag
        assigned.append(current)
    return assigned


def _counter(record: Mapping[str, Any], field: str) -> Optional[int]:
    value = record.get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _has_malformed_counter(records: Sequence[Mapping[str, Any]], field: str) -> bool:
    return any(field in record and _counter(record, field) is None for record in records)


def _is_settled_failure(record: Optional[Mapping[str, Any]]) -> bool:
    """The step's latest record is a failure with nothing in flight."""

    if not isinstance(record, Mapping):
        return False
    if str(record.get("status") or "") in {"", "ok"}:
        return False
    if any(str(key).startswith("capsule_pending_") for key in record):
        return False
    return not (
        record.get("step_llm_repair_history_invalid") is True
        or record.get("provider_call_budget_receipt_invalid") is True
    )


@dataclass(frozen=True)
class BudgetEpoch:
    """The epoch one attempt spends from, and the records that restore it."""

    epoch: int
    receipt_path: Path
    #: This epoch's prior attempt records, in order.
    records: tuple[Mapping[str, Any], ...]
    prior_record: Optional[Mapping[str, Any]]
    repair_count_offset: int
    #: Identities every earlier grant and attempt ran on.
    used_identities: tuple[AttemptIdentity, ...]
    #: The grant this selection opened (or, read-only, would open).
    opened: Optional[EpochGrant] = None
    #: Why the ledger, receipts or tags disagree; the step must fail closed.
    error: Optional[str] = None

    @property
    def counter_field(self) -> str:
        """The counter this epoch's budget is restored from."""

        return EPOCH_REPAIRS_FIELD if self.epoch else CUMULATIVE_REPAIRS_FIELD

    def tag(self, step_record: dict[str, Any], identity: Optional[AttemptIdentity]) -> None:
        """Mark one new attempt record as spending from this epoch."""

        if self.epoch:
            step_record[EPOCH_FIELD] = self.epoch
            if identity is not None:
                step_record[IDENTITY_FIELD] = identity.payload()


def _failed(run_dir: Path, step_id: str, records: tuple[Mapping[str, Any], ...], error: str) -> BudgetEpoch:
    return BudgetEpoch(
        epoch=0,
        receipt_path=epoch_receipt_path(run_dir, step_id=step_id, epoch=0),
        records=records,
        prior_record=next(iter(current_step_records(records)), None) if records else None,
        repair_count_offset=0,
        used_identities=(),
        error=error,
    )


def select_budget_epoch(
    *,
    run_dir: Path,
    step_id: str,
    records: Sequence[Mapping[str, Any]],
    latest_record: Optional[Mapping[str, Any]],
    current_identity: Optional[AttemptIdentity],
    explicit_rerun: bool,
    attempt_id: str,
    reserved_final_category: Optional[str],
    commit: bool,
) -> BudgetEpoch:
    """Choose the epoch a new attempt of ``step_id`` spends from.

    ``records`` are the step's prior attempt records in order and
    ``latest_record`` its current one.  A fresh epoch opens only for an
    explicit rerun of a settled failure whose current identity differs from
    every identity already granted or used, and only when the identity the
    original budget was spent under can be proved.  With ``commit`` the grant
    is written to the ledger before anything spends from it; without it (a
    read-only assessment) nothing is written.
    """

    run_dir = Path(run_dir)
    records = tuple(record for record in records if isinstance(record, Mapping))
    try:
        grants = load_epoch_ledger(run_dir, step_id=step_id)
        epochs = _record_epochs(records, last_epoch=len(grants))
    except BudgetEpochError as exc:
        return _failed(run_dir, step_id, records, str(exc))
    active = len(grants)
    next_receipt = epoch_receipt_path(run_dir, step_id=step_id, epoch=active + 1)
    if next_receipt.exists():
        return _failed(
            run_dir, step_id, records,
            f"budget epoch {active + 1} has a receipt but no ledger entry",
        )

    # Epoch 0's identity is fixed by the first grant once one exists;
    # before that it is the identity of the checkpoint-selected capsule.  Only
    # a selection that could open an epoch, or a read-only assessment of one,
    # reads the capsule; an ordinary step start does not pay for it.
    if grants:
        epoch_zero_identity: Optional[AttemptIdentity] = grants[0].superseded_identity
    elif explicit_rerun or not commit:
        epoch_zero_identity = checkpoint_capsule_identity(run_dir, step_id)
    else:
        epoch_zero_identity = None
    used: list[AttemptIdentity] = []
    if epoch_zero_identity is not None:
        used.append(epoch_zero_identity)
    used.extend(grant.identity for grant in grants)
    for record in records:
        if IDENTITY_FIELD in record:
            try:
                used.append(AttemptIdentity.from_payload(record[IDENTITY_FIELD]))
            except BudgetEpochError as exc:
                return _failed(run_dir, step_id, records, str(exc))

    active_records = tuple(record for record, epoch in zip(records, epochs) if epoch == active)
    counter_field = EPOCH_REPAIRS_FIELD if active else CUMULATIVE_REPAIRS_FIELD
    opened: Optional[EpochGrant] = None
    if (
        explicit_rerun
        and current_identity is not None
        and epoch_zero_identity is not None
        and _is_settled_failure(latest_record)
        and not _has_malformed_counter(active_records, counter_field)
        and earns_fresh_budget(current_identity, used)
    ):
        previous = epoch_receipt_path(run_dir, step_id=step_id, epoch=active)
        previous_ok = True
        if previous.exists():
            # The receipt being left must verify and hold no call in flight:
            # a pending transport belongs to that receipt and is recovered
            # from it, never abandoned for a fresh one.
            try:
                state = load_provider_call_budget_state(
                    previous,
                    step_id=step_id,
                    expected_reserved_final_category=reserved_final_category,
                )
            except ProviderCallBudgetReceiptError:
                previous_ok = False
            else:
                previous_ok = not any(
                    isinstance(entry, Mapping)
                    and isinstance(entry.get("transport"), Mapping)
                    and entry["transport"].get("state") == "pending"
                    for entry in (*state.logical_repairs, *state.initial_generations)
                )
        if previous_ok:
            superseded = grants[-1].identity if grants else epoch_zero_identity
            opened = EpochGrant(
                epoch=active + 1,
                identity=current_identity,
                superseded_identity=superseded,
                changed_components=current_identity.changes_since(superseded),
                receipt=next_receipt.relative_to(run_dir).as_posix(),
                previous_receipt=previous.relative_to(run_dir).as_posix(),
                previous_receipt_sha256=(
                    hashlib.sha256(previous.read_bytes()).hexdigest()
                    if previous.exists()
                    else None
                ),
                repair_count_offset=max(
                    (
                        value
                        for value in (
                            _counter(record, CUMULATIVE_REPAIRS_FIELD) for record in records
                        )
                        if value is not None
                    ),
                    default=0,
                ),
                opened_by_attempt_id=attempt_id,
            )
            if commit:
                _write_epoch_ledger(run_dir, step_id=step_id, grants=(*grants, opened))
            active += 1
            active_records = ()

    return BudgetEpoch(
        epoch=active,
        receipt_path=epoch_receipt_path(run_dir, step_id=step_id, epoch=active),
        records=active_records,
        prior_record=(
            next(iter(current_step_records(active_records)), None) if active_records else None
        ),
        repair_count_offset=(
            opened.repair_count_offset
            if opened is not None
            else (grants[active - 1].repair_count_offset if active else 0)
        ),
        used_identities=tuple(used),
        opened=opened,
    )


__all__ = [
    "AttemptIdentity",
    "BudgetEpoch",
    "BudgetEpochError",
    "CUMULATIVE_REPAIRS_FIELD",
    "EPOCH_FIELD",
    "EPOCH_REPAIRS_FIELD",
    "EpochGrant",
    "IDENTITY_FIELD",
    "checkpoint_capsule_identity",
    "current_attempt_identity",
    "earns_fresh_budget",
    "epoch_ledger_path",
    "epoch_receipt_path",
    "load_epoch_ledger",
    "runtime_attempt_identity",
    "select_budget_epoch",
]
