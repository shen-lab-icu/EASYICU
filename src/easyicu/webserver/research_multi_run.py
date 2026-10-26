"""Sequential, durable orchestration over independently governed single runs.

The immutable card is the human decision's complete scope. Mutable scheduling
state is never an approval. No caller-supplied plans, results or clearances are
accepted by this owner.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from easyicu.research_agent.canonical_json import canonical_sha256
from .agent_review_recovery import _locked, _write
from .research_append_source import read_json
from .research_multi_contracts import CombinedCard, MultiDatabaseError, MultiDatabaseRequest, PreparedTarget, failure_code


class SingleDatabaseOwner(Protocol):
    def design(self, request: MultiDatabaseRequest, job) -> dict: ...
    def verify_design(self, design: dict) -> None: ...
    def prepare(self, design: dict, target, *, batch_id: str, job) -> PreparedTarget: ...
    def verify(self, design: dict, target: PreparedTarget) -> None: ...
    def execute(self, target: PreparedTarget, *, reviewer: str, job, recovering: bool) -> dict: ...
    def result(self, target: PreparedTarget) -> dict: ...


class TargetJob:
    """Cancellation belongs to this target and survives a process restart."""
    def __init__(self, parent, batch, database):
        self.parent, self.batch, self.database = parent, batch, database
        self.id = getattr(parent, "id", batch.root.name) + "-" + database

    def is_cancel_requested(self):
        with _locked(self.batch.root / "state.lock"):
            requested = self.batch.state()["targets"][self.database]["cancel_requested"]
        checker = getattr(self.parent, "is_cancel_requested", lambda: False)
        return requested or checker() or bool(getattr(self.parent, "cancel_requested", False))

    @property
    def cancel_requested(self):
        return self.is_cancel_requested()

    def emit(self, event):
        if self.parent is not None:
            self.parent.emit({**event, "database": self.database, "multi_database_batch_id": self.batch.root.name})

    def __getattr__(self, name):
        if name == "provider_recovery":
            raise AttributeError(name)
        return getattr(self.parent, name)


class MultiDatabaseRun:
    def __init__(self, root: Path, owner: SingleDatabaseOwner):
        self.root, self.owner = Path(root), owner

    def state(self):
        return read_json(self.root / "state.json")

    def _save(self, state):
        _write(self.root / "state.json", state)

    def load(self):
        record = read_json(self.root / "request.json")
        if canonical_sha256(record["design"]) != record["design_sha256"]:
            raise MultiDatabaseError("multi_database_binding_changed")
        card = CombinedCard.model_validate(read_json(self.root / "card.json"))
        targets = tuple(PreparedTarget.model_validate(row) for row in record["targets"])
        expected = CombinedCard(batch_id=self.root.name, request_sha256=canonical_sha256(record["request"]),
                                design_sha256=record["design_sha256"],
                                targets=tuple(row.card() for row in targets))
        if card != expected:
            raise MultiDatabaseError("multi_database_binding_changed")
        return record, card, targets

    def prepare(self, request: MultiDatabaseRequest, *, job=None):
        with _locked(self.root / "prepare.lock"):
            if (self.root / "card.json").exists():
                record, card, _ = self.load()
                if record["request"] != request.model_dump(mode="json"):
                    raise MultiDatabaseError("multi_database_binding_changed")
                return card
            partial = self.root / "preparation.json"
            if partial.exists():
                saved = read_json(partial)
                if saved["request"] != request.model_dump(mode="json"):
                    raise MultiDatabaseError("multi_database_binding_changed")
                design = saved["design"]
                self.owner.verify_design(design)
                prepared = [PreparedTarget.model_validate(row) for row in saved["targets"]]
            else:
                design = self.owner.design(request, job)
                prepared = []
                reference = getattr(self.owner, "reference_target", lambda _: None)(design)
                requested = list(request.targets)
                if reference is not None:
                    if any(target.database == reference.target.database and target.source_id
                           and target.source_id != reference.target.source_id for target in requested):
                        raise MultiDatabaseError("multi_database_source_conflict")
                    requested = [target for target in requested if target.database != reference.target.database]
                    prepared.append(reference)
                if len(requested) + len(prepared) > 6:
                    raise MultiDatabaseError("multi_database_target_limit")
                saved = dict(request=request.model_dump(mode="json"), design=design,
                             target_requests=[row.model_dump(mode="json") for row in requested],
                             targets=[row.model_dump(mode="json") for row in prepared], reference_count=len(prepared))
                _write(partial, saved)
                self._save(dict(status="preparing", targets={row.database: dict(
                    state="pending", reason_code=None, cancel_requested=False, result=None)
                    for row in [*(item.target for item in prepared), *requested]}))
            from .research_multi_contracts import DatabaseTarget

            requested = [DatabaseTarget.model_validate(row) for row in saved["target_requests"]]
            for target in requested[len(prepared) - saved["reference_count"]:]:
                prepared.append(self.owner.prepare(design, target, batch_id=self.root.name,
                                                    job=TargetJob(job, self, target.database)))
                saved["targets"] = [row.model_dump(mode="json") for row in prepared]
                _write(partial, saved)
            targets = tuple(prepared)
            digest = canonical_sha256(design)
            record = dict(request=request.model_dump(mode="json"), design=design, design_sha256=digest,
                          targets=[row.model_dump(mode="json") for row in targets])
            card = CombinedCard(batch_id=self.root.name, request_sha256=canonical_sha256(record["request"]),
                                design_sha256=digest, targets=tuple(row.card() for row in targets))
            _write(self.root / "request.json", record)
            with _locked(self.root / "state.lock"):
                state = self.state()
                state["status"] = "review"
                for target in targets:
                    row = state["targets"][target.target.database]
                    if row["cancel_requested"]:
                        row.update(state="cancelled", reason_code="multi_database_cancelled")
                    else:
                        row.update(state="pending" if target.review else "not_run", reason_code=target.stop_code)
                self._save(state)
            _write(self.root / "card.json", card.model_dump(mode="json"))
            return card

    def _verify(self, record, targets):
        self.owner.verify_design(record["design"])
        for target in targets:
            if target.review:
                self.owner.verify(record["design"], target)

    def approve(self, *, card_sha256: str, reviewer: str):
        with _locked(self.root / "state.lock"):
            record, card, targets = self.load()
            if card.digest != card_sha256 or not reviewer:
                raise MultiDatabaseError("multi_database_approval_changed")
            if not any(target.review for target in targets):
                raise MultiDatabaseError("multi_database_no_runnable_target")
            self._verify(record, targets)
            path = self.root / "approval.json"
            if path.exists():
                if read_json(path)["card_sha256"] != card.digest:
                    raise MultiDatabaseError("multi_database_approval_changed")
            else:
                _write(path, dict(card_sha256=card.digest, reviewer=reviewer,
                                 decided_at=datetime.now(timezone.utc).isoformat()))
            state = self.state()
            if state["status"] == "review":
                state["status"] = "approved"
                self._save(state)
            return card

    def cancel(self, database):
        with _locked(self.root / "state.lock"):
            state = self.state()
            if database not in state["targets"]:
                raise MultiDatabaseError("multi_database_target_unknown")
            row = state["targets"][database]
            if row["state"] in {"pending", "running"}:
                row["cancel_requested"] = True
                if row["state"] == "pending":
                    row.update(state="cancelled", reason_code="multi_database_cancelled")
                self._save(state)

    def execute(self, *, job=None):
        # One process owns the sequence; the separate state lock lets another
        # request cancel a target while a fit is running.
        with _locked(self.root / "execution.lock"):
            record, card, targets = self.load()
            path = self.root / "approval.json"
            if not path.exists() or read_json(path)["card_sha256"] != card.digest:
                raise MultiDatabaseError("multi_database_approval_required")
            approval = read_json(path)
            self.owner.verify_design(record["design"])
            for target in targets:
                database = target.target.database
                with _locked(self.root / "state.lock"):
                    state = self.state()
                    row = state["targets"][database]
                    if row["state"] not in {"pending", "running"}:
                        continue
                    recovering = row["state"] == "running"
                    row["state"] = "running"
                    state["status"] = "running"
                    self._save(state)
                bound_job = TargetJob(job, self, database)
                try:
                    if bound_job.is_cancel_requested():
                        raise MultiDatabaseError("multi_database_cancelled")
                    # Verify the complete remaining approval scope before each
                    # dispatch, including the target's own source and provider.
                    self._verify(record, targets)
                    self.owner.execute(target, reviewer=approval["reviewer"], job=bound_job, recovering=recovering)
                    try:
                        result = self.owner.result(target)
                        outcome = dict(state="completed", reason_code=None, result=result)
                    except (ValueError, OSError, KeyError) as exc:
                        outcome = dict(state="completed", reason_code=failure_code(exc,
                            "multi_database_primary_estimate_unavailable"), result=None)
                except Exception as exc:
                    code = failure_code(exc)
                    outcome = dict(state="cancelled" if bound_job.is_cancel_requested() else "failed",
                                   reason_code="multi_database_cancelled" if bound_job.is_cancel_requested() else code, result=None)
                with _locked(self.root / "state.lock"):
                    state = self.state()
                    state["targets"][database].update(outcome)
                    self._save(state)
            with _locked(self.root / "state.lock"):
                state = self.state()
                state["status"] = "completed"
                self._save(state)
            return state
