"""Prepare and commit one Figure 2 formal trajectory without lease loss."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar
from uuid import uuid4

from easyicu.research_agent.authority.provider_hard_stop import TaskProviderHardStop

from .formal_provider_gate import FormalProviderSession
from .formal_scheduler import (
    consume_trajectory_lease,
    signed_output_root,
    signed_site_assignment,
    validate_trajectory_lease,
)


_T = TypeVar("_T")


class FormalTrajectoryLifecycleError(ValueError):
    """A formal trajectory cannot be prepared within its signed writable root."""

    reason_code = "FORMAL_TRAJECTORY_LIFECYCLE_INVALID"


class FormalTrajectoryLifecycle:
    """Own validation, implementation readiness, and single-use lease commit."""

    def __init__(
        self,
        *,
        lease_path: Path,
        scope: str,
        task_id: str,
        arm: str,
        execution_site: str,
        receipts: Mapping[str, Any],
    ) -> None:
        site_assignment = signed_site_assignment(receipts, scope=scope)
        output_root = signed_output_root(
            receipts,
            execution_site=execution_site,
        )
        payload = validate_trajectory_lease(
            lease_path,
            scope=scope,
            task_id=task_id,
            arm=arm,
            execution_site=execution_site,
            site_assignment=site_assignment,
            expected_output_root=output_root,
        )
        self._lease_path = Path(lease_path)
        self._scope = scope
        self._task_id = task_id
        self._arm = arm
        self._execution_site = execution_site
        self._site_assignment = site_assignment
        self._output_root = Path(output_root).resolve()
        self.output_dir = Path(payload["output_dir"]).resolve()
        self.workdir = (
            self._output_root / ".trajectory-work" / task_id / arm
        ).resolve()
        self._committed = False

    def require_workdir(self, workdir: Path) -> Path:
        """Require the exact derived scratch path under the signed site root."""

        candidate = Path(workdir)
        if candidate.is_symlink() or candidate.resolve() != self.workdir:
            raise FormalTrajectoryLifecycleError(
                "formal implementation workdir does not match the derived signed path"
            )
        if candidate.exists() and (
            not candidate.is_dir() or any(candidate.iterdir())
        ):
            raise FormalTrajectoryLifecycleError(
                "formal implementation workdir must be absent or empty"
            )
        return candidate.resolve()

    def initialize(
        self,
        *,
        workdir: Path,
        factory: Callable[[], _T],
    ) -> _T:
        """Construct the implementation before consuming the single-use lease."""

        validated_workdir = self.require_workdir(workdir)
        existed_before = validated_workdir.exists()
        try:
            implementation = factory()
            self.require_workdir(validated_workdir)
            self.commit()
        except BaseException:
            try:
                self._restore_retryable_workdir(
                    validated_workdir,
                    existed_before=existed_before,
                )
            except OSError as cleanup_exc:
                raise FormalTrajectoryLifecycleError(
                    "failed formal initialization could not be quarantined"
                ) from cleanup_exc
            raise
        return implementation

    def _restore_retryable_workdir(
        self,
        workdir: Path,
        *,
        existed_before: bool,
    ) -> None:
        """Preserve partial state while restoring the exact path for retry."""

        if workdir.is_dir() and not workdir.is_symlink() and not any(workdir.iterdir()):
            if not existed_before:
                workdir.rmdir()
            return
        if workdir.exists() or workdir.is_symlink():
            quarantine_root = (
                self._output_root
                / ".trajectory-failed"
                / self._task_id
                / self._arm
            )
            if quarantine_root.is_symlink():
                raise OSError("formal failure quarantine may not be a symlink")
            quarantine_root.mkdir(parents=True, exist_ok=True)
            quarantined = quarantine_root / uuid4().hex
            workdir.replace(quarantined)
        if existed_before:
            workdir.mkdir(parents=True, exist_ok=False)

    def commit(self) -> Mapping[str, Any]:
        """Atomically consume the lease after all local initialization passes."""

        if self._committed:
            raise FormalTrajectoryLifecycleError("trajectory lease is already committed")
        payload = consume_trajectory_lease(
            self._lease_path,
            scope=self._scope,
            task_id=self._task_id,
            arm=self._arm,
            execution_site=self._execution_site,
            site_assignment=self._site_assignment,
            expected_output_root=str(self._output_root),
        )
        self._committed = True
        return payload

    def require_output_dir(self, output_dir: Path) -> Path:
        """Keep the terminal seven-file bundle at the leased output path."""

        if Path(output_dir).resolve() != self.output_dir:
            raise FormalTrajectoryLifecycleError(
                "formal output directory does not match the committed lease"
            )
        return self.output_dir


class FormalExecutionSession:
    """Own one runner's lifecycle, Provider session, and formal identity."""

    def __init__(
        self,
        *,
        lease_path: Path,
        receipts: Mapping[str, Any],
        scope: str,
        task_id: str,
        arm: str,
        execution_site: str,
        provider_hard_stop: TaskProviderHardStop,
    ) -> None:
        self.provider = FormalProviderSession(
            receipts=receipts,
            scope=scope,
            task_id=task_id,
            arm=arm,
            execution_site=execution_site,
            provider_hard_stop=provider_hard_stop,
        )
        self._trajectory = FormalTrajectoryLifecycle(
            lease_path=lease_path,
            receipts=receipts,
            scope=scope,
            task_id=task_id,
            arm=arm,
            execution_site=execution_site,
        )

    @property
    def workdir(self) -> Path:
        return self._trajectory.workdir

    @property
    def output_dir(self) -> Path:
        return self._trajectory.output_dir

    def initialize(self, *, factory: Callable[[], _T]) -> _T:
        return self._trajectory.initialize(workdir=self.workdir, factory=factory)

    def require_workdir(self, workdir: Path) -> Path:
        return self._trajectory.require_workdir(workdir)

    def require_output_dir(self, output_dir: Path) -> Path:
        return self._trajectory.require_output_dir(output_dir)


__all__ = [
    "FormalExecutionSession",
    "FormalTrajectoryLifecycle",
    "FormalTrajectoryLifecycleError",
]
