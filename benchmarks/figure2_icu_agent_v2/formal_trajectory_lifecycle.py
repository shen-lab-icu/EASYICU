"""Prepare and commit one Figure 2 formal trajectory without lease loss."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

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
            if not existed_before and validated_workdir.is_dir():
                try:
                    validated_workdir.rmdir()
                except OSError:
                    pass
            raise
        return implementation

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


__all__ = ["FormalTrajectoryLifecycle", "FormalTrajectoryLifecycleError"]
