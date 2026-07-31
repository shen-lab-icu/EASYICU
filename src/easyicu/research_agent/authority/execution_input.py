"""Bind execute-phase cohort and trajectory inputs to one host authority state.

The state is deliberately science-neutral: it does not choose a cohort,
trajectory, method, exposure, outcome, or estimand.  It keeps the exact input
coordinates selected by the Planner/control plane, re-verifies immutable
trajectory bytes around sandbox execution, and carries the corruption latch
used to stop the remainder of a run.

This module must remain below orchestration.  In particular, it must not import
``pipeline`` or ``execution.phase``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Optional

from .analysis_cohort import (
    ExecutionCohortAuthority,
    bind_execution_cohort_authority,
)
from ..intake.materialized_metadata import (
    MaterializedCohortAuthorityRef,
    MaterializedMetadataError,
    load_verified_materialized_cohort_authority,
)
from ..intake.materialized_trajectory import (
    MaterializedTrajectoryError,
    StagedTrajectoryBinding,
    verify_materialized_trajectory_envelope,
)
from ..schema import ValidationFinding


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class ExecutionInputAuthorityState:
    """Exact execute-phase inputs plus the run-level corruption latch."""

    universe_path: Path
    analysis_path: Path
    cohort_authority: ExecutionCohortAuthority
    trajectory_binding: Optional[StagedTrajectoryBinding]
    run_dir: Path
    legacy_trajectory_verifier: Optional[Callable[..., None]] = None
    development_sample: Optional[Any] = None
    trajectory_bound_cohort_authority_ref: Optional[MaterializedCohortAuthorityRef] = (
        None
    )
    corrupted: bool = False
    step_id: Optional[str] = None

    @classmethod
    def bind(
        cls,
        *,
        universe_path: Path,
        analysis_path: Path,
        trajectory_binding: Optional[StagedTrajectoryBinding],
        run_dir: Path,
        legacy_trajectory_verifier: Optional[Callable[..., None]],
        plan: Any,
        context: Any,
        development_sample: Optional[Any] = None,
    ) -> "ExecutionInputAuthorityState":
        universe_path = Path(universe_path)
        analysis_path = Path(analysis_path)
        cohort_authority = bind_execution_cohort_authority(
            universe_path=universe_path,
            analysis_path=analysis_path,
            plan=plan,
            context=context,
        )
        state = cls(
            universe_path=universe_path,
            analysis_path=analysis_path,
            cohort_authority=cohort_authority,
            trajectory_binding=trajectory_binding,
            run_dir=Path(run_dir),
            legacy_trajectory_verifier=legacy_trajectory_verifier,
        )
        if development_sample is not None:
            state.apply_development_sample(development_sample)
        return state

    def apply_development_sample(self, binding: Any) -> None:
        """Select one verified non-paper child of the locked analysis cohort."""

        if self.cohort_authority.selected_path != Path(binding.parent_cohort_path):
            raise MaterializedMetadataError(
                "development sample is not descended from the selected locked "
                "analysis cohort"
            )
        sample_path = Path(binding.cohort_path)
        sample_authority = load_verified_materialized_cohort_authority(
            sample_path,
            expected_authority=binding.cohort_authority_ref,
        )
        if self.cohort_authority.analysis_authority is not None:
            if (
                sample_authority is None
                or sample_authority.authority.parent_authority_sha256
                != self.cohort_authority.analysis_authority.reference.sha256
            ):
                raise MaterializedMetadataError(
                    "typed development sample lost its analysis-cohort parent"
                )
        elif sample_authority is not None:
            raise MaterializedMetadataError(
                "untyped analysis cohort unexpectedly gained a typed sample"
            )
        self.cohort_authority = ExecutionCohortAuthority(
            selected_path=sample_path,
            universe_path=self.cohort_authority.universe_path,
            universe_authority=self.cohort_authority.universe_authority,
            analysis_authority=sample_authority,
        )
        self.development_sample = binding
        self.trajectory_binding = binding.trajectory_binding
        self.trajectory_bound_cohort_authority_ref = (
            binding.trajectory_bound_cohort_authority_ref
        )

    @property
    def selected_path(self) -> Path:
        return self.cohort_authority.selected_path

    @property
    def universe_authority_ref(self) -> Optional[MaterializedCohortAuthorityRef]:
        authority = self.cohort_authority.universe_authority
        return authority.reference if authority is not None else None

    @property
    def trajectory_sha256(self) -> Optional[str]:
        binding = self.trajectory_binding
        return binding.sha256 if binding is not None else None

    @property
    def trajectory_authority_sha256(self) -> Optional[str]:
        binding = self.trajectory_binding
        return binding.verified_authority_sha256 if binding is not None else None

    def rebind_cohort(self, *, plan: Any, context: Any) -> None:
        """Refresh the selected analysis-cohort child without changing inputs."""

        rebound = bind_execution_cohort_authority(
            universe_path=self.universe_path,
            analysis_path=self.analysis_path,
            plan=plan,
            context=context,
        )
        if self.development_sample is not None:
            if rebound.selected_path != Path(
                self.development_sample.parent_cohort_path
            ):
                raise MaterializedMetadataError(
                    "replanned cohort no longer matches the development sample parent"
                )
            sample = load_verified_materialized_cohort_authority(
                self.development_sample.cohort_path,
                expected_authority=self.development_sample.cohort_authority_ref,
            )
            if rebound.analysis_authority is not None and (
                sample is None
                or sample.authority.parent_authority_sha256
                != rebound.analysis_authority.reference.sha256
            ):
                raise MaterializedMetadataError(
                    "replanned typed cohort invalidated the development sample"
                )
            rebound = ExecutionCohortAuthority(
                selected_path=Path(self.development_sample.cohort_path),
                universe_path=rebound.universe_path,
                universe_authority=rebound.universe_authority,
                analysis_authority=sample,
            )
        self.cohort_authority = rebound

    def runner_bindings(self) -> dict[str, object]:
        """Return the exact host-owned coordinates accepted by ``_build_runner``."""

        binding = self.trajectory_binding
        runner_authority_ref = (
            self.trajectory_bound_cohort_authority_ref or self.universe_authority_ref
        )
        return {
            "universe_is_typed": self.cohort_authority.universe_is_typed,
            "universe_authority_ref": runner_authority_ref,
            "trajectory_path": binding.path if binding is not None else None,
            "trajectory_authority_ref": (
                binding.authority_ref if binding is not None else None
            ),
            "trajectory_legacy_capsule_receipt": (
                binding.legacy_capsule_receipt if binding is not None else None
            ),
        }

    def trajectory_integrity_finding(
        self, *, step_id: str
    ) -> Optional[ValidationFinding]:
        """Reverify the host-owned trajectory before or after execution."""

        binding = self.trajectory_binding
        if binding is None:
            return None
        observed_sha256: Optional[str] = None
        observed_size: Optional[int] = None
        authority_error: Optional[str] = None
        try:
            path = binding.path
            if path.is_symlink() or not path.is_file():
                raise MaterializedTrajectoryError(
                    "staged trajectory is missing or is not a regular file"
                )
            if binding.authority_ref is not None:
                universe_authority_ref = (
                    self.trajectory_bound_cohort_authority_ref
                    or self.universe_authority_ref
                )
                if universe_authority_ref is None:
                    raise MaterializedTrajectoryError(
                        "typed trajectory lost its universe authority"
                    )
                authority = verify_materialized_trajectory_envelope(
                    path,
                    expected_authority=binding.authority_ref,
                    expected_universe_authority=universe_authority_ref,
                )
                observed_sha256 = authority.trajectory_sha256
                observed_size = authority.trajectory_size
            elif binding.legacy_capsule_receipt is not None:
                universe_authority_ref = self.universe_authority_ref
                if (
                    universe_authority_ref is None
                    or binding.legacy_capsule_receipt.universe_authority_sha256
                    != universe_authority_ref.sha256
                ):
                    raise MaterializedTrajectoryError(
                        "legacy trajectory receipt lost its universe authority"
                    )
                verifier = self.legacy_trajectory_verifier
                if verifier is None:
                    raise MaterializedTrajectoryError(
                        "legacy trajectory receipt lost its verifier"
                    )
                observed_sha256, observed_size = verifier(
                    run_dir=self.run_dir,
                    trajectory_path=path,
                    receipt=binding.legacy_capsule_receipt,
                    expected_universe_authority=universe_authority_ref,
                )
            else:
                observed_size = int(path.stat().st_size)
                observed_sha256 = _sha256_file(path)
            if observed_sha256 != binding.sha256 or observed_size != binding.size:
                raise MaterializedTrajectoryError("staged trajectory bytes changed")
        except Exception as exc:
            authority_error = f"{type(exc).__name__}: {exc}"[:300]
        if authority_error is None:
            return None
        return ValidationFinding(
            validator="execution_input_authority_integrity",
            severity="error",
            message=(
                "The authoritative trajectory changed while step "
                f"{step_id} executed; all outputs from this attempt were rejected."
            ),
            detail={
                "step_id": step_id,
                "expected_trajectory_sha256": binding.sha256,
                "observed_trajectory_sha256": observed_sha256,
                "expected_trajectory_size": binding.size,
                "observed_trajectory_size": observed_size,
                "trajectory_authority_sha256": (
                    binding.authority_ref.sha256
                    if binding.authority_ref is not None
                    else None
                ),
                "error": authority_error,
            },
        )

    def require_trajectory_integrity(self, *, step_id: str) -> None:
        finding = self.trajectory_integrity_finding(step_id=step_id)
        if finding is not None:
            raise MaterializedTrajectoryError(finding.message)

    def mark_corrupted(self, *, step_id: str) -> None:
        self.corrupted = True
        self.step_id = step_id


__all__ = ["ExecutionInputAuthorityState"]
