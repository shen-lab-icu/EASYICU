"""Deterministic no-Provider scheduler for the 78 formal core trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROTOCOL_PATH = Path(__file__).with_name("experiment_protocol_v2_1.json")
ARMS = ("easyicu_full", "generic_code_agent")


class FormalScheduleError(ValueError):
    reason_code = "FORMAL_SCHEDULE_INVALID"


@dataclass(frozen=True)
class FormalTrajectory:
    sequence_number: int
    scope: str
    task_id: str
    arm: str
    output_dir: str


@dataclass(frozen=True)
class FormalScheduleDryRun:
    protocol_sha256: str
    core_trajectory_count: int
    provider_accessed: bool
    trajectories: tuple[FormalTrajectory, ...]

    def as_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "easyicu.figure2_schedule_dry_run/1",
            "protocol_sha256": self.protocol_sha256,
            "core_trajectory_count": self.core_trajectory_count,
            "provider_accessed": self.provider_accessed,
            "trajectories": [asdict(item) for item in self.trajectories],
        }


def _load_protocol(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FormalScheduleError("protocol must be a JSON object")
    return value


def _ordered_arms(first: str) -> tuple[str, str]:
    if first not in ARMS:
        raise FormalScheduleError(f"unknown first arm: {first!r}")
    return first, next(arm for arm in ARMS if arm != first)


def build_core_schedule_dry_run(
    output_root: Path,
    *,
    protocol_path: Path = PROTOCOL_PATH,
) -> FormalScheduleDryRun:
    """Validate and project the frozen core schedule without creating run dirs."""

    root = Path(output_root).resolve()
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise FormalScheduleError("output root must be an ordinary directory")
    if root.exists() and any(root.iterdir()):
        raise FormalScheduleError("formal output root must be empty")
    protocol = _load_protocol(protocol_path)
    schedule = protocol["formal_schedule"]
    rows: list[FormalTrajectory] = []

    heldout_order = schedule["heldout27_execution_order"]
    heldout_first = schedule["heldout27_arm_first"]
    if set(heldout_order) != set(heldout_first) or len(heldout_order) != 27:
        raise FormalScheduleError("Heldout27 schedule identity mismatch")
    for task_id in heldout_order:
        for arm in _ordered_arms(heldout_first[task_id]):
            rows.append(
                FormalTrajectory(
                    sequence_number=len(rows) + 1,
                    scope="core_wp2_wp3",
                    task_id=task_id,
                    arm=arm,
                    output_dir=str(root / task_id / arm),
                )
            )

    safety_order = schedule["safety12_execution_order"]
    safety_first = schedule["safety12_arm_first"]
    if len(safety_order) != 12 or len(safety_first) != 12:
        raise FormalScheduleError("Safety12 schedule identity mismatch")
    for task_id, first in zip(safety_order, safety_first, strict=True):
        for arm in _ordered_arms(first):
            rows.append(
                FormalTrajectory(
                    sequence_number=len(rows) + 1,
                    scope="core_wp2_wp3",
                    task_id=task_id,
                    arm=arm,
                    output_dir=str(root / task_id / arm),
                )
            )

    expected_count = protocol["formal_run_policy"]["core_formal_runs"]
    coordinates = {(row.task_id, row.arm) for row in rows}
    if len(rows) != expected_count or len(coordinates) != expected_count:
        raise FormalScheduleError("core schedule is not 78 unique task-arm pairs")
    occupied = [row.output_dir for row in rows if Path(row.output_dir).exists()]
    if occupied:
        raise FormalScheduleError(
            "formal trajectory output paths must not already exist: "
            + ", ".join(occupied[:3])
        )
    return FormalScheduleDryRun(
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        core_trajectory_count=len(rows),
        provider_accessed=False,
        trajectories=tuple(rows),
    )


__all__ = [
    "FormalScheduleDryRun",
    "FormalScheduleError",
    "FormalTrajectory",
    "build_core_schedule_dry_run",
]
