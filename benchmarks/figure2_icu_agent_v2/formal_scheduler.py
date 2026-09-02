"""Deterministic no-Provider scheduler for Figure 2 formal trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

from .review_bundle_semantics import CANONICAL_FILES


PROTOCOL_PATH = Path(__file__).with_name("experiment_protocol_v2_1.json")
EXECUTION_CONTRACT_PATH = Path(__file__).with_name(
    "execution_acceptance_contract_v1.json"
)
ARMS = ("easyicu_full", "generic_code_agent")


class FormalScheduleError(ValueError):
    reason_code = "FORMAL_SCHEDULE_INVALID"


@dataclass(frozen=True)
class FormalTrajectory:
    sequence_number: int
    pair_sequence_number: int
    site_pair_sequence_number: int
    arm_sequence_within_pair: int
    scope: str
    task_id: str
    arm: str
    execution_site: str
    output_dir: str
    predecessor_output_dir: str | None


@dataclass(frozen=True)
class FormalScheduleDryRun:
    protocol_sha256: str
    scope: str
    trajectory_count: int
    provider_accessed: bool
    site_pair_counts: Mapping[str, int]
    site_assignment_sha256: str
    trajectories: tuple[FormalTrajectory, ...]

    def as_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "easyicu.figure2_schedule_dry_run/1",
            "protocol_sha256": self.protocol_sha256,
            "scope": self.scope,
            "trajectory_count": self.trajectory_count,
            "provider_accessed": self.provider_accessed,
            "site_pair_counts": dict(self.site_pair_counts),
            "site_assignment_sha256": self.site_assignment_sha256,
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


def _prepare_output_roots(output_roots: Mapping[str, Path]) -> dict[str, Path]:
    execution_contract = _load_protocol(EXECUTION_CONTRACT_PATH)
    sites = tuple(execution_contract["logical_sites"])
    if sites != ("server", "laptop") or set(output_roots) != set(sites):
        raise FormalScheduleError("output roots must map server and laptop exactly")
    raw_roots = {site: Path(output_roots[site]) for site in sites}
    if any(root.is_symlink() for root in raw_roots.values()):
        raise FormalScheduleError("formal output roots may not be symlinks")
    roots = {site: raw_roots[site].resolve() for site in sites}
    if len(set(roots.values())) != len(sites):
        raise FormalScheduleError("logical sites require distinct output roots")
    for site, root in roots.items():
        if root.exists() and (root.is_symlink() or not root.is_dir()):
            raise FormalScheduleError(f"{site} output root must be an ordinary directory")
        if root.exists() and any(root.iterdir()):
            raise FormalScheduleError(f"{site} formal output root must be empty")
    return roots


def _finalize_dry_run(
    *,
    scope: str,
    rows: list[FormalTrajectory],
    site_pair_counts: Mapping[str, int],
    protocol_path: Path,
) -> FormalScheduleDryRun:
    for pair_index in range(0, len(rows), 2):
        first, second = rows[pair_index : pair_index + 2]
        if (
            first.task_id != second.task_id
            or first.execution_site != second.execution_site
            or first.arm == second.arm
            or first.arm_sequence_within_pair != 1
            or second.arm_sequence_within_pair != 2
            or second.predecessor_output_dir != first.output_dir
        ):
            raise FormalScheduleError("task-pair colocation or ordering drifted")
    occupied = [row.output_dir for row in rows if Path(row.output_dir).exists()]
    if occupied:
        raise FormalScheduleError(
            "formal trajectory output paths must not already exist: "
            + ", ".join(occupied[:3])
        )
    site_assignment = [
        {
            "pair_sequence_number": row.pair_sequence_number,
            "task_id": row.task_id,
            "execution_site": row.execution_site,
        }
        for row in rows[::2]
    ]
    return FormalScheduleDryRun(
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        scope=scope,
        trajectory_count=len(rows),
        provider_accessed=False,
        site_pair_counts=dict(site_pair_counts),
        site_assignment_sha256=hashlib.sha256(
            json.dumps(
                site_assignment,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        trajectories=tuple(rows),
    )


def build_core_schedule_dry_run(
    output_roots: Mapping[str, Path],
    *,
    protocol_path: Path = PROTOCOL_PATH,
) -> FormalScheduleDryRun:
    """Validate and project the frozen core schedule without creating run dirs."""

    protocol = _load_protocol(protocol_path)
    roots = _prepare_output_roots(output_roots)
    sites = tuple(roots)
    schedule = protocol["formal_schedule"]
    rows: list[FormalTrajectory] = []
    site_pair_counts = {site: 0 for site in sites}

    def append_pair(task_id: str, first_arm: str, site: str) -> None:
        pair_sequence = len(rows) // 2 + 1
        if site not in sites:
            raise FormalScheduleError(f"unsupported execution site: {site!r}")
        site_pair_counts[site] += 1
        arms = _ordered_arms(first_arm)
        first_output = roots[site] / task_id / arms[0]
        for arm_index, arm in enumerate(arms, start=1):
            rows.append(
                FormalTrajectory(
                    sequence_number=len(rows) + 1,
                    pair_sequence_number=pair_sequence,
                    site_pair_sequence_number=site_pair_counts[site],
                    arm_sequence_within_pair=arm_index,
                    scope="core_wp2_wp3",
                    task_id=task_id,
                    arm=arm,
                    execution_site=site,
                    output_dir=str(roots[site] / task_id / arm),
                    predecessor_output_dir=(
                        str(first_output) if arm_index == 2 else None
                    ),
                )
            )

    heldout_order = schedule["heldout27_execution_order"]
    heldout_first = schedule["heldout27_arm_first"]
    heldout_sites = schedule["heldout27_execution_site"]
    if (
        set(heldout_order) != set(heldout_first)
        or set(heldout_order) != set(heldout_sites)
        or len(heldout_order) != 27
    ):
        raise FormalScheduleError("Heldout27 schedule identity mismatch")
    for task_id in heldout_order:
        append_pair(task_id, heldout_first[task_id], heldout_sites[task_id])

    safety_order = schedule["safety12_execution_order"]
    safety_first = schedule["safety12_arm_first"]
    safety_sites = schedule["safety12_execution_site"]
    if len(safety_order) != 12 or len(safety_first) != 12 or len(safety_sites) != 12:
        raise FormalScheduleError("Safety12 schedule identity mismatch")
    for task_id, first, site in zip(
        safety_order,
        safety_first,
        safety_sites,
        strict=True,
    ):
        append_pair(task_id, first, site)

    expected_count = protocol["formal_run_policy"]["core_formal_runs"]
    coordinates = {(row.task_id, row.arm) for row in rows}
    if len(rows) != expected_count or len(coordinates) != expected_count:
        raise FormalScheduleError("core schedule is not 78 unique task-arm pairs")
    if site_pair_counts != {"server": 20, "laptop": 19}:
        raise FormalScheduleError(f"site pair balance drifted: {site_pair_counts!r}")
    return _finalize_dry_run(
        scope="core_wp2_wp3",
        rows=rows,
        site_pair_counts=site_pair_counts,
        protocol_path=protocol_path,
    )


def build_qualification_schedule_dry_run(
    task_ids: tuple[str, ...],
    output_roots: Mapping[str, Path],
    *,
    protocol_path: Path = PROTOCOL_PATH,
) -> FormalScheduleDryRun:
    """Project the post-unsealing Qualification12 schedule without Provider use."""

    if (
        len(task_ids) != 12
        or len(set(task_ids)) != 12
        or any(not isinstance(task_id, str) or not task_id.strip() for task_id in task_ids)
    ):
        raise FormalScheduleError("Qualification12 requires 12 unique task IDs")
    roots = _prepare_output_roots(output_roots)
    contract = _load_protocol(EXECUTION_CONTRACT_PATH)
    assignment = contract["qualification12_assignment"]
    ordered_ids = sorted(task_ids)
    random.Random(assignment["randomization_seed"]).shuffle(ordered_ids)
    first_arm_pattern = (
        "easyicu_full",
        "easyicu_full",
        "generic_code_agent",
        "generic_code_agent",
    )
    site_pair_counts = {site: 0 for site in roots}
    rows: list[FormalTrajectory] = []
    for pair_index, task_id in enumerate(ordered_ids, start=1):
        site = "server" if pair_index % 2 == 1 else "laptop"
        first_arm = first_arm_pattern[(pair_index - 1) % len(first_arm_pattern)]
        site_pair_counts[site] += 1
        arms = _ordered_arms(first_arm)
        first_output = roots[site] / task_id / arms[0]
        for arm_index, arm in enumerate(arms, start=1):
            rows.append(
                FormalTrajectory(
                    sequence_number=len(rows) + 1,
                    pair_sequence_number=pair_index,
                    site_pair_sequence_number=site_pair_counts[site],
                    arm_sequence_within_pair=arm_index,
                    scope="qualification12",
                    task_id=task_id,
                    arm=arm,
                    execution_site=site,
                    output_dir=str(roots[site] / task_id / arm),
                    predecessor_output_dir=(
                        str(first_output) if arm_index == 2 else None
                    ),
                )
            )
    if site_pair_counts != {"server": 6, "laptop": 6}:
        raise FormalScheduleError("Qualification12 site balance drifted")
    for site in roots:
        site_first = [
            row.arm
            for row in rows[::2]
            if row.execution_site == site
        ]
        if site_first.count("easyicu_full") != 3 or site_first.count(
            "generic_code_agent"
        ) != 3:
            raise FormalScheduleError("Qualification12 arm-first balance drifted")
    return _finalize_dry_run(
        scope="qualification12",
        rows=rows,
        site_pair_counts=site_pair_counts,
        protocol_path=protocol_path,
    )


def claim_trajectory_lease(
    trajectory: FormalTrajectory,
    *,
    logical_site: str,
    lease_root: Path,
) -> Path:
    """Claim one statically assigned trajectory exactly once on its site."""

    if logical_site != trajectory.execution_site:
        raise FormalScheduleError("trajectory cannot move to another logical site")
    root = Path(lease_root)
    if root.is_symlink() or not root.is_dir():
        raise FormalScheduleError("lease root must be an existing ordinary directory")
    if trajectory.arm_sequence_within_pair == 2:
        predecessor = Path(trajectory.predecessor_output_dir or "")
        if not predecessor.is_dir() or {
            path.name for path in predecessor.iterdir()
        } != set(CANONICAL_FILES):
            raise FormalScheduleError(
                "second arm requires the first arm's complete review bundle"
            )
        receipt_path = predecessor / "07_run_receipt.json"
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise FormalScheduleError(
                "second arm requires the first arm's terminal review bundle"
            )
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise FormalScheduleError("predecessor receipt is invalid") from exc
        if not isinstance(receipt, dict) or receipt.get("terminal_status") not in {
            "completed",
            "failed",
        }:
            raise FormalScheduleError("predecessor arm is not terminal")
    lease_path = root / f"{trajectory.task_id}__{trajectory.arm}.lease.json"
    payload = {
        "schema_version": "easyicu.figure2_trajectory_lease/1",
        "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "scope": trajectory.scope,
        "sequence_number": trajectory.sequence_number,
        "pair_sequence_number": trajectory.pair_sequence_number,
        "task_id": trajectory.task_id,
        "arm": trajectory.arm,
        "execution_site": trajectory.execution_site,
        "output_dir": trajectory.output_dir,
    }
    descriptor = os.open(
        lease_path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return lease_path


def validate_trajectory_lease(
    lease_path: Path,
    *,
    scope: str,
    task_id: str,
    arm: str,
    execution_site: str,
) -> Mapping[str, Any]:
    """Validate the exact lease required by a formal runner constructor."""

    path = Path(lease_path)
    if path.is_symlink() or not path.is_file():
        raise FormalScheduleError("trajectory lease must be an ordinary file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FormalScheduleError("trajectory lease is invalid JSON") from exc
    required = {
        "schema_version",
        "protocol_sha256",
        "scope",
        "sequence_number",
        "pair_sequence_number",
        "task_id",
        "arm",
        "execution_site",
        "output_dir",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise FormalScheduleError("trajectory lease fields do not match the schema")
    expected = {
        "schema_version": "easyicu.figure2_trajectory_lease/1",
        "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "scope": scope,
        "task_id": task_id,
        "arm": arm,
        "execution_site": execution_site,
    }
    for field, value in expected.items():
        if payload[field] != value:
            raise FormalScheduleError(f"trajectory lease {field} mismatch")
    return payload


def consume_trajectory_lease(
    lease_path: Path,
    *,
    scope: str,
    task_id: str,
    arm: str,
    execution_site: str,
) -> Path:
    """Atomically consume a lease so a formal trajectory cannot be restarted."""

    payload = validate_trajectory_lease(
        lease_path,
        scope=scope,
        task_id=task_id,
        arm=arm,
        execution_site=execution_site,
    )
    path = Path(lease_path)
    started_path = path.with_name(f"{path.name}.started")
    started_payload = {
        "schema_version": "easyicu.figure2_trajectory_start/1",
        "lease_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope": payload["scope"],
        "task_id": payload["task_id"],
        "arm": payload["arm"],
        "execution_site": payload["execution_site"],
    }
    descriptor = os.open(
        started_path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(started_payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return started_path


__all__ = [
    "FormalScheduleDryRun",
    "FormalScheduleError",
    "FormalTrajectory",
    "build_core_schedule_dry_run",
    "build_qualification_schedule_dry_run",
    "claim_trajectory_lease",
    "consume_trajectory_lease",
    "validate_trajectory_lease",
]
