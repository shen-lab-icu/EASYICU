"""Deterministic no-Provider scheduler for Figure 2 formal trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from .review_bundle_semantics import CANONICAL_FILES, validate_review_task_id
from .immutable_publication import publish_immutable_bytes


PROTOCOL_PATH = Path(__file__).with_name("experiment_protocol_v2_1.json")
EXECUTION_CONTRACT_PATH = Path(__file__).with_name(
    "execution_acceptance_contract_v1.json"
)
ARMS = ("easyicu_full", "generic_code_agent")
FORMAL_PAIR_SCOPES = ("qualification12", "core_wp2_wp3")


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
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise FormalScheduleError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, dict):
        raise FormalScheduleError("protocol must be a JSON object")
    return value


def _ordered_arms(first: str) -> tuple[str, str]:
    if first not in ARMS:
        raise FormalScheduleError(f"unknown first arm: {first!r}")
    return first, next(arm for arm in ARMS if arm != first)


def _site_assignment_sha256(assignment: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(assignment),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def expected_site_assignment(
    scope: str,
    *,
    task_ids: Sequence[str] = (),
    protocol_path: Path = PROTOCOL_PATH,
) -> tuple[dict[str, Any], ...]:
    """Return the only registered task-to-site assignment for a pair scope."""

    protocol = _load_protocol(protocol_path)
    if scope == "core_wp2_wp3":
        if task_ids:
            raise FormalScheduleError("core assignment does not accept task IDs")
        schedule = protocol["formal_schedule"]
        order = [
            *schedule["heldout27_execution_order"],
            *schedule["safety12_execution_order"],
        ]
        sites = {
            **schedule["heldout27_execution_site"],
            **dict(
                zip(
                    schedule["safety12_execution_order"],
                    schedule["safety12_execution_site"],
                    strict=True,
                )
            ),
        }
    elif scope == "qualification12":
        try:
            normalized_task_ids = tuple(
                validate_review_task_id(task_id) for task_id in task_ids
            )
        except ValueError as exc:
            raise FormalScheduleError(
                "Qualification12 task identity is invalid"
            ) from exc
        if (
            len(normalized_task_ids) != 12
            or len(set(normalized_task_ids)) != 12
        ):
            raise FormalScheduleError("Qualification12 requires 12 unique task IDs")
        order = sorted(normalized_task_ids)
        contract = _load_protocol(EXECUTION_CONTRACT_PATH)
        random.Random(
            contract["qualification12_assignment"]["randomization_seed"]
        ).shuffle(order)
        sites = {
            task_id: "server" if pair_index % 2 == 1 else "laptop"
            for pair_index, task_id in enumerate(order, start=1)
        }
    else:
        raise FormalScheduleError(f"unsupported pair scope: {scope!r}")
    return tuple(
        {
            "pair_sequence_number": pair_index,
            "task_id": task_id,
            "execution_site": sites[task_id],
        }
        for pair_index, task_id in enumerate(order, start=1)
    )


def expected_site_assignment_sha256(
    scope: str,
    *,
    task_ids: Sequence[str] = (),
    protocol_path: Path = PROTOCOL_PATH,
) -> str:
    return _site_assignment_sha256(
        expected_site_assignment(
            scope,
            task_ids=task_ids,
            protocol_path=protocol_path,
        )
    )


def _signed_declaration(receipts: Mapping[str, Any]) -> Mapping[str, Any]:
    declaration = receipts.get("atomic_declaration")
    if not isinstance(declaration, Mapping):
        raise FormalScheduleError("formal receipts lack an atomic declaration")
    return declaration


def signed_site_assignment_sha256(receipts: Mapping[str, Any]) -> str:
    """Read the assignment digest that the authority will verify before transport."""

    declaration = _signed_declaration(receipts)
    digest = declaration.get("site_assignment_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise FormalScheduleError("signed site assignment digest is invalid")
    return digest


def signed_site_assignment(
    receipts: Mapping[str, Any],
    *,
    scope: str,
    protocol_path: Path = PROTOCOL_PATH,
) -> tuple[dict[str, Any], ...]:
    """Rebuild and verify the registered pair assignment from the signed declaration."""

    declaration = _signed_declaration(receipts)
    if declaration.get("scope") != scope:
        raise FormalScheduleError("signed declaration scope mismatch")
    coordinates = declaration.get("authorized_call_coordinates")
    if not isinstance(coordinates, list) or not all(
        isinstance(coordinate, Mapping) for coordinate in coordinates
    ):
        raise FormalScheduleError("signed declaration coordinates are invalid")
    try:
        normalized = [
            {
                "scope": coordinate["scope"],
                "task_id": coordinate["task_id"],
                "arm": coordinate["arm"],
                "execution_site": coordinate["execution_site"],
                "call_id": coordinate["call_id"],
            }
            for coordinate in coordinates
        ]
    except KeyError as exc:
        raise FormalScheduleError("signed declaration coordinate is incomplete") from exc
    if any(
        not isinstance(value, str) or not value
        for coordinate in normalized
        for value in coordinate.values()
    ):
        raise FormalScheduleError("signed declaration coordinate is invalid")
    digest = signed_site_assignment_sha256(receipts)
    validate_authorized_site_coordinates(
        scope,
        normalized,
        declared_site_assignment_sha256=digest,
        protocol_path=protocol_path,
    )
    task_ids = tuple(sorted({coordinate["task_id"] for coordinate in normalized}))
    return expected_site_assignment(
        scope,
        task_ids=task_ids if scope == "qualification12" else (),
        protocol_path=protocol_path,
    )


def validate_output_root_by_site(value: Any) -> dict[str, str]:
    """Validate the exact two output roots carried by a signed declaration."""

    if not isinstance(value, Mapping) or set(value) != {"server", "laptop"}:
        raise FormalScheduleError("signed output roots must map server and laptop")
    roots: dict[str, str] = {}
    for site in ("server", "laptop"):
        raw = value[site]
        if not isinstance(raw, str) or not raw.strip() or raw != raw.strip():
            raise FormalScheduleError(f"signed {site} output root is invalid")
        path = Path(raw)
        if (
            not path.is_absolute()
            or str(path) != raw
            or ".." in path.parts
            or len(path.parts) < 3
        ):
            raise FormalScheduleError(f"signed {site} output root must be canonical")
        roots[site] = raw
    if roots["server"].casefold() == roots["laptop"].casefold():
        raise FormalScheduleError("signed output roots must be distinct")
    return roots


def signed_output_root(receipts: Mapping[str, Any], *, execution_site: str) -> str:
    declaration = _signed_declaration(receipts)
    roots = validate_output_root_by_site(declaration.get("output_root_by_site"))
    if execution_site not in roots:
        raise FormalScheduleError(f"unsupported execution site: {execution_site}")
    return roots[execution_site]


def validate_authorized_site_coordinates(
    scope: str,
    coordinates: Sequence[Mapping[str, str]],
    *,
    declared_site_assignment_sha256: str,
    protocol_path: Path = PROTOCOL_PATH,
) -> str:
    """Bind signed call coordinates to the registered pair assignment."""

    if scope not in FORMAL_PAIR_SCOPES:
        return declared_site_assignment_sha256
    task_ids = tuple(sorted({coordinate["task_id"] for coordinate in coordinates}))
    assignment = expected_site_assignment(
        scope,
        task_ids=task_ids if scope == "qualification12" else (),
        protocol_path=protocol_path,
    )
    expected_digest = _site_assignment_sha256(assignment)
    if declared_site_assignment_sha256 != expected_digest:
        raise FormalScheduleError("signed site assignment digest mismatch")
    expected_by_task = {item["task_id"]: item for item in assignment}
    arms_by_task: dict[str, set[str]] = {task_id: set() for task_id in expected_by_task}
    for coordinate in coordinates:
        task_id = coordinate["task_id"]
        expected = expected_by_task.get(task_id)
        if expected is None:
            raise FormalScheduleError(f"task is absent from site assignment: {task_id}")
        if coordinate["execution_site"] != expected["execution_site"]:
            raise FormalScheduleError(f"task site assignment mismatch: {task_id}")
        arm = coordinate["arm"]
        if arm not in ARMS:
            raise FormalScheduleError(f"unsupported arm in site assignment: {arm}")
        arms_by_task[task_id].add(arm)
    incomplete = sorted(
        task_id for task_id, arms in arms_by_task.items() if arms != set(ARMS)
    )
    if incomplete:
        raise FormalScheduleError(
            "signed assignment must contain both arms for every task: "
            + ", ".join(incomplete)
        )
    return expected_digest


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
    site_assignment = tuple(
        {
            "pair_sequence_number": row.pair_sequence_number,
            "task_id": row.task_id,
            "execution_site": row.execution_site,
        }
        for row in rows[::2]
    )
    task_ids = tuple(item["task_id"] for item in site_assignment)
    expected_assignment = expected_site_assignment(
        scope,
        task_ids=task_ids if scope == "qualification12" else (),
        protocol_path=protocol_path,
    )
    if site_assignment != expected_assignment:
        raise FormalScheduleError("generated site assignment drifted")
    return FormalScheduleDryRun(
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        scope=scope,
        trajectory_count=len(rows),
        provider_accessed=False,
        site_pair_counts=dict(site_pair_counts),
        site_assignment_sha256=_site_assignment_sha256(site_assignment),
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

    roots = _prepare_output_roots(output_roots)
    assignment = expected_site_assignment("qualification12", task_ids=task_ids)
    first_arm_pattern = (
        "easyicu_full",
        "easyicu_full",
        "generic_code_agent",
        "generic_code_agent",
    )
    site_pair_counts = {site: 0 for site in roots}
    rows: list[FormalTrajectory] = []
    for item in assignment:
        pair_index = item["pair_sequence_number"]
        task_id = item["task_id"]
        site = item["execution_site"]
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
    schedule: FormalScheduleDryRun,
    logical_site: str,
    lease_root: Path,
) -> Path:
    """Claim one statically assigned trajectory exactly once on its site."""

    if (
        trajectory not in schedule.trajectories
        or schedule.scope != trajectory.scope
        or schedule.protocol_sha256
        != hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    ):
        raise FormalScheduleError("trajectory is not in the registered dry run")
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
    lease_path = root / (
        f"{trajectory.scope}__{trajectory.task_id}__{trajectory.arm}.lease.json"
    )
    payload = {
        "schema_version": "easyicu.figure2_trajectory_lease/1",
        "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "site_assignment_sha256": schedule.site_assignment_sha256,
        "scope": trajectory.scope,
        "sequence_number": trajectory.sequence_number,
        "pair_sequence_number": trajectory.pair_sequence_number,
        "task_id": trajectory.task_id,
        "arm": trajectory.arm,
        "execution_site": trajectory.execution_site,
        "output_dir": trajectory.output_dir,
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    publish_immutable_bytes(encoded, lease_path)
    return lease_path


def validate_trajectory_lease(
    lease_path: Path,
    *,
    scope: str,
    task_id: str,
    arm: str,
    execution_site: str,
    site_assignment: Sequence[Mapping[str, Any]],
    expected_output_root: str,
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
        "site_assignment_sha256",
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
    if scope not in FORMAL_PAIR_SCOPES:
        raise FormalScheduleError(f"unsupported lease scope: {scope}")
    if arm not in ARMS:
        raise FormalScheduleError(f"unsupported lease arm: {arm}")
    canonical_assignment = tuple(dict(item) for item in site_assignment)
    assignment_task_ids = tuple(
        item.get("task_id") for item in canonical_assignment
    )
    expected_assignment = expected_site_assignment(
        scope,
        task_ids=assignment_task_ids if scope == "qualification12" else (),
    )
    if canonical_assignment != expected_assignment:
        raise FormalScheduleError("trajectory lease site assignment mismatch")
    site_assignment_sha256 = _site_assignment_sha256(canonical_assignment)
    expected = {
        "schema_version": "easyicu.figure2_trajectory_lease/1",
        "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "site_assignment_sha256": site_assignment_sha256,
        "scope": scope,
        "task_id": task_id,
        "arm": arm,
        "execution_site": execution_site,
    }
    for field, value in expected.items():
        if payload[field] != value:
            raise FormalScheduleError(f"trajectory lease {field} mismatch")
    if any(
        type(payload[field]) is not int or payload[field] <= 0
        for field in ("sequence_number", "pair_sequence_number")
    ):
        raise FormalScheduleError("trajectory lease sequence fields are invalid")
    expected_by_task = {item["task_id"]: item for item in expected_assignment}
    item = expected_by_task.get(task_id)
    if item is None:
        assignment_name = "core" if scope == "core_wp2_wp3" else "Qualification12"
        raise FormalScheduleError(
            f"task is absent from {assignment_name} assignment: {task_id}"
        )
    if execution_site != item["execution_site"]:
        raise FormalScheduleError("trajectory lease frozen site mismatch")
    if payload["pair_sequence_number"] != item["pair_sequence_number"]:
        raise FormalScheduleError("trajectory lease pair sequence mismatch")
    if scope == "core_wp2_wp3":
        protocol_schedule = _load_protocol(PROTOCOL_PATH)["formal_schedule"]
        first_arm_by_task = {
            **protocol_schedule["heldout27_arm_first"],
            **dict(
                zip(
                    protocol_schedule["safety12_execution_order"],
                    protocol_schedule["safety12_arm_first"],
                    strict=True,
                )
            ),
        }
        first_arm = first_arm_by_task[task_id]
    else:
        first_arm_pattern = (
            "easyicu_full",
            "easyicu_full",
            "generic_code_agent",
            "generic_code_agent",
        )
        first_arm = first_arm_pattern[(item["pair_sequence_number"] - 1) % 4]
    expected_sequence_number = 2 * item["pair_sequence_number"] - (
        1 if arm == first_arm else 0
    )
    if payload["sequence_number"] != expected_sequence_number:
        raise FormalScheduleError("trajectory lease sequence number mismatch")
    output_root = Path(expected_output_root)
    if (
        not output_root.is_absolute()
        or str(output_root) != expected_output_root
        or ".." in output_root.parts
        or len(output_root.parts) < 3
        or output_root.is_symlink()
    ):
        raise FormalScheduleError("expected formal output root is invalid")
    if not isinstance(payload["output_dir"], str):
        raise FormalScheduleError("trajectory lease output directory is invalid")
    output_dir = Path(payload["output_dir"])
    expected_output_dir = output_root / task_id / arm
    if (
        output_dir != expected_output_dir
        or output_dir.is_symlink()
        or output_dir.exists()
    ):
        raise FormalScheduleError("trajectory lease output directory mismatch")
    return payload


def consume_trajectory_lease(
    lease_path: Path,
    *,
    scope: str,
    task_id: str,
    arm: str,
    execution_site: str,
    site_assignment: Sequence[Mapping[str, Any]],
    expected_output_root: str,
) -> Mapping[str, Any]:
    """Atomically consume a lease so a formal trajectory cannot be restarted."""

    payload = validate_trajectory_lease(
        lease_path,
        scope=scope,
        task_id=task_id,
        arm=arm,
        execution_site=execution_site,
        site_assignment=site_assignment,
        expected_output_root=expected_output_root,
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
    encoded = (
        json.dumps(started_payload, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    publish_immutable_bytes(encoded, started_path)
    return payload


__all__ = [
    "FormalScheduleDryRun",
    "FormalScheduleError",
    "FormalTrajectory",
    "build_core_schedule_dry_run",
    "build_qualification_schedule_dry_run",
    "claim_trajectory_lease",
    "consume_trajectory_lease",
    "expected_site_assignment",
    "expected_site_assignment_sha256",
    "signed_output_root",
    "signed_site_assignment",
    "signed_site_assignment_sha256",
    "validate_authorized_site_coordinates",
    "validate_output_root_by_site",
    "validate_trajectory_lease",
]
