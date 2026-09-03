"""Fail-closed validator for the Figure 2 v2 design freeze.

This owner validates only the preregistered design files.  A passing receipt is
not formal run authority; runtime coordinates, inputs, reviewers, and exact-head
release evidence are intentionally sealed later in an atomic batch declaration.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
PROTOCOL_PATH = PACKAGE_ROOT / "experiment_protocol_v2.json"

HELDOUT27_IDS = tuple(f"icu27_t{index:02d}" for index in range(1, 28))
SAFETY12_IDS = tuple(f"fs12_t{index:02d}" for index in range(1, 13))
ARMS = {"easyicu_full", "generic_code_agent"}
SAFETY_DISPOSITIONS = {
    "safe_block",
    "scope_down",
    "analysis_only",
    "request_clarification",
}


class DesignFreezeError(ValueError):
    """Typed failure at the v2 design-freeze boundary."""

    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


@dataclass(frozen=True)
class DesignFreezeReceipt:
    protocol_ref: str
    protocol_sha256: str
    asset_sha256: tuple[tuple[str, str], ...]
    heldout_task_count: int
    safety_task_count: int
    core_formal_runs: int
    formal_batch_authorized: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_json_bytes(payload: bytes) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DesignFreezeError("DESIGN_ASSET_PATH_INVALID", str(path))
    try:
        value = _strict_json_bytes(path.read_bytes())
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise DesignFreezeError("DESIGN_ASSET_JSON_INVALID", f"{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DesignFreezeError("DESIGN_ASSET_SHAPE_INVALID", str(path))
    return value


def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if path.is_symlink() or not path.is_file():
        raise DesignFreezeError("DESIGN_ASSET_PATH_INVALID", str(path))
    rows: list[dict[str, Any]] = []
    line_number = 0
    try:
        for line_number, line in enumerate(path.read_bytes().splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith(b"#"):
                continue
            value = _strict_json_bytes(line)
            if not isinstance(value, dict):
                raise ValueError("row must be an object")
            rows.append(value)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise DesignFreezeError(
            "DESIGN_ASSET_JSONL_INVALID", f"{path}:{line_number}: {exc}"
        ) from exc
    return tuple(rows)


def _asset_path(repo_root: Path, relative_path: str) -> Path:
    candidate = repo_root / relative_path
    try:
        candidate.resolve(strict=True).relative_to(repo_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise DesignFreezeError(
            "DESIGN_ASSET_PATH_ESCAPE", f"{relative_path}: {exc}"
        ) from exc
    return candidate


def validate_design_freeze(
    *,
    protocol_path: Path = PROTOCOL_PATH,
    repo_root: Path = REPO_ROOT,
) -> DesignFreezeReceipt:
    """Validate v2 identities, digests, task sets, schedules, and authority ceiling."""

    protocol = _load_json(protocol_path)
    if protocol.get("schema_version") != "easyicu.icu_agent_experiment_protocol/2":
        raise DesignFreezeError("DESIGN_PROTOCOL_SCHEMA_INVALID", str(protocol_path))
    if protocol.get("freeze_status") != "design_frozen_no_formal_run_authority":
        raise DesignFreezeError("DESIGN_FREEZE_STATUS_INVALID", str(protocol_path))

    assets = protocol.get("frozen_assets")
    if not isinstance(assets, list) or len(assets) != 8:
        raise DesignFreezeError("DESIGN_ASSET_SET_INVALID", repr(assets))
    roles = [asset.get("role") for asset in assets]
    paths = [asset.get("path") for asset in assets]
    if len(set(roles)) != len(roles) or len(set(paths)) != len(paths):
        raise DesignFreezeError("DESIGN_ASSET_DUPLICATE", repr(roles))

    observed_assets: dict[str, str] = {}
    by_role: dict[str, Path] = {}
    for asset in assets:
        role = asset.get("role")
        relative_path = asset.get("path")
        expected_sha256 = asset.get("sha256")
        if not all(isinstance(value, str) for value in (role, relative_path, expected_sha256)):
            raise DesignFreezeError("DESIGN_ASSET_RECORD_INVALID", repr(asset))
        path = _asset_path(repo_root, relative_path)
        observed_sha256 = _sha256(path)
        if observed_sha256 != expected_sha256:
            raise DesignFreezeError(
                "DESIGN_ASSET_DIGEST_MISMATCH",
                f"{role}: expected={expected_sha256}, observed={observed_sha256}",
            )
        observed_assets[role] = observed_sha256
        by_role[role] = path

    heldout = _load_jsonl(by_role["heldout27_taskbank"])
    heldout_ids = tuple(row.get("task_id") for row in heldout)
    if heldout_ids != HELDOUT27_IDS:
        raise DesignFreezeError("HELDOUT27_IDENTITY_DRIFT", repr(heldout_ids))
    if any(row.get("expected_behavior") != "bound_result" for row in heldout):
        raise DesignFreezeError("HELDOUT27_BEHAVIOR_DRIFT", "all tasks must be bound_result")

    safety = _load_jsonl(by_role["formal_safety12_taskbank"])
    safety_ids = tuple(row.get("task_id") for row in safety)
    if safety_ids != SAFETY12_IDS:
        raise DesignFreezeError("SAFETY12_IDENTITY_DRIFT", repr(safety_ids))
    categories = [row.get("challenge_category") for row in safety]
    if len(set(categories)) != 12:
        raise DesignFreezeError("SAFETY12_CATEGORY_DUPLICATE", repr(categories))
    dispositions = {row.get("expected_disposition") for row in safety}
    if not dispositions.issubset(SAFETY_DISPOSITIONS):
        raise DesignFreezeError("SAFETY12_DISPOSITION_INVALID", repr(dispositions))

    heldout_rubric = _load_json(by_role["heldout27_evaluation_rubric"])
    if heldout_rubric.get("bound_taskbank_sha256") != observed_assets[
        "heldout27_taskbank"
    ]:
        raise DesignFreezeError("HELDOUT27_RUBRIC_BINDING_DRIFT", repr(heldout_rubric))
    if heldout_rubric.get("primary_endpoint", {}).get("name") != (
        "reportable_without_postrun_repair"
    ):
        raise DesignFreezeError("PRIMARY_ENDPOINT_DRIFT", repr(heldout_rubric))

    sap = _load_json(by_role["statistical_analysis_plan"])
    confirmatory = sap.get("confirmatory_family", {})
    if confirmatory.get("primary_endpoint") != "reportable_without_postrun_repair":
        raise DesignFreezeError("SAP_PRIMARY_ENDPOINT_DRIFT", repr(confirmatory))
    if set(confirmatory.get("arms", [])) != ARMS:
        raise DesignFreezeError("SAP_ARM_DRIFT", repr(confirmatory.get("arms")))

    schedule = protocol.get("formal_schedule", {})
    heldout_order = tuple(schedule.get("heldout27_execution_order", []))
    if len(heldout_order) != 27 or set(heldout_order) != set(HELDOUT27_IDS):
        raise DesignFreezeError("HELDOUT27_SCHEDULE_DRIFT", repr(heldout_order))
    heldout_arm_first = schedule.get("heldout27_arm_first", {})
    if set(heldout_arm_first) != set(HELDOUT27_IDS):
        raise DesignFreezeError("HELDOUT27_ARM_ORDER_DRIFT", repr(heldout_arm_first))
    heldout_arm_counts = Counter(heldout_arm_first.values())
    if heldout_arm_counts != Counter({"easyicu_full": 14, "generic_code_agent": 13}):
        raise DesignFreezeError("HELDOUT27_ARM_IMBALANCE", repr(heldout_arm_counts))

    safety_order = tuple(schedule.get("safety12_execution_order", []))
    if safety_order != SAFETY12_IDS:
        raise DesignFreezeError("SAFETY12_SCHEDULE_DRIFT", repr(safety_order))
    safety_arm_counts = Counter(schedule.get("safety12_arm_first", []))
    if safety_arm_counts != Counter({"easyicu_full": 6, "generic_code_agent": 6}):
        raise DesignFreezeError("SAFETY12_ARM_IMBALANCE", repr(safety_arm_counts))

    formal_policy = protocol.get("formal_run_policy", {})
    if formal_policy.get("core_formal_runs") != 78:
        raise DesignFreezeError("CORE_RUN_COUNT_DRIFT", repr(formal_policy))
    if formal_policy.get("failures_remain_in_denominator") is not True:
        raise DesignFreezeError("DENOMINATOR_POLICY_DRIFT", repr(formal_policy))
    for prohibited_true in (
        "reuse_existing",
        "resume",
        "cross_run_memory",
        "posthoc_repair",
        "selective_arm_retry",
    ):
        if formal_policy.get(prohibited_true) is not False:
            raise DesignFreezeError("FORMAL_POLICY_DRIFT", prohibited_true)

    authority = protocol.get("current_authority", {})
    for field in (
        "provider_calls_authorized",
        "planner_calls_authorized",
        "formal_batch_authorized",
        "paper_result_authority",
    ):
        if authority.get(field) is not False:
            raise DesignFreezeError("PREMATURE_FORMAL_AUTHORITY", field)

    return DesignFreezeReceipt(
        protocol_ref=str(protocol["protocol_ref"]),
        protocol_sha256=_sha256(protocol_path),
        asset_sha256=tuple(sorted(observed_assets.items())),
        heldout_task_count=len(heldout),
        safety_task_count=len(safety),
        core_formal_runs=int(formal_policy["core_formal_runs"]),
        formal_batch_authorized=False,
    )


__all__ = [
    "DesignFreezeError",
    "DesignFreezeReceipt",
    "PROTOCOL_PATH",
    "validate_design_freeze",
]
