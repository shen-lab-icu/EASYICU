"""Shared structural semantics for both Figure 2 review-bundle producers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence


CANONICAL_FILES = (
    "01_plan.json",
    "02_cohort.json",
    "03_results.json",
    "04_diagnostics.json",
    "05_evidence_manifest.json",
    "06_report.md",
    "07_run_receipt.json",
)
ARTIFACT_REFERENCE_FILES = frozenset(CANONICAL_FILES[:4] + ("06_report.md",))
SUBSTANTIVE_OUTPUT_FILES = (
    "02_cohort.json",
    "03_results.json",
    "04_diagnostics.json",
    "06_report.md",
)
RESOURCE_RECEIPT_FIELDS = (
    "resource_receipt_schema_version",
    "within_frozen_budget",
    "provider_calls",
    "provider_tokens",
    "accounted_cost_upper_bound_usd",
    "reported_billed_cost_usd",
    "model_turns",
    "tool_calls",
    "wall_seconds",
)


class TerminalStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class FailureCategory(str, Enum):
    AGENT_OUTPUT_CONTRACT_ERROR = "agent_output_contract_error"
    BUDGET_EXHAUSTED = "budget_exhausted"
    EXECUTION_FAILURE = "execution_failure"
    EXECUTION_TIMEOUT = "execution_timeout"
    PLAN_REJECTED = "plan_rejected"


@dataclass(frozen=True)
class TerminalOutcome:
    """Closed, arm-neutral terminal meaning for one review bundle."""

    status: TerminalStatus
    failure_category: FailureCategory | None

    def __post_init__(self) -> None:
        if (self.status is TerminalStatus.COMPLETED) != (
            self.failure_category is None
        ):
            raise ValueError(
                "completed requires no failure category; failed requires one"
            )

    @classmethod
    def completed(cls) -> TerminalOutcome:
        return cls(TerminalStatus.COMPLETED, None)

    @classmethod
    def failed(cls, category: FailureCategory) -> TerminalOutcome:
        if not isinstance(category, FailureCategory):
            raise TypeError("failure category must be FailureCategory")
        return cls(TerminalStatus.FAILED, category)

    @classmethod
    def parse(cls, status: Any, category: Any) -> TerminalOutcome:
        try:
            normalized_status = TerminalStatus(status)
            normalized_category = (
                None if category is None else FailureCategory(category)
            )
            return cls(normalized_status, normalized_category)
        except (TypeError, ValueError) as exc:
            raise ValueError("terminal outcome is outside the closed vocabulary") from exc

    def receipt_fields(self) -> dict[str, str | None]:
        return {
            "terminal_status": self.status.value,
            "failure_category": (
                None
                if self.failure_category is None
                else self.failure_category.value
            ),
        }


def _optional_nonnegative_number(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a non-negative finite number or null")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{field} must be a non-negative finite number or null")
    return normalized


def _optional_nonnegative_integer(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer or null")
    return value


@dataclass(frozen=True)
class ReviewResourceReceipt:
    """Fixed raw resource evidence shared by both experiment arms."""

    within_frozen_budget: bool
    provider_calls: int | None = None
    provider_tokens: int | None = None
    accounted_cost_upper_bound_usd: float | None = None
    reported_billed_cost_usd: float | None = None
    model_turns: int | None = None
    tool_calls: int | None = None
    wall_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.within_frozen_budget, bool):
            raise ValueError("within_frozen_budget must be a boolean")
        for field in ("provider_calls", "provider_tokens", "model_turns", "tool_calls"):
            _optional_nonnegative_integer(getattr(self, field), field=field)
        for field in (
            "accounted_cost_upper_bound_usd",
            "reported_billed_cost_usd",
            "wall_seconds",
        ):
            _optional_nonnegative_number(getattr(self, field), field=field)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, Any],
        *,
        model_turns: int | None = None,
        tool_calls: int | None = None,
        wall_seconds: float | None = None,
    ) -> ReviewResourceReceipt:
        if "within_frozen_budget" not in snapshot:
            raise ValueError("resource snapshot is missing within_frozen_budget")
        billed = snapshot.get(
            "reported_billed_cost_usd",
            snapshot.get("billed_cost"),
        )
        return cls(
            within_frozen_budget=snapshot["within_frozen_budget"],
            provider_calls=_optional_nonnegative_integer(
                snapshot.get("provider_calls"), field="provider_calls"
            ),
            provider_tokens=_optional_nonnegative_integer(
                snapshot.get("provider_tokens"), field="provider_tokens"
            ),
            accounted_cost_upper_bound_usd=_optional_nonnegative_number(
                snapshot.get("accounted_cost_upper_bound_usd"),
                field="accounted_cost_upper_bound_usd",
            ),
            reported_billed_cost_usd=_optional_nonnegative_number(
                billed,
                field="reported_billed_cost_usd",
            ),
            model_turns=_optional_nonnegative_integer(
                model_turns if model_turns is not None else snapshot.get("model_turns"),
                field="model_turns",
            ),
            tool_calls=_optional_nonnegative_integer(
                tool_calls if tool_calls is not None else snapshot.get("tool_calls"),
                field="tool_calls",
            ),
            wall_seconds=_optional_nonnegative_number(
                wall_seconds if wall_seconds is not None else snapshot.get("wall_seconds"),
                field="wall_seconds",
            ),
        )

    @classmethod
    def from_provider_accounting(
        cls,
        accounting: Mapping[str, Any],
        *,
        within_frozen_budget: bool,
        reported_billed_cost_usd: float | None = None,
    ) -> ReviewResourceReceipt:
        conservative = accounting.get("conservative_upper_bound")
        if not isinstance(conservative, Mapping):
            raise ValueError("provider accounting lacks conservative_upper_bound")
        return cls(
            within_frozen_budget=within_frozen_budget,
            provider_calls=_optional_nonnegative_integer(
                conservative.get("n_calls"), field="provider_calls"
            ),
            provider_tokens=_optional_nonnegative_integer(
                conservative.get("total_tokens"), field="provider_tokens"
            ),
            accounted_cost_upper_bound_usd=_optional_nonnegative_number(
                conservative.get("estimated_cost_usd"),
                field="accounted_cost_upper_bound_usd",
            ),
            reported_billed_cost_usd=_optional_nonnegative_number(
                reported_billed_cost_usd,
                field="reported_billed_cost_usd",
            ),
        )

    def as_dict(self) -> dict[str, bool | int | float | None | str]:
        return {
            "resource_receipt_schema_version": "easyicu.figure2_resource_receipt/1",
            "within_frozen_budget": self.within_frozen_budget,
            "provider_calls": self.provider_calls,
            "provider_tokens": self.provider_tokens,
            "accounted_cost_upper_bound_usd": self.accounted_cost_upper_bound_usd,
            "reported_billed_cost_usd": self.reported_billed_cost_usd,
            "model_turns": self.model_turns,
            "tool_calls": self.tool_calls,
            "wall_seconds": self.wall_seconds,
        }


def normalize_artifact_inventory(
    inventory: Mapping[str, Any],
    mandatory_artifacts: Sequence[str],
) -> dict[str, list[str]]:
    """Validate only references to canonical bundle files, without judging science."""

    labels = tuple(mandatory_artifacts)
    if set(inventory) != set(labels):
        raise ValueError("artifact_inventory must map every frozen mandatory artifact")
    normalized: dict[str, list[str]] = {}
    for label in labels:
        references = inventory[label]
        if (
            not isinstance(references, list)
            or not references
            or not all(
                isinstance(reference, str)
                and reference in ARTIFACT_REFERENCE_FILES
                for reference in references
            )
        ):
            raise ValueError(f"artifact_inventory has invalid references for {label!r}")
        normalized[label] = list(dict.fromkeys(references))
    return normalized


def substantive_file_flags(
    *,
    plan: Mapping[str, Any],
    cohort: Mapping[str, Any],
    results: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    report: str,
) -> dict[str, bool]:
    """Return file-level non-emptiness flags; these are not scientific validation."""

    values = {
        "01_plan.json": bool(plan),
        "02_cohort.json": bool(cohort),
        "03_results.json": bool(results),
        "04_diagnostics.json": bool(diagnostics),
        "06_report.md": bool(report.strip()),
    }
    return {name: values[name] for name in SUBSTANTIVE_OUTPUT_FILES}


def asserted_artifact_presence(
    inventory: Mapping[str, Sequence[str]],
    *,
    plan: Mapping[str, Any],
    cohort: Mapping[str, Any],
    results: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    report: str,
) -> dict[str, bool]:
    """Project producer assertions; human gates still determine adequacy."""

    substantive = {
        "01_plan.json": bool(plan),
        **substantive_file_flags(
            plan=plan,
            cohort=cohort,
            results=results,
            diagnostics=diagnostics,
            report=report,
        ),
    }
    return {
        label: all(substantive[reference] for reference in references)
        for label, references in inventory.items()
    }


__all__ = [
    "ARTIFACT_REFERENCE_FILES",
    "CANONICAL_FILES",
    "FailureCategory",
    "RESOURCE_RECEIPT_FIELDS",
    "ReviewResourceReceipt",
    "SUBSTANTIVE_OUTPUT_FILES",
    "TerminalOutcome",
    "TerminalStatus",
    "asserted_artifact_presence",
    "normalize_artifact_inventory",
    "substantive_file_flags",
]
