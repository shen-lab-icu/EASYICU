"""Step-level repair coordination for the A2 control-plane split.

This module is a LINE-FOR-LINE extraction of four accounting closures that
lived inside ``pipeline_execute.run_execute_phase``'s step worker, plus the
authorized deterministic concept-repair helper.  Behavior preservation is the
contract:

* ``step_record`` is the SAME dict the caller owns; every key name, value and
  write order matches the original closures exactly (characterization and
  resume tests replay these keys — ``step_llm_repair_classes`` is a persisted
  contract verified by ``_monotonic_step_llm_repair_history`` on resume).
* The provider probe uses ``can_consume`` (never ``consume``), so a refused
  reservation is not misrecorded as a real paid attempt and the durable
  provider receipt is untouched.
* ``authorized_deterministic_concept_repair`` keeps its all-or-nothing
  semantics; the authorization side effects (repair ledger + findings) stay
  with the injected ``authorize`` callback, which remains defined at the call
  site.

The LLM transport ladder also lives here.  It is deliberately science-neutral:
the Coder still owns the repair prompt and the Planner-owned scientific
coordinates, while :class:`RepairCoordinator` decides only whether the current
provider allowance can afford patch-then-rewrite or must spend its sole
non-audit slot directly on a full rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .code_patch import CodePatchError, apply_code_patch
from .code_repair import deterministic_concept_audit_repair
from .provider_budget import (
    PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION,
    StepProviderCallBudget,
    complete_with_provider_budget,
)


class StepRepairBudget:
    """Logical LLM-repair allowance + provider-call probe for one step.

    Wraps the two coupled budgets a repair must clear:

    1. the per-step LOGICAL repair allowance
       (``pipeline._max_step_llm_repair_attempts``), whose consumption is
       recorded in ``step_record`` and replayed monotonically on resume; and
    2. the per-step PROVIDER-call budget (:class:`StepProviderCallBudget`),
       probed neutrally before any repair reservation.
    """

    def __init__(
        self,
        *,
        provider_budget: StepProviderCallBudget,
        step_record: Dict[str, Any],
        max_llm_repairs: int,
        initial_llm_repair_attempts: int = 0,
        provider_receipt_relative_path: Optional[str] = None,
    ) -> None:
        self._provider_budget = provider_budget
        self._step_record = step_record
        self._max_llm_repairs = int(max_llm_repairs)
        self._llm_repair_attempts = int(initial_llm_repair_attempts)
        self._provider_receipt_relative_path = provider_receipt_relative_path

    @property
    def llm_repair_attempts(self) -> int:
        return self._llm_repair_attempts

    @property
    def provider_budget(self) -> StepProviderCallBudget:
        return self._provider_budget

    def sync_provider(self) -> None:
        """Project the provider-budget snapshot into the step record."""

        snapshot = self._provider_budget.snapshot()
        step_record = self._step_record
        step_record["step_provider_call_budget_scope"] = (
            "coder_generation_repair_concept_audit_and_analyzer"
        )
        step_record["step_provider_call_budget"] = snapshot["limit"]
        step_record["step_provider_call_attempts"] = snapshot["used"]
        step_record["step_provider_call_remaining"] = snapshot["remaining"]
        step_record["step_provider_call_budget_exhausted"] = snapshot["exhausted"]
        step_record["step_provider_call_categories"] = snapshot["categories"]
        step_record["step_provider_call_reserved_category"] = snapshot[
            "reserved_final_category"
        ]
        step_record["step_provider_call_reservation_released"] = snapshot[
            "reservation_released"
        ]
        step_record["step_provider_call_receipt_version"] = (
            PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION
        )
        step_record["step_provider_call_receipt"] = (
            self._provider_receipt_relative_path if snapshot["used"] else None
        )

    def logical_available(self) -> bool:
        return self._llm_repair_attempts < self._max_llm_repairs

    def provider_available(self) -> bool:
        # Every Coder repair starts with a non-audit patch reservation.  The
        # exact category name does not affect the reserved-final-audit rule,
        # so a neutral probe prevents a refused reservation from being
        # misrecorded as a real logical repair attempt.
        available = self._provider_budget.can_consume("llm_repair_budget_probe")
        if not available:
            self._step_record["step_provider_call_repair_unavailable"] = True
            self.sync_provider()
        return available

    def available(self) -> bool:
        return self.logical_available() and self.provider_available()

    def consume(self, repair_class: str) -> bool:
        if not self.logical_available():
            self._step_record["step_llm_repair_budget_exhausted"] = True
            self._step_record["step_llm_repair_budget"] = self._max_llm_repairs
            return False
        if not self.provider_available():
            return False
        self._llm_repair_attempts += 1
        self._step_record["step_llm_repair_attempts"] = self._llm_repair_attempts
        self._step_record["step_llm_repair_budget"] = self._max_llm_repairs
        self._step_record.setdefault("step_llm_repair_classes", []).append(
            str(repair_class)
        )
        return True


@dataclass(frozen=True)
class RepairTransportResult:
    """Result of one mechanical patch/full-rewrite transport transaction."""

    code: str
    mode: str
    provider_calls: int


class RepairCoordinator:
    """Execute the mechanical patch → optional full-rewrite ladder once.

    The coordinator never chooses a model, variable, cohort, product, or
    estimand.  It receives already-built Coder prompts as callables and owns
    only transport accounting plus deterministic response parsing.

    When the shared budget has exactly one non-audit call left while a final
    audit slot remains reserved, attempting a patch is unsafe: a malformed
    patch would consume the only repair slot and make its required fallback
    impossible.  In that state the coordinator goes directly to full rewrite.
    """

    def __init__(
        self,
        *,
        provider_budget: Optional[StepProviderCallBudget],
        provider_category: str,
        normalize_script: Callable[[str], str],
        is_executable_script: Callable[[str], bool],
    ) -> None:
        self._provider_budget = provider_budget
        self._provider_category = str(provider_category).strip() or "repair"
        self._normalize_script = normalize_script
        self._is_executable_script = is_executable_script

    def _must_skip_patch(self) -> bool:
        if self._provider_budget is None:
            return False
        snapshot = self._provider_budget.snapshot()
        reservation_active = bool(
            snapshot.get("reserved_final_category")
            and not snapshot.get("reservation_released")
        )
        if not reservation_active:
            return False
        non_audit_remaining = max(0, int(snapshot["remaining"]) - 1)
        return non_audit_remaining == 1

    def _call(self, suffix: str, call: Callable[[], str]) -> str:
        return complete_with_provider_budget(
            budget=self._provider_budget,
            category=f"{self._provider_category}_{suffix}",
            call=call,
        )

    def _parse_full_rewrite(self, *, code: str, raw: str) -> tuple[str, str]:
        try:
            # Some providers ignore the full-script instruction but still emit
            # a valid exact patch. Applying it remains safer than executing a
            # JSON payload as Python.
            return apply_code_patch(code, raw), "full_rewrite_response_patch"
        except CodePatchError:
            return self._normalize_script(str(raw or "").strip()), "full_rewrite"

    def repair(
        self,
        *,
        code: str,
        patch_call: Callable[[], str],
        full_rewrite_call: Callable[[str], str],
    ) -> RepairTransportResult:
        calls_before = self._provider_budget.used if self._provider_budget else 0
        repaired: str
        mode: str

        if self._must_skip_patch():
            raw = self._call(
                "full_rewrite",
                lambda: full_rewrite_call(
                    "minimal patch skipped because only one non-audit provider "
                    "call remained while the mandatory final audit stayed reserved"
                ),
            )
            repaired, mode = self._parse_full_rewrite(code=code, raw=raw)
        else:
            raw_patch = self._call("patch", patch_call)
            try:
                repaired = apply_code_patch(code, raw_patch)
                mode = "minimal_patch"
            except CodePatchError as patch_error:
                # If a provider ignored the JSON instruction but already
                # returned a valid complete script, that response is the one
                # allowed fallback; do not buy another full rewrite.
                direct_script = self._normalize_script(str(raw_patch or "").strip())
                if self._is_executable_script(direct_script):
                    repaired = direct_script
                    mode = "patch_response_full_script"
                else:
                    fallback_reason = str(patch_error)
                    raw = self._call(
                        "full_rewrite",
                        lambda: full_rewrite_call(fallback_reason),
                    )
                    repaired, mode = self._parse_full_rewrite(code=code, raw=raw)

        if not self._is_executable_script(repaired):
            raise ValueError(
                "Coder repair returned non-script output; refusing to replace "
                "the previous analysis script."
            )
        calls_after = self._provider_budget.used if self._provider_budget else 0
        return RepairTransportResult(
            code=repaired,
            mode=mode,
            provider_calls=(calls_after - calls_before) if self._provider_budget else 0,
        )


def authorized_deterministic_concept_repair(
    script_text: str,
    error_messages: Sequence[str],
    *,
    authorize: Callable[..., Optional[Any]],
    step: Any,
    source: str,
) -> Tuple[str, List[str]]:
    """Return an all-or-nothing centrally authorized mechanical repair."""

    candidate_code, repair_names = deterministic_concept_audit_repair(
        script_text,
        error_messages,
    )
    if not repair_names or candidate_code == script_text:
        return script_text, []
    for repair_name in repair_names:
        if (
            authorize(
                (repair_name, candidate_code),
                step=step,
                source=source,
                before_code=script_text,
            )
            is None
        ):
            return script_text, []
    return candidate_code, list(repair_names)


__all__ = [
    "RepairCoordinator",
    "RepairTransportResult",
    "StepRepairBudget",
    "authorized_deterministic_concept_repair",
]
