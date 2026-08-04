"""Step-level repair coordination for the A2 control-plane split.

This module is a LINE-FOR-LINE extraction of four accounting closures that
lived inside ``execution.phase.run_execute_phase``'s step worker, plus the
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

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .patch import CodePatchError, apply_code_patch
from .binary_domain import patch_observed_binary_primary_exposure_guard
from .source import (
    _deterministic_runner_repair,
    _deterministic_summary_repair,
    deterministic_concept_audit_repair,
)
from .reasons import RepairReason
from ..schema import ValidationFinding
from ..authority.provider_budget import (
    PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION,
    ProviderCallBudgetReceiptError,
    StepProviderCallBudget,
    complete_with_provider_budget,
)

REPAIR_AUTHORITY_BINDING_SCHEMA_VERSION = "easyicu.repair_authority_binding/2"

#: Repair classes raised by the LAST gate a step can fail -- the concept audit,
#: which reads the code only after it has run.  Every other class (runtime,
#: contract, compatibility, visual, critic_resume) is discovered earlier and
#: draws from the same pool, so the terminal gate is the only one that can be
#: starved by a failure that came before it.  See
#: :meth:`StepRepairBudget.effective_limit`.
TERMINAL_GATE_REPAIR_CLASSES = frozenset({"concept", "post_mutation_concept"})

#: The stages whose failures cannot be known until everything before them has
#: already had its chance to spend the pool.  Each gets ONE repair of its own,
#: once, on top of the shared allowance.
#:
#: * the concept audit -- it reads the code only after it has run;
#: * execution itself -- a traceback exists only once the script runs, and
#:   every pre-execution gate (contract, compatibility, visual, and the audit)
#:   draws from the same pool first.
#:
#: MEASURED over every recorded run: 89 steps ended ``execution_failed`` and 20
#: of them -- across 7 of the 9 tasks -- had spent ZERO runtime repairs while
#: provider calls remained, the pool having gone to ``contract`` (11),
#: ``post_mutation_concept`` (5) and ``concept`` (3+1).  A runtime repair is
#: worth attempting: of the 211 steps that ever spent one, 90 (43 %) finished
#: ``ok`` -- a better rate than the concept gate's 38 %.  A traceback is the
#: most specific repair signal the pipeline produces, and it was the one most
#: often unaffordable.
TERMINAL_STAGE_REPAIR_CLASSES: Tuple[frozenset, ...] = (
    TERMINAL_GATE_REPAIR_CLASSES,
    frozenset({"runtime"}),
)


def resume_deterministic_repair_candidate(
    *,
    code: str,
    step_dir: Path,
    analysis_family: str | None,
) -> tuple[tuple[str, str], str, dict[str, Any]] | None:
    """Select a proven runtime/summary repair for an explicitly resumed step."""

    run_log_path = step_dir / "run.log"
    if run_log_path.is_file():
        run_log = run_log_path.read_text(encoding="utf-8", errors="replace")
        repair = _deterministic_runner_repair(
            code=code,
            run_log=run_log,
            previous_repair=None,
            analysis_family=analysis_family,
        )
        if repair is not None:
            source = "resume_runner_repair_preflight"
            return (
                repair,
                source,
                {
                    "source": source,
                    "run_log_path": str(run_log_path),
                },
            )

    summary_path = step_dir / "outputs" / "step_summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(summary, dict) or not summary:
        return None
    repair = _deterministic_summary_repair(
        code=code,
        step_summary=summary,
        previous_repair=None,
        analysis_family=analysis_family or summary.get("analysis_family"),
    )
    if repair is None:
        return None
    source = "resume_summary_repair_preflight"
    return (
        repair,
        source,
        {
            "source": source,
            "step_summary_path": str(summary_path),
            "step_summary_keys": sorted(str(key) for key in summary),
        },
    )


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True)
class RepairAuthorityBinding:
    """Content-addressed authority coordinates for one logical repair.

    The binding contains only host-owned identities.  It does not choose any
    scientific coordinate; it proves which Planner step, candidate code,
    typed inputs/context, repair ticket, gate implementation and prompt pack
    authorized a paid repair attempt.
    """

    step_id: str
    attempt_id: int
    repair_class: str
    provider_category: str
    before_code_sha256: str
    step_spec_sha256: str
    resolved_inputs_sha256: str
    coder_context_sha256: str
    repair_ticket_sha256: str
    engine_validator_sha256: str
    prompt_pack_version: str
    run_input_capsule_sha256: Optional[str] = None
    schema_version: str = REPAIR_AUTHORITY_BINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPAIR_AUTHORITY_BINDING_SCHEMA_VERSION:
            raise ValueError("repair authority binding schema is unsupported")
        if not str(self.step_id).strip():
            raise ValueError("repair authority step_id must be non-empty")
        if (
            isinstance(self.attempt_id, bool)
            or not isinstance(self.attempt_id, int)
            or self.attempt_id < 1
        ):
            raise ValueError("repair authority attempt_id must be >= 1")
        if not str(self.repair_class).strip():
            raise ValueError("repair authority class must be non-empty")
        provider_category = self.provider_category
        if (
            not isinstance(provider_category, str)
            or not provider_category.strip()
            or provider_category != provider_category.strip()
        ):
            raise ValueError("repair authority provider category must be non-empty")
        if not str(self.prompt_pack_version).strip():
            raise ValueError("repair authority prompt version must be non-empty")
        digest_fields = (
            "before_code_sha256",
            "step_spec_sha256",
            "resolved_inputs_sha256",
            "coder_context_sha256",
            "repair_ticket_sha256",
            "engine_validator_sha256",
        )
        for field_name in digest_fields:
            value = str(getattr(self, field_name)).strip().lower()
            if not _is_sha256_hex(value):
                raise ValueError(
                    f"repair authority {field_name} must be a SHA-256 hex digest"
                )
            object.__setattr__(self, field_name, value)
        if self.run_input_capsule_sha256 is not None:
            capsule_digest = str(self.run_input_capsule_sha256).strip().lower()
            if not _is_sha256_hex(capsule_digest):
                raise ValueError(
                    "repair authority run_input_capsule_sha256 must be a "
                    "SHA-256 hex digest"
                )
            object.__setattr__(self, "run_input_capsule_sha256", capsule_digest)

    def payload(self) -> Dict[str, object]:
        """Return the canonical JSON-safe receipt payload."""

        return asdict(self)

    @property
    def sha256(self) -> str:
        canonical = json.dumps(
            self.payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
        initial_repair_classes: Sequence[str] = (),
        provider_receipt_relative_path: Optional[str] = None,
    ) -> None:
        self._provider_budget = provider_budget
        self._step_record = step_record
        self._max_llm_repairs = int(max_llm_repairs)
        prior_classes = tuple(str(item).strip() for item in initial_repair_classes)
        if prior_classes:
            if len(prior_classes) != int(initial_llm_repair_attempts):
                raise ValueError("initial logical repair attempts and classes disagree")
            self._provider_budget.migrate_logical_repairs(prior_classes)
        durable_classes = self._provider_budget.logical_repair_classes
        if prior_classes and (
            len(prior_classes) > len(durable_classes)
            or durable_classes[: len(prior_classes)] != prior_classes
        ):
            raise ValueError("durable logical repair history conflicts with snapshot")
        self._llm_repair_attempts = max(
            int(initial_llm_repair_attempts),
            len(durable_classes),
        )
        if durable_classes:
            self._step_record["step_llm_repair_attempts"] = len(durable_classes)
            self._step_record["step_llm_repair_budget"] = self._max_llm_repairs
            self._step_record["step_llm_repair_classes"] = list(durable_classes)
            provider_snapshot = self._provider_budget.snapshot()
            self._step_record["step_llm_repair_bindings"] = list(
                provider_snapshot["logical_repair_binding_sha256"]
            )
            self._step_record["step_llm_repair_transport_states"] = list(
                provider_snapshot["logical_repair_transport_states"]
            )
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
        step_record["step_llm_repair_transport_states"] = snapshot[
            "logical_repair_transport_states"
        ]
        step_record["step_provider_call_receipt"] = (
            self._provider_receipt_relative_path
            if (
                snapshot["used"]
                or snapshot["logical_repair_attempts"]
                or snapshot["initial_generation_transport_state"] is not None
            )
            else None
        )

    def effective_limit(self, repair_class: Optional[str] = None) -> int:
        """Return the logical allowance visible to ``repair_class``.

        The concept audit is the LAST gate a step can fail and the only one
        whose finding cannot be known before the code runs.  It nevertheless
        drew from the same flat pool as every earlier class, so a step could
        spend its whole allowance on runtime or contract failures and then be
        refused at a gate it was never given one chance to answer.

        MEASURED over every recorded run: 80 steps ended
        ``blocked_by_concept_audit``; 30 of them -- across 7 of the 9 tasks --
        had spent ZERO concept-class repairs, the pool having gone to
        ``runtime`` (14), ``contract`` (6+3), ``compatibility`` (2) and others.
        28 of those 30 still had unspent provider calls.  A concept repair is
        worth attempting: of the 171 steps that ever spent one, 65 (38 %)
        finished ``ok``.

        The reserve is ADDITIVE, not a partition of the existing pool.  The
        strict alternative -- holding one of the two back -- was measured and
        rejected: 40 steps that currently finish ``ok`` spent two non-concept
        repairs and would have lost one.

        This mirrors the rule the provider-call budget already applies one
        level down (``reserved_final_category = "concept_audit"``); the logical
        allowance simply never had it.
        """

        limit = self._max_llm_repairs
        normalized = str(repair_class or "").strip()
        stage = next(
            (item for item in TERMINAL_STAGE_REPAIR_CLASSES if normalized in item),
            None,
        )
        if stage is None:
            return limit
        spent = [str(item).strip() for item in self._provider_budget.logical_repair_classes]
        if any(item in stage for item in spent):
            return limit
        # One repair for this stage, whenever it is reached. Expressed against
        # the attempts already made rather than as a fixed "+1" so that each
        # stage's reserve is independent: a step that has already spent its
        # concept reserve must still be able to answer a traceback, and vice
        # versa. Bounded by construction -- a stage that has been paid takes
        # the branch above and never gets a second.
        return max(limit, len(spent) + 1)

    def logical_available(self, repair_class: Optional[str] = None) -> bool:
        limit = self.effective_limit(repair_class)
        next_attempt_id = self._provider_budget.next_logical_repair_attempt_id()
        durable_attempts = int(
            self._provider_budget.snapshot()["logical_repair_attempts"]
        )
        if next_attempt_id <= durable_attempts:
            return next_attempt_id <= limit
        return max(self._llm_repair_attempts, durable_attempts) < limit

    @property
    def next_attempt_id(self) -> int:
        """Return the next new or exact unpaid-resume attempt identifier."""

        return self._provider_budget.next_logical_repair_attempt_id()

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

    def available(self, repair_class: Optional[str] = None) -> bool:
        return self.logical_available(repair_class) and self.provider_available()

    def consume(
        self,
        repair_class: str,
        *,
        authority_binding: Optional[RepairAuthorityBinding] = None,
    ) -> bool:
        normalized_class = str(repair_class).strip()
        if not self.logical_available(normalized_class):
            self._step_record["step_llm_repair_budget_exhausted"] = True
            self._step_record["step_llm_repair_budget"] = self._max_llm_repairs
            return False
        expected_attempt_id = self.next_attempt_id
        if authority_binding is not None:
            if authority_binding.attempt_id != expected_attempt_id:
                raise ValueError(
                    "repair authority attempt_id does not match the next durable "
                    "logical repair attempt"
                )
            if authority_binding.repair_class != normalized_class:
                raise ValueError(
                    "repair authority class does not match the requested repair"
                )
        durable_attempts_before = int(
            self._provider_budget.snapshot()["logical_repair_attempts"]
        )
        attempt_id = self._provider_budget.reserve_logical_repair(
            normalized_class,
            max_repairs=self.effective_limit(normalized_class),
            binding=(
                authority_binding.payload() if authority_binding is not None else None
            ),
            binding_sha256=(
                authority_binding.sha256 if authority_binding is not None else None
            ),
        )
        if attempt_id is None:
            if not self.logical_available(normalized_class):
                self._step_record["step_llm_repair_budget_exhausted"] = True
                self._step_record["step_llm_repair_budget"] = self._max_llm_repairs
            else:
                self._step_record["step_provider_call_repair_unavailable"] = True
                self.sync_provider()
            return False
        resumed_unpaid_attempt = attempt_id <= durable_attempts_before
        self._llm_repair_attempts = max(self._llm_repair_attempts, attempt_id)
        self._step_record["step_llm_repair_attempts"] = self._llm_repair_attempts
        self._step_record["step_llm_repair_budget"] = self._max_llm_repairs
        if attempt_id > self._max_llm_repairs:
            # Otherwise the record reads "budget 2, attempts 3" with nothing
            # saying why. The reserve is a host decision and belongs in the
            # manifest beside the allowance it extends.
            self._step_record.setdefault(
                "step_llm_repair_terminal_gate_reserve", []
            )
            reserves = self._step_record["step_llm_repair_terminal_gate_reserve"]
            if isinstance(reserves, list):
                reserves.append(normalized_class)
            else:  # a pre-existing scalar record from an earlier run
                self._step_record["step_llm_repair_terminal_gate_reserve"] = [
                    reserves,
                    normalized_class,
                ]
        if not resumed_unpaid_attempt:
            self._step_record.setdefault("step_llm_repair_classes", []).append(
                normalized_class
            )
            if authority_binding is not None:
                self._step_record.setdefault("step_llm_repair_bindings", []).append(
                    authority_binding.sha256
                )
        self.sync_provider()
        return True


class PatchTransportUnavailable(Exception):
    """The minimal patch could not be *posed* -- not answered wrongly.

    The distinction decides who pays. A patch the provider answered badly is
    evidence about the code and rightly spends a repair. A patch the host
    could not even assemble is evidence about the host, and the full-rewrite
    transport beside it is still available.

    Measured 2026-07-29: a real E1 primary step lost its last repair to
    ``CoderPromptBudgetError`` with ``provider_calls: 0``. The preceding repair
    had produced a 30,105-byte script, so the *patch* prompt built around it
    crossed its 30,000-byte envelope. Nothing was sent, nothing was answered,
    and the step died reported as ``repair_failed`` -- pointing at the science
    rather than at the prompt that was never built.
    """


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
        finalize_script: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._provider_budget = provider_budget
        self._provider_category = str(provider_category).strip() or "repair"
        self._normalize_script = normalize_script
        self._is_executable_script = is_executable_script
        self._finalize_script = finalize_script or (lambda value: value)

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
        patch_preflight: Optional[Callable[[], None]] = None,
        full_rewrite_preflight: Optional[Callable[[str], None]] = None,
        logical_repair_attempt_id: Optional[int] = None,
        persist_result: Optional[Callable[[str, str], object]] = None,
    ) -> RepairTransportResult:
        if logical_repair_attempt_id is not None and self._provider_budget is None:
            raise ValueError(
                "logical repair transport binding requires a provider budget"
            )
        if logical_repair_attempt_id is not None:
            assert self._provider_budget is not None
            self._provider_budget.assert_logical_repair_provider_category(
                attempt_id=logical_repair_attempt_id,
                provider_category=self._provider_category,
            )

        try:
            result = self._repair_once(
                code=code,
                patch_call=patch_call,
                full_rewrite_call=full_rewrite_call,
                patch_preflight=patch_preflight,
                full_rewrite_preflight=full_rewrite_preflight,
            )
            finalized_code = self._finalize_script(result.code)
            if not self._is_executable_script(finalized_code):
                raise ValueError("finalized Coder repair is not an executable script")
            result = RepairTransportResult(
                code=finalized_code,
                mode=result.mode,
                provider_calls=result.provider_calls,
            )
        except ProviderCallBudgetReceiptError:
            raise
        except Exception as exc:
            if logical_repair_attempt_id is not None:
                assert self._provider_budget is not None
                self._provider_budget.fail_logical_repair_transport(
                    attempt_id=logical_repair_attempt_id,
                    error_type=type(exc).__name__,
                )
            raise

        persisted_result: object = None
        if persist_result is not None:
            try:
                persisted_result = persist_result(result.code, result.mode)
            except ProviderCallBudgetReceiptError:
                raise
            except Exception as exc:
                if logical_repair_attempt_id is not None:
                    assert self._provider_budget is not None
                    self._provider_budget.fail_logical_repair_transport(
                        attempt_id=logical_repair_attempt_id,
                        error_type=type(exc).__name__,
                    )
                raise

        if logical_repair_attempt_id is not None:
            assert self._provider_budget is not None
            self._provider_budget.complete_logical_repair_transport(
                attempt_id=logical_repair_attempt_id,
                mode=result.mode,
                after_code_sha256=hashlib.sha256(
                    result.code.encode("utf-8")
                ).hexdigest(),
                after_code_size_bytes=(
                    int(getattr(persisted_result, "size_bytes"))
                    if getattr(persisted_result, "size_bytes", None) is not None
                    else None
                ),
            )
        return result

    def _repair_once(
        self,
        *,
        code: str,
        patch_call: Callable[[], str],
        full_rewrite_call: Callable[[str], str],
        patch_preflight: Optional[Callable[[], None]],
        full_rewrite_preflight: Optional[Callable[[str], None]],
    ) -> RepairTransportResult:
        calls_before = self._provider_budget.used if self._provider_budget else 0
        repaired: str
        mode: str

        # Three ways to have no minimal patch, one answer. The patch may be
        # unaffordable (below), unassemblable (``PatchTransportUnavailable``),
        # or unusable once answered (``CodePatchError`` further down). Each
        # falls through to the full-rewrite transport, which carries its own
        # preflight and its own envelope. Until 2026-07-29 the middle case was
        # the odd one out and killed the attempt outright: a real E1 primary
        # step lost its last repair to a prompt that was never built, and was
        # then reported as ``repair_failed`` -- naming the science instead of
        # the host.
        skip_patch_reason: Optional[str] = None
        if self._must_skip_patch():
            skip_patch_reason = (
                "minimal patch skipped because only one non-audit provider "
                "call remained while the mandatory final audit stayed reserved"
            )
        elif patch_preflight is not None:
            try:
                patch_preflight()
            except PatchTransportUnavailable as preflight_error:
                skip_patch_reason = str(preflight_error)

        if skip_patch_reason is not None:
            if full_rewrite_preflight is not None:
                full_rewrite_preflight(skip_patch_reason)
            raw = self._call(
                "full_rewrite",
                lambda: full_rewrite_call(skip_patch_reason),
            )
            repaired, mode = self._parse_full_rewrite(code=code, raw=raw)
        else:
            raw_patch = self._call("patch", patch_call)
            try:
                repaired = apply_code_patch(code, raw_patch)
                mode = "minimal_patch"
            except CodePatchError as patch_error:
                # Patch transport sees only selected code blocks. A provider
                # response that ignores PATCH_FORMAT therefore has no authority
                # to replace the complete candidate, even if it parses as
                # executable Python. Only the explicit full-rewrite transport
                # receives the complete script and scoped science authority.
                fallback_reason = str(patch_error)
                if full_rewrite_preflight is not None:
                    full_rewrite_preflight(fallback_reason)
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
    repair_reasons: Sequence[RepairReason] = (),
    repair_findings: Sequence[ValidationFinding] = (),
    authorize: Callable[..., Optional[Any]],
    step: Any,
    source: str,
    context: Optional[Any] = None,
) -> Tuple[str, List[str]]:
    """Return an all-or-nothing centrally authorized mechanical repair."""

    candidate_code, repair_names = deterministic_concept_audit_repair(
        script_text,
        error_messages,
        repair_reasons=repair_reasons,
        repair_findings=repair_findings,
    )
    if context is not None:
        binary_guarded = patch_observed_binary_primary_exposure_guard(
            candidate_code,
            context=context,
            repair_findings=repair_findings,
        )
        if binary_guarded != candidate_code:
            candidate_code = binary_guarded
            repair_names.append("observed_binary_primary_exposure_guard_v1")
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
    "REPAIR_AUTHORITY_BINDING_SCHEMA_VERSION",
    "PatchTransportUnavailable",
    "RepairAuthorityBinding",
    "RepairCoordinator",
    "RepairTransportResult",
    "StepRepairBudget",
    "authorized_deterministic_concept_repair",
    "resume_deterministic_repair_candidate",
]
