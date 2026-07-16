"""Thread-safe per-step budget for repair/audit LLM provider calls.

The budget is intentionally transport-agnostic.  Callers consume one unit
immediately before invoking ``llm.complete`` and attach a stable category so
the execution layer can report where the finite provider-call allowance went.
The first logical call and every transport or fallback attempt consume the same
finite allowance.  A persisted category history can be restored on resume so a
failed step cannot obtain a fresh budget simply by starting another process.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from threading import Lock
from typing import (
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

_T = TypeVar("_T")
_RESERVATION_UNSPECIFIED = object()
PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION = 5
_SUPPORTED_RECEIPT_SCHEMA_VERSIONS = {1, 2, 3, 4, 5}
_LOGICAL_REPAIR_RECEIPT_SCHEMA_VERSIONS = {3, 4, 5}
_LOGICAL_REPAIR_TRANSPORT_STATES = {
    "pending",
    "completed",
    "failed",
    "legacy_untracked",
}
_REPAIR_TRANSPORT_PROVIDER_SUFFIXES = ("patch", "full_rewrite")
_REPAIR_AUTHORITY_BINDING_SCHEMA_V2 = "easyicu.repair_authority_binding/2"


class ProviderCallBudgetError(RuntimeError):
    """Base class for fail-closed provider budget errors."""


class ProviderCallBudgetExhausted(ProviderCallBudgetError):
    """Raised before a provider call that would exceed a step budget."""

    def __init__(
        self,
        *,
        category: str,
        limit: int,
        used: int,
        step_id: Optional[str] = None,
        reserved_for: Optional[str] = None,
    ) -> None:
        self.category = category
        self.limit = limit
        self.used = used
        self.step_id = step_id
        self.reserved_for = reserved_for
        scope = f" for step {step_id!r}" if step_id else ""
        reservation = (
            f"; final slot reserved for {reserved_for!r}" if reserved_for else ""
        )
        super().__init__(
            f"LLM provider-call budget unavailable{scope}: "
            f"category={category!r}, used={used}, limit={limit}{reservation}."
        )


class ProviderCallBudgetReceiptError(ProviderCallBudgetError):
    """Raised when a durable provider-call receipt cannot be trusted."""


@dataclass(frozen=True)
class ProviderCallBudgetReceiptState:
    """Verified durable state for one step's single provider/repair ledger."""

    schema_version: int
    limit: int
    categories: Tuple[str, ...]
    reserved_final_category: Optional[str]
    logical_repairs: Tuple[Dict[str, object], ...]
    required_reservation_token: Optional[str]
    reservation_bound_provider_history_len: Optional[int]
    completed_reservation_token: Optional[str]
    reservation_released: bool


def provider_call_budget_receipt_path(
    run_dir: Path,
    *,
    step_id: str,
) -> Path:
    """Return a traversal-safe receipt path for one run-local step."""

    normalized = str(step_id).strip()
    if not normalized:
        raise ValueError("provider-call receipt step_id must be non-empty")
    suffix = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return Path(run_dir) / ".runtime" / "provider_call_budgets" / f"{suffix}.json"


def _receipt_digest(payload: Dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _category_history_digest(categories: Sequence[str]) -> str:
    return _receipt_digest({"categories": [str(item) for item in categories]})


def _is_sha256_hex(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _bound_repair_provider_category(binding: object) -> Optional[str]:
    """Return the receipt-bound provider category for a logical repair.

    Binding v1 receipts did not record this coordinate. They retain the
    conservative legacy rule that any later provider call makes a pending
    transport ambiguous. New bindings name the exact category so unrelated
    audit/analyzer calls cannot be misattributed to the repair attempt.
    """

    if not isinstance(binding, dict):
        return None
    binding_schema = binding.get("schema_version")
    has_provider_category = "provider_category" in binding
    if binding_schema != _REPAIR_AUTHORITY_BINDING_SCHEMA_V2:
        if has_provider_category:
            raise ProviderCallBudgetReceiptError(
                "Legacy repair authority binding unexpectedly declares a "
                "provider category"
            )
        return None
    if not has_provider_category:
        raise ProviderCallBudgetReceiptError(
            "Repair authority binding v2 is missing its provider category"
        )
    provider_category = binding.get("provider_category")
    if (
        not isinstance(provider_category, str)
        or not provider_category.strip()
        or provider_category != provider_category.strip()
    ):
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt repair provider category is invalid"
        )
    return provider_category


def _repair_owned_provider_calls(
    categories: Sequence[str],
    *,
    reserved_history_len: int,
    history_len: int,
    provider_category: Optional[str],
) -> int:
    """Count calls owned by one repair transport, not unrelated step calls."""

    suffix = categories[reserved_history_len:history_len]
    if provider_category is None:
        return len(suffix)
    owned_categories = {
        f"{provider_category}_{transport_suffix}"
        for transport_suffix in _REPAIR_TRANSPORT_PROVIDER_SUFFIXES
    }
    return sum(category in owned_categories for category in suffix)


def _verified_repair_transport(
    raw_transport: object,
    *,
    categories: Tuple[str, ...],
    reserved_history_len: int,
    provider_category: Optional[str],
    required: bool,
) -> Optional[Dict[str, object]]:
    if raw_transport is None and not required:
        return None
    if not isinstance(raw_transport, dict):
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt logical repair transport is invalid"
        )
    state = raw_transport.get("state")
    if state not in _LOGICAL_REPAIR_TRANSPORT_STATES:
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt logical repair transport state is invalid"
        )
    allowed_keys = {"state"}
    if state in {"completed", "failed"}:
        allowed_keys.update(
            {
                "provider_history_len",
                "provider_history_sha256",
                "provider_calls",
            }
        )
    if state == "completed":
        allowed_keys.update({"mode", "after_code_sha256"})
    elif state == "failed":
        allowed_keys.add("error_type")
    if set(raw_transport) != allowed_keys:
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt logical repair transport fields are inconsistent"
        )
    if state in {"pending", "legacy_untracked"}:
        return dict(raw_transport)

    history_len = raw_transport.get("provider_history_len")
    history_sha256 = raw_transport.get("provider_history_sha256")
    provider_calls = raw_transport.get("provider_calls")
    expected_provider_calls = _repair_owned_provider_calls(
        categories,
        reserved_history_len=reserved_history_len,
        history_len=(history_len if isinstance(history_len, int) else 0),
        provider_category=provider_category,
    )
    if (
        isinstance(history_len, bool)
        or not isinstance(history_len, int)
        or not reserved_history_len <= history_len <= len(categories)
        or history_sha256 != _category_history_digest(categories[:history_len])
        or isinstance(provider_calls, bool)
        or not isinstance(provider_calls, int)
        or provider_calls != expected_provider_calls
        or (state == "completed" and provider_calls < 1)
    ):
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt logical repair transport history is inconsistent"
        )
    if state == "completed":
        mode = raw_transport.get("mode")
        if (
            not isinstance(mode, str)
            or not mode.strip()
            or mode != mode.strip()
            or not _is_sha256_hex(raw_transport.get("after_code_sha256"))
        ):
            raise ProviderCallBudgetReceiptError(
                "Provider-call receipt completed repair transport is invalid"
            )
    else:
        error_type = raw_transport.get("error_type")
        if (
            not isinstance(error_type, str)
            or not error_type.strip()
            or error_type != error_type.strip()
        ):
            raise ProviderCallBudgetReceiptError(
                "Provider-call receipt failed repair transport is invalid"
            )
    return dict(raw_transport)


def _verified_logical_repairs(
    raw_entries: object,
    *,
    categories: Tuple[str, ...],
    require_transport: bool = False,
) -> Tuple[Dict[str, object], ...]:
    if not isinstance(raw_entries, list):
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt logical repair ledger is invalid"
        )
    verified: list[Dict[str, object]] = []
    for index, raw_entry in enumerate(raw_entries, start=1):
        if not isinstance(raw_entry, dict):
            raise ProviderCallBudgetReceiptError(
                "Provider-call receipt logical repair entry is invalid"
            )
        attempt_id = raw_entry.get("attempt_id")
        repair_class = raw_entry.get("repair_class")
        history_len = raw_entry.get("provider_history_len")
        history_sha256 = raw_entry.get("provider_history_sha256")
        binding = raw_entry.get("binding")
        binding_sha256 = raw_entry.get("binding_sha256")
        binding_pair_invalid = (binding is None) != (binding_sha256 is None)
        if binding is not None and not isinstance(binding, dict):
            binding_pair_invalid = True
        if isinstance(binding, dict) and isinstance(binding_sha256, str):
            binding_pair_invalid = (
                binding_pair_invalid or binding_sha256 != _receipt_digest(dict(binding))
            )
        if (
            isinstance(attempt_id, bool)
            or attempt_id != index
            or not isinstance(repair_class, str)
            or not repair_class.strip()
            or isinstance(history_len, bool)
            or not isinstance(history_len, int)
            or history_len < 0
            or history_len > len(categories)
            or not isinstance(history_sha256, str)
            or history_sha256 != _category_history_digest(categories[:history_len])
            or binding_pair_invalid
            or (
                binding_sha256 is not None
                and (
                    not isinstance(binding_sha256, str)
                    or len(binding_sha256) != 64
                    or any(char not in "0123456789abcdef" for char in binding_sha256)
                )
            )
        ):
            raise ProviderCallBudgetReceiptError(
                "Provider-call receipt logical repair history is inconsistent"
            )
        provider_category = _bound_repair_provider_category(binding)
        transport = _verified_repair_transport(
            raw_entry.get("transport"),
            categories=categories,
            reserved_history_len=history_len,
            provider_category=provider_category,
            required=require_transport,
        )
        if (
            transport is not None
            and transport.get("state") == "pending"
            and index != len(raw_entries)
        ):
            raise ProviderCallBudgetReceiptError(
                "Provider-call receipt pending logical repair must be the final "
                "ledger entry"
            )
        normalized_entry = dict(raw_entry)
        if transport is not None:
            normalized_entry["transport"] = transport
        verified.append(normalized_entry)
    return tuple(verified)


def load_provider_call_budget_state(
    path: Path,
    *,
    step_id: str,
    expected_reserved_final_category: object = _RESERVATION_UNSPECIFIED,
) -> ProviderCallBudgetReceiptState:
    """Load the complete single-ledger state, failing closed on corruption."""

    receipt_path = Path(path)
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProviderCallBudgetReceiptError(
            f"Provider-call receipt is unreadable: {receipt_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ProviderCallBudgetReceiptError("Provider-call receipt must be an object")
    digest = payload.pop("sha256", None)
    if not isinstance(digest, str) or digest != _receipt_digest(payload):
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt digest is missing or invalid"
        )
    schema_version = payload.get("schema_version")
    if schema_version not in _SUPPORTED_RECEIPT_SCHEMA_VERSIONS:
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt schema version is unsupported"
        )
    if str(payload.get("step_id") or "") != str(step_id):
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt belongs to a different step"
        )
    limit = payload.get("limit")
    categories = payload.get("categories")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 0
        or not isinstance(categories, list)
    ):
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt has invalid limit or categories"
        )
    normalized = tuple(str(item).strip() for item in categories)
    if any(not item for item in normalized) or len(normalized) > limit:
        raise ProviderCallBudgetReceiptError("Provider-call receipt history is invalid")

    stored_reservation: Optional[str] = None
    if schema_version in {2, 3, 4, PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION}:
        raw_reservation = payload.get("reserved_final_category")
        if raw_reservation is not None:
            if not isinstance(raw_reservation, str) or not raw_reservation.strip():
                raise ProviderCallBudgetReceiptError(
                    "Provider-call receipt has an invalid final reservation"
                )
            stored_reservation = raw_reservation.strip()
    if expected_reserved_final_category is not _RESERVATION_UNSPECIFIED:
        if expected_reserved_final_category is None:
            expected_reservation = None
        elif (
            isinstance(expected_reserved_final_category, str)
            and expected_reserved_final_category.strip()
        ):
            expected_reservation = expected_reserved_final_category.strip()
        else:
            raise ValueError("expected final reservation must be non-empty or None")
        if schema_version == 1:
            if expected_reservation is None:
                raise ProviderCallBudgetReceiptError(
                    "Legacy provider-call receipt does not bind final-audit policy"
                )
        elif stored_reservation != expected_reservation:
            raise ProviderCallBudgetReceiptError(
                "Provider-call receipt final-audit policy changed on resume"
            )

    logical_repairs = (
        _verified_logical_repairs(
            payload.get("logical_repairs"),
            categories=normalized,
            require_transport=(
                schema_version == PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION
            ),
        )
        if schema_version in _LOGICAL_REPAIR_RECEIPT_SCHEMA_VERSIONS
        else ()
    )
    required_reservation_token: Optional[str] = None
    reservation_bound_provider_history_len: Optional[int] = None
    completed_reservation_token: Optional[str] = None
    reservation_released = False
    if schema_version in {4, PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION}:
        raw_state = payload.get("final_reservation_state")
        if not isinstance(raw_state, dict):
            raise ProviderCallBudgetReceiptError(
                "Provider-call receipt final reservation state is invalid"
            )
        required_reservation_token = raw_state.get("required_token")
        reservation_bound_provider_history_len = raw_state.get(
            "bound_provider_history_len"
        )
        bound_provider_history_sha256 = raw_state.get("bound_provider_history_sha256")
        completed_reservation_token = raw_state.get("completed_token")
        reservation_released = raw_state.get("released")
        if required_reservation_token is None:
            if (
                reservation_bound_provider_history_len is not None
                or bound_provider_history_sha256 is not None
                or completed_reservation_token is not None
                or reservation_released is not False
            ):
                raise ProviderCallBudgetReceiptError(
                    "Provider-call receipt has an inconsistent empty final "
                    "reservation state"
                )
        elif (
            not isinstance(required_reservation_token, str)
            or not required_reservation_token.strip()
            or required_reservation_token != required_reservation_token.strip()
            or isinstance(reservation_bound_provider_history_len, bool)
            or not isinstance(reservation_bound_provider_history_len, int)
            or not 0 <= reservation_bound_provider_history_len <= len(normalized)
            or bound_provider_history_sha256
            != _category_history_digest(
                normalized[:reservation_bound_provider_history_len]
            )
            or completed_reservation_token not in {None, required_reservation_token}
            or not isinstance(reservation_released, bool)
            or (reservation_released and completed_reservation_token is None)
        ):
            raise ProviderCallBudgetReceiptError(
                "Provider-call receipt final reservation state is inconsistent"
            )
        if stored_reservation is None and required_reservation_token is not None:
            raise ProviderCallBudgetReceiptError(
                "Provider-call receipt binds an audit without a final reservation"
            )
    elif payload.get("final_reservation_state") is not None:
        raise ProviderCallBudgetReceiptError(
            "Legacy provider-call receipt unexpectedly declares final "
            "reservation state"
        )
    return ProviderCallBudgetReceiptState(
        schema_version=int(schema_version),
        limit=limit,
        categories=normalized,
        reserved_final_category=stored_reservation,
        logical_repairs=logical_repairs,
        required_reservation_token=required_reservation_token,
        reservation_bound_provider_history_len=(reservation_bound_provider_history_len),
        completed_reservation_token=completed_reservation_token,
        reservation_released=reservation_released,
    )


def load_provider_call_budget_receipt(
    path: Path,
    *,
    step_id: str,
    expected_reserved_final_category: object = _RESERVATION_UNSPECIFIED,
) -> Tuple[int, Tuple[str, ...]]:
    """Load and verify a durable receipt, failing closed on any corruption."""

    state = load_provider_call_budget_state(
        path,
        step_id=step_id,
        expected_reserved_final_category=expected_reserved_final_category,
    )
    return state.limit, state.categories


class StepProviderCallBudget:
    """Atomically account for real provider calls made for one analysis step."""

    def __init__(
        self,
        limit: int,
        *,
        step_id: Optional[str] = None,
        consumed_categories: Tuple[str, ...] = (),
        logical_repair_entries: Sequence[Mapping[str, object]] = (),
        receipt_path: Optional[Path] = None,
        reserved_final_category: Optional[str] = None,
        required_reservation_token: Optional[str] = None,
        reservation_bound_provider_history_len: Optional[int] = None,
        completed_reservation_token: Optional[str] = None,
        reservation_released: bool = False,
    ) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("provider-call budget limit must be an integer")
        if limit < 0:
            raise ValueError("provider-call budget limit must be non-negative")
        self._limit = limit
        self._step_id = str(step_id).strip() if step_id else None
        restored = tuple(str(item).strip() for item in consumed_categories)
        if any(not item for item in restored):
            raise ValueError("restored provider-call categories must be non-empty")
        self._categories: list[str] = list(restored)
        restored_logical_repairs = [dict(entry) for entry in logical_repair_entries]
        for entry in restored_logical_repairs:
            entry.setdefault("transport", {"state": "legacy_untracked"})
        self._logical_repairs: list[Dict[str, object]] = [
            dict(entry)
            for entry in _verified_logical_repairs(
                restored_logical_repairs,
                categories=restored,
                require_transport=True,
            )
        ]
        self._receipt_path = Path(receipt_path) if receipt_path is not None else None
        self._reserved_final_category = (
            str(reserved_final_category).strip() if reserved_final_category else None
        )
        # A historical call in the same category is not proof that the current
        # code + authority binding was audited.  The reservation is released
        # only after the caller binds and completes one exact final token.
        required_token = (
            str(required_reservation_token).strip()
            if required_reservation_token is not None
            else None
        )
        completed_token = (
            str(completed_reservation_token).strip()
            if completed_reservation_token is not None
            else None
        )
        if required_token is None:
            if (
                reservation_bound_provider_history_len is not None
                or completed_token is not None
                or reservation_released is not False
            ):
                raise ValueError("restored final reservation state is inconsistent")
        elif (
            self._reserved_final_category is None
            or not required_token
            or isinstance(reservation_bound_provider_history_len, bool)
            or not isinstance(reservation_bound_provider_history_len, int)
            or not 0 <= reservation_bound_provider_history_len <= len(self._categories)
            or completed_token not in {None, required_token}
            or not isinstance(reservation_released, bool)
            or (reservation_released and completed_token is None)
        ):
            raise ValueError("restored final reservation state is inconsistent")
        self._required_reservation_token = required_token
        self._reservation_bound_provider_history_len = (
            reservation_bound_provider_history_len
        )
        self._completed_reservation_token = completed_token
        self._reservation_released = reservation_released
        self._lock = Lock()

    def _can_consume_locked(self, category: str) -> bool:
        used = len(self._categories)
        if used >= self._limit:
            return False
        if (
            self._reserved_final_category
            and not self._reservation_released
            and category != self._reserved_final_category
            and self._limit - used <= 1
        ):
            return False
        return True

    def _persist_locked(self) -> None:
        if self._receipt_path is None:
            return
        payload: Dict[str, object] = {
            "schema_version": PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION,
            "step_id": self._step_id,
            "limit": self._limit,
            "categories": list(self._categories),
            "reserved_final_category": self._reserved_final_category,
            "logical_repairs": [dict(entry) for entry in self._logical_repairs],
            "final_reservation_state": {
                "required_token": self._required_reservation_token,
                "bound_provider_history_len": (
                    self._reservation_bound_provider_history_len
                ),
                "bound_provider_history_sha256": (
                    _category_history_digest(
                        self._categories[: self._reservation_bound_provider_history_len]
                    )
                    if self._reservation_bound_provider_history_len is not None
                    else None
                ),
                "completed_token": self._completed_reservation_token,
                "released": self._reservation_released,
            },
        }
        payload["sha256"] = _receipt_digest(payload)
        path = self._receipt_path
        temp_path = path.with_name(f".{path.name}.{os.getpid()}.{id(self)}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temp_path, path)
        except Exception as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise ProviderCallBudgetReceiptError(
                f"Could not persist provider-call receipt: {path}"
            ) from exc

    def consume(self, category: str) -> int:
        """Reserve one call and return its one-based sequence number.

        Reservation happens before the provider call.  A provider exception
        therefore still consumes the unit, matching the real attempted-call
        cost.  An exhausted reservation does not mutate the budget.
        """

        normalized = str(category).strip()
        if not normalized:
            raise ValueError("provider-call category must be non-empty")
        with self._lock:
            used = len(self._categories)
            if not self._can_consume_locked(normalized):
                reserved_for = (
                    self._reserved_final_category
                    if (
                        self._reserved_final_category
                        and not self._reservation_released
                        and normalized != self._reserved_final_category
                    )
                    else None
                )
                raise ProviderCallBudgetExhausted(
                    category=normalized,
                    limit=self._limit,
                    used=used,
                    step_id=self._step_id,
                    reserved_for=reserved_for,
                )
            self._categories.append(normalized)
            try:
                # Write before returning the reservation. A crash after this
                # point cannot make the paid attempt disappear on resume.
                self._persist_locked()
            except Exception:
                self._categories.pop()
                raise
            return used + 1

    def can_consume(self, category: str) -> bool:
        """Return whether ``category`` may reserve the next provider call."""

        normalized = str(category).strip()
        if not normalized:
            return False
        with self._lock:
            return self._can_consume_locked(normalized)

    def migrate_logical_repairs(self, repair_classes: Sequence[str]) -> None:
        """Seed the v3 ledger once from a verified legacy step snapshot.

        Legacy v1/v2 receipts counted provider calls but not logical repairs.
        The caller has already verified the monotonic step-record history; this
        one-way migration copies that exact history into the same durable
        receipt.  Existing v3 entries are never replaced or shortened.
        """

        normalized = tuple(str(item).strip() for item in repair_classes)
        if any(not item for item in normalized):
            raise ProviderCallBudgetReceiptError(
                "Legacy logical repair history contains an empty class"
            )
        if not normalized:
            return
        with self._lock:
            existing = tuple(
                str(entry["repair_class"]) for entry in self._logical_repairs
            )
            if existing:
                if (
                    len(normalized) > len(existing)
                    or existing[: len(normalized)] != normalized
                ):
                    raise ProviderCallBudgetReceiptError(
                        "Durable logical repair ledger conflicts with step history"
                    )
                return
            history_len = len(self._categories)
            history_sha256 = _category_history_digest(self._categories)
            migrated = [
                {
                    "attempt_id": index,
                    "repair_class": repair_class,
                    "provider_history_len": history_len,
                    "provider_history_sha256": history_sha256,
                    "migrated_from_step_snapshot": True,
                    "transport": {"state": "legacy_untracked"},
                }
                for index, repair_class in enumerate(normalized, start=1)
            ]
            self._logical_repairs.extend(migrated)
            try:
                self._persist_locked()
            except Exception:
                del self._logical_repairs[-len(migrated) :]
                raise

    def reserve_logical_repair(
        self,
        repair_class: str,
        *,
        max_repairs: int,
        binding: Optional[Mapping[str, object]] = None,
        binding_sha256: Optional[str] = None,
    ) -> Optional[int]:
        """Durably reserve one logical repair before any provider call.

        The logical attempt and provider-call categories now share one atomic
        receipt.  A crash after this method cannot grant a fresh logical
        attempt on resume, even if no provider call was made yet.
        """

        normalized = str(repair_class).strip()
        if not normalized:
            raise ValueError("logical repair class must be non-empty")
        if isinstance(max_repairs, bool) or not isinstance(max_repairs, int):
            raise TypeError("logical repair limit must be an integer")
        if max_repairs < 0:
            raise ValueError("logical repair limit must be non-negative")
        normalized_binding_payload: Optional[Dict[str, object]] = None
        normalized_binding: Optional[str] = None
        if binding is not None:
            try:
                normalized_binding_payload = json.loads(
                    json.dumps(
                        dict(binding),
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "logical repair binding must be canonical JSON data"
                ) from exc
            if not isinstance(normalized_binding_payload, dict):
                raise ValueError("logical repair binding must be an object")
            _bound_repair_provider_category(normalized_binding_payload)
            computed_binding = _receipt_digest(normalized_binding_payload)
            if binding_sha256 is None:
                binding_sha256 = computed_binding
            elif str(binding_sha256).strip().lower() != computed_binding:
                raise ValueError(
                    "logical repair binding digest does not match its payload"
                )
        if binding_sha256 is not None:
            normalized_binding = str(binding_sha256).strip().lower()
            if len(normalized_binding) != 64 or any(
                char not in "0123456789abcdef" for char in normalized_binding
            ):
                raise ValueError("logical repair binding must be a SHA-256 hex digest")
        with self._lock:
            if self._logical_repairs:
                pending_entry = self._logical_repairs[-1]
                pending_transport = pending_entry.get("transport")
                if (
                    isinstance(pending_transport, dict)
                    and pending_transport.get("state") == "pending"
                ):
                    pending_history_len = int(pending_entry["provider_history_len"])
                    pending_provider_category = _bound_repair_provider_category(
                        pending_entry.get("binding")
                    )
                    if _repair_owned_provider_calls(
                        self._categories,
                        reserved_history_len=pending_history_len,
                        history_len=len(self._categories),
                        provider_category=pending_provider_category,
                    ):
                        raise ProviderCallBudgetReceiptError(
                            "A logical repair has paid provider calls but no durable "
                            "transport result; refusing to pay for a duplicate repair"
                        )
                    if (
                        str(pending_entry["repair_class"]) != normalized
                        or pending_entry.get("binding_sha256") != normalized_binding
                    ):
                        raise ProviderCallBudgetReceiptError(
                            "An unpaid logical repair reservation belongs to different "
                            "authority; refusing to replace it on resume"
                        )
                    return int(pending_entry["attempt_id"])
            if len(self._logical_repairs) >= max_repairs:
                return None
            if not self._can_consume_locked("llm_repair_budget_probe"):
                return None
            entry: Dict[str, object] = {
                "attempt_id": len(self._logical_repairs) + 1,
                "repair_class": normalized,
                "provider_history_len": len(self._categories),
                "provider_history_sha256": _category_history_digest(self._categories),
                "transport": {"state": "pending"},
            }
            if normalized_binding is not None:
                if normalized_binding_payload is None:
                    raise ValueError(
                        "logical repair binding digest requires its canonical payload"
                    )
                entry["binding"] = normalized_binding_payload
                entry["binding_sha256"] = normalized_binding
            self._logical_repairs.append(entry)
            try:
                self._persist_locked()
            except Exception:
                self._logical_repairs.pop()
                raise
            return int(entry["attempt_id"])

    def next_logical_repair_attempt_id(self) -> int:
        """Return the new or safely resumable logical-attempt identifier.

        A pending reservation can be reused only while no attempt-owned provider
        call is visible. Once that repair has paid calls, the missing result is
        unknowable and resume must fail closed instead of paying twice.
        """

        with self._lock:
            if self._logical_repairs:
                entry = self._logical_repairs[-1]
                transport = entry.get("transport")
                if isinstance(transport, dict) and transport.get("state") == "pending":
                    history_len = int(entry["provider_history_len"])
                    provider_category = _bound_repair_provider_category(
                        entry.get("binding")
                    )
                    if _repair_owned_provider_calls(
                        self._categories,
                        reserved_history_len=history_len,
                        history_len=len(self._categories),
                        provider_category=provider_category,
                    ):
                        raise ProviderCallBudgetReceiptError(
                            "A logical repair has paid provider calls but no durable "
                            "transport result; refusing to resume ambiguously"
                        )
                    return int(entry["attempt_id"])
            return len(self._logical_repairs) + 1

    def assert_logical_repair_provider_category(
        self,
        *,
        attempt_id: int,
        provider_category: str,
    ) -> None:
        """Verify a coordinator will spend against the reserved repair route.

        The check runs before the coordinator reserves a provider call. Legacy
        bindings have no route coordinate and retain their conservative
        history-based accounting; v2 bindings must match exactly.
        """

        normalized_category = str(provider_category)
        if (
            not normalized_category.strip()
            or normalized_category != normalized_category.strip()
        ):
            raise ValueError("logical repair provider category must be non-empty")
        with self._lock:
            if (
                isinstance(attempt_id, bool)
                or not isinstance(attempt_id, int)
                or attempt_id != len(self._logical_repairs)
            ):
                raise ProviderCallBudgetReceiptError(
                    "Logical repair provider route does not target the current attempt"
                )
            entry = self._logical_repairs[attempt_id - 1]
            transport = entry.get("transport")
            if not isinstance(transport, dict) or transport.get("state") != "pending":
                raise ProviderCallBudgetReceiptError(
                    "Logical repair provider route is already terminal or untracked"
                )
            bound_category = _bound_repair_provider_category(entry.get("binding"))
            if bound_category is not None and bound_category != normalized_category:
                raise ProviderCallBudgetReceiptError(
                    "Logical repair provider category conflicts with its authority "
                    "binding"
                )

    def _record_logical_repair_transport(
        self,
        *,
        attempt_id: int,
        transport: Mapping[str, object],
    ) -> None:
        with self._lock:
            if (
                isinstance(attempt_id, bool)
                or not isinstance(attempt_id, int)
                or attempt_id != len(self._logical_repairs)
            ):
                raise ProviderCallBudgetReceiptError(
                    "Logical repair transport does not target the current attempt"
                )
            entry = self._logical_repairs[attempt_id - 1]
            current = entry.get("transport")
            if not isinstance(current, dict) or current.get("state") != "pending":
                raise ProviderCallBudgetReceiptError(
                    "Logical repair transport is already terminal or untracked"
                )
            history_len = len(self._categories)
            reserved_history_len = int(entry["provider_history_len"])
            provider_category = _bound_repair_provider_category(entry.get("binding"))
            candidate = dict(transport)
            candidate.update(
                {
                    "provider_history_len": history_len,
                    "provider_history_sha256": _category_history_digest(
                        self._categories
                    ),
                    "provider_calls": _repair_owned_provider_calls(
                        self._categories,
                        reserved_history_len=reserved_history_len,
                        history_len=history_len,
                        provider_category=provider_category,
                    ),
                }
            )
            verified = _verified_repair_transport(
                candidate,
                categories=tuple(self._categories),
                reserved_history_len=reserved_history_len,
                provider_category=provider_category,
                required=True,
            )
            if verified is None:
                raise AssertionError("verified logical repair transport disappeared")
            entry["transport"] = verified
            try:
                self._persist_locked()
            except Exception:
                entry["transport"] = current
                raise

    def complete_logical_repair_transport(
        self,
        *,
        attempt_id: int,
        mode: str,
        after_code_sha256: str,
    ) -> None:
        """Bind one paid repair result to its exact output code digest."""

        normalized_mode = str(mode).strip()
        normalized_digest = str(after_code_sha256).strip().lower()
        if not normalized_mode:
            raise ValueError("logical repair transport mode must be non-empty")
        if not _is_sha256_hex(normalized_digest):
            raise ValueError("after-code digest must be a SHA-256 hex digest")
        self._record_logical_repair_transport(
            attempt_id=attempt_id,
            transport={
                "state": "completed",
                "mode": normalized_mode,
                "after_code_sha256": normalized_digest,
            },
        )

    def fail_logical_repair_transport(
        self,
        *,
        attempt_id: int,
        error_type: str,
    ) -> None:
        """Persist a terminal transport failure without storing error prose."""

        normalized_error_type = str(error_type).strip()
        if not normalized_error_type:
            raise ValueError("logical repair error type must be non-empty")
        self._record_logical_repair_transport(
            attempt_id=attempt_id,
            transport={"state": "failed", "error_type": normalized_error_type},
        )

    def bind_reserved_category(self, category: str, *, token: str) -> None:
        """Bind the final reservation to one exact code/authority audit token."""

        normalized = str(category).strip()
        normalized_token = str(token).strip()
        if not normalized_token:
            raise ValueError("reservation token must be non-empty")
        with self._lock:
            if normalized != self._reserved_final_category:
                raise ValueError("category does not own this provider reservation")
            if self._required_reservation_token != normalized_token:
                previous = (
                    self._required_reservation_token,
                    self._reservation_bound_provider_history_len,
                    self._completed_reservation_token,
                    self._reservation_released,
                )
                self._required_reservation_token = normalized_token
                self._reservation_bound_provider_history_len = len(self._categories)
                self._completed_reservation_token = None
                self._reservation_released = False
                try:
                    self._persist_locked()
                except Exception:
                    (
                        self._required_reservation_token,
                        self._reservation_bound_provider_history_len,
                        self._completed_reservation_token,
                        self._reservation_released,
                    ) = previous
                    raise

    def complete_reserved_category(self, category: str, *, token: str) -> None:
        """Record that the exact bound audit token passed its final gate."""

        normalized = str(category).strip()
        normalized_token = str(token).strip()
        with self._lock:
            if normalized != self._reserved_final_category:
                raise ValueError("category does not own this provider reservation")
            if (
                not normalized_token
                or normalized_token != self._required_reservation_token
            ):
                raise ValueError("reservation token is not the current bound audit")
            previous = self._completed_reservation_token
            self._completed_reservation_token = normalized_token
            try:
                self._persist_locked()
            except Exception:
                self._completed_reservation_token = previous
                raise

    def release_reserved_category(self, category: str, *, token: str) -> None:
        """Release the slot only for the exact audit token that passed."""

        normalized = str(category).strip()
        normalized_token = str(token).strip()
        with self._lock:
            if normalized != self._reserved_final_category:
                raise ValueError("category does not own this provider reservation")
            if (
                not normalized_token
                or normalized_token != self._required_reservation_token
                or normalized_token != self._completed_reservation_token
            ):
                raise ValueError("reservation token has not completed the final audit")
            previous = self._reservation_released
            self._reservation_released = True
            try:
                self._persist_locked()
            except Exception:
                self._reservation_released = previous
                raise

    def reservation_status(self, category: str, *, token: str) -> str:
        """Return the durable phase for one exact final-audit authority token."""

        normalized = str(category).strip()
        normalized_token = str(token).strip()
        with self._lock:
            if (
                normalized != self._reserved_final_category
                or not normalized_token
                or normalized_token != self._required_reservation_token
            ):
                return "unbound"
            if self._reservation_released:
                return "released"
            if self._completed_reservation_token == normalized_token:
                return "completed"
            bound_len = self._reservation_bound_provider_history_len
            if bound_len is not None and len(self._categories) > bound_len:
                return "attempted_incomplete"
            return "bound_unpaid"

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def step_id(self) -> Optional[str]:
        return self._step_id

    @property
    def used(self) -> int:
        with self._lock:
            return len(self._categories)

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self._limit - len(self._categories))

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return len(self._categories) >= self._limit

    @property
    def categories(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(self._categories)

    @property
    def logical_repair_classes(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(str(entry["repair_class"]) for entry in self._logical_repairs)

    @property
    def logical_repair_transport_states(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(
                str(dict(entry.get("transport") or {}).get("state") or "")
                for entry in self._logical_repairs
            )

    def snapshot(self) -> Dict[str, object]:
        """Return a JSON-serializable, internally consistent counter snapshot."""

        with self._lock:
            categories = tuple(self._categories)
            counts = dict(Counter(categories))
            return {
                "step_id": self._step_id,
                "limit": self._limit,
                "used": len(categories),
                "remaining": max(0, self._limit - len(categories)),
                "exhausted": len(categories) >= self._limit,
                "categories": list(categories),
                "category_counts": counts,
                "logical_repair_attempts": len(self._logical_repairs),
                "logical_repair_classes": [
                    str(entry["repair_class"]) for entry in self._logical_repairs
                ],
                "logical_repair_binding_sha256": [
                    entry.get("binding_sha256") for entry in self._logical_repairs
                ],
                "logical_repair_transport_states": [
                    str(dict(entry.get("transport") or {}).get("state") or "")
                    for entry in self._logical_repairs
                ],
                "reserved_final_category": self._reserved_final_category,
                "reservation_bound": self._required_reservation_token is not None,
                "reservation_completed": self._completed_reservation_token is not None,
                "reservation_released": self._reservation_released,
                "reservation_bound_provider_history_len": (
                    self._reservation_bound_provider_history_len
                ),
            }


class _ActiveProviderCall:
    def __init__(
        self,
        *,
        budget: StepProviderCallBudget,
        category: str,
    ) -> None:
        self.budget = budget
        self.category = category
        self._transport_attempts = 0
        self._lock = Lock()

    def consume_transport_attempt(self) -> None:
        # The outer complete call reserves the first attempt before entering
        # the scope. Every subsequent transport retry must reserve another.
        with self._lock:
            self._transport_attempts += 1
            already_reserved = self._transport_attempts == 1
        if not already_reserved:
            self.budget.consume(self.category)


_ACTIVE_PROVIDER_CALL: ContextVar[Optional[_ActiveProviderCall]] = ContextVar(
    "easyicu_active_provider_call",
    default=None,
)


@contextmanager
def provider_call_scope(
    budget: StepProviderCallBudget,
    category: str,
) -> Iterator[None]:
    """Expose one pre-reserved logical call to transport retry accounting."""

    state = _ActiveProviderCall(budget=budget, category=category)
    token = _ACTIVE_PROVIDER_CALL.set(state)
    try:
        yield
    finally:
        _ACTIVE_PROVIDER_CALL.reset(token)


def consume_active_transport_attempt() -> None:
    """Charge retries for a budget-scoped transport call, if one is active."""

    state = _ACTIVE_PROVIDER_CALL.get()
    if state is not None:
        state.consume_transport_attempt()


def active_provider_retry_available() -> bool:
    """Return whether a scoped transport call can afford another attempt."""

    state = _ACTIVE_PROVIDER_CALL.get()
    return state is None or state.budget.can_consume(state.category)


def complete_with_provider_budget(
    *,
    budget: Optional[StepProviderCallBudget],
    category: str,
    call: Callable[[], _T],
) -> _T:
    """Reserve and execute one LLM call with retry-aware accounting."""

    if budget is None:
        return call()
    budget.consume(category)
    with provider_call_scope(budget, category):
        return call()
