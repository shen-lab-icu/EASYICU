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
PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION = 8
_SUPPORTED_RECEIPT_SCHEMA_VERSIONS = {1, 2, 3, 4, 5, 6, 7, 8}
_LOGICAL_REPAIR_RECEIPT_SCHEMA_VERSIONS = {3, 4, 5, 6, 7, 8}
_LOGICAL_REPAIR_TRANSPORT_STATES = {
    "pending",
    "completed",
    "failed",
    "legacy_untracked",
}
_REPAIR_TRANSPORT_PROVIDER_SUFFIXES = ("patch", "full_rewrite")
_REPAIR_AUTHORITY_BINDING_SCHEMA_V2 = "easyicu.repair_authority_binding/2"
_INITIAL_GENERATION_TRANSPORT_STATES = {"pending", "completed", "failed"}
_MAX_DETERMINISTIC_RESERVED_CATEGORY_EXTENSIONS = 3


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
    initial_generations: Tuple[Dict[str, object], ...]
    required_reservation_token: Optional[str]
    reservation_bound_provider_history_len: Optional[int]
    completed_reservation_token: Optional[str]
    reservation_released: bool
    reserved_category_extensions: Tuple[Dict[str, object], ...]

    @property
    def initial_generation(self) -> Optional[Dict[str, object]]:
        """Return the current generation while preserving the append-only ledger."""

        return self.initial_generations[-1] if self.initial_generations else None


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


def _verified_initial_generation(
    raw_entry: object,
    *,
    categories: Tuple[str, ...],
) -> Optional[Dict[str, object]]:
    """Verify the one provider result that creates a step's first candidate."""

    if raw_entry is None:
        return None
    if not isinstance(raw_entry, dict) or set(raw_entry) != {
        "binding",
        "binding_sha256",
        "provider_history_len",
        "provider_history_sha256",
        "provider_transport_id",
        "transport",
    }:
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt initial-generation entry is invalid"
        )
    binding = raw_entry.get("binding")
    binding_sha256 = raw_entry.get("binding_sha256")
    reserved_history_len = raw_entry.get("provider_history_len")
    reserved_history_sha256 = raw_entry.get("provider_history_sha256")
    transport_id = raw_entry.get("provider_transport_id")
    if (
        not isinstance(binding, dict)
        or not _is_sha256_hex(binding_sha256)
        or binding_sha256 != _receipt_digest(dict(binding))
        or isinstance(reserved_history_len, bool)
        or not isinstance(reserved_history_len, int)
        or not 0 <= reserved_history_len <= len(categories)
        or reserved_history_sha256
        != _category_history_digest(categories[:reserved_history_len])
        or not isinstance(transport_id, str)
        or not transport_id.startswith("initial_generation:")
        or not transport_id.removeprefix("initial_generation:").isdigit()
    ):
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt initial-generation authority is inconsistent"
        )
    raw_transport = raw_entry.get("transport")
    if not isinstance(raw_transport, dict):
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt initial-generation transport is invalid"
        )
    state = raw_transport.get("state")
    if state not in _INITIAL_GENERATION_TRANSPORT_STATES:
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt initial-generation transport state is invalid"
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
        allowed_keys.update({"after_code_sha256", "after_code_size_bytes"})
    elif state == "failed":
        allowed_keys.add("error_type")
    if set(raw_transport) != allowed_keys:
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt initial-generation transport fields are inconsistent"
        )
    if state in {"completed", "failed"}:
        history_len = raw_transport.get("provider_history_len")
        history_sha256 = raw_transport.get("provider_history_sha256")
        provider_calls = raw_transport.get("provider_calls")
        if (
            isinstance(history_len, bool)
            or not isinstance(history_len, int)
            or not reserved_history_len <= history_len <= len(categories)
            or history_sha256 != _category_history_digest(categories[:history_len])
            or isinstance(provider_calls, bool)
            or not isinstance(provider_calls, int)
            or provider_calls
            != sum(
                category == "initial_generation"
                for category in categories[reserved_history_len:history_len]
            )
        ):
            raise ProviderCallBudgetReceiptError(
                "Provider-call receipt initial-generation history is inconsistent"
            )
        if state == "completed" and (
            provider_calls < 1
            or not _is_sha256_hex(raw_transport.get("after_code_sha256"))
            or isinstance(raw_transport.get("after_code_size_bytes"), bool)
            or not isinstance(raw_transport.get("after_code_size_bytes"), int)
            or int(raw_transport["after_code_size_bytes"]) < 0
        ):
            raise ProviderCallBudgetReceiptError(
                "Provider-call receipt completed initial generation is invalid"
            )
        if state == "failed":
            error_type = raw_transport.get("error_type")
            if (
                not isinstance(error_type, str)
                or not error_type.strip()
                or error_type != error_type.strip()
            ):
                raise ProviderCallBudgetReceiptError(
                    "Provider-call receipt failed initial generation is invalid"
                )
    return {
        **dict(raw_entry),
        "binding": dict(binding),
        "transport": dict(raw_transport),
    }


def _verified_initial_generations(
    raw_entries: object,
    *,
    categories: Tuple[str, ...],
) -> Tuple[Dict[str, object], ...]:
    """Verify an append-only sequence of explicit initial-generation epochs."""

    if not isinstance(raw_entries, list):
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt initial-generation ledger is invalid"
        )
    verified: list[Dict[str, object]] = []
    seen_transport_ids: set[str] = set()
    previous_transport_number = 0
    previous_terminal_history_len = 0
    for index, raw_entry in enumerate(raw_entries):
        entry = _verified_initial_generation(raw_entry, categories=categories)
        if entry is None:
            raise ProviderCallBudgetReceiptError(
                "Provider-call receipt initial-generation ledger contains an empty entry"
            )
        transport_id = str(entry["provider_transport_id"])
        transport_number = int(transport_id.removeprefix("initial_generation:"))
        transport = dict(entry["transport"])
        reserved_history_len = int(entry["provider_history_len"])
        if (
            transport_id in seen_transport_ids
            or transport_number <= 0
            or transport_number <= previous_transport_number
            or reserved_history_len < previous_terminal_history_len
            or (transport.get("state") == "pending" and index != len(raw_entries) - 1)
            or (
                index != len(raw_entries) - 1
                and transport.get("state") not in {"completed", "failed"}
            )
        ):
            raise ProviderCallBudgetReceiptError(
                "Provider-call receipt initial-generation epochs are inconsistent"
            )
        seen_transport_ids.add(transport_id)
        previous_transport_number = transport_number
        if transport.get("state") in {"completed", "failed"}:
            previous_terminal_history_len = int(transport["provider_history_len"])
        verified.append(entry)
    return tuple(verified)


def _verified_repair_transport(
    raw_transport: object,
    *,
    categories: Tuple[str, ...],
    reserved_history_len: int,
    provider_category: Optional[str],
    required: bool,
    receipt_schema_version: int,
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
        if receipt_schema_version >= 6:
            allowed_keys.add("result_persistence")
            if raw_transport.get("result_persistence") == "content_addressed":
                allowed_keys.add("after_code_size_bytes")
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
        result_persistence = raw_transport.get("result_persistence")
        if (
            not isinstance(mode, str)
            or not mode.strip()
            or mode != mode.strip()
            or not _is_sha256_hex(raw_transport.get("after_code_sha256"))
            or (
                receipt_schema_version >= 6
                and result_persistence
                not in {
                    "content_addressed",
                    "legacy_untracked",
                    "untracked",
                }
            )
            or (
                result_persistence == "content_addressed"
                and (
                    isinstance(raw_transport.get("after_code_size_bytes"), bool)
                    or not isinstance(raw_transport.get("after_code_size_bytes"), int)
                    or int(raw_transport["after_code_size_bytes"]) < 0
                )
            )
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
    receipt_schema_version: int = PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION,
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
            receipt_schema_version=receipt_schema_version,
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


def _verified_reserved_category_extensions(
    raw_extensions: object,
    *,
    base_limit: int,
    categories: Sequence[str],
    reserved_final_category: Optional[str],
) -> Tuple[Dict[str, object], ...]:
    """Verify bounded, audit-only calls beyond the ordinary step ceiling.

    Each extension is authorized only after the base budget is exhausted and
    funds exactly one call in the already-reserved final category.  Binding it
    to the category-history digest makes the grant append-only and prevents a
    resume from turning it into general repair or generation headroom.
    """

    if not isinstance(raw_extensions, list):
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt reserved-category extensions are invalid"
        )
    if len(raw_extensions) > _MAX_DETERMINISTIC_RESERVED_CATEGORY_EXTENSIONS:
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt has too many reserved-category extensions"
        )
    if raw_extensions and reserved_final_category is None:
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt extends a missing final reservation"
        )

    verified: list[Dict[str, object]] = []
    seen_tokens: set[str] = set()
    for index, raw_entry in enumerate(raw_extensions):
        if not isinstance(raw_entry, Mapping):
            raise ProviderCallBudgetReceiptError(
                "Provider-call receipt reserved-category extension must be an object"
            )
        token = raw_entry.get("token")
        history_len = raw_entry.get("provider_history_len")
        history_sha256 = raw_entry.get("provider_history_sha256")
        expected_history_len = base_limit + index
        if (
            not isinstance(token, str)
            or not token.strip()
            or token != token.strip()
            or token in seen_tokens
            or isinstance(history_len, bool)
            or not isinstance(history_len, int)
            or history_len != expected_history_len
            or history_len > len(categories)
            or not isinstance(history_sha256, str)
            or history_sha256
            != _category_history_digest(categories[:expected_history_len])
        ):
            raise ProviderCallBudgetReceiptError(
                "Provider-call receipt reserved-category extension is inconsistent"
            )
        if (
            history_len < len(categories)
            and categories[history_len] != reserved_final_category
        ):
            raise ProviderCallBudgetReceiptError(
                "Provider-call receipt spent a reserved-category extension elsewhere"
            )
        seen_tokens.add(token)
        verified.append(
            {
                "token": token,
                "provider_history_len": history_len,
                "provider_history_sha256": history_sha256,
            }
        )
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
    if any(not item for item in normalized):
        raise ProviderCallBudgetReceiptError("Provider-call receipt history is invalid")

    stored_reservation: Optional[str] = None
    if schema_version in {2, 3, 4, 5, 6, 7, 8}:
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

    if schema_version == 8:
        reserved_category_extensions = _verified_reserved_category_extensions(
            payload.get("reserved_category_extensions"),
            base_limit=limit,
            categories=normalized,
            reserved_final_category=stored_reservation,
        )
    else:
        if payload.get("reserved_category_extensions") is not None:
            raise ProviderCallBudgetReceiptError(
                "Legacy provider-call receipt unexpectedly declares "
                "reserved-category extensions"
            )
        reserved_category_extensions = ()
    effective_limit = limit + len(reserved_category_extensions)
    if len(normalized) > effective_limit or (
        len(normalized) > limit
        and (
            stored_reservation is None
            or any(
                category != stored_reservation for category in normalized[limit:]
            )
        )
    ):
        raise ProviderCallBudgetReceiptError("Provider-call receipt history is invalid")

    logical_repairs = (
        _verified_logical_repairs(
            payload.get("logical_repairs"),
            categories=normalized,
            require_transport=(schema_version in {5, 6, 7, 8}),
            receipt_schema_version=int(schema_version),
        )
        if schema_version in _LOGICAL_REPAIR_RECEIPT_SCHEMA_VERSIONS
        else ()
    )
    if schema_version == 5:
        migrated_repairs: list[Dict[str, object]] = []
        for entry in logical_repairs:
            migrated = dict(entry)
            transport = dict(migrated.get("transport") or {})
            if transport.get("state") == "completed":
                transport["result_persistence"] = "legacy_untracked"
                migrated["transport"] = transport
            migrated_repairs.append(migrated)
        logical_repairs = tuple(migrated_repairs)
    if schema_version == 6:
        if payload.get("initial_generations") is not None:
            raise ProviderCallBudgetReceiptError(
                "Schema-v6 provider-call receipt unexpectedly declares an "
                "initial-generation ledger"
            )
        legacy_initial_generation = _verified_initial_generation(
            payload.get("initial_generation"),
            categories=normalized,
        )
        initial_generations = (
            (legacy_initial_generation,)
            if legacy_initial_generation is not None
            else ()
        )
    elif schema_version in {7, 8}:
        if payload.get("initial_generation") is not None:
            raise ProviderCallBudgetReceiptError(
                "Current provider-call receipt unexpectedly declares the "
                "legacy initial-generation field"
            )
        initial_generations = _verified_initial_generations(
            payload.get("initial_generations"),
            categories=normalized,
        )
    else:
        initial_generations = ()
    if schema_version not in {6, 7, 8} and (
        payload.get("initial_generation") is not None
        or payload.get("initial_generations") is not None
    ):
        raise ProviderCallBudgetReceiptError(
            "Legacy provider-call receipt unexpectedly declares initial generation"
        )
    required_reservation_token: Optional[str] = None
    reservation_bound_provider_history_len: Optional[int] = None
    completed_reservation_token: Optional[str] = None
    reservation_released = False
    if schema_version in {4, 5, 6, 7, 8}:
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
            "Legacy provider-call receipt unexpectedly declares final reservation state"
        )
    return ProviderCallBudgetReceiptState(
        schema_version=int(schema_version),
        limit=limit,
        categories=normalized,
        reserved_final_category=stored_reservation,
        logical_repairs=logical_repairs,
        initial_generations=initial_generations,
        required_reservation_token=required_reservation_token,
        reservation_bound_provider_history_len=(reservation_bound_provider_history_len),
        completed_reservation_token=completed_reservation_token,
        reservation_released=reservation_released,
        reserved_category_extensions=reserved_category_extensions,
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
        initial_generation_entry: Optional[Mapping[str, object]] = None,
        initial_generation_entries: Sequence[Mapping[str, object]] = (),
        allow_terminal_initial_generation_restart: bool = False,
        receipt_path: Optional[Path] = None,
        reserved_final_category: Optional[str] = None,
        required_reservation_token: Optional[str] = None,
        reservation_bound_provider_history_len: Optional[int] = None,
        completed_reservation_token: Optional[str] = None,
        reservation_released: bool = False,
        reserved_category_extensions: Sequence[Mapping[str, object]] = (),
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
            transport = dict(entry.get("transport") or {})
            if (
                transport.get("state") == "completed"
                and "result_persistence" not in transport
            ):
                transport["result_persistence"] = "legacy_untracked"
                entry["transport"] = transport
        self._logical_repairs: list[Dict[str, object]] = [
            dict(entry)
            for entry in _verified_logical_repairs(
                restored_logical_repairs,
                categories=restored,
                require_transport=True,
                receipt_schema_version=PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION,
            )
        ]
        if initial_generation_entry is not None and initial_generation_entries:
            raise ValueError(
                "restore either one legacy initial generation or the generation "
                "ledger, not both"
            )
        restored_initial_generations = (
            [dict(entry) for entry in initial_generation_entries]
            if initial_generation_entries
            else (
                [dict(initial_generation_entry)]
                if initial_generation_entry is not None
                else []
            )
        )
        self._initial_generations: list[Dict[str, object]] = [
            dict(entry)
            for entry in _verified_initial_generations(
                restored_initial_generations,
                categories=restored,
            )
        ]
        if not isinstance(allow_terminal_initial_generation_restart, bool):
            raise TypeError(
                "terminal initial-generation restart authorization must be boolean"
            )
        self._terminal_initial_generation_restart_available = (
            allow_terminal_initial_generation_restart
        )
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
        self._reserved_category_extensions: list[Dict[str, object]] = list(
            _verified_reserved_category_extensions(
                [dict(entry) for entry in reserved_category_extensions],
                base_limit=self._limit,
                categories=self._categories,
                reserved_final_category=self._reserved_final_category,
            )
        )
        self._lock = Lock()

    def _effective_limit_locked(self) -> int:
        return self._limit + len(self._reserved_category_extensions)

    def _can_consume_locked(self, category: str) -> bool:
        used = len(self._categories)
        if used >= self._effective_limit_locked():
            return False
        if used >= self._limit:
            extension_index = used - self._limit
            extension = self._reserved_category_extensions[extension_index]
            return bool(
                category == self._reserved_final_category
                and not self._reservation_released
                and self._required_reservation_token == extension.get("token")
                and extension.get("provider_history_len") == used
            )
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
            "initial_generations": [
                {
                    **dict(entry),
                    "binding": dict(entry["binding"]),
                    "transport": dict(entry["transport"]),
                }
                for entry in self._initial_generations
            ],
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
            "reserved_category_extensions": [
                dict(entry) for entry in self._reserved_category_extensions
            ],
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
                    limit=self._effective_limit_locked(),
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

    def reserve_initial_generation(
        self,
        binding: Mapping[str, object],
    ) -> str:
        """Reserve the first candidate before paying for its provider call."""

        try:
            normalized_binding = json.loads(
                json.dumps(
                    dict(binding),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "initial-generation binding must be canonical JSON data"
            ) from exc
        if not isinstance(normalized_binding, dict):
            raise ValueError("initial-generation binding must be an object")
        binding_sha256 = _receipt_digest(normalized_binding)
        with self._lock:
            existing = (
                self._initial_generations[-1] if self._initial_generations else None
            )
            if existing is not None:
                transport = dict(existing.get("transport") or {})
                state = transport.get("state")
                if state == "pending":
                    if (
                        existing.get("binding_sha256") != binding_sha256
                        or existing.get("binding") != normalized_binding
                    ):
                        raise ProviderCallBudgetReceiptError(
                            "Initial-generation reservation belongs to different "
                            "authority"
                        )
                    history_len = int(existing["provider_history_len"])
                    paid_calls = sum(
                        category == "initial_generation"
                        for category in self._categories[history_len:]
                    )
                    if paid_calls:
                        raise ProviderCallBudgetReceiptError(
                            "Initial generation has paid provider calls but no durable "
                            "result; refusing to pay twice"
                        )
                    return str(existing["provider_transport_id"])
                if not self._terminal_initial_generation_restart_available:
                    if (
                        existing.get("binding_sha256") != binding_sha256
                        or existing.get("binding") != normalized_binding
                    ):
                        raise ProviderCallBudgetReceiptError(
                            "Initial-generation reservation belongs to different "
                            "authority"
                        )
                    raise ProviderCallBudgetReceiptError(
                        "Initial-generation transport is already terminal"
                    )

            previous_transport_number = max(
                (
                    int(
                        str(entry["provider_transport_id"]).removeprefix(
                            "initial_generation:"
                        )
                    )
                    for entry in self._initial_generations
                ),
                default=len(self._categories),
            )
            transport_id = f"initial_generation:{previous_transport_number + 1}"
            entry: Dict[str, object] = {
                "binding": normalized_binding,
                "binding_sha256": binding_sha256,
                "provider_history_len": len(self._categories),
                "provider_history_sha256": _category_history_digest(self._categories),
                "provider_transport_id": transport_id,
                "transport": {"state": "pending"},
            }
            restart_was_available = self._terminal_initial_generation_restart_available
            self._initial_generations.append(entry)
            self._terminal_initial_generation_restart_available = False
            try:
                self._persist_locked()
            except Exception:
                self._initial_generations.pop()
                self._terminal_initial_generation_restart_available = (
                    restart_was_available
                )
                raise
            return transport_id

    def _record_initial_generation_transport(
        self,
        *,
        provider_transport_id: str,
        transport: Mapping[str, object],
    ) -> None:
        with self._lock:
            entry = self._initial_generations[-1] if self._initial_generations else None
            if entry is None or entry.get("provider_transport_id") != str(
                provider_transport_id
            ):
                raise ProviderCallBudgetReceiptError(
                    "Initial-generation transport does not match its reservation"
                )
            current = entry.get("transport")
            if not isinstance(current, dict) or current.get("state") != "pending":
                raise ProviderCallBudgetReceiptError(
                    "Initial-generation transport is already terminal"
                )
            history_len = len(self._categories)
            reserved_history_len = int(entry["provider_history_len"])
            candidate = dict(transport)
            candidate.update(
                {
                    "provider_history_len": history_len,
                    "provider_history_sha256": _category_history_digest(
                        self._categories
                    ),
                    "provider_calls": sum(
                        category == "initial_generation"
                        for category in self._categories[
                            reserved_history_len:history_len
                        ]
                    ),
                }
            )
            replacement = {**dict(entry), "transport": candidate}
            verified = _verified_initial_generation(
                replacement,
                categories=tuple(self._categories),
            )
            if verified is None:
                raise AssertionError(
                    "verified initial-generation transport disappeared"
                )
            self._initial_generations[-1] = verified
            try:
                self._persist_locked()
            except Exception:
                self._initial_generations[-1] = entry
                raise

    def complete_initial_generation_transport(
        self,
        *,
        provider_transport_id: str,
        after_code_sha256: str,
        after_code_size_bytes: int,
    ) -> None:
        """Bind the paid initial provider result to persisted candidate bytes."""

        digest = str(after_code_sha256).strip().lower()
        if not _is_sha256_hex(digest):
            raise ValueError("initial-generation code digest must be SHA-256")
        if (
            isinstance(after_code_size_bytes, bool)
            or not isinstance(after_code_size_bytes, int)
            or after_code_size_bytes < 0
        ):
            raise ValueError("initial-generation code size must be non-negative")
        self._record_initial_generation_transport(
            provider_transport_id=provider_transport_id,
            transport={
                "state": "completed",
                "after_code_sha256": digest,
                "after_code_size_bytes": after_code_size_bytes,
            },
        )

    def fail_initial_generation_transport(
        self,
        *,
        provider_transport_id: str,
        error_type: str,
    ) -> None:
        """Persist a terminal initial-generation failure without prose."""

        normalized = str(error_type).strip()
        if not normalized:
            raise ValueError("initial-generation error type must be non-empty")
        self._record_initial_generation_transport(
            provider_transport_id=provider_transport_id,
            transport={"state": "failed", "error_type": normalized},
        )

    def authorize_failed_initial_generation_retry(
        self,
        *,
        error_type: str,
        max_generation_epochs: int,
    ) -> bool:
        """Authorize one bounded retry of a locally rejected provider result.

        The failed epoch must already be durable, must name the exact local
        validation error the caller is handling, and must leave provider-call
        capacity for the retry. Provider exceptions, interruptions, and
        paid-pending transports therefore cannot enter this path.
        """

        normalized = str(error_type).strip()
        if not normalized:
            raise ValueError("initial-generation retry error type must be non-empty")
        if (
            isinstance(max_generation_epochs, bool)
            or not isinstance(max_generation_epochs, int)
            or max_generation_epochs < 1
        ):
            raise ValueError("maximum initial-generation epochs must be positive")
        with self._lock:
            entry = self._initial_generations[-1] if self._initial_generations else None
            transport = dict(entry.get("transport") or {}) if entry else {}
            if (
                entry is None
                or len(self._initial_generations) >= max_generation_epochs
                or transport.get("state") != "failed"
                or transport.get("error_type") != normalized
                or not self._can_consume_locked("initial_generation")
            ):
                return False
            self._terminal_initial_generation_restart_available = True
            return True

    def initial_generation_resume_status(self) -> str:
        """Return the verified crash-resume state of initial generation."""

        with self._lock:
            entry = self._initial_generations[-1] if self._initial_generations else None
            if entry is None:
                return "absent"
            transport = dict(entry.get("transport") or {})
            state = str(transport.get("state") or "")
            if state != "pending":
                return state
            history_len = int(entry["provider_history_len"])
            paid_calls = sum(
                category == "initial_generation"
                for category in self._categories[history_len:]
            )
            return "paid_pending" if paid_calls else "unpaid_pending"

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

    def assert_logical_repair_prompt_binding(
        self,
        *,
        attempt_id: int,
        repair_ticket_sha256: str,
    ) -> None:
        """Join the actual repair prompt to the pending receipt before payment."""

        normalized_digest = str(repair_ticket_sha256 or "").strip().lower()
        if len(normalized_digest) != 64 or any(
            char not in "0123456789abcdef" for char in normalized_digest
        ):
            raise ValueError("repair prompt binding must be a SHA-256 hex digest")
        with self._lock:
            if (
                isinstance(attempt_id, bool)
                or not isinstance(attempt_id, int)
                or attempt_id != len(self._logical_repairs)
            ):
                raise ProviderCallBudgetReceiptError(
                    "Repair prompt binding does not target the current attempt"
                )
            entry = self._logical_repairs[attempt_id - 1]
            transport = entry.get("transport")
            if not isinstance(transport, dict) or transport.get("state") != "pending":
                raise ProviderCallBudgetReceiptError(
                    "Repair prompt binding is already terminal or untracked"
                )
            binding = entry.get("binding")
            if not isinstance(binding, dict):
                raise ProviderCallBudgetReceiptError(
                    "Repair prompt binding is absent from the provider receipt"
                )
            bound_digest = binding.get("repair_ticket_sha256")
            if bound_digest != normalized_digest:
                raise ProviderCallBudgetReceiptError(
                    "Actual repair prompt conflicts with its authority receipt"
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
                receipt_schema_version=(PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION),
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
        after_code_size_bytes: Optional[int] = None,
    ) -> None:
        """Bind one paid repair result to its exact output code digest."""

        normalized_mode = str(mode).strip()
        normalized_digest = str(after_code_sha256).strip().lower()
        if not normalized_mode:
            raise ValueError("logical repair transport mode must be non-empty")
        if not _is_sha256_hex(normalized_digest):
            raise ValueError("after-code digest must be a SHA-256 hex digest")
        if after_code_size_bytes is not None and (
            isinstance(after_code_size_bytes, bool)
            or not isinstance(after_code_size_bytes, int)
            or after_code_size_bytes < 0
        ):
            raise ValueError("after-code size must be non-negative")
        completed_transport: Dict[str, object] = {
            "state": "completed",
            "mode": normalized_mode,
            "after_code_sha256": normalized_digest,
            "result_persistence": (
                "content_addressed"
                if after_code_size_bytes is not None
                else "untracked"
            ),
        }
        if after_code_size_bytes is not None:
            completed_transport["after_code_size_bytes"] = after_code_size_bytes
        self._record_logical_repair_transport(
            attempt_id=attempt_id,
            transport=completed_transport,
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

    def authorize_deterministic_reserved_category_extension(
        self,
        category: str,
        *,
        token: str,
    ) -> bool:
        """Fund one digest-bound re-audit after a deterministic concept repair.

        This is not general budget ratcheting. It is available only after the
        ordinary ceiling is fully spent, only for the existing final reserved
        category, only for its currently bound token, and at most three times.
        The caller owns proof that an authorized deterministic repair produced
        that token; this ledger owns the one-call, crash-safe spend boundary.
        """

        normalized = str(category).strip()
        normalized_token = str(token).strip()
        with self._lock:
            if normalized != self._reserved_final_category:
                raise ValueError("category does not own this provider reservation")
            if (
                not normalized_token
                or normalized_token != self._required_reservation_token
            ):
                raise ValueError("extension token is not the current bound audit")
            used = len(self._categories)
            if used < self._limit:
                return False
            if any(
                extension.get("token") == normalized_token
                for extension in self._reserved_category_extensions
            ):
                # A provider attempt for this exact audit may already have
                # failed after spending its extension.  Retrying with the same
                # token would turn crash-safe accounting into an unbounded
                # provider retry loop, so the append-only ledger refuses it.
                return False
            if used != self._effective_limit_locked():
                raise ProviderCallBudgetReceiptError(
                    "A reserved-category extension is already awaiting its provider call"
                )
            if (
                len(self._reserved_category_extensions)
                >= _MAX_DETERMINISTIC_RESERVED_CATEGORY_EXTENSIONS
            ):
                raise ProviderCallBudgetExhausted(
                    category=normalized,
                    limit=self._effective_limit_locked(),
                    used=used,
                    step_id=self._step_id,
                )
            entry: Dict[str, object] = {
                "token": normalized_token,
                "provider_history_len": used,
                "provider_history_sha256": _category_history_digest(
                    self._categories
                ),
            }
            self._reserved_category_extensions.append(entry)
            try:
                self._persist_locked()
            except Exception:
                self._reserved_category_extensions.pop()
                raise
            return True

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
        with self._lock:
            return self._effective_limit_locked()

    @property
    def base_limit(self) -> int:
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
            return max(0, self._effective_limit_locked() - len(self._categories))

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return len(self._categories) >= self._effective_limit_locked()

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

    @property
    def initial_generation_entry(self) -> Optional[Dict[str, object]]:
        """Return a detached copy of the current initial-generation epoch."""

        with self._lock:
            if not self._initial_generations:
                return None
            entry = self._initial_generations[-1]
            return {
                **dict(entry),
                "binding": dict(entry["binding"]),
                "transport": dict(entry["transport"]),
            }

    @property
    def initial_generation_entries(self) -> Tuple[Dict[str, object], ...]:
        """Return detached copies of every append-only generation epoch."""

        with self._lock:
            return tuple(
                {
                    **dict(entry),
                    "binding": dict(entry["binding"]),
                    "transport": dict(entry["transport"]),
                }
                for entry in self._initial_generations
            )

    @property
    def terminal_initial_generation_restart_allowed(self) -> bool:
        """Return whether this explicit execution window may append one epoch."""

        with self._lock:
            return self._terminal_initial_generation_restart_available

    def snapshot(self) -> Dict[str, object]:
        """Return a JSON-serializable, internally consistent counter snapshot."""

        with self._lock:
            categories = tuple(self._categories)
            counts = dict(Counter(categories))
            effective_limit = self._effective_limit_locked()
            return {
                "step_id": self._step_id,
                "limit": effective_limit,
                "base_limit": self._limit,
                "used": len(categories),
                "remaining": max(0, effective_limit - len(categories)),
                "exhausted": len(categories) >= effective_limit,
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
                "initial_generation_epochs": len(self._initial_generations),
                "initial_generation_transport_state": (
                    str(
                        dict(self._initial_generations[-1].get("transport") or {}).get(
                            "state"
                        )
                        or ""
                    )
                    if self._initial_generations
                    else None
                ),
                "reserved_final_category": self._reserved_final_category,
                "reservation_bound": self._required_reservation_token is not None,
                "reservation_completed": self._completed_reservation_token is not None,
                "reservation_released": self._reservation_released,
                "reservation_bound_provider_history_len": (
                    self._reservation_bound_provider_history_len
                ),
                "reserved_category_extension_count": len(
                    self._reserved_category_extensions
                ),
            }


class _ActiveProviderCall:
    """The one logical call a scoped transport is currently serving.

    Retrying the SAME request used to consume another slot of this step's
    budget. That budget is funded in logical attempts -- ``config.py``
    certifies "1 initial generation + N code repairs + M LLM repairs + 1
    reserved concept audit" -- so an HTTP retry spent one of the repairs the
    certificate had just promised. With ``--transport-max-attempts 8`` a single
    flaky generation could take 8 of 9 before the script had run once, and the
    step then failed the way a scientifically broken step fails, which is the
    exact outcome the certificate beside it exists to prevent.

    Removing that charge weakens no bound. Retries are limited by
    ``providers/llm.py``'s own ``manual_attempts`` loop, and total spend by the
    run/batch hard stop, which :func:`consume_active_transport_attempt`
    reserves first and independently.

    A *handoff* is the other thing that used to share this counter and is not
    the same event: ``FallbackLLMClient`` moving the same question to a
    DIFFERENT supplier is a new logical call, and still costs a slot. The two
    were routed through one function, which is how one rule ended up governing
    both.
    """

    def __init__(
        self,
        *,
        budget: StepProviderCallBudget,
        category: str,
    ) -> None:
        self.budget = budget
        self.category = category
        self._suppliers = 0
        self._lock = Lock()

    def consume_provider_handoff(self) -> None:
        """Charge a second supplier answering the question already paid for.

        The outer complete call reserves the first supplier before entering the
        scope; every further one must reserve its own.
        """

        with self._lock:
            self._suppliers += 1
            already_reserved = self._suppliers == 1
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


def consume_active_transport_attempt() -> Optional[float]:
    """Charge one real HTTP attempt of the SAME request to the stop-loss.

    The per-step allowance is deliberately NOT charged here: it counts logical
    asks, and a retry is the same ask. See :class:`_ActiveProviderCall`. Use
    :func:`consume_active_provider_handoff` when a different supplier takes
    the question over.
    """

    # The run/batch stop-loss is independent from the per-step allowance.
    # Reserve it first so a paid transport cannot start unless the outer
    # ceilings were durably recorded.
    from .provider_hard_stop import consume_active_provider_hard_stop_attempt

    return consume_active_provider_hard_stop_attempt()


def consume_active_provider_handoff() -> Optional[float]:
    """Charge a different supplier taking over the same question.

    This is a new logical call, not a retry, so it costs the step a slot as
    well as the run/batch stop-loss.
    """

    from .provider_hard_stop import consume_active_provider_hard_stop_attempt

    hard_stop_remaining = consume_active_provider_hard_stop_attempt()
    state = _ACTIVE_PROVIDER_CALL.get()
    if state is not None:
        state.consume_provider_handoff()
    return hard_stop_remaining


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
