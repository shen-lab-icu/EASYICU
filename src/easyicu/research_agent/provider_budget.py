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
import hashlib
import json
import os
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, Iterator, Optional, Tuple, TypeVar


_T = TypeVar("_T")


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
    ) -> None:
        self.category = category
        self.limit = limit
        self.used = used
        self.step_id = step_id
        scope = f" for step {step_id!r}" if step_id else ""
        super().__init__(
            f"LLM provider-call budget exhausted{scope}: "
            f"category={category!r}, used={used}, limit={limit}."
        )


class ProviderCallBudgetReceiptError(ProviderCallBudgetError):
    """Raised when a durable provider-call receipt cannot be trusted."""


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


def load_provider_call_budget_receipt(
    path: Path,
    *,
    step_id: str,
) -> Tuple[int, Tuple[str, ...]]:
    """Load and verify a durable receipt, failing closed on any corruption."""

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
    if payload.get("schema_version") != 1:
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
        raise ProviderCallBudgetReceiptError(
            "Provider-call receipt history is invalid"
        )
    return limit, normalized


class StepProviderCallBudget:
    """Atomically account for real provider calls made for one analysis step."""

    def __init__(
        self,
        limit: int,
        *,
        step_id: Optional[str] = None,
        consumed_categories: Tuple[str, ...] = (),
        receipt_path: Optional[Path] = None,
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
        self._receipt_path = Path(receipt_path) if receipt_path is not None else None
        self._lock = Lock()

    def _persist_locked(self) -> None:
        if self._receipt_path is None:
            return
        payload: Dict[str, object] = {
            "schema_version": 1,
            "step_id": self._step_id,
            "limit": self._limit,
            "categories": list(self._categories),
        }
        payload["sha256"] = _receipt_digest(payload)
        path = self._receipt_path
        temp_path = path.with_name(
            f".{path.name}.{os.getpid()}.{id(self)}.tmp"
        )
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
            if used >= self._limit:
                raise ProviderCallBudgetExhausted(
                    category=normalized,
                    limit=self._limit,
                    used=used,
                    step_id=self._step_id,
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
    return state is None or state.budget.remaining > 0


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
