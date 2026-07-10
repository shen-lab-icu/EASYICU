"""Bounded pseudonymous entity navigation for Patient Review."""

from __future__ import annotations

import hashlib
import secrets
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_ENTITY_PAGE_SIZE = 12
MAX_ENTITY_PAGE_SIZE = 24


def entity_ref(path: Path, entity_id: str) -> str:
    """Return the stable one-way browser token for one local export entity."""
    token = f"{path.resolve()}::{entity_id}"
    return "ent_" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]


def entity_page_request(
    body: Dict[str, Any], total_entities: int
) -> Dict[str, Any]:
    """Normalize one bounded navigator request without accepting identifiers."""
    page_size = _bounded_int(
        body.get("entity_page_size"),
        DEFAULT_ENTITY_PAGE_SIZE,
        1,
        MAX_ENTITY_PAGE_SIZE,
    )
    page_count = max(1, (max(0, total_entities) + page_size - 1) // page_size)
    randomized = body.get("random_page") is True
    if randomized:
        page = secrets.randbelow(page_count) + 1
    else:
        page = _bounded_int(body.get("entity_page"), 1, 1, page_count)
    return {
        "page": page,
        "page_size": page_size,
        "page_count": page_count,
        "offset": (page - 1) * page_size,
        "randomized": randomized,
    }


def entity_navigation_payload(
    path: Path,
    entity_rows: Iterable[Tuple[int, str]],
    *,
    total_entities: int,
    page: int,
    page_size: int,
    page_count: int,
    selected_ref: str | None = None,
    selected_ordinal: int | None = None,
    randomized: bool = False,
) -> Dict[str, Any]:
    """Build a row-free navigator page from ordinal/entity-id pairs."""
    options: List[Dict[str, Any]] = []
    for ordinal, entity_id in entity_rows:
        if not entity_id:
            continue
        ref = entity_ref(path, entity_id)
        options.append(
            {
                "ref": ref,
                "label": f"Entity {ordinal}",
                "ordinal": ordinal,
                "selected": bool(selected_ref and ref == selected_ref),
            }
        )
    row_start = options[0]["ordinal"] if options else 0
    row_end = options[-1]["ordinal"] if options else 0
    return {
        "options": options,
        "page": page,
        "page_size": page_size,
        "page_count": page_count,
        "row_start": row_start,
        "row_end": row_end,
        "total_entities": max(0, int(total_entities)),
        "selected_ref": selected_ref,
        "selected_ordinal": selected_ordinal,
        "has_previous": page > 1,
        "has_next": page < page_count,
        "randomized": randomized,
        "identifier_policy": "pseudonymous_entity_token_plus_ordinal",
        "payload_scope": "bounded_pseudonymous_entity_navigation_page",
    }


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


__all__ = [
    "DEFAULT_ENTITY_PAGE_SIZE",
    "MAX_ENTITY_PAGE_SIZE",
    "entity_navigation_payload",
    "entity_page_request",
    "entity_ref",
]
