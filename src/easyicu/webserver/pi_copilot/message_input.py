"""Host-side normalization for user messages that name local data sources.

Filesystem paths are useful input to the EasyICU host, but they must never be
forwarded to the conversational model.  This owner recognizes only exact,
validated registry paths and replaces them with a path-free source label.  Any
other host path remains blocked by the existing sensitive-message boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .contracts import PiCopilotError
from .projections import reject_sensitive_message


_HOST_PATH_PATTERN = re.compile(
    r"(?P<quoted>[`\"'])(?P<quoted_path>/(?:Users|home|private|tmp|var|etc|opt|Volumes)/[^`\"']+)(?P=quoted)"
    r"|(?P<plain_path>/(?:Users|home|private|tmp|var|etc|opt|Volumes)/[^\s。；，,;!?！？]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PreparedUserMessage:
    """One path-free provider message plus an optional exact source receipt."""

    provider_message: str
    registered_source: Optional[Mapping[str, Any]] = None


def _normalized_path(value: str) -> str:
    try:
        return str(Path(value).expanduser().resolve(strict=False))
    except OSError:
        return str(Path(value).expanduser())


def prepare_user_message(
    message: str,
    *,
    registered_sources: Iterable[Mapping[str, Any]],
) -> PreparedUserMessage:
    """Keep exact registered data paths host-side and reject all other paths."""

    text = str(message or "").strip()
    matches = list(_HOST_PATH_PATTERN.finditer(text))
    if not matches:
        reject_sensitive_message(text)
        return PreparedUserMessage(provider_message=text)

    by_path = {
        _normalized_path(str(row.get("path") or "").strip()): row
        for row in registered_sources
        if isinstance(row, Mapping)
        and bool(row.get("ok"))
        and str(row.get("path") or "").strip()
    }
    selected: list[Mapping[str, Any]] = []
    replacements: list[tuple[int, int, str]] = []
    for match in matches:
        candidate = str(
            match.group("quoted_path") or match.group("plain_path") or ""
        ).strip()
        source = by_path.get(_normalized_path(candidate))
        if source is None:
            raise PiCopilotError(
                "pi_message_local_path_unregistered",
                (
                    "The local path is not an exact validated EasyICU data source. "
                    "Choose it through the local data-folder workflow first."
                ),
                status_code=400,
            )
        selected.append(source)
        label = str(source.get("label") or source.get("database") or "local data")
        replacements.append(
            (
                match.start(),
                match.end(),
                f"[EasyICU host-verified local data source: {label[:160]}]",
            )
        )

    source_ids = {
        str(row.get("id") or row.get("path") or "").strip() for row in selected
    }
    if len(source_ids) != 1:
        raise PiCopilotError(
            "pi_message_multiple_local_sources",
            "Use one exact local data source for a single research message.",
            status_code=400,
        )

    provider_text = text
    for start, end, replacement in reversed(replacements):
        provider_text = provider_text[:start] + replacement + provider_text[end:]
    reject_sensitive_message(provider_text)
    return PreparedUserMessage(
        provider_message=provider_text,
        registered_source=dict(selected[0]),
    )


__all__ = ["PreparedUserMessage", "prepare_user_message"]
