"""Opaque, server-issued coordinates for local run directories.

The conversation's run files never carry a run's absolute directory: the run
history issues a ``run_ref`` for each directory it lists, and the review,
artifact, download and sign-off routes resolve it back here.  A ref is an
HMAC of the resolved directory under a per-process secret, so it reveals
nothing about the path, names only a directory this process listed, and
lapses with the process -- the page lists the runs again after a restart.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

_SECRET = secrets.token_bytes(32)
#: Runs one process lists across its studies; the oldest refs lapse first.
_MAX_REFS = 4096
_REF = re.compile(r"rr_[0-9a-f]{32}")
#: Fields of a run row or response that name host directories.
PATH_FIELDS = ("project_dir", "project_root", "project_roots")

_lock = threading.Lock()
_refs: "OrderedDict[str, str]" = OrderedDict()


def issue(project_dir: str) -> str:
    """The ref of one listed run directory."""

    path = str(Path(project_dir).expanduser().resolve())
    digest = hmac.new(_SECRET, path.encode("utf-8"), hashlib.sha256).hexdigest()
    ref = "rr_" + digest[:32]
    with _lock:
        _refs[ref] = path
        _refs.move_to_end(ref)
        while len(_refs) > _MAX_REFS:
            _refs.popitem(last=False)
    return ref


def resolve(ref: Any) -> Optional[str]:
    """The directory a ref this process issued names; ``None`` otherwise."""

    text = str(ref or "").strip()
    if not _REF.fullmatch(text):
        return None
    with _lock:
        return _refs.get(text)


def path_free_row(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """One listed run with its directory replaced by its ref."""

    project_dir = str(row.get("project_dir") or "").strip()
    if not project_dir:
        return None
    public = {key: value for key, value in row.items() if key not in PATH_FIELDS}
    public["run_ref"] = issue(project_dir)
    return public


def _without_path(row: Any) -> Any:
    if not isinstance(row, Mapping):
        return row
    return {key: value for key, value in row.items() if key != "path"}


def path_free_result(result: Mapping[str, Any], ref: str) -> Dict[str, Any]:
    """A run route's result addressed by the ref it was asked with.

    Artifact rows keep their names, digests and run-relative paths; the
    review's public payloads keep their source's label but not its path.
    """

    public = {key: value for key, value in result.items() if key not in PATH_FIELDS}
    if isinstance(public.get("artifacts"), list):
        public["artifacts"] = [_without_path(row) for row in public["artifacts"]]
    if isinstance(public.get("artifact"), Mapping):
        public["artifact"] = _without_path(public["artifact"])
    payloads = public.get("artifact_payloads")
    if isinstance(payloads, Mapping):
        public["artifact_payloads"] = {
            name: (
                {**payload, "source": _without_path(payload.get("source"))}
                if isinstance(payload, Mapping) and isinstance(payload.get("source"), Mapping)
                else payload
            )
            for name, payload in payloads.items()
        }
    public["run_ref"] = ref
    return public


__all__ = ["PATH_FIELDS", "issue", "path_free_result", "path_free_row", "resolve"]
