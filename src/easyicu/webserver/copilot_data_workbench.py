"""Immutable browser-only snapshots for conversational Data Workbench views.

The clinical/data owners still compute every payload.  This module owns only
the project-scoped replay coordinate used by Copilot to reopen a bounded view
without putting patient timelines, rows, or host paths in the model-visible
tool receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from easyicu.webserver import state_paths
from easyicu.webserver.data_package_review import (
    DataPackageReviewError,
    verify_path_free_snapshot,
)


SCHEMA_VERSION = "easyicu.copilot-data-workbench/1"
ALLOWED_VIEWS = frozenset(
    {
        "cohort_summary",
        "feature_distribution",
        "patient_timeline",
        "crossdb_comparison",
    }
)
MAX_SNAPSHOT_BYTES = 768 * 1024


class CopilotDataWorkbenchError(RuntimeError):
    """Owner-attributable snapshot failure."""

    def __init__(
        self, code: str, message: str, *, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.details = dict(details or {})


def _stable_project_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 160:
        raise CopilotDataWorkbenchError(
            "copilot_data_workbench_project_invalid",
            "The Data Workbench snapshot requires a bounded project id.",
        )
    return text


def _stable_view(value: Any) -> str:
    text = str(value or "").strip()
    if text not in ALLOWED_VIEWS:
        raise CopilotDataWorkbenchError(
            "copilot_data_workbench_view_invalid",
            "The requested conversational Data Workbench view is not supported.",
            details={"view": text},
        )
    return text


def _digest_payload(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("snapshot_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def build_snapshot(
    *,
    project_id: str,
    view: str,
    title: str,
    payload: Mapping[str, Any],
    privacy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Seal one path-free owner payload for browser-only conversation replay."""

    snapshot: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project_id": _stable_project_id(project_id),
        "view": _stable_view(view),
        "title": str(title or "Data Workbench")[:160],
        "payload": dict(payload),
        "privacy": dict(privacy or {}),
    }
    try:
        verify_path_free_snapshot(snapshot)
    except DataPackageReviewError as exc:
        raise CopilotDataWorkbenchError(
            "copilot_data_workbench_path_forbidden",
            "The Data Workbench owner payload contains a host-path-shaped field.",
            details=exc.details,
        ) from exc
    snapshot["snapshot_sha256"] = _digest_payload(snapshot)
    return snapshot


class CopilotDataWorkbenchSnapshotStore:
    """Persist immutable project-scoped Data Workbench browser snapshots."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = (
            Path(root)
            if root is not None
            else state_paths.state_root() / "copilot-data-workbench"
        )
        self._lock = threading.RLock()

    @staticmethod
    def _coordinates(payload: Mapping[str, Any]) -> tuple[str, str, str]:
        project_id = _stable_project_id(payload.get("project_id"))
        view = _stable_view(payload.get("view"))
        digest = str(payload.get("snapshot_sha256") or "").strip().lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise CopilotDataWorkbenchError(
                "copilot_data_workbench_coordinates_invalid",
                "The Data Workbench snapshot has invalid immutable coordinates.",
            )
        return project_id, view, digest

    def _path(self, project_id: str, digest: str) -> Path:
        project_key = hashlib.sha256(project_id.encode("utf-8")).hexdigest()[:24]
        return self.root / project_key / f"{digest}.json"

    def persist(self, payload: Mapping[str, Any]) -> Path:
        project_id, _view, digest = self._coordinates(payload)
        if _digest_payload(payload) != digest:
            raise CopilotDataWorkbenchError(
                "copilot_data_workbench_digest_invalid",
                "The Data Workbench snapshot does not match its owner digest.",
            )
        try:
            verify_path_free_snapshot(payload)
        except DataPackageReviewError as exc:
            raise CopilotDataWorkbenchError(
                "copilot_data_workbench_path_forbidden",
                "The Data Workbench snapshot contains a host-path-shaped field.",
                details=exc.details,
            ) from exc
        encoded = json.dumps(
            dict(payload), ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8")
        if len(encoded) > MAX_SNAPSHOT_BYTES:
            raise CopilotDataWorkbenchError(
                "copilot_data_workbench_snapshot_too_large",
                "The Data Workbench snapshot exceeds its bounded browser contract.",
            )
        target = self._path(project_id, digest)
        with self._lock:
            if target.exists():
                if target.read_bytes() != encoded:
                    raise CopilotDataWorkbenchError(
                        "copilot_data_workbench_identity_drift",
                        "An immutable Data Workbench coordinate already has different bytes.",
                    )
                return target
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            handle = tempfile.NamedTemporaryFile(
                mode="wb",
                dir=str(target.parent),
                prefix=".copilot-data-workbench-",
                suffix=".tmp",
                delete=False,
            )
            temporary = Path(handle.name)
            try:
                with handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.chmod(0o600)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return target

    def load(self, *, project_id: str, digest: str) -> Dict[str, Any]:
        clean_project = _stable_project_id(project_id)
        clean_digest = str(digest or "").strip().lower()
        if len(clean_digest) != 64 or any(
            c not in "0123456789abcdef" for c in clean_digest
        ):
            raise CopilotDataWorkbenchError(
                "copilot_data_workbench_coordinates_invalid",
                "The requested Data Workbench coordinates are invalid.",
            )
        target = self._path(clean_project, clean_digest)
        try:
            if target.stat().st_size > MAX_SNAPSHOT_BYTES:
                raise CopilotDataWorkbenchError(
                    "copilot_data_workbench_snapshot_too_large",
                    "The Data Workbench snapshot exceeds its bounded browser contract.",
                )
            payload = json.loads(target.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CopilotDataWorkbenchError(
                "copilot_data_workbench_snapshot_not_found",
                "The immutable Data Workbench snapshot is unavailable.",
            ) from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CopilotDataWorkbenchError(
                "copilot_data_workbench_snapshot_unreadable",
                "The Data Workbench snapshot cannot be read.",
            ) from exc
        if not isinstance(payload, dict):
            raise CopilotDataWorkbenchError(
                "copilot_data_workbench_snapshot_invalid",
                "The Data Workbench snapshot has an invalid shape.",
            )
        loaded_project, _view, loaded_digest = self._coordinates(payload)
        if loaded_project != clean_project or loaded_digest != clean_digest:
            raise CopilotDataWorkbenchError(
                "copilot_data_workbench_snapshot_mismatch",
                "The Data Workbench snapshot does not match the requested project.",
            )
        if _digest_payload(payload) != clean_digest:
            raise CopilotDataWorkbenchError(
                "copilot_data_workbench_digest_invalid",
                "The Data Workbench snapshot digest is invalid.",
            )
        return payload


__all__ = [
    "ALLOWED_VIEWS",
    "CopilotDataWorkbenchError",
    "CopilotDataWorkbenchSnapshotStore",
    "SCHEMA_VERSION",
    "build_snapshot",
]
