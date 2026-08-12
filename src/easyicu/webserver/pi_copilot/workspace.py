"""Project-scoped artifact workspace for Pi's authoring mode.

This owner deliberately exposes relative project files rather than arbitrary
host paths.  It is the only filesystem boundary used by Pi workspace tools and
the browser preview endpoints. Compare-and-swap writes are serialized by both a
process-local reentrant lock and an OS-released per-project file lock.

The declared workspace entry itself must not be a symbolic link. Stable
ancestor links are supported so an operator may relocate ``~/.easyicu`` as one
unit; their resolved identity is sealed at construction and any later retarget
fails closed with ``pi_workspace_base_root_changed``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
import threading
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional

from .contracts import PiCopilotError
from .locking import exclusive_file_lock

MAX_FILE_BYTES = 256 * 1024
MAX_PROJECT_BYTES = 5 * 1024 * 1024
MAX_FILES = 200
MAX_READ_CHARS = 24_000
MAX_RELATIVE_FILE_CHARS = 240
TEXT_SUFFIXES = frozenset(
    {
        ".css",
        ".csv",
        ".html",
        ".htm",
        ".js",
        ".json",
        ".md",
        ".mjs",
        ".py",
        ".svg",
        ".txt",
        ".yaml",
        ".yml",
    }
)
PREVIEW_SUFFIXES = frozenset({".html", ".htm"})
WORKSPACE_ARTIFACT_AUTHORITY: Mapping[str, Any] = MappingProxyType(
    {
        "authority_class": "workspace_artifact",
        "scientific_evidence": False,
        "validation_status": "unvalidated",
        "claim_ceiling": "unsupported",
    }
)

_PROJECT_LOCKS: dict[tuple[str, str], threading.RLock] = {}
_PROJECT_LOCKS_GUARD = threading.Lock()
_PROJECT_LOCK_NAME = ".easyicu-workspace-write.lock"


def _authority_metadata() -> Dict[str, Any]:
    return dict(WORKSPACE_ARTIFACT_AUTHORITY)


def _node_check_environment() -> Dict[str, str]:
    """Return the small non-secret environment needed by ``node --check``."""

    allowed = {"PATH", "HOME", "TMPDIR", "LANG"}
    return {
        key: value
        for key, value in os.environ.items()
        if key in allowed or key.startswith("LC_")
    }


def project_workspace_id(project_id: str) -> str:
    clean = str(project_id or "").strip()
    if not clean or len(clean) > 160:
        raise PiCopilotError(
            "pi_workspace_project_required",
            "A valid EasyICU project is required for workspace access.",
        )
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:32]


def _relative_file(value: Any) -> PurePosixPath:
    text = str(value or "").strip().replace("\\", "/")
    if not text or len(text) > MAX_RELATIVE_FILE_CHARS or "\x00" in text:
        raise PiCopilotError(
            "pi_workspace_file_invalid",
            "The project file name is invalid or exceeds its bounded contract.",
        )
    relative = PurePosixPath(text)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise PiCopilotError(
            "pi_workspace_path_escape",
            "Project files must stay inside the isolated Pi workspace.",
        )
    if relative.suffix.lower() not in TEXT_SUFFIXES:
        raise PiCopilotError(
            "pi_workspace_file_type_unsupported",
            "This Pi workspace currently accepts bounded text and web files only.",
            details={"suffix": relative.suffix.lower()},
        )
    return relative


class _HTMLCheck(HTMLParser):
    def error(self, message: str) -> None:  # pragma: no cover - legacy hook
        raise ValueError(message)


class ProjectWorkspace:
    """Read and mutate bounded UTF-8 artifacts under one project root."""

    def __init__(self, base_dir: Path) -> None:
        declared = Path(base_dir).expanduser().absolute()
        if declared.is_symlink():
            raise PiCopilotError(
                "pi_workspace_base_root_symlink_blocked",
                "The Pi workspace root must not be a symbolic link.",
            )
        self.declared_base_dir = declared
        self.base_dir = declared.resolve()

    @staticmethod
    def _ensure_directory(
        path: Path,
        *,
        code: str,
        message: str,
        parents: bool = False,
    ) -> None:
        """Create an expected workspace directory with an owner-stable error."""

        try:
            path.mkdir(parents=parents, exist_ok=True, mode=0o700)
        except OSError as exc:
            raise PiCopilotError(code, message, details={"path_name": path.name}) from exc
        if not path.is_dir():
            raise PiCopilotError(code, message, details={"path_name": path.name})

    def _ensure_base_root(self) -> None:
        if self.declared_base_dir.is_symlink():
            raise PiCopilotError(
                "pi_workspace_base_root_symlink_blocked",
                "The Pi workspace root must not be a symbolic link.",
            )
        if self.declared_base_dir.resolve(strict=False) != self.base_dir:
            raise PiCopilotError(
                "pi_workspace_base_root_changed",
                "The Pi workspace root changed after this workspace was opened.",
            )
        self._ensure_directory(
            self.declared_base_dir,
            code="pi_workspace_base_root_not_directory",
            message="The Pi workspace root is not an accessible directory.",
            parents=True,
        )
        if self.declared_base_dir.is_symlink():
            raise PiCopilotError(
                "pi_workspace_base_root_symlink_blocked",
                "The Pi workspace root must not be a symbolic link.",
            )
        if self.declared_base_dir.resolve() != self.base_dir:
            raise PiCopilotError(
                "pi_workspace_base_root_changed",
                "The Pi workspace root changed after this workspace was opened.",
            )

    def _project_lock(self, project_id: str) -> threading.RLock:
        key = (str(self.base_dir), project_workspace_id(project_id))
        with _PROJECT_LOCKS_GUARD:
            return _PROJECT_LOCKS.setdefault(key, threading.RLock())

    def _project_file_lock(self, project_id: str):
        return exclusive_file_lock(
            self.project_root(project_id) / _PROJECT_LOCK_NAME,
            code="pi_workspace_write_lock_unavailable",
        )

    def project_root(self, project_id: str) -> Path:
        self._ensure_base_root()
        projects_root = self.base_dir / "projects"
        if projects_root.is_symlink():
            raise PiCopilotError(
                "pi_workspace_projects_root_symlink_blocked",
                "The Pi projects directory must not be a symbolic link.",
            )
        self._ensure_directory(
            projects_root,
            code="pi_workspace_projects_root_not_directory",
            message="The Pi projects path is not an accessible directory.",
        )
        if projects_root.is_symlink():
            raise PiCopilotError(
                "pi_workspace_projects_root_symlink_blocked",
                "The Pi projects directory must not be a symbolic link.",
            )
        try:
            resolved_projects = projects_root.resolve()
            resolved_projects.relative_to(self.base_dir)
        except ValueError as exc:
            raise PiCopilotError(
                "pi_workspace_projects_root_escape",
                "The Pi projects directory escaped its host-owned workspace root.",
            ) from exc

        root = projects_root / project_workspace_id(project_id)
        if root.is_symlink():
            raise PiCopilotError(
                "pi_workspace_project_root_symlink_blocked",
                "The Pi project root must not be a symbolic link.",
            )
        self._ensure_directory(
            root,
            code="pi_workspace_project_root_not_directory",
            message="The Pi project path is not an accessible directory.",
        )
        if root.is_symlink():
            raise PiCopilotError(
                "pi_workspace_project_root_symlink_blocked",
                "The Pi project root must not be a symbolic link.",
            )
        resolved_root = root.resolve()
        if resolved_root.parent != resolved_projects:
            raise PiCopilotError(
                "pi_workspace_project_root_escape",
                "The Pi project root escaped its host-owned projects directory.",
            )
        return resolved_root

    def _candidate(
        self,
        project_id: str,
        relative_file: Any,
        *,
        create_parent: bool = False,
    ) -> tuple[Path, str]:
        relative = _relative_file(relative_file)
        root = self.project_root(project_id)
        candidate = root.joinpath(*relative.parts)
        if create_parent:
            # Validate the nearest existing ancestor before mkdir. Otherwise a
            # symlinked directory could create paths outside the project before
            # the later resolve check rejects it.
            existing_parent = candidate.parent
            while existing_parent != root and not existing_parent.exists():
                existing_parent = existing_parent.parent
            if existing_parent.is_symlink() or not existing_parent.is_dir():
                raise PiCopilotError(
                    "pi_workspace_path_escape",
                    "Project files must stay inside the isolated Pi workspace.",
                )
            try:
                existing_parent.resolve().relative_to(root)
            except ValueError as exc:
                raise PiCopilotError(
                    "pi_workspace_path_escape",
                    "Project files must stay inside the isolated Pi workspace.",
                ) from exc
            self._ensure_directory(
                candidate.parent,
                code="pi_workspace_parent_not_directory",
                message="The Pi project file parent is not an accessible directory.",
                parents=True,
            )
        try:
            candidate.parent.resolve().relative_to(root)
        except ValueError as exc:
            raise PiCopilotError(
                "pi_workspace_path_escape",
                "Project files must stay inside the isolated Pi workspace.",
            ) from exc
        if candidate.is_symlink():
            raise PiCopilotError(
                "pi_workspace_symlink_blocked",
                "Symbolic links are not accepted by the Pi project workspace.",
            )
        if candidate.exists():
            try:
                candidate.resolve().relative_to(root)
            except ValueError as exc:
                raise PiCopilotError(
                    "pi_workspace_path_escape",
                    "Project files must stay inside the isolated Pi workspace.",
                ) from exc
        return candidate, relative.as_posix()

    @staticmethod
    def _media_type(path: Path) -> str:
        guessed = mimetypes.guess_type(path.name)[0]
        return guessed or "text/plain"

    def _file_rows(self, project_id: str) -> list[Dict[str, Any]]:
        root = self.project_root(project_id)
        rows: list[Dict[str, Any]] = []
        for candidate in sorted(root.rglob("*")):
            if candidate.is_symlink() or not candidate.is_file():
                continue
            try:
                relative = candidate.resolve().relative_to(root).as_posix()
            except ValueError:
                continue
            if candidate.suffix.lower() not in TEXT_SUFFIXES:
                continue
            rows.append(
                {
                    "file": relative,
                    "size": candidate.stat().st_size,
                    "media_type": self._media_type(candidate),
                    **_authority_metadata(),
                }
            )
            if len(rows) >= MAX_FILES:
                break
        return rows

    def list_files(self, project_id: str) -> list[Dict[str, Any]]:
        return self._file_rows(project_id)

    def read_file(
        self,
        project_id: str,
        relative_file: Any,
        *,
        start_line: int = 1,
        end_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        with self._project_lock(project_id):
            candidate, relative, text, raw = self._read_complete_file(
                project_id, relative_file
            )
            lines = text.splitlines()
            first = max(1, int(start_line or 1))
            last = min(len(lines), int(end_line or min(len(lines), first + 399)))
            if last < first:
                last = first
            selected = "\n".join(lines[first - 1 : last])
            truncated = len(selected) > MAX_READ_CHARS
            selected = selected[:MAX_READ_CHARS]
            return {
                "file": relative,
                "media_type": self._media_type(candidate),
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "start_line": first,
                "end_line": last,
                "total_lines": len(lines),
                "text": selected,
                "truncated": truncated or last < len(lines),
                **_authority_metadata(),
            }

    @staticmethod
    def _read_text(candidate: Path, relative: str) -> str:
        try:
            return candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise PiCopilotError(
                "pi_workspace_file_not_text",
                "The requested Pi project file is not valid UTF-8 text.",
                details={"file": relative},
            ) from exc

    def _read_complete_file(
        self,
        project_id: str,
        relative_file: Any,
    ) -> tuple[Path, str, str, bytes]:
        candidate, relative = self._candidate(project_id, relative_file)
        if not candidate.is_file():
            raise PiCopilotError(
                "pi_workspace_file_not_found",
                "The requested Pi project file does not exist.",
                status_code=404,
                details={"file": relative},
            )
        raw = candidate.read_bytes()
        if len(raw) > MAX_FILE_BYTES:
            raise PiCopilotError(
                "pi_workspace_file_too_large",
                "The requested Pi project file exceeds the bounded file limit.",
                details={"file": relative, "max_bytes": MAX_FILE_BYTES},
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PiCopilotError(
                "pi_workspace_file_not_text",
                "The requested Pi project file is not valid UTF-8 text.",
                details={"file": relative},
            ) from exc
        return candidate, relative, text, raw

    def _project_size_without(self, project_id: str, target: Path) -> int:
        root = self.project_root(project_id)
        total = 0
        count = 0
        for candidate in root.rglob("*"):
            if (
                candidate.is_symlink()
                or not candidate.is_file()
                or candidate == target
                or candidate.name == _PROJECT_LOCK_NAME
            ):
                continue
            total += candidate.stat().st_size
            count += 1
            if count >= MAX_FILES or total > MAX_PROJECT_BYTES:
                break
        return total

    def write_file(
        self,
        project_id: str,
        relative_file: Any,
        content: Any,
    ) -> Dict[str, Any]:
        with self._project_lock(project_id), self._project_file_lock(project_id):
            candidate, relative = self._candidate(
                project_id,
                relative_file,
                create_parent=True,
            )
            encoded = str(content if content is not None else "").encode("utf-8")
            if len(encoded) > MAX_FILE_BYTES:
                raise PiCopilotError(
                    "pi_workspace_file_too_large",
                    "The Pi project file exceeds the write limit.",
                    details={"file": relative, "max_bytes": MAX_FILE_BYTES},
                )
            rows = self._file_rows(project_id)
            existing = candidate.exists()
            if existing:
                raise PiCopilotError(
                    "pi_workspace_write_create_only",
                    "Write creates new project files only; use an exact edit for an existing file.",
                    status_code=409,
                    details={"file": relative},
                )
            if not existing and len(rows) >= MAX_FILES:
                raise PiCopilotError(
                    "pi_workspace_file_limit",
                    "The Pi project workspace reached its bounded file limit.",
                    details={"max_files": MAX_FILES},
                )
            if (
                self._project_size_without(project_id, candidate) + len(encoded)
                > MAX_PROJECT_BYTES
            ):
                raise PiCopilotError(
                    "pi_workspace_size_limit",
                    "The Pi project workspace reached its bounded size limit.",
                    details={"max_bytes": MAX_PROJECT_BYTES},
                )
            self._atomic_write(candidate, encoded)
            return {
                "file": relative,
                "media_type": self._media_type(candidate),
                "size": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "created": not existing,
                **_authority_metadata(),
            }

    @staticmethod
    def _atomic_write(candidate: Path, encoded: bytes) -> None:
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(candidate.parent),
            prefix=".pi-workspace-",
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
            temporary.replace(candidate)
            candidate.chmod(0o600)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _expected_sha256(value: Any) -> Optional[str]:
        if value is None or str(value).strip() == "":
            return None
        digest = str(value).strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise PiCopilotError(
                "pi_workspace_expected_sha256_invalid",
                "The expected project file SHA-256 digest is invalid.",
            )
        return digest

    def edit_file(
        self,
        project_id: str,
        relative_file: Any,
        *,
        old_text: Any,
        new_text: Any,
        expected_sha256: Any = None,
    ) -> Dict[str, Any]:
        with self._project_lock(project_id), self._project_file_lock(project_id):
            candidate, relative, source, raw = self._read_complete_file(
                project_id, relative_file
            )
            supplied_digest = self._expected_sha256(expected_sha256)
            current_digest = hashlib.sha256(raw).hexdigest()
            if supplied_digest is None:
                raise PiCopilotError(
                    "pi_workspace_expected_sha256_required",
                    "Editing a project file requires its current SHA-256 digest.",
                    status_code=409,
                    details={"file": relative, "current_sha256": current_digest},
                )
            if supplied_digest != current_digest:
                raise PiCopilotError(
                    "pi_workspace_file_changed",
                    "The project file changed after it was read; read it again before editing it.",
                    status_code=409,
                    details={"file": relative, "current_sha256": current_digest},
                )
            needle = str(old_text if old_text is not None else "")
            if not needle:
                raise PiCopilotError(
                    "pi_workspace_edit_target_required",
                    "An exact non-empty edit target is required.",
                )
            occurrences = source.count(needle)
            if occurrences != 1:
                raise PiCopilotError(
                    "pi_workspace_edit_target_not_unique",
                    "The exact edit target must occur once in the project file.",
                    details={"file": relative, "occurrences": occurrences},
                )
            encoded = source.replace(
                needle, str(new_text if new_text is not None else ""), 1
            ).encode("utf-8")
            if len(encoded) > MAX_FILE_BYTES:
                raise PiCopilotError(
                    "pi_workspace_file_too_large",
                    "The Pi project file exceeds the edit limit.",
                    details={"file": relative, "max_bytes": MAX_FILE_BYTES},
                )
            if (
                self._project_size_without(project_id, candidate) + len(encoded)
                > MAX_PROJECT_BYTES
            ):
                raise PiCopilotError(
                    "pi_workspace_size_limit",
                    "The Pi project workspace reached its bounded size limit.",
                    details={"max_bytes": MAX_PROJECT_BYTES},
                )
            self._atomic_write(candidate, encoded)
            return {
                "file": relative,
                "media_type": self._media_type(candidate),
                "size": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "created": False,
                "replacements": 1,
                **_authority_metadata(),
            }

    def check_file(self, project_id: str, relative_file: Any) -> Dict[str, Any]:
        with self._project_lock(project_id):
            candidate, relative, text, _raw = self._read_complete_file(
                project_id, relative_file
            )
        suffix = Path(relative).suffix.lower()
        try:
            if suffix in {".html", ".htm"}:
                parser = _HTMLCheck(convert_charrefs=True)
                parser.feed(text)
                parser.close()
                checker = "html.parser"
            elif suffix == ".json":
                json.loads(text)
                checker = "json"
            elif suffix == ".py":
                ast.parse(text, filename=relative)
                checker = "python-ast"
            elif suffix in {".js", ".mjs"}:
                node = shutil.which("node")
                if not node:
                    raise PiCopilotError(
                        "pi_workspace_checker_unavailable",
                        "Node.js is unavailable for the JavaScript syntax check.",
                    )
                result = subprocess.run(
                    [node, "--check", str(candidate)],
                    cwd=self.project_root(project_id),
                    env=_node_check_environment(),
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if result.returncode:
                    raise ValueError((result.stderr or result.stdout).strip()[:2000])
                checker = "node --check"
            elif suffix == ".css":
                if text.count("{") != text.count("}"):
                    raise ValueError("CSS braces are unbalanced")
                checker = "css-braces"
            else:
                checker = "utf-8"
        except PiCopilotError:
            raise
        except Exception as exc:
            raise PiCopilotError(
                "pi_workspace_check_failed",
                "The project file did not pass its bounded static check.",
                details={"file": relative, "reason": str(exc)[:2000]},
            ) from exc
        return {
            "file": relative,
            "media_type": self._media_type(candidate),
            "checker": checker,
            "valid": True,
            "check_scope": "bounded_static_syntax",
            **_authority_metadata(),
        }

    def preview_file(self, project_id: str, relative_file: Any) -> Dict[str, Any]:
        with self._project_lock(project_id):
            candidate, relative, text, raw = self._read_complete_file(
                project_id, relative_file
            )
        if Path(relative).suffix.lower() not in PREVIEW_SUFFIXES:
            raise PiCopilotError(
                "pi_workspace_preview_type_unsupported",
                "Only bounded HTML project artifacts can open as a live web preview.",
                details={"file": relative},
            )
        return {
            "file": relative,
            "media_type": self._media_type(candidate),
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "text": text,
            "truncated": False,
            **_authority_metadata(),
        }


__all__ = [
    "MAX_FILE_BYTES",
    "MAX_FILES",
    "MAX_PROJECT_BYTES",
    "PREVIEW_SUFFIXES",
    "WORKSPACE_ARTIFACT_AUTHORITY",
    "ProjectWorkspace",
    "project_workspace_id",
]
