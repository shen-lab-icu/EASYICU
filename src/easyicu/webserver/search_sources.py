"""Bounded discovery of registered project metadata, never a recursive crawler."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException

from . import state_paths
from .hosted import require_instance_path
from .run_file_guide import run_file_guide
from .run_refs import issue as issue_run_ref
from .search_contracts import SearchDocument, text_value
from .search_projection import messages, run_documents, transcript

MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_TRANSCRIPT_BYTES = 16 * 1024 * 1024
MAX_LINE_BYTES = 256 * 1024
MAX_ENTRIES = 10_000
MAX_DIRECTORY_ENTRIES = 512
MAX_SOURCES = 4096
MAX_CATALOG_DOCUMENTS = 10_000
MAX_CATALOG_TEXT_BYTES = 8 * 1024 * 1024
MAX_DISCOVERY_ENTRIES = 4096
MAX_REFRESH_READ_BYTES = 64 * 1024 * 1024
RUN_TEXT_FILES = ("run_context.json", "agent_plan.json", "manuscript_draft.json", "result_tables.json")
# Only browser-owned aggregate/result documents, never arbitrary run files.
RUN_FILES = (
    "result_tables.json", "figure_gallery.json", "cohort_summary.json", "table1_summary.json",
    "missingness_audit.json", "roc_curve.json", "calibration_curve.json",
    "manuscript_draft.json", "manuscript_scaffold.pdf", "manuscript_scaffold.tex",
    "manuscript_scaffold.bib", "manuscript_revision.pdf", "manuscript_provenance.json",
)


@dataclass(frozen=True)
class Source:
    path: Path
    root: Path
    project: str
    kind: str
    conversation: str = ""
    run: str = ""
    wrapper: str = ""


class SearchCatalog:
    """Read one instance's registries and stat only known metadata coordinates."""

    def __init__(self):
        self.state = state_paths.state_root().absolute()
        self.projects = state_paths.projects_root().absolute()
        require_instance_path(self.state)
        require_instance_path(self.projects)
        self.issues: set[str] = set()
        self.documents: list[SearchDocument] = []
        self.sources: list[Source] = []
        self._document_bytes = 0
        self._entries = 0
        self._read_bytes = 0

    def add_document(self, document: SearchDocument):
        size = len(document.searchable.encode("utf-8"))
        if (len(self.documents) >= MAX_CATALOG_DOCUMENTS
                or self._document_bytes + size > MAX_CATALOG_TEXT_BYTES):
            self.issues.add("catalog_limit")
            return
        self._document_bytes += size
        self.documents.append(document)

    def safe(self, root: Path, path: Path) -> bool:
        try:
            require_instance_path(path)
            relative = path.relative_to(root)
            if ".." in relative.parts or root.is_symlink():
                return False
            return not any(p.is_symlink() for p in (path, *path.parents) if p == root or root in p.parents)
        except (OSError, ValueError, RuntimeError, HTTPException):
            return False

    def stamp(self, root: Path, path: Path) -> tuple | None:
        if not self.safe(root, path):
            return None
        try:
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                return None
            return info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
        except OSError:
            return None

    def read(self, root: Path, path: Path, *, lines: bool = False):
        stamp = self.stamp(root, path)
        if stamp is None:
            return [] if lines else {}
        maximum = MAX_TRANSCRIPT_BYTES if lines else MAX_METADATA_BYTES
        if stamp[1] > maximum:
            self.issues.add("source_too_large")
            return [] if lines else {}
        if self._read_bytes + stamp[1] > MAX_REFRESH_READ_BYTES:
            self.issues.add("read_limit")
            return [] if lines else {}
        self._read_bytes += stamp[1]
        try:
            # Bound the read itself as well as stat: a writer may grow the file.
            with path.open("rb") as handle:
                raw = handle.read(maximum + 1)
            if len(raw) > maximum:
                self.issues.add("source_too_large")
                return [] if lines else {}
            if lines:
                rows = []
                for line in raw.splitlines():
                    if len(rows) >= MAX_ENTRIES or len(line) > MAX_LINE_BYTES:
                        self.issues.add("transcript_limit")
                        return []  # Do not guess a current branch from a partial tree.
                    if line.strip():
                        row = json.loads(line)
                        if isinstance(row, dict):
                            rows.append(row)
                return rows
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except (OSError, UnicodeError, ValueError, RecursionError):
            self.issues.add("source_unreadable")
            return [] if lines else {}

    def directories(self, root: Path, path: Path):
        if not self.safe(root, path):
            return
        try:
            with os.scandir(path) as entries:
                for index, entry in enumerate(entries):
                    self._entries += 1
                    if index >= MAX_DIRECTORY_ENTRIES or self._entries > MAX_DISCOVERY_ENTRIES:
                        self.issues.add("directory_limit")
                        break
                    if entry.is_dir(follow_symlinks=False):
                        yield Path(entry.path)
        except (OSError, ValueError):
            return

    def add_source(self, source: Source):
        if len(self.sources) >= MAX_SOURCES:
            self.issues.add("source_limit")
        elif self.stamp(source.root, source.path) is not None:
            self.sources.append(source)

    def project(self, project: str, title: str):
        self.add_document(SearchDocument(project, "project", title[:240], "",
                                             dict(kind="project", project_id=project)))

    def runs(self, root: Path, study: Path, project: str):
        for wrapper in self.directories(root, study):
            if not wrapper.name.startswith("run_"):
                continue
            context = self.read(root, wrapper / "run_context.json")
            run = text_value(context.get("run_id"))[:160]
            if not run:
                continue
            # Reuse the existing review/download coordinate, not a new path API.
            key = issue_run_ref(str(wrapper))
            for name in RUN_TEXT_FILES:
                self.add_source(Source(wrapper / name, root, project, "run", run=run, wrapper=key))
            names = [name for name in RUN_FILES if self.stamp(root, wrapper / name) is not None]
            gate = self.read(root, wrapper / "quality_gate.json").get("gate")
            checks = gate.get("checks") if isinstance(gate, dict) else None
            for entry in run_file_guide(names, gate_checks=checks if isinstance(checks, list) else ()):
                self.add_document(SearchDocument(
                    project, "file", entry.title_en, f"{entry.name}\n{entry.title_zh}",
                    dict(kind="file", project_id=project, run_id=run, run_ref=key,
                         file_id=entry.name, artifact=entry.name),
                ))

    def discover(self):
        guided = self.read(self.state, self.state / "webserver_guided_drafts.json")
        registered: dict[str, Path] = {}
        for row in self.rows(guided, "drafts"):
            project = text_value(row.get("id"))[:160]
            path = Path(text_value(row.get("project_dir")))
            if not project or not path.is_absolute() or not self.safe(path, path) or not path.is_dir():
                continue
            # External desktop folders must carry the existing Guided marker.
            if not path.is_relative_to(self.projects) and not self.stamp(path, path / "guided_draft.json"):
                continue
            registered[project] = path
            self.project(project, text_value(row.get("title")) or path.name)
            question = text_value(row.get("question"))
            if question:
                self.add_document(SearchDocument(project, "question", "", question,
                                                  dict(kind="project", project_id=project)))
            self.add_source(Source(path / "guided_copilot_session.json", path, project, "guided"))
            for study in self.directories(path, path):
                self.runs(path, study, project)

        for filename, kind in (("webserver_guided_drafts.json", "guided"),
                               ("webserver_copilot_sessions.json", "copilot")):
            data = guided if kind == "guided" else self.read(self.state, self.state / filename)
            for row in self.rows(data, "sessions"):
                folder = Path(text_value(row.get("project_dir")))
                if not folder.is_absolute() or not self.safe(folder, folder) or not folder.is_dir():
                    continue
                project = next((pid for pid, path in registered.items() if path == folder), "")
                if not project and kind == "guided":
                    continue  # Unregistering a Guided project also hides its sessions.
                conversation = text_value(row.get("id"))[:160]
                project = project or conversation
                if not project or not conversation:
                    continue
                if kind == "copilot":
                    self.project(project, text_value(row.get("title")) or folder.name)
                # The registry is the legacy session writer's source of truth.
                for document in messages(self.rows(row, "messages"), project, conversation, kind):
                    self.add_document(document)

        for row in self.rows(self.read(self.state, self.state / "pi_copilot_sessions.json"), "sessions"):
            project = text_value(row.get("project_id"))[:160]
            conversation = text_value(row.get("session_id"))[:160]
            if project not in registered or not conversation:
                continue
            self.add_document(SearchDocument(project, "conversation", text_value(row.get("title")), "",
                                                  dict(kind="conversation", project_id=project,
                                                       conversation_id=conversation, source="pi")))
            path = Path(text_value(row.get("pi_session_file")))
            if path.is_relative_to(self.state / "pi-agent/sessions") and path.suffix == ".jsonl":
                self.add_source(Source(path, self.state, project, "pi", conversation=conversation))

        bindings = []
        for filename in ("pi_project_authority.json", "pi_project_authority_archive.json"):
            bindings.extend(self.rows(self.read(self.state, self.state / filename), "bindings"))
        for row in bindings:
            project = text_value(row.get("project_id"))[:160]
            study_id = text_value(row.get("study_context_id"))[:160]
            if project not in registered or not study_id:
                continue
            key = hashlib.sha256(study_id.encode()).hexdigest()[:32]
            base = self.state / "pi-agent/workspace" / key
            for study in self.directories(self.state, base):
                self.runs(self.state, study, project)
        # Legacy Agent runs live directly under the desktop projects root.
        # Never rediscover a removed Guided folder through this fallback.
        for study in self.directories(self.projects, self.projects):
            if study in registered.values() or any((study / name).exists() for name in
                ("guided_draft.json", "guided_copilot_session.json", "page_guide_session.json", "copilot_session.json")):
                continue
            before = len(self.sources)
            self.runs(self.projects, study, study.name)
            if len(self.sources) > before:
                self.project(study.name, study.name)

    def rows(self, value: dict, field: str) -> list[dict]:
        rows = value.get(field)
        if not isinstance(rows, list):
            return []
        if len(rows) > MAX_DIRECTORY_ENTRIES:
            self.issues.add("registry_limit")
        return [row for row in rows[:MAX_DIRECTORY_ENTRIES] if isinstance(row, dict)]

    def load(self, source: Source):
        value = self.read(source.root, source.path, lines=source.kind == "pi")
        if not value:
            return iter(())
        if source.kind == "pi":
            return transcript(value, source.project, source.conversation)
        if source.kind == "guided":
            return messages(self.rows(value, "messages"), source.project, text_value(value.get("id")), "guided")
        return run_documents(value, source.project, source.run, source.path.name, source.wrapper)
