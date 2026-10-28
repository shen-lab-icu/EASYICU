"""Bounded, passive Skill packages. Never extract an upload onto a filesystem."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import PurePosixPath
import stat
from typing import Mapping
import zipfile
import zlib

from .contracts import ExtensionRegistryError

MAX_ARCHIVE_BYTES = 256_000
MAX_PACKAGE_BYTES = 48_000
MAX_PACKAGE_FILES = 64


def _invalid(code: str, message: str) -> ExtensionRegistryError:
    return ExtensionRegistryError(f"extension_skill_package_{code}", message)


def _path(value: str) -> str:
    parts = value.rstrip("/").split("/")
    if (not value or len(value) > 256 or "\\" in value or ":" in value
            or any(part in {"", ".", ".."} for part in parts)
            or any(ord(char) < 32 for char in value)):
        raise _invalid("path_invalid", "Package path must be a safe relative path.")
    return "/".join(parts)


def read_archive(content: bytes) -> dict[str, bytes]:
    if len(content) > MAX_ARCHIVE_BYTES:
        raise _invalid("too_large", "ZIP exceeds 256 KB.")
    try:
        with zipfile.ZipFile(BytesIO(content)) as package:
            entries = package.infolist()
            if len(entries) > MAX_PACKAGE_FILES:
                raise _invalid("too_many_files", "Package exceeds 64 entries.")
            total = 0
            files = {}
            names = set()
            for entry in entries:
                name = _path(entry.orig_filename)
                if name in names or stat.S_ISLNK(entry.external_attr >> 16):
                    raise _invalid("path_invalid", "Package path is duplicated or is a symbolic link.")
                names.add(name)
                total += entry.file_size
                if total > MAX_PACKAGE_BYTES:
                    raise _invalid("too_large", "Unpacked package exceeds 48 KB.")
                if not entry.is_dir():
                    # Bound actual inflation too; metadata alone grants no size authority.
                    with package.open(entry) as handle:
                        data = handle.read(MAX_PACKAGE_BYTES + 1)
                    if len(data) != entry.file_size:
                        raise _invalid("too_large", "Package member exceeds its size limit.")
                    files[name] = data
            return files
    except (zipfile.BadZipFile, RuntimeError, OSError, NotImplementedError, EOFError, UnicodeError, zlib.error) as exc:
        raise _invalid("invalid", "Cannot read this ZIP package.") from exc


def package_files(files: Mapping[str, str | bytes]) -> tuple[str, dict[str, str], list[str]]:
    """Admit one SKILL.md and text references, ignoring executable payloads."""
    if not files or len(files) > MAX_PACKAGE_FILES:
        raise _invalid("too_many_files", "Choose a package with at most 64 files.")
    safe = {}
    total = 0
    for path, value in files.items():
        name = _path(path)
        try:
            data = value.encode("utf-8") if isinstance(value, str) else value
        except UnicodeError as exc:
            raise _invalid("text_invalid", "Skill documents must be UTF-8 text.") from exc
        total += len(data)
        if total > MAX_PACKAGE_BYTES:
            raise _invalid("too_large", "Unpacked package exceeds 48 KB.")
        safe[name] = data
    # Browser directory uploads and common ZIPs have exactly one wrapper folder.
    if "SKILL.md" not in safe:
        roots = {name.split("/")[0] for name in safe}
        if len(roots) == 1 and all("/" in name for name in safe):
            safe = {name.split("/", 1)[1]: data for name, data in safe.items()}
    if "SKILL.md" not in safe:
        raise _invalid("main_required", "Package must contain SKILL.md.")
    documents = {}
    ignored = []
    for name, data in sorted(safe.items()):
        reference = name.startswith("references/") and PurePosixPath(name).suffix.lower() in {".md", ".txt"}
        if name != "SKILL.md" and not reference:
            ignored.append(name)
            continue
        try:
            documents[name] = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _invalid("text_invalid", "Skill documents must be UTF-8 text.") from exc
    main = documents.pop("SKILL.md")
    return main, documents, ignored


def with_references(parsed: dict, references: Mapping[str, str] | None) -> dict:
    """Bind all reference bytes to the same frozen digest and instruction budget."""
    from .registry import _MODEL_VISIBLE_SECRET_OR_HOST_PATH

    if not references:
        return {**parsed, "reference_files": {}}
    main, clean, ignored = package_files({"SKILL.md": parsed["canonical_markdown"], **references})
    if ignored or set(clean) != set(references):
        raise _invalid("path_invalid", "Reference path must be under references/ and end in .md or .txt.")
    if any(_MODEL_VISIBLE_SECRET_OR_HOST_PATH.search(text) or "\x00" in text for text in clean.values()):
        raise _invalid("text_invalid", "References must not contain credentials, host paths or binary content.")
    # Include full documents, never a mutable filesystem reference or executable path.
    instructions = parsed["instructions"] + "".join(
        f"\n\n## Reference: {path}\n\n{text}" for path, text in sorted(clean.items())
    )
    if len(instructions.encode("utf-8")) > MAX_PACKAGE_BYTES:
        raise _invalid("too_large", "Combined instructions exceed 48 KB.")
    identity = json.dumps({"SKILL.md": main, **clean}, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {**parsed, "reference_files": clean, "instructions": instructions,
            "digest": hashlib.sha256(identity).hexdigest(),
            "size_bytes": max(len(main.encode("utf-8")) + sum(len(text.encode("utf-8")) for text in clean.values()),
                              len(instructions.encode("utf-8")))}


def write_archive(main: str, references: Mapping[str, str]) -> bytes:
    out = BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for path, text in [("SKILL.md", main), *sorted(references.items())]:
            package.writestr(path, text.encode("utf-8"))
    return out.getvalue()
