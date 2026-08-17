"""Content identity for code that can execute inside the Runner image.

The Research Agent control plane changes much more often than its deterministic
statistics kernel.  A Docker image therefore binds the transitive local source
closure of the modules that generated analysis may import, rather than every
Planner, Provider, and orchestration module installed in the image.

The closure starts from deterministic runners, methods, and figure renderers.
It follows normal EasyICU imports, package initializers, literal dynamic imports,
and full module names embedded by those code-generating seeds.  Any new local
execution dependency therefore enters the digest automatically.  Host identity
remains separately bound by Git/prompt/schema receipts.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


EXECUTION_KERNEL_IDENTITY_SCHEMA = "easyicu.execution_kernel_identity/1"
_MODULE_PREFIX = "easyicu"
_SEED_PREFIXES = (
    "research_agent/execution/runners/",
    "research_agent/methods/",
    "research_agent/figures/",
)
_DYNAMIC_MODULE_RE = re.compile(r"easyicu(?:\.[A-Za-z_][A-Za-z0-9_]*)+")


class ExecutionKernelIdentityError(RuntimeError):
    """The local execution-kernel source contract could not be constructed."""


@dataclass(frozen=True)
class ExecutionKernelIdentity:
    """Immutable Host expectation for one Runner execution kernel."""

    schema_version: str
    source_sha256: str
    files_sha256: str
    file_count: int
    requirements_lock_sha256: str
    identity_sha256: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def _module_name(package_root: Path, path: Path) -> str:
    relative = path.relative_to(package_root).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    suffix = ".".join(parts)
    return _MODULE_PREFIX + (f".{suffix}" if suffix else "")


def _module_index(package_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(package_root.rglob("*.py")):
        if path.is_symlink() or not path.is_file():
            raise ExecutionKernelIdentityError(
                f"execution kernel source is not a regular file: {path}"
            )
        module_name = _module_name(package_root, path)
        if module_name in index:
            raise ExecutionKernelIdentityError(
                f"duplicate execution kernel module: {module_name}"
            )
        index[module_name] = path
    if _MODULE_PREFIX not in index:
        raise ExecutionKernelIdentityError(
            "execution kernel package root must contain easyicu/__init__.py"
        )
    return index


def _seed_modules(
    package_root: Path,
    module_index: dict[str, Path],
) -> set[str]:
    seeds = {
        module_name
        for module_name, path in module_index.items()
        if any(
            path.relative_to(package_root).as_posix().startswith(prefix)
            for prefix in _SEED_PREFIXES
        )
    }
    if not seeds:
        raise ExecutionKernelIdentityError(
            "execution kernel seed modules are unavailable"
        )
    return seeds


def _package_import_nodes(tree: ast.AST) -> Iterator[ast.AST]:
    """Walk package import-time code without traversing lazy API functions."""

    pending = list(getattr(tree, "body", ()))
    while pending:
        node = pending.pop()
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            continue
        yield node
        pending.extend(ast.iter_child_nodes(node))


def _from_import_base(
    *,
    current_module: str,
    package_initializer: bool,
    node: ast.ImportFrom,
) -> str:
    if node.level == 0:
        return str(node.module or "")
    package = (
        current_module if package_initializer else current_module.rpartition(".")[0]
    )
    parts = package.split(".")
    if node.level > 1:
        trim = node.level - 1
        if trim >= len(parts):
            return ""
        parts = parts[:-trim]
    if node.module:
        parts.extend(str(node.module).split("."))
    return ".".join(parts)


def _local_module_target(
    module_name: str,
    module_index: dict[str, Path],
) -> str | None:
    cleaned = str(module_name or "").strip(".")
    if not cleaned.startswith(_MODULE_PREFIX):
        return None
    if cleaned in module_index:
        return cleaned
    parts = cleaned.split(".")
    while len(parts) > 1:
        parts.pop()
        candidate = ".".join(parts)
        if candidate in module_index:
            return candidate
    return None


def _import_targets(
    *,
    module_name: str,
    path: Path,
    tree: ast.AST,
    seed: bool,
    module_index: dict[str, Path],
) -> set[str]:
    package_initializer = path.name == "__init__.py"
    nodes: Iterable[ast.AST] = (
        _package_import_nodes(tree) if package_initializer else ast.walk(tree)
    )
    targets: set[str] = set()
    for node in nodes:
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _from_import_base(
                current_module=module_name,
                package_initializer=package_initializer,
                node=node,
            )
            candidates.append(base)
            candidates.extend(
                f"{base}.{alias.name}"
                for alias in node.names
                if base and alias.name != "*"
            )
        elif isinstance(node, ast.Call) and node.args:
            function_name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else node.func.id
                if isinstance(node.func, ast.Name)
                else ""
            )
            first = node.args[0]
            if (
                function_name == "import_module"
                and isinstance(first, ast.Constant)
                and isinstance(first.value, str)
            ):
                candidates.append(first.value)
        elif seed and isinstance(node, ast.Constant) and isinstance(node.value, str):
            candidates.extend(_DYNAMIC_MODULE_RE.findall(node.value))
        for candidate in candidates:
            target = _local_module_target(candidate, module_index)
            if target is not None:
                targets.add(target)
    return targets


def execution_kernel_relative_paths(package_root: Path) -> tuple[str, ...]:
    """Return the deterministic transitive source manifest for one checkout."""

    root = Path(package_root).resolve()
    module_index = _module_index(root)
    seeds = _seed_modules(root, module_index)
    selected: set[str] = set()
    pending = list(seeds)
    while pending:
        module_name = pending.pop()
        if module_name in selected:
            continue
        path = module_index.get(module_name)
        if path is None:
            raise ExecutionKernelIdentityError(
                f"execution kernel module disappeared: {module_name}"
            )
        selected.add(module_name)

        package_parts = module_name.split(".")
        if path.name != "__init__.py":
            package_parts.pop()
        while package_parts:
            package_name = ".".join(package_parts)
            if package_name in module_index and package_name not in selected:
                pending.append(package_name)
            package_parts.pop()

        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise ExecutionKernelIdentityError(
                f"execution kernel module could not be parsed: {module_name}"
            ) from exc
        pending.extend(
            target
            for target in _import_targets(
                module_name=module_name,
                path=path,
                tree=tree,
                seed=module_name in seeds,
                module_index=module_index,
            )
            if target not in selected
        )

    return tuple(
        sorted(
            module_index[module_name].relative_to(root).as_posix()
            for module_name in selected
        )
    )


def _source_digest(package_root: Path, relative_paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for relative_text in relative_paths:
        relative = Path(relative_text)
        path = package_root / relative
        if path.is_symlink() or not path.is_file():
            raise ExecutionKernelIdentityError(
                f"execution kernel manifest entry is unavailable: {relative_text}"
            )
        relative_bytes = relative.as_posix().encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def build_execution_kernel_identity(
    package_root: Path,
    *,
    relative_paths: Iterable[str] | None = None,
) -> ExecutionKernelIdentity:
    """Build the Host expectation for source plus locked Runner dependencies."""

    root = Path(package_root).resolve()
    manifest = (
        execution_kernel_relative_paths(root)
        if relative_paths is None
        else tuple(str(value) for value in relative_paths)
    )
    if manifest != tuple(sorted(set(manifest))) or any(
        Path(value).is_absolute()
        or ".." in Path(value).parts
        or Path(value).suffix != ".py"
        for value in manifest
    ):
        raise ExecutionKernelIdentityError(
            "execution kernel manifest must contain sorted unique Python paths"
        )
    files_payload = "\n".join(manifest).encode("utf-8")
    lock_path = root / "research_agent" / "runner_image" / "requirements.lock"
    if lock_path.is_symlink() or not lock_path.is_file():
        raise ExecutionKernelIdentityError(
            "execution Runner requirements.lock is unavailable"
        )
    payload: dict[str, str | int] = {
        "schema_version": EXECUTION_KERNEL_IDENTITY_SCHEMA,
        "source_sha256": _source_digest(root, manifest),
        "files_sha256": hashlib.sha256(files_payload).hexdigest(),
        "file_count": len(manifest),
        "requirements_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
    }
    identity_sha256 = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return ExecutionKernelIdentity(
        **payload,
        identity_sha256=identity_sha256,
    )


__all__ = [
    "EXECUTION_KERNEL_IDENTITY_SCHEMA",
    "ExecutionKernelIdentity",
    "ExecutionKernelIdentityError",
    "build_execution_kernel_identity",
    "execution_kernel_relative_paths",
]
