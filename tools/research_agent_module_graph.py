#!/usr/bin/env python3
"""Deterministic import-architecture gate for ``easyicu.research_agent``.

The control-plane refactor moves implementation into responsibility subpackages
while preserving legacy import surfaces.  This tool records the import graph
before those moves so a mechanical reorganization cannot silently introduce a
new cycle, drop an archived import target, or erase a literal dynamic import.

Only syntax is inspected; importing research-agent modules would execute provider
and environment setup and would make the baseline dependent on the host machine.

Usage::

    python tools/research_agent_module_graph.py
    python tools/research_agent_module_graph.py \
        --emit tools/arch_baselines/research_agent_module_graph.json
    python tools/research_agent_module_graph.py \
        --diff tools/arch_baselines/research_agent_module_graph.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

TOOL_VERSION = "1.1.0"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_DIR = REPO_ROOT / "src" / "easyicu" / "research_agent"
DEFAULT_PACKAGE_NAME = "easyicu.research_agent"

LEGACY_TARGET_MODULES: Tuple[str, ...] = (
    "easyicu.research_agent.gate_evaluator",
    "easyicu.research_agent.contract_gate",
    "easyicu.research_agent.concept_gate",
    "easyicu.research_agent.concept_audit_execution",
    "easyicu.research_agent.figure_contract_preparation",
    "easyicu.research_agent.publication_figure_execution",
    "easyicu.research_agent.evidence_registration",
    "easyicu.research_agent.deterministic_causal",
    "easyicu.research_agent.deterministic_clustering",
    "easyicu.research_agent.deterministic_cohort_flow",
    "easyicu.research_agent.deterministic_descriptive",
    "easyicu.research_agent.deterministic_missingness",
    "easyicu.research_agent.deterministic_ordinal",
    "easyicu.research_agent.deterministic_robustness",
    "easyicu.research_agent.deterministic_sensitivity",
    "easyicu.research_agent.deterministic_survival",
    "easyicu.research_agent.trajectory_stability_executor",
    "easyicu.research_agent.idea_mining_schema",
    "easyicu.research_agent.idea_mining_pubmed",
    "easyicu.research_agent.idea_scope",
    "easyicu.research_agent.idea_registry",
    "easyicu.research_agent.hypothesis_generator",
    "easyicu.research_agent.idea_mining_data_first",
    "easyicu.research_agent.idea_mining_feasibility_tier",
    "easyicu.research_agent.concept_proposal",
    "easyicu.research_agent.idea_mining",
    "easyicu.research_agent.idea_mining_priorart",
    "easyicu.research_agent.idea_mining_funnel",
    "easyicu.research_agent.idea_mining_extended_feasibility",
    "easyicu.research_agent.idea_mining_eval",
    "easyicu.research_agent.discovery_handoff",
    "easyicu.research_agent.discovery_package",
    "easyicu.research_agent.discovery_story_figure",
    "easyicu.research_agent.pdf_render",
    "easyicu.research_agent.reporting_checklist",
    "easyicu.research_agent.reviewer",
    "easyicu.research_agent.review_artifacts",
    "easyicu.research_agent.display_suite",
    "easyicu.research_agent.article_contract",
    "easyicu.research_agent.bibtex",
    "easyicu.research_agent.latex",
    "easyicu.research_agent.manuscript_post",
    "easyicu.research_agent.provider_budget",
    "easyicu.research_agent.code_preflight",
    "easyicu.research_agent.code_repair",
    "easyicu.research_agent.code_repair_helpers",
    "easyicu.research_agent.repair_reasons",
    "easyicu.research_agent.repair_coordination",
    "easyicu.research_agent.code_patch",
    "easyicu.research_agent.summary_repair",
)


@dataclass(frozen=True)
class ModuleSource:
    """One Python module discovered below the package root."""

    name: str
    path: Path
    relative_path: str
    is_package: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_name_for_path(
    path: Path,
    *,
    package_dir: Path,
    package_name: str,
) -> Tuple[str, bool]:
    relative = path.relative_to(package_dir)
    is_package = relative.name == "__init__.py"
    parts = list(relative.parts[:-1])
    if not is_package:
        parts.append(relative.stem)
    suffix = ".".join(parts)
    return (package_name if not suffix else f"{package_name}.{suffix}", is_package)


def discover_modules(package_dir: Path, package_name: str) -> Dict[str, ModuleSource]:
    """Return every ``*.py`` module below ``package_dir`` in stable name order."""

    package_dir = package_dir.resolve()
    if not (package_dir / "__init__.py").is_file():
        raise ValueError(f"package root has no __init__.py: {package_dir}")

    modules: Dict[str, ModuleSource] = {}
    for path in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        name, is_package = _module_name_for_path(
            path,
            package_dir=package_dir,
            package_name=package_name,
        )
        if name in modules:
            raise ValueError(f"duplicate module name {name!r}")
        modules[name] = ModuleSource(
            name=name,
            path=path,
            relative_path=path.relative_to(package_dir).as_posix(),
            is_package=is_package,
        )
    return dict(sorted(modules.items()))


def _source_package(source: ModuleSource) -> str:
    if source.is_package:
        return source.name
    return source.name.rpartition(".")[0]


def _resolve_import_from_base(
    source: ModuleSource, node: ast.ImportFrom
) -> Optional[str]:
    if node.level == 0:
        return node.module

    package_parts = _source_package(source).split(".")
    parents_to_drop = node.level - 1
    if parents_to_drop >= len(package_parts):
        return None
    if parents_to_drop:
        package_parts = package_parts[:-parents_to_drop]
    if node.module:
        package_parts.extend(node.module.split("."))
    return ".".join(package_parts)


def _nearest_existing_module(
    name: Optional[str], modules: Mapping[str, ModuleSource]
) -> Optional[str]:
    """Resolve ``name`` to the most specific discovered module prefix."""

    candidate = name
    while candidate:
        if candidate in modules:
            return candidate
        candidate = candidate.rpartition(".")[0]
    return None


def _static_edges(
    source: ModuleSource,
    tree: ast.AST,
    modules: Mapping[str, ModuleSource],
    package_name: str,
) -> Set[str]:
    targets: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == package_name or alias.name.startswith(
                    f"{package_name}."
                ):
                    target = _nearest_existing_module(alias.name, modules)
                    if target is not None:
                        targets.add(target)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_import_from_base(source, node)
            if base is None or not (
                base == package_name or base.startswith(f"{package_name}.")
            ):
                continue
            base_target = _nearest_existing_module(base, modules)
            for alias in node.names:
                if alias.name == "*":
                    target = base_target
                else:
                    target = _nearest_existing_module(f"{base}.{alias.name}", modules)
                    if target is None:
                        target = base_target
                if target is not None:
                    targets.add(target)
    return targets


def _dynamic_import_aliases(tree: ast.AST) -> Tuple[Set[str], Set[str]]:
    importlib_names = {"importlib"}
    import_module_names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_names.add(alias.asname or "importlib")
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    import_module_names.add(alias.asname or alias.name)
    return importlib_names, import_module_names


def _resolve_dynamic_target(
    literal: str,
    *,
    source: ModuleSource,
    modules: Mapping[str, ModuleSource],
) -> Optional[str]:
    if literal.startswith("."):
        level = len(literal) - len(literal.lstrip("."))
        remainder = literal[level:]
        synthetic = ast.ImportFrom(module=remainder or None, names=[], level=level)
        resolved = _resolve_import_from_base(source, synthetic)
    else:
        resolved = literal
    return _nearest_existing_module(resolved, modules)


def _dynamic_literal_imports(
    source: ModuleSource,
    tree: ast.AST,
    modules: Mapping[str, ModuleSource],
) -> List[Dict[str, Optional[str]]]:
    importlib_names, import_module_names = _dynamic_import_aliases(tree)
    records: Set[Tuple[str, str, Optional[str]]] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue

        kind: Optional[str] = None
        if isinstance(node.func, ast.Name):
            if node.func.id == "__import__":
                kind = "__import__"
            elif node.func.id in import_module_names:
                kind = "importlib.import_module"
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in importlib_names
        ):
            kind = "importlib.import_module"
        if kind is None:
            continue

        literal = first.value
        records.add(
            (
                kind,
                literal,
                _resolve_dynamic_target(literal, source=source, modules=modules),
            )
        )

    return [
        {"source": source.name, "kind": kind, "target": target, "resolved": resolved}
        for kind, target, resolved in sorted(
            records, key=lambda item: (item[0], item[1], item[2] or "")
        )
    ]


def _literal_all(tree: ast.Module) -> Optional[List[str]]:
    """Return a statically literal module ``__all__``, otherwise ``None``."""

    value: Optional[List[str]] = None
    for node in tree.body:
        candidate: Optional[ast.AST] = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            candidate = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            candidate = node.value
        if candidate is None:
            continue
        try:
            raw = ast.literal_eval(candidate)
        except (ValueError, TypeError):
            return None
        if not isinstance(raw, (list, tuple)) or not all(
            isinstance(item, str) for item in raw
        ):
            return None
        value = list(raw)
    return value


def _strongly_connected_components(
    nodes: Iterable[str], edges: Iterable[Tuple[str, str]]
) -> List[List[str]]:
    """Tarjan SCCs with deterministic traversal and output ordering."""

    adjacency: Dict[str, Set[str]] = {node: set() for node in nodes}
    for source, target in edges:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set())

    index = 0
    indices: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    stack: List[str] = []
    on_stack: Set[str] = set()
    components: List[List[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for target in sorted(adjacency[node]):
            if target not in indices:
                visit(target)
                lowlink[node] = min(lowlink[node], lowlink[target])
            elif target in on_stack:
                lowlink[node] = min(lowlink[node], indices[target])

        if lowlink[node] != indices[node]:
            return
        component: List[str] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        components.append(sorted(component))

    for node in sorted(adjacency):
        if node not in indices:
            visit(node)
    return sorted(components, key=lambda component: tuple(component))


def build_snapshot(
    package_dir: Path = DEFAULT_PACKAGE_DIR,
    package_name: str = DEFAULT_PACKAGE_NAME,
    legacy_targets: Sequence[str] = LEGACY_TARGET_MODULES,
) -> Dict[str, Any]:
    """Build the complete deterministic graph snapshot."""

    modules = discover_modules(package_dir, package_name)
    trees: Dict[str, ast.Module] = {}
    edges: Set[Tuple[str, str]] = set()
    dynamic_records: List[Dict[str, Optional[str]]] = []
    for module_name, source in modules.items():
        try:
            tree = ast.parse(
                source.path.read_text(encoding="utf-8"), filename=str(source.path)
            )
        except (OSError, SyntaxError) as exc:
            raise ValueError(f"cannot parse {source.relative_path}: {exc}") from exc
        trees[module_name] = tree
        edges.update(
            (module_name, target)
            for target in _static_edges(source, tree, modules, package_name)
        )
        for record in _dynamic_literal_imports(source, tree, modules):
            dynamic_records.append(record)
            resolved = record["resolved"]
            if resolved is not None:
                edges.add((module_name, resolved))

    sorted_edges = sorted(edges)
    components = _strongly_connected_components(modules, sorted_edges)
    edge_set = set(sorted_edges)
    cyclic_components = [
        component
        for component in components
        if len(component) > 1 or (component[0], component[0]) in edge_set
    ]

    packages = sorted(name for name, source in modules.items() if source.is_package)
    legacy_surfaces: Dict[str, Dict[str, Any]] = {}
    for target in legacy_targets:
        source = modules.get(target)
        if source is None:
            legacy_surfaces[target] = {"exists": False, "literal_all": None}
        else:
            legacy_surfaces[target] = {
                "exists": True,
                "relative_path": source.relative_path,
                "literal_all": _literal_all(trees[target]),
            }

    module_files = {
        name: source.relative_path for name, source in sorted(modules.items())
    }
    top_level_count = sum(
        1 for source in modules.values() if "/" not in source.relative_path
    )
    largest_scc_size = max(
        (len(component) for component in cyclic_components), default=0
    )
    cyclic_module_count = sum(len(component) for component in cyclic_components)

    return {
        "schema_version": 2,
        "tool_version": TOOL_VERSION,
        "tool_sha256": _sha256(Path(__file__)),
        "package_name": package_name,
        "metrics": {
            "module_count": len(modules),
            "top_level_module_count": top_level_count,
            "package_count": len(packages),
            "edge_count": len(sorted_edges),
            "cyclic_scc_count": len(cyclic_components),
            "cyclic_module_count": cyclic_module_count,
            "largest_scc_size": largest_scc_size,
        },
        "modules": module_files,
        "packages": packages,
        "edges": [list(edge) for edge in sorted_edges],
        "cyclic_sccs": cyclic_components,
        "dynamic_literal_imports": sorted(
            dynamic_records,
            key=lambda record: (
                str(record["source"]),
                str(record["kind"]),
                str(record["target"]),
                str(record["resolved"]),
            ),
        ),
        "legacy_surfaces": legacy_surfaces,
    }


def _dynamic_identity(record: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        str(record.get("source")),
        str(record.get("kind")),
        str(record.get("target")),
    )


def compare_snapshots(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> List[str]:
    """Return fail-closed architecture regressions relative to ``baseline``."""

    errors: List[str] = []
    current_metrics = current.get("metrics", {})
    baseline_metrics = baseline.get("metrics", {})
    # Splitting one giant SCC into several smaller SCCs increases the component
    # count while reducing the actual cyclic burden.  Gate the number of modules
    # participating in cycles and the largest component; report SCC count only.
    for metric in ("cyclic_module_count", "largest_scc_size"):
        before = int(baseline_metrics.get(metric, 0))
        after = int(current_metrics.get(metric, 0))
        if after > before:
            errors.append(f"{metric} increased: {before} -> {after}")

    current_surfaces = current.get("legacy_surfaces", {})
    for module, baseline_surface in baseline.get("legacy_surfaces", {}).items():
        current_surface = current_surfaces.get(module)
        if not isinstance(current_surface, Mapping) or not current_surface.get(
            "exists"
        ):
            errors.append(f"legacy target module disappeared: {module}")
            continue
        baseline_all = baseline_surface.get("literal_all")
        if isinstance(baseline_all, list):
            current_all = current_surface.get("literal_all")
            if not isinstance(current_all, list):
                errors.append(f"legacy target lost literal __all__: {module}")
                continue
            missing = sorted(set(baseline_all) - set(current_all))
            if missing:
                errors.append(
                    f"legacy target {module} lost __all__ symbols: {', '.join(missing)}"
                )

    baseline_dynamic = {
        _dynamic_identity(record)
        for record in baseline.get("dynamic_literal_imports", [])
        if isinstance(record, Mapping)
    }
    current_dynamic = {
        _dynamic_identity(record)
        for record in current.get("dynamic_literal_imports", [])
        if isinstance(record, Mapping)
    }
    for source, kind, target in sorted(baseline_dynamic - current_dynamic):
        errors.append(
            f"dynamic literal import disappeared: {source} {kind}({target!r})"
        )
    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--emit", type=Path, help="write the current snapshot to PATH")
    mode.add_argument(
        "--diff", type=Path, help="compare current state with baseline PATH"
    )
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--package-name", default=DEFAULT_PACKAGE_NAME)
    parser.add_argument(
        "--legacy-target",
        action="append",
        dest="legacy_targets",
        help="legacy module to snapshot (repeatable; defaults to the production set)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    legacy_targets = tuple(args.legacy_targets or LEGACY_TARGET_MODULES)
    current = build_snapshot(args.package_dir, args.package_name, legacy_targets)
    rendered = json.dumps(current, indent=2, sort_keys=True) + "\n"

    if args.emit is not None:
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_text(rendered, encoding="utf-8")
        return 0
    if args.diff is not None:
        baseline = json.loads(args.diff.read_text(encoding="utf-8"))
        errors = compare_snapshots(current, baseline)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        return 0

    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
