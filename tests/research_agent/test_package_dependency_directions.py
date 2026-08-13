"""Package-to-package import DIRECTION gate for ``easyicu.research_agent``.

The module-graph tool guards cycles and the canonical surface, but a
wrong-direction acyclic edge (for example a read-only gate importing an
execution runner) passes it silently — that exact violation shipped in
``gates/contract.py`` until 2026-07-21. This test pins the responsibility
directions from the architecture contract:

* ``execution`` executes already-authorized work — only the top-level entry
  surface (``pipeline.py`` and friends) and ``execution`` itself may import it;
* nothing below the top-level entry surface may import ``pipeline``;
* ``methods`` is a leaf statistical-kernel package.

Any import counts — lazy function-level imports still create the dependency.
If a new edge is genuinely intended, move the shared helper to a leaf package
(``methods``/``contracts``/``robustness``) instead of allowlisting the edge.
"""

from __future__ import annotations

import ast
from pathlib import Path

import easyicu.research_agent as research_agent
from easyicu.research_agent.agents import plan_payload
from easyicu.research_agent.planning import literature_bindings

PACKAGE_ROOT = Path(research_agent.__file__).resolve().parent
PACKAGE_NAME = "easyicu.research_agent"

# Top-level modules are the entry/orchestration surface; they may import
# anything inside the package.
_TOP = "<top>"


def _source_package(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT)
    return relative.parts[0] if len(relative.parts) > 1 else _TOP


def _imported_top_level_targets(path: Path) -> set[str]:
    """Resolve every static import in ``path`` to a research_agent member."""

    relative = path.relative_to(PACKAGE_ROOT)
    # Number of package levels between this file and research_agent itself:
    # <top>/mod.py -> 0, pkg/mod.py -> 1, pkg/sub/mod.py -> 2.
    depth = len(relative.parts) - 1
    targets: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith(PACKAGE_NAME + "."):
                    targets.add(name.split(".")[2])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module and node.module.startswith(PACKAGE_NAME + "."):
                    targets.add(node.module.split(".")[2])
                continue
            # Relative import: level 1 strips the module filename, each extra
            # level strips one package directory, so level == depth + 1 lands
            # exactly on research_agent and the first module segment names a
            # top-level member. A smaller level stays inside the current
            # subpackage; a larger level leaves research_agent (easyicu.*) —
            # both are out of scope here.
            if node.level == depth + 1 and node.module:
                targets.add(node.module.split(".")[0])
            elif node.level == depth + 1 and not node.module:
                # ``from . import X`` at top level / ``from .. import X``
                # inside a subpackage: each alias is a top-level member.
                for alias in node.names:
                    targets.add(alias.name)
    return targets


def _package_edges() -> dict[str, set[str]]:
    edges: dict[str, set[str]] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        source = _source_package(path)
        for target in _imported_top_level_targets(path):
            if target != source:
                edges.setdefault(source, set()).add(target)
    return edges


def test_resolver_detects_the_original_gate_violation(tmp_path) -> None:
    """Negative control: the exact pre-fix import shape must be resolvable.

    ``gates/contract.py`` imported ``from ..execution.runners.<mod> import x``
    (depth 1, level 2). A resolver that misses this shape makes every
    direction assertion below pass vacuously.
    """

    probe = tmp_path / "research_agent" / "gates" / "contract_probe.py"
    probe.parent.mkdir(parents=True)
    probe.write_text(
        "from ..execution.runners.deterministic_robustness import x\n"
        "from ...outside import y\n"
        "from .visual import z\n"
        "from easyicu.research_agent.pipeline import run\n"
        "import easyicu.research_agent.orchestration.resume\n",
        encoding="utf-8",
    )
    relative_depth = 1  # gates/contract_probe.py
    tree = ast.parse(probe.read_text(encoding="utf-8"))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PACKAGE_NAME + "."):
                    targets.add(alias.name.split(".")[2])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module and node.module.startswith(PACKAGE_NAME + "."):
                    targets.add(node.module.split(".")[2])
            elif node.level == relative_depth + 1 and node.module:
                targets.add(node.module.split(".")[0])
    assert targets == {"execution", "pipeline", "orchestration"}


def test_only_the_entry_surface_may_import_execution() -> None:
    """gates/authority/figures/... must never depend on the execution layer."""

    edges = _package_edges()
    # ``agents`` is deliberately allowed: the Coder prompt must describe the
    # ACTUAL validated sandbox allow-list, and that runtime capability
    # snapshot is execution-owned state (``execution/method_capabilities``).
    # Verification/authority layers (gates, authority, audits, figures,
    # methods, ...) stay strictly independent of the execution layer.
    allowed = {_TOP, "execution", "agents"}
    importers = sorted(
        source
        for source, targets in edges.items()
        if "execution" in targets and source not in allowed
    )
    assert importers == [], (
        "Only the entry surface (and the documented agents prompt edge) may "
        "import 'execution'; move the shared helper to a leaf package "
        f"instead. Violations: {importers}"
    )


def test_no_subpackage_imports_the_pipeline_module() -> None:
    edges = _package_edges()
    importers = sorted(
        source
        for source, targets in edges.items()
        if "pipeline" in targets and source != _TOP
    )
    assert importers == [], (
        "No responsibility subpackage may import 'pipeline'; violations: "
        f"{importers}"
    )


def test_read_only_gates_do_not_import_action_layers() -> None:
    """Gates emit findings; they must not depend on execution or orchestration."""

    edges = _package_edges()
    gate_targets = edges.get("gates", set())
    forbidden = sorted(gate_targets & {"execution", "orchestration", "pipeline"})
    assert forbidden == [], (
        f"gates/ imports action-layer packages {forbidden}; gates must stay "
        "read-only consumers of leaf contracts."
    )


def test_planning_does_not_import_agent_or_action_layers() -> None:
    """Planning contracts must be reusable without loading their adapters."""

    edges = _package_edges()
    planning_targets = edges.get("planning", set())
    forbidden = sorted(
        planning_targets & {"agents", "execution", "orchestration", "pipeline"}
    )
    assert forbidden == [], (
        "planning/ must own dependency-neutral contracts instead of importing "
        f"agent adapters or action layers; violations: {forbidden}"
    )
    assert (
        plan_payload.validate_literature_citation_bindings
        is literature_bindings.validate_literature_citation_bindings
    )


def test_methods_package_stays_a_leaf() -> None:
    edges = _package_edges()
    method_targets = edges.get("methods", set())
    forbidden = sorted(
        method_targets
        & {"execution", "pipeline", "gates", "repairs", "orchestration", "figures"}
    )
    assert (
        forbidden == []
    ), f"methods/ is a deterministic-kernel leaf; it must not import {forbidden}."
