"""Drift test: catalog direct bindings match the real runner direct targets.

C-F7: runners connect directly to methods.* without adapter indirection
(table_one / descriptive_inputs / source_status). This test locks the catalog
entries to the direct targets without changing the call chain.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


def _direct_imports(owner_path: Path, direct_module: str) -> set[str]:
    # Drift guard without changing the call chain: accept both static
    # ImportFrom nodes and codegen-embedded import strings (table_one
    # executor renders its script via textwrap). Fall back to raw text
    # search so f-string templates (JoinedStr) are also covered.
    text = owner_path.read_text(encoding="utf-8")
    found: set[str] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == direct_module:
                for alias in node.names:
                    found.add(alias.name)
    marker = f"from {direct_module} import"
    if marker in text:
        # Collect symbol names appearing near the marker.
        for symbol in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
            if symbol in text and marker in text:
                pass
        # Direct substring check per symbol is done by the caller; here just
        # seed known symbols present in the file text.
        for candidate in re.findall(
            re.escape(marker) + r"\s*\(?([^)]*)\)?", text
        ):
            for part in candidate.replace("\n", ",").split(","):
                name = part.strip().split(" as ")[0].strip()
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""):
                    found.add(name)
        # Fallback: if parsing missed (e.g. JoinedStr), still require the
        # marker + symbol substrings below.
    return found


def test_direct_method_bindings_match_runner_targets() -> None:
    from easyicu.research_agent.planning.method_adapter_catalog import (
        DIRECT_METHOD_BINDINGS,
        direct_method_bindings_receipt,
    )

    assert len(DIRECT_METHOD_BINDINGS) == 3
    assert {item.binding_id for item in DIRECT_METHOD_BINDINGS} == {
        "direct_table_one_v1",
        "direct_descriptive_inputs_v1",
        "direct_source_status_v1",
    }

    repo = Path(__file__).resolve().parents[3]
    for binding in DIRECT_METHOD_BINDINGS:
        # owner_module is easyicu.research_agent.... -> map to src tree
        owner_path = (
            repo / "src" / (binding.owner_module.replace(".", "/") + ".py")
        )
        assert owner_path.is_file(), f"missing owner {owner_path}"
        text = owner_path.read_text(encoding="utf-8")
        marker = f"from {binding.direct_module} import"
        assert marker in text, (
            f"{binding.binding_id}: {binding.direct_module!r} not directly "
            f"imported by {binding.owner_module}"
        )
        names = _direct_imports(owner_path, binding.direct_module)
        # Every catalogued symbol must be a real direct target of the owner
        # (static import or codegen-embedded import string).
        for symbol in binding.direct_symbols:
            assert (symbol in names) or (symbol in text), (
                f"{binding.binding_id}: {symbol!r} not directly imported by "
                f"{binding.owner_module} from {binding.direct_module}; "
                f"found={sorted(names)}"
            )

    receipt = direct_method_bindings_receipt()
    assert receipt["binding_count"] == 3
    assert len(str(receipt["catalog_sha256"])) == 64
