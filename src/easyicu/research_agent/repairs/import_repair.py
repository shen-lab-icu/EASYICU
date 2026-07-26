"""Host/package checks used by deterministic import repairs."""

from __future__ import annotations

import ast
import importlib.util
import re

_KNOWN_HOST_HELPER_RELOCATIONS = {
    (
        "easyicu.research_agent.methods.validation",
        "strict_numeric_input",
    ): "easyicu.research_agent.methods.descriptive_inputs",
    (
        "easyicu.research_agent.methods.measurement_provenance_receipt",
        "measurement_provenance_receipt",
    ): "easyicu.research_agent.methods.descriptive_inputs",
}


def host_module_is_available(module_name: str) -> bool:
    """Distinguish sandbox image drift from a hallucinated EasyICU module."""

    try:
        return importlib.util.find_spec(module_name) is not None
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
        return False


def insert_after_imports(source: str, block: str) -> str:
    """Insert ``block`` after the leading import block, never inside one."""

    parsed = ast.parse(source)
    lines = source.splitlines()
    body = list(parsed.body)
    insert_at = 0
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        insert_at = int(body.pop(0).end_lineno or 0)
    for node in body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            break
        insert_at = int(node.end_lineno or node.lineno)
    lines.insert(insert_at, block)
    return "\n".join(lines)


def patch_known_host_helper_import(source: str, diagnostic: str) -> str | None:
    """Relocate one exact known host helper without inventing an implementation.

    This handles both the original bad import and the one-line fail-closed stub
    previously emitted by the generic missing-module repair.  Only a closed
    source-module/helper mapping is accepted.
    """

    lowered = str(diagnostic or "").lower()
    for (old_module, helper), new_module in _KNOWN_HOST_HELPER_RELOCATIONS.items():
        old_import = f"from {old_module} import {helper}"
        new_import = f"from {new_module} import {helper}"
        if (
            old_import in source
            and f"no module named '{old_module}'" in lowered
            and source.count(old_import) == 1
        ):
            repaired = source.replace(old_import, new_import, 1)
        else:
            stub_pattern = re.compile(
                r"# auto-stubs for stripped fake imports\n"
                + rf"def {re.escape(helper)}\(\*args, \*\*kwargs\): "
                + rf"raise NotImplementedError\(\"{re.escape(helper)} from "
                + rf"{re.escape(old_module)} is not available; "
                + r"reimplement inline using numpy/scipy/statsmodels\.\"\)\n?"
            )
            stripped_comment = f"# stripped: import from non-existent {old_module}"
            if not (
                stub_pattern.search(source)
                and stripped_comment in source
                and f"{helper} from {old_module} is not available" in lowered
            ):
                continue
            repaired = stub_pattern.sub(new_import + "\n", source, count=1)
            repaired = repaired.replace(stripped_comment, "", 1)
        try:
            ast.parse(repaired)
        except SyntaxError:
            return None
        return repaired
    return None


__all__ = [
    "host_module_is_available",
    "insert_after_imports",
    "patch_known_host_helper_import",
]
