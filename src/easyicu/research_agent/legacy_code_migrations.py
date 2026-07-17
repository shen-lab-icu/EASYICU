"""Exact one-way migrations for already-sealed generated code.

These are compatibility shims, not analysis capabilities. New generated code
must follow the current host API contract instead of depending on a migration.
"""

from __future__ import annotations

import ast
import re
import textwrap

_LEGACY_PUBLICATION_HELPER_ADAPTER = textwrap.indent(
    textwrap.dedent("""
        # Use the EasyICU publication helper when its installed signature is known.
        signature = inspect.signature(save_publication_figure)
        kwargs = {}
        positional = []
        for name, parameter in signature.parameters.items():
            if name in {"fig", "figure"}:
                kwargs[name] = fig
            elif name in {"contract", "figure_contract"}:
                kwargs[name] = contract
            elif name in {"out_dir", "output_dir", "directory"}:
                kwargs[name] = out_dir
            elif name in {"stem", "figure_stem", "filename_stem"}:
                kwargs[name] = stem
            elif parameter.default is inspect.Parameter.empty and name not in kwargs:
                if not positional:
                    positional.append(fig)
                elif len(positional) == 1:
                    positional.append(contract)
                elif len(positional) == 2:
                    positional.append(out_dir)
                elif len(positional) == 3:
                    positional.append(stem)

        try:
            save_publication_figure(*positional, **kwargs)
        except TypeError:
            # The explicit exports below remain the same source figure and ensure
            # the requested files exist even if helper signatures differ by version.
            pass
        """).strip("\n"),
    "    ",
)

_DIRECT_PUBLICATION_HELPER_CALL = textwrap.indent(
    textwrap.dedent("""
        # Use the stable host-owned helper API; it owns version compatibility.
        try:
            save_publication_figure(
                fig=fig,
                out_dir=out_dir,
                stem=stem,
                contract=contract,
            )
        except TypeError:
            pass
        """).strip("\n"),
    "    ",
)


def migrate_legacy_publication_helper_adapter_v1(code: str) -> str:
    """Migrate one exact obsolete adapter; all deviations fail closed."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if code.count(_LEGACY_PUBLICATION_HELPER_ADAPTER) != 1:
        return code
    block_offset = code.index(_LEGACY_PUBLICATION_HELPER_ADAPTER)
    block_start_line = code.count("\n", 0, block_offset) + 1
    block_end_line = block_start_line + _LEGACY_PUBLICATION_HELPER_ADAPTER.count("\n")
    executable_signature_call = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "inspect"
        and node.func.attr == "signature"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "save_publication_figure"
        and block_start_line <= int(node.lineno) <= block_end_line
        for node in ast.walk(tree)
    )
    if not executable_signature_call:
        return code
    inspect_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.Import)
        and [(alias.name, alias.asname) for alias in node.names] == [("inspect", None)]
    ]
    helper_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module == "easyicu.research_agent.publication_figures"
        and sum(
            alias.name == "save_publication_figure" and alias.asname is None
            for alias in node.names
        )
        == 1
    ]
    if len(inspect_imports) != 1 or len(helper_imports) != 1:
        return code
    if (
        len(re.findall(r"\binspect\b", code)) != 3
        or len(re.findall(r"\bsave_publication_figure\b", code)) != 3
    ):
        return code
    inspect_import = inspect_imports[0]
    lines = code.splitlines(keepends=True)
    if lines[inspect_import.lineno - 1].strip() != "import inspect":
        return code
    del lines[inspect_import.lineno - 1]
    repaired = "".join(lines).replace(
        _LEGACY_PUBLICATION_HELPER_ADAPTER,
        _DIRECT_PUBLICATION_HELPER_CALL,
        1,
    )
    try:
        repaired_tree = ast.parse(repaired)
    except SyntaxError:
        return code
    from .code_preflight import _builtin_int_binding_is_unmodified

    return repaired if _builtin_int_binding_is_unmodified(repaired_tree) else code


__all__ = ["migrate_legacy_publication_helper_adapter_v1"]
