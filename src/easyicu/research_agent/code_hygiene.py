"""Deterministic pre-flight lints for agent-generated Python scripts.

Small language models (notably qwen3-coder-30b) sometimes emit
``def helper(...)`` near the bottom of an ``analysis.py`` but reference
``helper`` from earlier top-level statements, which triggers
``NameError: name 'helper' is not defined`` at execution time. The
pipeline's LLM self-repair loop can get stuck on this because the
model's "fix" is usually to define the helper *again* (still at the
bottom), not to reorder.

:func:`reorder_forward_references` is a zero-LLM lint that hoists every
top-level ``FunctionDef`` / ``AsyncFunctionDef`` / ``ClassDef`` whose
name is referenced by an earlier top-level statement so that execution
order becomes legal. It runs on the raw source text the Coder agent
produced, immediately before the runner writes ``analysis.py`` to disk.

Design notes:
- Only top-level defs are hoisted. Nested defs and name bindings inside
  other functions are unaffected, because they are only looked up at
  call time and the ``NameError`` pattern we are targeting is strictly
  module-scope.
- Line ranges from :mod:`ast` are used to splice the original source
  text, so comments and formatting that :func:`ast.unparse` would
  discard are preserved.
- The rewrite is idempotent: applying it twice leaves the second call a
  no-op.
- On ``SyntaxError`` the source is returned untouched; the runner will
  then surface the real syntax error to the agent's repair loop.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

__all__ = ["reorder_forward_references", "forward_reference_report"]


_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class _ForwardRef:
    name: str
    def_index: int  # index in ``tree.body``
    first_reference_index: int  # earliest body index that references the name


def _referenced_names(node: ast.AST) -> Set[str]:
    """Collect every identifier loaded anywhere inside ``node``."""
    names: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            names.add(child.id)
    return names


def _find_forward_references(tree: ast.Module) -> List[_ForwardRef]:
    """Return top-level defs that are referenced before they are defined."""
    body = tree.body
    top_defs = {
        node.name: idx
        for idx, node in enumerate(body)
        if isinstance(node, _DEF_TYPES)
    }
    if not top_defs:
        return []

    first_ref_index: dict[str, int] = {}
    for idx, node in enumerate(body):
        if isinstance(node, _DEF_TYPES):
            # References inside a def body are resolved at call time;
            # they do not cause module-import NameErrors, so skip them.
            continue
        for name in _referenced_names(node):
            if name in top_defs and name not in first_ref_index:
                first_ref_index[name] = idx

    forwards: List[_ForwardRef] = []
    for name, def_idx in top_defs.items():
        ref_idx = first_ref_index.get(name)
        if ref_idx is not None and ref_idx < def_idx:
            forwards.append(
                _ForwardRef(
                    name=name,
                    def_index=def_idx,
                    first_reference_index=ref_idx,
                )
            )
    return forwards


def forward_reference_report(source: str) -> List[_ForwardRef]:
    """Diagnostic helper that lists every forward reference without rewriting."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return _find_forward_references(tree)


def _insertion_line(tree: ast.Module, lines: List[str]) -> int:
    """Return the 0-based line index right after docstring + imports.

    The goal is to put hoisted defs in a natural position: below the
    module docstring and all initial imports, but above the first
    statement that uses them. Falls back to 0 if there is no header.
    """
    body = tree.body
    insert_line = 0  # 0-based
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            insert_line = max(insert_line, (node.end_lineno or node.lineno))
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and node is body[0]
        ):
            # Module-level docstring.
            insert_line = max(insert_line, (node.end_lineno or node.lineno))
        else:
            break
    # ``end_lineno`` is 1-based inclusive; we want the index right after it.
    return insert_line


def _extract_def_block(
    node: ast.AST, lines: List[str]
) -> Tuple[int, int, List[str]]:
    """Return (start_line_0based, end_line_0based_exclusive, block_lines)."""
    start = (node.lineno or 1) - 1
    end_inclusive = (node.end_lineno or node.lineno or 1) - 1
    # Capture any trailing blank lines that visually belong to this block,
    # up to but not including the next non-blank line.
    end_exclusive = end_inclusive + 1
    while end_exclusive < len(lines) and lines[end_exclusive].strip() == "":
        end_exclusive += 1
        # Only absorb one trailing blank line; more would steal the gap
        # between this def and the next statement.
        break
    return start, end_exclusive, lines[start:end_exclusive]


def reorder_forward_references(source: str) -> str:
    """Return ``source`` with forward-referenced top-level defs hoisted.

    Returns the input unchanged when there is nothing to fix or when
    the source fails to parse.
    """
    if not source:
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    forwards = _find_forward_references(tree)
    if not forwards:
        return source

    body = tree.body
    lines = source.splitlines(keepends=True)

    # Sort by original body order so we splice them out back-to-front
    # (splicing from the end keeps earlier line indices stable).
    forwards_sorted = sorted(forwards, key=lambda f: f.def_index, reverse=True)

    extracted_blocks: List[Tuple[int, List[str]]] = []  # (def_index, block_lines)
    mutable_lines = list(lines)
    for fwd in forwards_sorted:
        def_node = body[fwd.def_index]
        start, end_exclusive, block_lines = _extract_def_block(def_node, mutable_lines)
        extracted_blocks.append((fwd.def_index, block_lines))
        del mutable_lines[start:end_exclusive]

    # Restore original relative order among the hoisted defs.
    extracted_blocks.sort(key=lambda item: item[0])
    hoisted_lines: List[str] = []
    header = (
        "# --- easyicu code_hygiene: hoisted forward-referenced "
        f"{'definition' if len(extracted_blocks) == 1 else 'definitions'} "
        f"({', '.join(f.name for f in sorted(forwards, key=lambda x: x.def_index))}) "
        "---\n"
    )
    footer = "# --- end hoisted definitions ---\n\n"
    hoisted_lines.append(header)
    for _, block_lines in extracted_blocks:
        hoisted_lines.extend(block_lines)
        if block_lines and not block_lines[-1].endswith("\n"):
            hoisted_lines.append("\n")
    hoisted_lines.append(footer)

    # Recompute insertion line against the mutated lines. We rely on the
    # fact that hoisted defs, being top-level, never overlap with the
    # import/docstring header, so ``_insertion_line`` is stable.
    insertion_line = _insertion_line(tree, lines)
    # ``insertion_line`` from the original tree corresponds to the same
    # logical line in ``mutable_lines`` because we only removed def blocks
    # further down the file.
    rewritten = (
        "".join(mutable_lines[:insertion_line])
        + "".join(hoisted_lines)
        + "".join(mutable_lines[insertion_line:])
    )
    return rewritten
