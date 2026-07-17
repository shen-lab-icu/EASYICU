"""Deterministic application of small coder repair patches."""

from __future__ import annotations

import ast
import json
import re

from .repair_reasons import StructuredRepairMetadata

PATCH_FORMAT = "easyicu.code_patch/1"


class CodePatchError(ValueError):
    """Raised when an LLM patch cannot be applied exactly and safely."""


def _json_payload(raw: str) -> object:
    text = str(raw or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise CodePatchError(f"repair response is not valid patch JSON: {exc}") from exc


def apply_code_patch(code: str, raw_patch: str) -> str:
    """Apply exact, unique replacements and require valid changed Python."""

    payload = _json_payload(raw_patch)
    if not isinstance(payload, dict) or payload.get("format") != PATCH_FORMAT:
        raise CodePatchError(f"repair response must use format {PATCH_FORMAT!r}")
    edits = payload.get("edits")
    if not isinstance(edits, list) or not 1 <= len(edits) <= 12:
        raise CodePatchError("repair patch must contain 1-12 edits")

    patched = str(code)
    seen_old: set[str] = set()
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            raise CodePatchError(f"edit {index} is not an object")
        old = edit.get("old")
        new = edit.get("new")
        expected_count = edit.get("expected_count", 1)
        if not isinstance(old, str) or not old:
            raise CodePatchError(f"edit {index} has an empty old block")
        if not isinstance(new, str) or expected_count != 1:
            raise CodePatchError(f"edit {index} must replace exactly one block")
        if old in seen_old:
            raise CodePatchError(f"edit {index} repeats an earlier old block")
        seen_old.add(old)
        count = patched.count(old)
        if count != 1:
            raise CodePatchError(
                f"edit {index} old block occurs {count} times; exactly one is required"
            )
        patched = patched.replace(old, new, 1)

    if patched == code:
        raise CodePatchError("repair patch made no change")
    try:
        ast.parse(patched)
    except SyntaxError as exc:
        raise CodePatchError(f"repair patch produced invalid Python: {exc}") from exc
    return patched


def looks_like_executable_python(text: str) -> bool:
    """Reject prose/literal payloads even when they contain code-like strings."""

    stripped = str(text or "").strip()
    if not stripped or stripped in {"{}", "[]", "null", "None"}:
        return False
    try:
        payload = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict) and payload.get("format") == PATCH_FORMAT:
        return False
    try:
        tree = ast.parse(stripped)
    except SyntaxError:
        return False
    try:
        compile(tree, "<easyicu-candidate>", "exec")
    except (SyntaxError, ValueError, TypeError):
        return False
    if not tree.body:
        return False
    if all(
        isinstance(node, ast.Expr)
        and isinstance(
            node.value, (ast.Constant, ast.Dict, ast.List, ast.Set, ast.Tuple)
        )
        for node in tree.body
    ):
        return False
    # A complete candidate may intentionally raise on its first pass or may
    # consist of a direct call such as ``main()``/``print(...)``. Those are
    # executable programs whose runtime/output gates must decide success. Raw
    # prose, lone names, and inert arithmetic/literal expressions are not.
    return any(
        not isinstance(node, ast.Expr)
        or isinstance(node.value, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom))
        for node in tree.body
    )


def repair_code_excerpt(
    code: str,
    *,
    repair_metadata: StructuredRepairMetadata | None = None,
    char_limit: int = 10_000,
) -> str:
    """Select exact, diagnosis-relevant source slices within ``char_limit``.

    Host-owned typed coordinates take priority over human validator prose.
    Imports, named helper definitions, and complete sibling statements around
    reported source lines are retained.  Oversized functions are never added
    whole and then truncated, because a partial AST block hides the very
    definitions and guards a minimal patch needs.
    """

    text = str(code or "")
    if len(text) <= char_limit:
        return text
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return (
            text[: char_limit // 2] + "\n# ... omitted ...\n" + text[-char_limit // 2 :]
        )

    metadata = repair_metadata or StructuredRepairMetadata(
        reasons=frozenset(),
        helper_names=frozenset(),
        failure_modes=frozenset(),
        line_anchors=frozenset(),
    )
    structured_terms = {
        *metadata.reasons,
        *metadata.helper_names,
        *metadata.failure_modes,
    }
    token_sources = structured_terms
    stop_tokens = {
        "after",
        "all",
        "and",
        "any",
        "before",
        "but",
        "can",
        "closed",
        "completed",
        "count",
        "detail",
        "does",
        "error",
        "evidence",
        "fail",
        "failed",
        "finding",
        "from",
        "helper",
        "line",
        "message",
        "module",
        "none",
        "not",
        "occurrence",
        "occurrences",
        "reason",
        "script",
        "severity",
        "status",
        "step",
        "that",
        "the",
        "this",
        "true",
        "validator",
        "with",
        "without",
    }
    tokens: set[str] = set()
    for source in token_sources:
        candidates = [source, *str(source).split("_")]
        tokens.update(
            candidate
            for candidate in candidates
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{2,}", candidate)
            and candidate.lower() not in stop_tokens
        )

    lines = text.splitlines(keepends=True)
    blocks: list[tuple[int, int, int, ast.stmt]] = []
    for node in tree.body:
        start = max(1, int(getattr(node, "lineno", 1)))
        end = max(start, int(getattr(node, "end_lineno", start)))
        block = "".join(lines[start - 1 : end])
        score = sum(4 for token in tokens if token in block)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            score += 2
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = getattr(node, "name", "")
            if name in tokens:
                score += 20
        blocks.append((score, start, end, node))

    separator = "# ... unrelated code omitted; exact line blocks preserved ...\n"
    chosen: list[tuple[int, int]] = []

    def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    def _rendered_size(ranges: list[tuple[int, int]]) -> int:
        return sum(
            sum(len(line) for line in lines[start - 1 : end]) for start, end in ranges
        ) + max(0, len(ranges) - 1) * len(separator)

    def _try_add_range(start: int, end: int) -> bool:
        nonlocal chosen
        start = max(1, start)
        end = min(len(lines), max(start, end))
        candidate = _merge_ranges([*chosen, (start, end)])
        if _rendered_size(candidate) > char_limit:
            return False
        chosen = candidate
        return True

    # Imports are required repair context, not low-scoring optional blocks.
    for _score, start, end, node in blocks:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _try_add_range(start, end)

    # Include the exact definition of a helper named by the validator when it
    # fits.  This lets a patch change both a call and its failure contract.
    named_helpers = {
        name
        for name in metadata.helper_names
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
    }
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in named_helpers:
            continue
        _try_add_range(int(node.lineno), int(node.end_lineno or node.lineno))

    def _sibling_context(line_number: int) -> bool:
        suites: list[tuple[int, list[ast.stmt], int]] = []
        for parent in ast.walk(tree):
            for _field, value in ast.iter_fields(parent):
                if not isinstance(value, list):
                    continue
                statements = [item for item in value if isinstance(item, ast.stmt)]
                for index, statement in enumerate(statements):
                    start = int(getattr(statement, "lineno", 0) or 0)
                    end = int(getattr(statement, "end_lineno", start) or start)
                    if start <= line_number <= end:
                        suites.append((end - start, statements, index))
        if not suites:
            return False
        _span, statements, index = min(suites, key=lambda item: item[0])
        for radius in range(4, -1, -1):
            selected = statements[
                max(0, index - radius) : min(len(statements), index + radius + 1)
            ]
            start = int(selected[0].lineno)
            end = int(selected[-1].end_lineno or selected[-1].lineno)
            if _try_add_range(start, end):
                return True
        return False

    for line_number in sorted(metadata.line_anchors):
        _sibling_context(line_number)

    # Runtime stdout/stderr is not a source of repair authority.  When the host
    # has no typed coordinates, select deterministic complete AST statements
    # rather than mining attacker-controlled diagnostic tokens.  Alternate
    # early/late statements and use nested sibling slices when a top-level
    # function is too large to fit whole.
    if not structured_terms and not metadata.line_anchors:
        non_import_blocks = [
            block
            for block in blocks
            if not isinstance(block[3], (ast.Import, ast.ImportFrom))
        ]
        ordered_blocks: list[tuple[int, int, int, ast.stmt]] = []
        left = 0
        right = len(non_import_blocks) - 1
        while left <= right:
            ordered_blocks.append(non_import_blocks[left])
            if right != left:
                ordered_blocks.append(non_import_blocks[right])
            left += 1
            right -= 1
        for _score, start, end, node in ordered_blocks:
            if _try_add_range(start, end):
                continue
            nested_statements = sorted(
                (
                    statement
                    for statement in ast.walk(node)
                    if isinstance(statement, ast.stmt) and statement is not node
                ),
                key=lambda statement: int(getattr(statement, "lineno", 0) or 0),
            )
            if nested_statements:
                _sibling_context(int(nested_statements[0].lineno))
                _sibling_context(int(nested_statements[-1].lineno))

    # With no source line (for example a module-scope finding), take compact
    # neighborhoods around the strongest typed-token blocks.  Neighboring
    # assignments and guards remain visible instead of isolated loops.
    ranked_indices = sorted(
        range(len(blocks)),
        key=lambda index: (-blocks[index][0], blocks[index][1]),
    )
    seeds: list[int] = []
    for index in ranked_indices:
        score, _start, _end, node = blocks[index]
        if score <= 0 or isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if any(abs(index - seed) <= 4 for seed in seeds):
            continue
        added = False
        for radius in range(4, -1, -1):
            neighborhood = blocks[
                max(0, index - radius) : min(len(blocks), index + radius + 1)
            ]
            if _try_add_range(neighborhood[0][1], neighborhood[-1][2]):
                added = True
                break
        if not added:
            # The top-level node is itself too large.  Use a complete sibling
            # slice around its strongest matching nested statement.
            nested = []
            for statement in ast.walk(node):
                if not isinstance(statement, ast.stmt) or statement is node:
                    continue
                start = int(getattr(statement, "lineno", 0) or 0)
                end = int(getattr(statement, "end_lineno", start) or start)
                source = "".join(lines[start - 1 : end])
                nested_score = sum(4 for token in tokens if token in source)
                if nested_score > 0:
                    nested.append((nested_score, end - start, start))
            if nested:
                _nested_score, _nested_span, nested_line = max(
                    nested,
                    key=lambda item: (item[0], -item[1], -item[2]),
                )
                added = _sibling_context(nested_line)
        if added:
            seeds.append(index)
        if len(seeds) >= 3:
            break

    # Use any remaining capacity for other exact high-scoring blocks.
    for index in ranked_indices:
        score, start, end, node = blocks[index]
        if score <= 0 or isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        _try_add_range(start, end)

    if not chosen:
        # Parsed code with no usable diagnostic token: retain complete leading
        # top-level statements rather than slicing through an AST node.
        for _score, start, end, _node in blocks:
            if not _try_add_range(start, end):
                break

    chosen.sort()
    parts: list[str] = []
    previous_end = 0
    for start, end in chosen:
        if previous_end and start > previous_end + 1:
            parts.append(separator)
        parts.extend(lines[start - 1 : end])
        previous_end = end
    return "".join(parts)


__all__ = [
    "CodePatchError",
    "PATCH_FORMAT",
    "apply_code_patch",
    "looks_like_executable_python",
    "repair_code_excerpt",
]
