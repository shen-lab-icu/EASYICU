"""Deterministic application of small coder repair patches."""

from __future__ import annotations

import ast
import json
import re


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


def repair_code_excerpt(code: str, run_log: str, *, char_limit: int = 10_000) -> str:
    """Select imports and the most diagnosis-relevant top-level AST blocks."""

    text = str(code or "")
    if len(text) <= char_limit:
        return text
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text[: char_limit // 2] + "\n# ... omitted ...\n" + text[-char_limit // 2 :]

    tokens = {
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(run_log or ""))
        if token.lower() not in {"error", "script", "column", "finding", "step"}
    }
    lines = text.splitlines(keepends=True)
    blocks: list[tuple[int, int, int]] = []
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
        blocks.append((score, start, end))

    chosen: list[tuple[int, int]] = []
    used = 0
    for _score, start, end in sorted(blocks, key=lambda item: (-item[0], item[1])):
        block_len = sum(len(line) for line in lines[start - 1 : end])
        if chosen and used + block_len > char_limit:
            continue
        chosen.append((start, end))
        used += block_len
        if used >= char_limit * 0.85:
            break
    chosen.sort()
    parts = []
    previous_end = 0
    for start, end in chosen:
        if previous_end and start > previous_end + 1:
            parts.append("# ... unrelated code omitted; exact line blocks preserved ...\n")
        parts.extend(lines[start - 1 : end])
        previous_end = end
    excerpt = "".join(parts)
    return excerpt[:char_limit]


__all__ = ["CodePatchError", "PATCH_FORMAT", "apply_code_patch", "repair_code_excerpt"]
