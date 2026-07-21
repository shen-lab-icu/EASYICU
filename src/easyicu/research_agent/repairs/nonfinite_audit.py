"""Narrow repairs for self-declared non-finite value audits."""

from __future__ import annotations

import ast
from collections.abc import Sequence

from ..schema import ValidationFinding


def _import_alias(tree: ast.Module, module: str) -> str | None:
    aliases = {
        alias.asname or module
        for statement in tree.body
        if isinstance(statement, ast.Import)
        for alias in statement.names
        if alias.name == module
    }
    if len(aliases) != 1:
        return None
    name = next(iter(aliases))
    if any(
        (
            isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, ast.Store)
        )
        or (isinstance(node, ast.arg) and node.arg == name)
        for node in ast.walk(tree)
    ):
        return None
    return name


def _root_name(node: ast.AST) -> str | None:
    current = node
    while True:
        if isinstance(current, ast.Name):
            return current.id
        if isinstance(current, ast.Call):
            current = current.func
            continue
        if isinstance(current, ast.Attribute):
            current = current.value
            continue
        if isinstance(current, ast.Subscript):
            current = current.value
            continue
        return None


def _is_mask_sum(node: ast.AST, mask_name: str) -> bool:
    return any(
        isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Attribute)
        and candidate.func.attr == "sum"
        and isinstance(candidate.func.value, ast.Name)
        and candidate.func.value.id == mask_name
        for candidate in ast.walk(node)
    )


def _declares_nonfinite_output(tree: ast.Module, mask_name: str) -> bool:
    count_names = {
        target.id
        for statement in ast.walk(tree)
        if isinstance(statement, (ast.Assign, ast.AnnAssign))
        for target in (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        if isinstance(target, ast.Name)
        and statement.value is not None
        and _is_mask_sum(statement.value, mask_name)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        entries = {
            key.value: value
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        status = entries.get("status")
        count = entries.get("count")
        if not (
            isinstance(status, ast.Constant)
            and isinstance(status.value, str)
            and "nonfinite" in status.value.lower().replace("-", "")
            and count is not None
        ):
            continue
        if _is_mask_sum(count, mask_name) or (
            isinstance(count, ast.Name) and count.id in count_names
        ):
            return True
    return False


def patch_strict_numeric_nonfinite_audit_conflict(
    code: str,
    *,
    audit_messages: Sequence[str] = (),
    repair_findings: Sequence[ValidationFinding] = (),
) -> str:
    """Preserve non-finite values when code explicitly promises to audit them.

    The repair is deliberately structural and fail-closed.  It applies only
    when one blocking concept finding exists and exactly one code path both:

    * feeds a ``strict_numeric`` result into a derived series; and
    * publishes a named non-finite count that is currently hard-coded from an
      all-False mask.

    Other strict numeric inputs keep their fail-closed behavior.  The audited
    series still rejects lossy string-to-number coercion, while observed
    infinities remain available for the script's own exclusion and audit.
    """

    authorized = bool(audit_messages) or any(
        finding.validator == "llm_concept_auditor" and finding.severity == "error"
        for finding in repair_findings
    )
    if not authorized:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    pandas_alias = _import_alias(tree, "pandas")
    numpy_alias = _import_alias(tree, "numpy")
    if pandas_alias is None or numpy_alias is None:
        return code

    assignments: dict[str, ast.Assign] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assignments[node.targets[0].id] = node

    candidates: list[tuple[ast.Assign, ast.Assign, str, str, ast.AST]] = []
    for mask_name, mask_assignment in assignments.items():
        value = mask_assignment.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == pandas_alias
            and value.func.attr == "Series"
            and value.args
            and isinstance(value.args[0], ast.Constant)
            and value.args[0].value is False
            and _declares_nonfinite_output(tree, mask_name)
        ):
            continue
        index_keyword = next(
            (keyword.value for keyword in value.keywords if keyword.arg == "index"),
            None,
        )
        if not (
            isinstance(index_keyword, ast.Attribute)
            and index_keyword.attr == "index"
            and isinstance(index_keyword.value, ast.Name)
        ):
            continue
        derived_name = index_keyword.value.id
        derived_assignment = assignments.get(derived_name)
        if derived_assignment is None:
            continue
        strict_name = _root_name(derived_assignment.value)
        strict_assignment = assignments.get(strict_name or "")
        if strict_assignment is None:
            continue
        strict_call = strict_assignment.value
        if not (
            isinstance(strict_call, ast.Call)
            and isinstance(strict_call.func, ast.Name)
            and strict_call.func.id == "strict_numeric"
            and len(strict_call.args) >= 2
            and isinstance(strict_call.args[1], ast.Constant)
            and isinstance(strict_call.args[1].value, str)
        ):
            continue
        candidates.append(
            (
                strict_assignment,
                mask_assignment,
                strict_name or "",
                derived_name,
                strict_call.args[0],
            )
        )
    if len(candidates) != 1:
        return code

    strict_assignment, mask_assignment, strict_name, derived_name, source = candidates[
        0
    ]
    if strict_assignment.end_lineno is None or mask_assignment.end_lineno is None:
        return code
    source_text = ast.get_source_segment(code, source)
    if not source_text:
        return code
    raw_name = f"_easyicu_{strict_name}_raw_nonfinite_audit_v1"
    loss_name = f"_easyicu_{strict_name}_coercion_loss_v1"
    all_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)
    }
    if raw_name in all_names or loss_name in all_names:
        return code

    lines = code.splitlines(keepends=True)
    strict_indent = lines[strict_assignment.lineno - 1][: strict_assignment.col_offset]
    body_indent = strict_indent + "    "
    strict_replacement = (
        f"{strict_indent}{raw_name} = ({source_text}).copy()\n"
        f"{strict_indent}{strict_name} = {pandas_alias}.to_numeric("
        f'{raw_name}, errors="coerce")\n'
        f"{strict_indent}{loss_name} = {raw_name}.notna() & "
        f"{strict_name}.isna()\n"
        f"{strict_indent}if int({loss_name}.sum()) > 0:\n"
        f'{body_indent}raise ValueError("lossy numeric coercion in audited input")\n'
    )
    mask_indent = lines[mask_assignment.lineno - 1][: mask_assignment.col_offset]
    mask_replacement = (
        f"{mask_indent}{mask_assignment.targets[0].id} = "
        f"{derived_name}.notna() & ~{numpy_alias}.isfinite({derived_name})\n"
    )
    replacements = [
        (
            strict_assignment.lineno - 1,
            strict_assignment.end_lineno,
            strict_replacement,
        ),
        (
            mask_assignment.lineno - 1,
            mask_assignment.end_lineno,
            mask_replacement,
        ),
    ]
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    repaired = "".join(lines)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_strict_numeric_nonfinite_audit_conflict"]
