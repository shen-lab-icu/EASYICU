"""Small, case-neutral AST facts shared by mechanical gates."""

from __future__ import annotations

import ast
from collections.abc import Collection, Mapping

from ..schema import ValidationFinding

DYNAMIC_NAMESPACE_PRIMITIVES = frozenset(
    {
        "__import__",
        "compile",
        "eval",
        "exec",
        "getattr",
        "globals",
        "import_module",
        "locals",
        "vars",
    }
)
DYNAMIC_NAMESPACE_MUTATORS = frozenset(
    {"__delattr__", "__setattr__", "delattr", "setattr"}
)


def call_name(node: ast.AST) -> str:
    """Return the dotted direct-call name represented by an AST node."""

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return call_name(node.func)
    return ""


def execution_cohort_row_count_findings(
    tree: ast.Module,
) -> list[ValidationFinding]:
    """Reject positive literals used as the current cohort row count.

    ``ResearchContext.cohort.n_stays`` describes the run input before a locked
    cohort producer applies eligibility. The current step can therefore have
    fewer rows. This conservative data-flow check only follows a parquet frame
    loaded from the host-owned ``COHORT_PARQUET`` environment coordinate and a
    direct ``len(frame)`` alias; unrelated table-size assertions are ignored.
    """

    def _cohort_environment_lookup(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "os"
            and node.value.attr == "environ"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value
            in {
                "COHORT_PARQUET",
                "COHORT_PATH",
                "EASYICU_COHORT_PATH",
                "EASYICU_COHORT_PARQUET",
            }
        )

    cohort_path_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
            (
                node.targets[0]
                if isinstance(node, ast.Assign) and node.targets
                else node.target if isinstance(node, ast.AnnAssign) else None
            ),
            ast.Name,
        ):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            value = node.value
            if value is not None and any(
                _cohort_environment_lookup(candidate) for candidate in ast.walk(value)
            ):
                cohort_path_names.add(target.id)

    cohort_frame_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        value = node.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "read_parquet"
            and value.args
        ):
            continue
        source = value.args[0]
        if (
            _cohort_environment_lookup(source)
            or isinstance(source, ast.Name)
            and source.id in cohort_path_names
            or any(
                _cohort_environment_lookup(candidate) for candidate in ast.walk(source)
            )
        ):
            cohort_frame_names.add(targets[0].id)
    if not cohort_frame_names:
        return []

    def _loaded_cohort_len(node: ast.AST) -> bool:
        candidate = node
        if (
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Name)
            and candidate.func.id == "int"
            and len(candidate.args) == 1
        ):
            candidate = candidate.args[0]
        return (
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Name)
            and candidate.func.id == "len"
            and len(candidate.args) == 1
            and isinstance(candidate.args[0], ast.Name)
            and candidate.args[0].id in cohort_frame_names
        )

    count_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if (
            len(targets) == 1
            and isinstance(targets[0], ast.Name)
            and node.value is not None
            and _loaded_cohort_len(node.value)
        ):
            count_names.add(targets[0].id)

    def _is_count_expression(node: ast.AST) -> bool:
        return _loaded_cohort_len(node) or (
            isinstance(node, ast.Name) and node.id in count_names
        )

    findings: list[ValidationFinding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        operands = (node.left, node.comparators[0])
        if _is_count_expression(operands[0]):
            literal = operands[1]
        elif _is_count_expression(operands[1]):
            literal = operands[0]
        else:
            continue
        if not (
            isinstance(literal, ast.Constant)
            and isinstance(literal.value, int)
            and not isinstance(literal.value, bool)
            and literal.value > 0
            and literal.end_lineno is not None
            and literal.end_col_offset is not None
        ):
            continue
        findings.append(
            ValidationFinding(
                validator="mechanical_code_preflight",
                severity="error",
                message=(
                    "The current execution-cohort row count is compared with a "
                    "hard-coded positive integer. Use the host-owned "
                    "EASYICU_COHORT_ROWS runtime coordinate; the initial "
                    "ResearchContext cardinality may precede eligibility."
                ),
                detail={
                    "reason": "execution_cohort_row_count_hardcoded",
                    "line": int(literal.lineno),
                    "column": int(literal.col_offset),
                    "end_line": int(literal.end_lineno),
                    "end_column": int(literal.end_col_offset),
                    "literal": int(literal.value),
                },
            )
        )
    return sorted(
        findings,
        key=lambda finding: (
            int(finding.detail["line"]),
            int(finding.detail["column"]),
        ),
    )


def runtime_context_opaque_level_findings(
    tree: ast.Module,
) -> list[ValidationFinding]:
    """Reject treating the outbound-safe projection as the runtime context.

    Provider prompts expose ``observed_shape.opaque_levels`` so categorical
    literals stay private. The digest-verified local ResearchContext keeps the
    real execution binding under ``observed_domain.levels``.
    """

    def _string_slice(node: ast.AST, value: str) -> ast.Constant | None:
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == value
            and node.slice.end_lineno is not None
            and node.slice.end_col_offset is not None
        ):
            return node.slice
        return None

    findings: list[ValidationFinding] = []
    for node in ast.walk(tree):
        outer_key = _string_slice(node, "opaque_levels")
        if outer_key is None or not isinstance(node, ast.Subscript):
            continue
        inner_key = _string_slice(node.value, "observed_shape")
        if inner_key is None:
            continue
        findings.append(
            ValidationFinding(
                validator="mechanical_code_preflight",
                severity="error",
                message=(
                    "Generated code reads outbound-only "
                    "observed_shape.opaque_levels from the local ResearchContext. "
                    "Use the digest-verified local observed_domain.levels binding; "
                    "never copy private level literals into generated source."
                ),
                detail={
                    "reason": "runtime_context_opaque_levels_projection",
                    "observed_shape_key": {
                        "line": int(inner_key.lineno),
                        "column": int(inner_key.col_offset),
                        "end_line": int(inner_key.end_lineno),
                        "end_column": int(inner_key.end_col_offset),
                    },
                    "opaque_levels_key": {
                        "line": int(outer_key.lineno),
                        "column": int(outer_key.col_offset),
                        "end_line": int(outer_key.end_lineno),
                        "end_column": int(outer_key.end_col_offset),
                    },
                },
            )
        )
    return sorted(
        findings,
        key=lambda finding: (
            int(finding.detail["observed_shape_key"]["line"]),
            int(finding.detail["observed_shape_key"]["column"]),
        ),
    )


def literal_observational_getattr(
    node: ast.AST,
    *,
    protected_names: Collection[str] = (),
) -> bool:
    """Recognize a fixed, non-magic attribute read without granting reflection."""

    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) in {2, 3}
        and not node.keywords
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        return False
    attribute = node.args[1].value
    return bool(
        attribute
        and attribute not in protected_names
        and attribute not in DYNAMIC_NAMESPACE_PRIMITIVES
        and attribute not in DYNAMIC_NAMESPACE_MUTATORS
        and not (attribute.startswith("__") and attribute.endswith("__"))
    )


def contains_literal_provenance_audit_row(
    scope: ast.AST,
    *,
    tree: ast.Module,
    parents: Mapping[ast.AST, ast.AST],
    failure_keys: Collection[str],
) -> bool:
    """Return whether one scope contains a concrete numeric provenance row."""

    expected_function = None if scope is tree else scope

    def nearest_function(node: ast.AST) -> ast.AST | None:
        current = parents.get(node)
        while current is not None and not isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            current = parents.get(current)
        return current

    for node in ast.walk(scope):
        if (
            not isinstance(node, ast.Dict)
            or nearest_function(node) is not expected_function
        ):
            continue
        fields = {
            str(key.value): value
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if all(
            isinstance(fields.get(key), ast.Constant) and fields[key].value is None
            for key in failure_keys
        ):
            continue
        role = fields.get("role")
        if (
            set(failure_keys) <= fields.keys()
            and isinstance(role, ast.Constant)
            and str(role.value).strip().lower() == "audit_only"
        ):
            return True
    return False


__all__ = [
    "DYNAMIC_NAMESPACE_MUTATORS",
    "DYNAMIC_NAMESPACE_PRIMITIVES",
    "call_name",
    "contains_literal_provenance_audit_row",
    "execution_cohort_row_count_findings",
    "literal_observational_getattr",
    "runtime_context_opaque_level_findings",
]
