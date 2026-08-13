"""Science-neutral source repairs for host-issued typed-input bindings."""

from __future__ import annotations

import ast
import re
from typing import Any, Mapping, Sequence

from ..gates.typed_input import resolved_input_shadowed_by_cohort_env_findings
from ..gates.typed_input_symbols import environment_key_node
from ..schema import ValidationFinding


def patch_bound_panel_measurement_status_alias(
    code: str,
    run_log: str,
    *,
    resolved_input_bindings: Mapping[str, Any] | None,
) -> str:
    """Use the exact measurement-status spelling declared by a bound panel.

    Generated clustering code occasionally derives ``<feature>_measured``
    from feature names that already end in ``_first``, yielding the impossible
    ``<feature>_first_measured``.  This repair is allowed only when the runner
    reported that exact missing-column family, the script names one literal
    ``PANEL_KEY``, and that binding's typed product contract contains every
    corresponding ``<feature>_measured`` column.  It never invents a column or
    falls back to a different input.
    """

    missing_match = re.search(
        r"Required (?:measurement-status|declared panel) columns are unavailable"
        r"[^\[]*\[(?P<columns>[^\]]+)\]",
        str(run_log or ""),
    )
    if missing_match is None or not isinstance(resolved_input_bindings, Mapping):
        return code
    missing_columns = re.findall(
        r"['\"]([^'\"]+)['\"]", missing_match.group("columns")
    )
    if not missing_columns or any(
        not column.endswith("_first_measured") for column in missing_columns
    ):
        return code

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    panel_key_candidates: list[str] = []
    feature_assignment_candidates: list[tuple[str, tuple[str, ...]]] = []
    status_assignment_candidates: list[tuple[ast.Assign, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if (
            isinstance(node.value, ast.Constant)
            and target.id == "PANEL_KEY"
            and isinstance(node.value.value, str)
        ):
            panel_key_candidates.append(node.value.value)
        if target.id in {"FEATURE_NAMES", "feature_names"} and isinstance(
            node.value, (ast.List, ast.Tuple)
        ):
            values = tuple(
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant)
                and isinstance(element.value, str)
            )
            if len(values) == len(node.value.elts):
                feature_assignment_candidates.append((target.id, values))
        if target.id not in {"registered_status_columns", "status_columns"}:
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "tuple"
            and len(value.args) == 1
            and isinstance(value.args[0], ast.GeneratorExp)
        ):
            generator = value.args[0]
            container = "tuple"
        elif isinstance(value, ast.ListComp):
            generator = value
            container = "list"
        else:
            continue
        if len(generator.generators) != 1:
            continue
        comprehension = generator.generators[0]
        if not (
            isinstance(comprehension.target, ast.Name)
            and isinstance(comprehension.iter, ast.Name)
            and isinstance(generator.elt, ast.JoinedStr)
            and len(generator.elt.values) == 2
            and isinstance(generator.elt.values[0], ast.FormattedValue)
            and isinstance(generator.elt.values[0].value, ast.Name)
            and generator.elt.values[0].value.id == comprehension.target.id
            and isinstance(generator.elt.values[1], ast.Constant)
            and generator.elt.values[1].value == "_measured"
        ):
            continue
        status_assignment_candidates.append((node, container))

    if (
        len(panel_key_candidates) != 1
        or len(feature_assignment_candidates) != 1
        or len(status_assignment_candidates) != 1
    ):
        return code
    panel_key = panel_key_candidates[0]
    feature_name, feature_names = feature_assignment_candidates[0]
    if not feature_names or any(not value.endswith("_first") for value in feature_names):
        return code
    expected_missing = {f"{value}_measured" for value in feature_names}
    if set(missing_columns) != expected_missing:
        return code
    expected_bound_columns = {
        f"{value.removesuffix('_first')}_measured" for value in feature_names
    }
    binding = resolved_input_bindings.get(panel_key)
    if not isinstance(binding, Mapping):
        return code
    product_contract = binding.get("product_contract")
    bound_columns = product_contract.get("columns") if isinstance(product_contract, Mapping) else None
    if not isinstance(bound_columns, list) or any(
        not isinstance(column, str) for column in bound_columns
    ):
        return code
    if not expected_bound_columns.issubset(set(bound_columns)):
        return code

    status_assignment, container = status_assignment_candidates[0]
    value = status_assignment.value
    if (
        container == "tuple"
        and (
            not isinstance(value, ast.Call)
            or not isinstance(value.args[0], ast.GeneratorExp)
        )
    ) or (container == "list" and not isinstance(value, ast.ListComp)):
        return code
    generator = value.args[0] if container == "tuple" else value
    if (
        len(generator.generators) != 1
        or not isinstance(generator.generators[0].iter, ast.Name)
        or generator.generators[0].iter.id != feature_name
    ):
        return code
    lines = code.splitlines(keepends=True)
    if status_assignment.value.end_lineno is None or status_assignment.value.end_col_offset is None:
        return code
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def absolute_offset(lineno: int, utf8_column: int) -> int:
        line = lines[lineno - 1]
        character_column = len(line.encode("utf-8")[:utf8_column].decode("utf-8"))
        return line_starts[lineno - 1] + character_column

    start = absolute_offset(value.lineno, value.col_offset)
    end = absolute_offset(value.end_lineno, value.end_col_offset)
    element_name = generator.generators[0].target.id
    replacement = (
        f"tuple(f\"{{{element_name}.removesuffix('_first')}}_measured\" "
        f"for {element_name} in {feature_name})"
        if container == "tuple"
        else f"[f\"{{{element_name}.removesuffix('_first')}}_measured\" "
        f"for {element_name} in {feature_name}]"
    )
    repaired = code[:start] + replacement + code[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_resolved_input_consumption_contract_owner(
    code: str,
    run_log: str,
    *,
    resolved_input_bindings: Mapping[str, Any] | None,
) -> str:
    """Read a verified consumption contract from its binding, not its product.

    Resolved-input manifests place ``consumption_contract`` beside
    ``product_contract`` on the typed binding.  This repair only corrects a
    generated script after the exact all-rows runtime failure, when the host
    binding named in that failure proves the top-level contract is ``all_rows``
    and the product contract does not contain a competing nested contract.
    The existing fail-closed all-rows check remains unchanged.
    """

    failure = re.search(
        r"RuntimeError:\s+Input\s+(?P<input_key>\S+)\s+does not have the required "
        r"all_rows contract",
        str(run_log or ""),
        flags=re.IGNORECASE,
    )
    if failure is None:
        return code
    input_key = failure.group("input_key")
    binding = (resolved_input_bindings or {}).get(input_key)
    if not isinstance(binding, Mapping):
        return code
    consumption_contract = binding.get("consumption_contract")
    product_contract = binding.get("product_contract")
    identity_row = binding.get("identity_row")
    if (
        not isinstance(consumption_contract, Mapping)
        or consumption_contract.get("mode") != "all_rows"
        or not isinstance(product_contract, Mapping)
        or "consumption_contract" in product_contract
        or not isinstance(identity_row, Mapping)
        or identity_row.get("declared_kind") != "table"
        or identity_row.get("input_key") != input_key
    ):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    product_owners: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        value = node.value
        binding_name: str | None = None
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.attr == "get"
            and value.args
            and isinstance(value.args[0], ast.Constant)
            and value.args[0].value == "product_contract"
        ):
            binding_name = value.func.value.id
        elif (
            isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Name)
            and isinstance(value.slice, ast.Constant)
            and value.slice.value == "product_contract"
        ):
            binding_name = value.value.id
        if binding_name is not None and binding_name != targets[0].id:
            product_owners.setdefault(targets[0].id, set()).add(binding_name)

    candidates: list[tuple[ast.Name, str]] = []
    for node in ast.walk(tree):
        owner: ast.Name | None = None
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "consumption_contract"
        ):
            owner = node.func.value
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "consumption_contract"
        ):
            owner = node.value
        if owner is None:
            continue
        binding_names = product_owners.get(owner.id, set())
        if len(binding_names) == 1:
            candidates.append((owner, next(iter(binding_names))))
    if len(candidates) != 1:
        return code
    owner, binding_name = candidates[0]
    if not all(
        isinstance(value, int)
        for value in (
            owner.lineno,
            owner.col_offset,
            owner.end_lineno,
            owner.end_col_offset,
        )
    ):
        return code

    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    cursor = 0
    for line in lines:
        line_starts.append(cursor)
        cursor += len(line)

    def _offset(lineno: int, utf8_column: int) -> int:
        line = lines[lineno - 1]
        char_column = len(line.encode("utf-8")[:utf8_column].decode("utf-8"))
        return line_starts[lineno - 1] + char_column

    start = _offset(owner.lineno, owner.col_offset)
    end = _offset(owner.end_lineno, owner.end_col_offset)
    repaired = code[:start] + binding_name + code[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_all_rows_outcome_coordinate_filter(
    code: str,
    run_log: str,
    *,
    resolved_input_bindings: Mapping[str, Any] | None,
) -> str:
    """Keep every row of a target-bound outcome-incidence product.

    A producer may use a reader-facing label in its ``outcome`` column while
    the ResearchContext uses the canonical source coordinate.  A rendering
    consumer must not equate those two namespaces and silently remove every
    row.  This repair is intentionally closed to one exact runtime failure,
    one exact AST filter/guard pair, and a host binding that proves the input
    is the complete ``table:outcome_incidence`` product in ``all_rows`` mode.
    It does not choose an outcome: the unique-label runtime guard verifies
    that the already bound table represents exactly one outcome coordinate.
    """

    if (
        "runtimeerror: no bound outcome-incidence rows for target outcome"
        not in str(run_log or "").lower()
    ):
        return code
    binding = dict((resolved_input_bindings or {}).get("table:outcome_incidence") or {})
    consumption = binding.get("consumption_contract")
    product_contract = binding.get("product_contract")
    identity = binding.get("identity_row")
    if (
        not isinstance(consumption, Mapping)
        or consumption.get("mode") != "all_rows"
        or not isinstance(product_contract, Mapping)
        or "outcome" not in list(product_contract.get("columns") or [])
        or not isinstance(identity, Mapping)
        or identity.get("declared_kind") != "table"
        or identity.get("product") != "outcome_incidence"
        or identity.get("input_key") != "table:outcome_incidence"
    ):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    def _matches_filter(node: ast.Assign) -> str | None:
        if (
            len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
            or not isinstance(node.value, ast.Call)
            or node.value.args
            or node.value.keywords
            or not isinstance(node.value.func, ast.Attribute)
            or node.value.func.attr != "copy"
            or not isinstance(node.value.func.value, ast.Subscript)
            or not isinstance(node.value.func.value.value, ast.Attribute)
            or node.value.func.value.value.attr != "loc"
        ):
            return None
        frame_name = node.targets[0].id
        loc_owner = node.value.func.value.value.value
        predicate = node.value.func.value.slice
        expected = ast.parse(
            f'{frame_name}["outcome"].astype(str).eq(str(target_outcome))',
            mode="eval",
        ).body
        if (
            not isinstance(loc_owner, ast.Name)
            or loc_owner.id != frame_name
            or ast.dump(predicate, include_attributes=False)
            != ast.dump(expected, include_attributes=False)
        ):
            return None
        return frame_name

    def _matches_empty_guard(node: ast.stmt, frame_name: str) -> bool:
        return bool(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute)
            and isinstance(node.test.value, ast.Name)
            and node.test.value.id == frame_name
            and node.test.attr == "empty"
            and len(node.body) == 1
            and isinstance(node.body[0], ast.Raise)
            and isinstance(node.body[0].exc, ast.Call)
            and isinstance(node.body[0].exc.func, ast.Name)
            and node.body[0].exc.func.id == "RuntimeError"
            and len(node.body[0].exc.args) == 1
            and isinstance(node.body[0].exc.args[0], ast.Constant)
            and node.body[0].exc.args[0].value
            == "No bound outcome-incidence rows for target outcome"
        )

    candidates: list[tuple[ast.Assign, ast.If, str]] = []
    for owner in ast.walk(tree):
        body = getattr(owner, "body", None)
        if not isinstance(body, list):
            continue
        for index, statement in enumerate(body[:-1]):
            if not isinstance(statement, ast.Assign):
                continue
            frame_name = _matches_filter(statement)
            guard = body[index + 1]
            if frame_name is not None and _matches_empty_guard(guard, frame_name):
                candidates.append((statement, guard, frame_name))
    if len(candidates) != 1:
        return code
    assignment, guard, frame_name = candidates[0]
    label_name = "_easyicu_bound_outcome_labels"
    if any(
        isinstance(node, ast.Name) and node.id == label_name for node in ast.walk(tree)
    ):
        return code
    if not all(
        isinstance(value, int)
        for value in (
            assignment.lineno,
            assignment.col_offset,
            guard.end_lineno,
            guard.end_col_offset,
        )
    ):
        return code

    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    cursor = 0
    for line in lines:
        line_starts.append(cursor)
        cursor += len(line)

    def _offset(lineno: int, utf8_column: int) -> int:
        line = lines[lineno - 1]
        char_column = len(line.encode("utf-8")[:utf8_column].decode("utf-8"))
        return line_starts[lineno - 1] + char_column

    start = _offset(assignment.lineno, assignment.col_offset)
    end = _offset(guard.end_lineno, guard.end_col_offset)
    indent = " " * int(assignment.col_offset)
    replacement = (
        f'{label_name} = {frame_name}["outcome"].dropna().astype(str).unique().tolist()\n'
        f"{indent}if len({label_name}) != 1:\n"
        f'{indent}    raise RuntimeError("Bound outcome-incidence table must contain exactly one outcome coordinate")\n'
        f"{indent}{frame_name} = {frame_name}.copy()"
    )
    repaired = code[:start] + replacement + code[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_resolved_input_manifest_env(code: str, run_log: str) -> str:
    """Use the host-issued resolved-input manifest path after an exact failure.

    The generated script sometimes derives ``resolved_inputs.json`` from the
    run directory even though the host provides its authoritative path through
    ``EASYICU_RESOLVED_INPUTS_JSON``.  Another narrow failure mode treats that
    path string as the JSON document itself.  A third joins the manifest's
    run-root-relative ``evidence/...`` path to ``EASYICU_EVIDENCE_DIR``, thereby
    producing ``evidence/evidence/...`` inside the sandbox.  All repairs are
    deliberately limited to exact runtime failures and structurally unique AST
    shapes.
    """

    lowered_log = (run_log or "").lower()
    missing_manifest_failure = "resolved input manifest not found:" in lowered_log
    path_parsed_as_json_failure = (
        "easyicu_resolved_inputs_json must contain valid json" in lowered_log
        and "jsondecodeerror:" in lowered_log
    )
    relative_path_wrong_root_failure = (
        "bound typed input file is unavailable: evidence/" in lowered_log
    )
    if not (
        missing_manifest_failure
        or path_parsed_as_json_failure
        or relative_path_wrong_root_failure
    ):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    def env_subscript(node: ast.AST, env_name: str) -> bool:
        return (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "os"
            and node.value.attr == "environ"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == env_name
        )

    if relative_path_wrong_root_failure:
        if not any(
            isinstance(node, ast.Constant)
            and node.value == "EASYICU_RESOLVED_INPUTS_JSON"
            for node in ast.walk(tree)
        ):
            return code

        relative_names: set[str] = set()
        for node in ast.walk(tree):
            if not (
                isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
            ):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                continue
            if any(
                (
                    isinstance(descendant, ast.Call)
                    and isinstance(descendant.func, ast.Attribute)
                    and descendant.func.attr == "get"
                    and descendant.args
                    and isinstance(descendant.args[0], ast.Constant)
                    and descendant.args[0].value == "relative_path"
                )
                or (
                    isinstance(descendant, ast.Subscript)
                    and isinstance(descendant.slice, ast.Constant)
                    and descendant.slice.value == "relative_path"
                )
                for descendant in ast.walk(node.value)
            ):
                relative_names.add(targets[0].id)
        if not relative_names:
            return code

        def evidence_environment_key(node: ast.AST) -> ast.AST | None:
            # Same matcher the gate uses. A repair that recognised a different
            # set of shapes than the gate it satisfies would rewrite code the
            # gate still rejects, or miss code the gate blocks.
            return environment_key_node(node, "EASYICU_EVIDENCE_DIR")

        candidates: list[ast.Constant] = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Div)
                and isinstance(node.right, ast.Name)
                and node.right.id in relative_names
                and isinstance(node.left, ast.Call)
                and isinstance(node.left.func, ast.Name)
                and node.left.func.id == "Path"
                and len(node.left.args) == 1
            ):
                continue
            key = evidence_environment_key(node.left.args[0])
            if key is not None:
                candidates.append(key)
        if (
            len(candidates) != 1
            or candidates[0].end_lineno is None
            or candidates[0].end_col_offset is None
        ):
            return code
        candidate = candidates[0]
        lines = code.splitlines(keepends=True)
        line_starts: list[int] = []
        offset = 0
        for line in lines:
            line_starts.append(offset)
            offset += len(line)

        def absolute_offset(lineno: int, utf8_col: int) -> int:
            line = lines[lineno - 1]
            char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
            return line_starts[lineno - 1] + char_col

        start = absolute_offset(candidate.lineno, candidate.col_offset)
        end = absolute_offset(candidate.end_lineno, candidate.end_col_offset)
        repaired = code[:start] + repr("EASYICU_RUN_DIR") + code[end:]
        try:
            ast.parse(repaired)
        except SyntaxError:
            return code
        return repaired

    if path_parsed_as_json_failure:
        path_imported = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "pathlib"
            and any(
                alias.name == "Path" and alias.asname is None for alias in node.names
            )
            for node in tree.body
        )
        if not path_imported:
            return code
        env_bindings = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and env_subscript(node.value, "EASYICU_RESOLVED_INPUTS_JSON")
        ]
        if len(env_bindings) != 1:
            return code
        binding_name = env_bindings[0].targets[0].id
        loads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "json"
            and node.func.attr == "loads"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == binding_name
            and node.args[0].end_lineno is not None
            and node.args[0].end_col_offset is not None
        ]
        binding_reads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == binding_name
        ]
        if len(loads) != 1 or len(binding_reads) != 1:
            return code
        candidate = loads[0].args[0]
        lines = code.splitlines(keepends=True)
        line_starts: list[int] = []
        offset = 0
        for line in lines:
            line_starts.append(offset)
            offset += len(line)

        def absolute_offset(lineno: int, utf8_col: int) -> int:
            line = lines[lineno - 1]
            char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
            return line_starts[lineno - 1] + char_col

        start = absolute_offset(candidate.lineno, candidate.col_offset)
        end = absolute_offset(candidate.end_lineno, candidate.end_col_offset)
        replacement = f'Path({binding_name}).read_text(encoding="utf-8")'
        repaired = code[:start] + replacement + code[end:]
        try:
            ast.parse(repaired)
        except SyntaxError:
            return code
        return repaired

    def env_path_name(node: ast.AST, env_name: str) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Path"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Subscript)
            and isinstance(node.args[0].value, ast.Attribute)
            and isinstance(node.args[0].value.value, ast.Name)
            and node.args[0].value.value.id == "os"
            and node.args[0].value.attr == "environ"
            and isinstance(node.args[0].slice, ast.Constant)
            and node.args[0].slice.value == env_name
        )

    run_dir_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
        and env_path_name(node.value, "EASYICU_RUN_DIR")
    }
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and isinstance(node.left, ast.Name)
        and node.left.id in run_dir_names
        and isinstance(node.right, ast.Constant)
        and node.right.value == "resolved_inputs.json"
        and node.end_lineno is not None
        and node.end_col_offset is not None
    ]
    if len(candidates) != 1:
        return code
    candidate = candidates[0]
    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    start = absolute_offset(candidate.lineno, candidate.col_offset)
    end = absolute_offset(candidate.end_lineno, candidate.end_col_offset)
    replacement = 'Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"])'
    repaired = code[:start] + replacement + code[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_resolved_input_relative_path_root(
    code: str,
    *,
    repair_findings: Sequence[ValidationFinding],
) -> str:
    """Join host-issued run-relative input paths to ``EASYICU_RUN_DIR``."""

    coordinates: list[tuple[tuple[int, int, int, int], str]] = []
    matching_findings = 0
    for finding in repair_findings:
        detail = finding.detail or {}
        if detail.get("reason") != "resolved_input_relative_path_wrong_root":
            continue
        matching_findings += 1
        occurrences = detail.get("occurrences")
        if not isinstance(occurrences, list) or not occurrences:
            return code
        for occurrence in occurrences:
            if not isinstance(occurrence, dict):
                return code
            values = (
                occurrence.get("line"),
                occurrence.get("column"),
                occurrence.get("end_line"),
                occurrence.get("end_column"),
            )
            if not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in values
            ):
                return code
            kind = occurrence.get("kind", "environment_key")
            if kind not in {"environment_key", "root_parameter"}:
                return code
            coordinates.append(
                (tuple(int(value) for value in values), str(kind))
            )
    if (
        matching_findings != 1
        or not coordinates
        or len(coordinates) != len(set(coordinates))
    ):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    constants = {
        (
            int(node.lineno),
            int(node.col_offset),
            int(node.end_lineno),
            int(node.end_col_offset),
        ): node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and node.value == "EASYICU_EVIDENCE_DIR"
        and node.end_lineno is not None
        and node.end_col_offset is not None
    }
    names = {
        (
            int(node.lineno),
            int(node.col_offset),
            int(node.end_lineno),
            int(node.end_col_offset),
        ): node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.end_lineno is not None
        and node.end_col_offset is not None
    }
    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    replacements: list[tuple[int, int, str]] = []
    for coordinate, kind in coordinates:
        node = (
            constants.get(coordinate)
            if kind == "environment_key"
            else names.get(coordinate)
        )
        if node is None or (
            kind == "root_parameter"
            and not isinstance(node.ctx, ast.Load)
        ):
            return code
        replacements.append(
            (
                absolute_offset(coordinate[0], coordinate[1]),
                absolute_offset(coordinate[2], coordinate[3]),
                (
                    repr("EASYICU_RUN_DIR")
                    if kind == "environment_key"
                    else 'os.environ["EASYICU_RUN_DIR"]'
                ),
            )
        )
    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_resolved_input_cohort_env_shadow(
    code: str,
    *,
    repair_findings: Sequence[ValidationFinding],
) -> str:
    """Keep an exact manifest-bound artifact as the physical typed input."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    authoritative = resolved_input_shadowed_by_cohort_env_findings(tree)
    supplied = [
        finding
        for finding in repair_findings
        if (finding.detail or {}).get("reason")
        == "resolved_typed_input_shadowed_by_cohort_env"
    ]
    if (
        len(authoritative) != 1
        or len(supplied) != 1
        or supplied[0].detail != authoritative[0].detail
    ):
        return code
    occurrences = authoritative[0].detail.get("occurrences")
    if not isinstance(occurrences, list) or not occurrences:
        return code
    calls = {
        (
            int(node.lineno),
            int(node.col_offset),
            int(node.end_lineno),
            int(node.end_col_offset),
        ): node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.end_lineno is not None
        and node.end_col_offset is not None
    }
    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    replacements: list[tuple[int, int, str]] = []
    for occurrence in occurrences:
        if not isinstance(occurrence, dict):
            return code
        values = (
            occurrence.get("line"),
            occurrence.get("column"),
            occurrence.get("end_line"),
            occurrence.get("end_column"),
        )
        replacement_name = occurrence.get("replacement_name")
        if (
            not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in values
            )
            or not isinstance(replacement_name, str)
            or not replacement_name.isidentifier()
        ):
            return code
        coordinate = tuple(int(value) for value in values)
        call = calls.get(coordinate)
        if (
            call is None
            or not isinstance(call.func, ast.Name)
            or call.func.id != "Path"
        ):
            return code
        replacements.append(
            (
                absolute_offset(coordinate[0], coordinate[1]),
                absolute_offset(coordinate[2], coordinate[3]),
                replacement_name,
            )
        )
    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = [
    "patch_all_rows_outcome_coordinate_filter",
    "patch_bound_panel_measurement_status_alias",
    "patch_resolved_input_consumption_contract_owner",
    "patch_resolved_input_manifest_env",
    "patch_resolved_input_cohort_env_shadow",
    "patch_resolved_input_relative_path_root",
]
