"""Deterministic completion of value-verifiable figure source bundles.

The repair in this module only re-materializes already loaded typed tables or
already written same-step tables, plus side-effect-free scalar receipts.  It
does not derive new scientific values or infer relationships between tables.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
import textwrap
from typing import Dict, List, Sequence, Set


def find_figure_contract_source_assignment(
    tree: ast.AST,
    *,
    source_value_type: type[ast.AST] | tuple[type[ast.AST], ...],
) -> tuple[ast.Assign, ast.keyword] | None:
    """Return the single supported figure-contract source assignment."""

    candidates: List[tuple[ast.Assign, ast.keyword]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and (
                (
                    isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "make_figure_contract"
                )
                or (
                    isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "make_figure_contract"
                )
            )
        ):
            continue
        source_keywords = [
            keyword
            for keyword in node.value.keywords
            if keyword.arg == "source_data"
            and isinstance(keyword.value, source_value_type)
        ]
        if len(source_keywords) == 1:
            candidates.append((node, source_keywords[0]))
    return candidates[0] if len(candidates) == 1 else None


def _safe_existing_statistic_expression(node: ast.AST) -> bool:
    """Return whether replaying an already-written scalar is side-effect free."""

    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _safe_existing_statistic_expression(node.operand)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"float", "int"}
        and len(node.args) == 1
        and not node.keywords
    ):
        return _safe_existing_statistic_expression(node.args[0])
    return False


def patch_complete_bound_figure_source_bundle(
    code: str,
    *,
    missing_table_names: Sequence[str],
    missing_statistic_ids: Sequence[str],
    invalid_source_names: Sequence[str],
) -> str:
    """Complete a figure bundle from exact loaded frames and scalar receipts."""

    table_names = list(dict.fromkeys(str(value) for value in missing_table_names))
    statistic_ids = list(dict.fromkeys(str(value) for value in missing_statistic_ids))
    invalid_names = list(dict.fromkeys(str(value) for value in invalid_source_names))
    if not table_names and not statistic_ids:
        return code
    if any(
        not value
        or Path(value).name != value
        or Path(value).suffix.lower() not in {".csv", ".parquet"}
        for value in table_names
    ):
        return code
    if any(
        not value or Path(value).name != value or Path(value).suffix.lower() != ".csv"
        for value in invalid_names
    ):
        return code
    if any(
        re.fullmatch(r"same_step:statistic:[A-Za-z0-9_]+", value) is None
        for value in statistic_ids
    ):
        return code
    statistic_names = [value.rsplit(":", 1)[-1] for value in statistic_ids]
    if len(statistic_names) != len(set(statistic_names)):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not any(
        isinstance(node, ast.Import)
        and any(alias.name == "pandas" and alias.asname == "pd" for alias in node.names)
        for node in tree.body
    ):
        return code

    source_assignment = find_figure_contract_source_assignment(
        tree,
        source_value_type=(ast.Constant, ast.Attribute, ast.List, ast.Name),
    )
    if source_assignment is None:
        return code
    contract_statement, source_keyword = source_assignment
    existing_source_expression = ast.unparse(source_keyword.value)

    output_dir_names = {
        keyword.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (
                isinstance(node.func, ast.Name)
                and node.func.id == "save_publication_figure"
            )
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "save_publication_figure"
            )
        )
        for keyword in node.keywords
        if keyword.arg == "out_dir" and isinstance(keyword.value, ast.Name)
    }
    if len(output_dir_names) != 1:
        return code
    output_dir_name = next(iter(output_dir_names))

    static_path_names: Dict[str, str] = {}
    typed_path_names: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.BinOp)
            and isinstance(value.op, ast.Div)
            and isinstance(value.right, ast.Constant)
            and isinstance(value.right.value, str)
            and Path(value.right.value).name == value.right.value
        ):
            static_path_names[target.id] = value.right.value
        if not (
            isinstance(target, (ast.Tuple, ast.List))
            and target.elts
            and isinstance(target.elts[0], ast.Name)
            and isinstance(value, ast.Call)
            and (
                (
                    isinstance(value.func, ast.Name)
                    and value.func.id == "resolve_bound_product"
                )
                or (
                    isinstance(value.func, ast.Attribute)
                    and value.func.attr == "resolve_bound_product"
                )
            )
            and any(
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and re.fullmatch(
                    r"(?:artifact|dataset|model|statistic|table):[A-Za-z0-9_.-]+",
                    argument.value,
                )
                is not None
                for argument in value.args
            )
        ):
            continue
        typed_path_names.add(target.elts[0].id)

    table_candidates: List[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
            if (
                isinstance(target, ast.Name)
                and isinstance(value, ast.Call)
                and value.args
                and isinstance(value.args[0], ast.Name)
                and value.args[0].id in typed_path_names
                and (
                    (
                        isinstance(value.func, ast.Name)
                        and value.func.id in {"load_bound_table", "load_tabular"}
                    )
                    or (
                        isinstance(value.func, ast.Attribute)
                        and value.func.attr in {"load_bound_table", "load_tabular"}
                    )
                )
            ):
                table_candidates.append((node.lineno, value.args[0].id, target.id))
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"to_csv", "to_parquet"}
            and isinstance(node.func.value, ast.Name)
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            continue
        path_name = node.args[0].id
        if path_name in static_path_names:
            table_candidates.append((node.lineno, path_name, node.func.value.id))
    table_candidates = list(
        dict.fromkeys(
            (path_name, frame_name)
            for _lineno, path_name, frame_name in sorted(table_candidates)
        )
    )
    if not table_candidates and table_names:
        return code

    output_candidate_names = {
        static_path_names[path_name]
        for path_name, _frame_name in table_candidates
        if path_name in static_path_names
    }
    materialized_table_names = list(
        dict.fromkeys(
            table_names
            + [value for value in invalid_names if value in output_candidate_names]
        )
    )
    stat_candidates: Dict[str, tuple[str, ast.AST]] = {}
    duplicate_statistics: Set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "write_json")
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "write_json"
                )
            )
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and isinstance(node.args[1], ast.Dict)
        ):
            continue
        payload = {
            key.value: value
            for key, value in zip(node.args[1].keys, node.args[1].values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        name_node = payload.get("name")
        value_node = payload.get("value")
        if not (
            isinstance(name_node, ast.Constant)
            and isinstance(name_node.value, str)
            and isinstance(value_node, ast.AST)
            and _safe_existing_statistic_expression(value_node)
        ):
            continue
        statistic_name = str(name_node.value)
        if statistic_name in stat_candidates:
            duplicate_statistics.add(statistic_name)
            continue
        stat_candidates[statistic_name] = (node.args[0].id, value_node)
    if duplicate_statistics or any(
        name not in stat_candidates for name in statistic_names
    ):
        return code

    indent = " " * int(contract_statement.col_offset)
    marker = "_easyicu_completed_bound_source_files"
    if marker in code:
        return code
    candidate_literal = ",\n".join(
        f"({path_name}, {frame_name})" for path_name, frame_name in table_candidates
    )
    projection = (
        f"{indent}_easyicu_existing_source_files = {existing_source_expression}\n"
        f"{indent}if isinstance(_easyicu_existing_source_files, str):\n"
        f"{indent}    _easyicu_existing_source_files = [_easyicu_existing_source_files]\n"
        f"{indent}else:\n"
        f"{indent}    _easyicu_existing_source_files = list(_easyicu_existing_source_files)\n"
        f"{indent}_easyicu_invalid_source_names = set({invalid_names!r})\n"
        f"{indent}{marker} = [\n"
        f"{indent}    name for name in _easyicu_existing_source_files\n"
        f"{indent}    if name not in _easyicu_invalid_source_names\n"
        f"{indent}]\n"
        f"{indent}_easyicu_required_table_names = set({materialized_table_names!r})\n"
        f"{indent}_easyicu_matched_table_names = set()\n"
        f"{indent}_easyicu_bound_table_candidates = [\n"
        + textwrap.indent(candidate_literal, indent + " " * 4)
        + f"\n{indent}]\n"
        f"{indent}for _easyicu_bound_path, _easyicu_bound_frame in _easyicu_bound_table_candidates:\n"
        f"{indent}    _easyicu_bound_name = _easyicu_bound_path.name\n"
        f"{indent}    if _easyicu_bound_name not in _easyicu_required_table_names:\n"
        f"{indent}        continue\n"
        f"{indent}    if _easyicu_bound_name in _easyicu_matched_table_names:\n"
        f'{indent}        raise RuntimeError("Ambiguous bound table for figure source completion")\n'
        f"{indent}    _easyicu_bound_copy = _easyicu_bound_frame.copy(deep=True).reset_index(drop=True)\n"
        f'{indent}    if {{"source_row_index", "source_table"}} & set(_easyicu_bound_copy.columns):\n'
        f'{indent}        raise RuntimeError("Bound parent already uses reserved figure provenance columns")\n'
        f'{indent}    _easyicu_bound_copy.insert(0, "source_table", _easyicu_bound_name)\n'
        f'{indent}    _easyicu_bound_copy.insert(0, "source_row_index", range(len(_easyicu_bound_copy)))\n'
        f'{indent}    _easyicu_bound_token = "".join(\n'
        f'{indent}        character if character.isalnum() else "_"\n'
        f"{indent}        for character in _easyicu_bound_name\n"
        f'{indent}    ).strip("_")\n'
        f"{indent}    _easyicu_bound_source_name = (\n"
        f'{indent}        f"bound_{{len(_easyicu_matched_table_names):03d}}_{{_easyicu_bound_token}}_source_data.csv"\n'
        f"{indent}    )\n"
        f"{indent}    _easyicu_bound_copy.to_csv(\n"
        f"{indent}        {output_dir_name} / _easyicu_bound_source_name, index=False\n"
        f"{indent}    )\n"
        f"{indent}    {marker}.append(_easyicu_bound_source_name)\n"
        f"{indent}    _easyicu_matched_table_names.add(_easyicu_bound_name)\n"
        f"{indent}if _easyicu_matched_table_names != _easyicu_required_table_names:\n"
        f'{indent}    raise RuntimeError("Could not uniquely materialize every bound figure table")\n'
    )
    for index, statistic_name in enumerate(statistic_names):
        _path_name, value_node = stat_candidates[statistic_name]
        local_name = f"bound_stat_{index:03d}_{statistic_name}_source_data.csv"
        projection += (
            f"{indent}_easyicu_statistic_source = pd.DataFrame([{{\n"
            f'{indent}    "statistic": {statistic_name!r},\n'
            f'{indent}    "value": {ast.unparse(value_node)},\n'
            f"{indent}}}])\n"
            f"{indent}_easyicu_statistic_source.to_csv(\n"
            f"{indent}    {output_dir_name} / {local_name!r}, index=False\n"
            f"{indent})\n"
            f"{indent}{marker}.append({local_name!r})\n"
        )
    projection += "\n"

    lines = code.splitlines(keepends=True)
    line_starts: List[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def absolute_offset(line_number: int, byte_column: int) -> int:
        line = lines[line_number - 1]
        character_column = len(line.encode("utf-8")[:byte_column].decode("utf-8"))
        return line_starts[line_number - 1] + character_column

    replacements: List[tuple[int, int, str]] = [
        (
            absolute_offset(
                source_keyword.value.lineno, source_keyword.value.col_offset
            ),
            absolute_offset(
                source_keyword.value.end_lineno, source_keyword.value.end_col_offset
            ),
            marker,
        ),
        (
            line_starts[contract_statement.lineno - 1],
            line_starts[contract_statement.lineno - 1],
            projection,
        ),
    ]
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == "source_data_files"
                and ast.dump(value) == ast.dump(source_keyword.value)
            ):
                replacements.append(
                    (
                        absolute_offset(value.lineno, value.col_offset),
                        absolute_offset(value.end_lineno, value.end_col_offset),
                        marker,
                    )
                )
    repaired = code
    for start, end, replacement in sorted(
        replacements, key=lambda item: item[0], reverse=True
    ):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired
