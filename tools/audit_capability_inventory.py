#!/usr/bin/env python3
"""Require an explicit disposition for every package-local zero-inbound module."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path


ALLOWED_STATUSES = frozenset(
    {"production_reachable", "experimental", "disabled"}
)

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TEST_REFERENCE_PATTERN = re.compile(
    r"^(tests/[A-Za-z0-9_./-]+\.py)::(test_[A-Za-z0-9_]+)$"
)


@dataclass(frozen=True, slots=True)
class CapabilityRow:
    module: str
    loc: int
    status: str
    owner: str
    activation: str
    tests: str
    proof: str
    review: str

    def covers(self, relative_path: str) -> bool:
        if self.module.endswith("/*"):
            return relative_path.startswith(self.module[:-1])
        if self.module.endswith("/"):
            return relative_path.startswith(self.module)
        return relative_path == self.module


def parse_inventory(path: Path) -> tuple[CapabilityRow, ...]:
    rows: list[CapabilityRow] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 8:
            continue
        module = cells[0].strip("`")
        status = cells[2].strip("`")
        try:
            loc = int(cells[1])
        except ValueError:
            continue
        rows.append(
            CapabilityRow(
                module=module,
                loc=loc,
                status=status,
                owner=cells[3],
                activation=cells[4],
                tests=cells[5],
                proof=cells[6],
                review=cells[7],
            )
        )
    return tuple(rows)


def _current_graph(repo_root: Path) -> dict:
    local_python = repo_root / ".venv" / "bin" / "python"
    python = local_python if local_python.is_file() else Path(sys.executable)
    result = subprocess.run(
        [
            str(python),
            str(repo_root / "tools" / "research_agent_module_graph.py"),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _production_test_reference(
    repo_root: Path,
    raw_reference: str,
) -> str | None:
    """Return a stable finding when a production reachability test is invalid."""

    reference = raw_reference.strip().strip("`")
    match = TEST_REFERENCE_PATTERN.fullmatch(reference)
    if match is None:
        return "must be an exact tests/...py::test_name reference"
    relative_path, test_name = match.groups()
    test_path = repo_root / relative_path
    if not test_path.is_file():
        return f"points to missing test file: {relative_path}"
    source = test_path.read_text(encoding="utf-8")
    if re.search(rf"^def {re.escape(test_name)}\s*\(", source, flags=re.MULTILINE) is None:
        return f"points to missing test function: {reference}"
    return None


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _production_route_proof(
    repo_root: Path,
    *,
    raw_reference: str,
    raw_proof: str,
) -> str | None:
    """Verify that the named test invokes and asserts its declared route."""

    reference = raw_reference.strip().strip("`")
    match = TEST_REFERENCE_PATTERN.fullmatch(reference)
    if match is None:
        return "cannot inspect route proof without an exact test reference"
    proof = raw_proof.strip().strip("`")
    tokens = tuple(item.strip() for item in proof.split(";") if item.strip())
    if not tokens or proof == "-":
        return "must declare at least one call:<public_entrypoint> proof token"
    invalid = [
        token
        for token in tokens
        if not token.startswith(("call:", "trace:")) or not token.split(":", 1)[1]
    ]
    if invalid:
        return f"contains invalid proof tokens: {invalid!r}"
    if not any(token.startswith("call:") for token in tokens):
        return "must declare at least one call:<public_entrypoint> proof token"

    relative_path, test_name = match.groups()
    tree = ast.parse((repo_root / relative_path).read_text(encoding="utf-8"))
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == test_name
    ]
    if len(functions) != 1:
        return f"cannot uniquely parse test function: {reference}"
    function = functions[0]
    call_names = {
        _call_name(node.func)
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    }
    asserted_strings = {
        str(node.value)
        for assertion in ast.walk(function)
        if isinstance(assertion, ast.Assert)
        for node in ast.walk(assertion.test)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    for token in tokens:
        kind, expected = token.split(":", 1)
        if kind == "call" and not any(
            observed == expected or observed.endswith("." + expected)
            for observed in call_names
        ):
            return f"declared public call is absent from test AST: {expected}"
        if kind == "trace" and expected not in asserted_strings:
            return f"declared downstream trace is not asserted by the test: {expected}"
    return None


def zero_inbound_leaf_paths(graph: dict) -> tuple[str, ...]:
    modules = graph.get("modules") or {}
    indegree = {name: 0 for name in modules}
    for edge in graph.get("edges") or ():
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        target = edge[1]
        if target in indegree:
            indegree[target] += 1
    return tuple(
        sorted(
            relative
            for module, relative in modules.items()
            if indegree.get(module) == 0
            and isinstance(relative, str)
            and not relative.endswith("/__init__.py")
            and relative != "__init__.py"
        )
    )


def audit_capability_inventory(
    repo_root: Path,
    *,
    today: date | None = None,
    graph: dict | None = None,
) -> tuple[str, ...]:
    root = repo_root.resolve()
    inventory_path = root / "docs" / "research_agent_capability_inventory.md"
    rows = parse_inventory(inventory_path)
    findings: list[str] = []
    current_date = today or date.today()

    if not rows:
        findings.append("capability inventory has no parseable rows")
        return tuple(findings)

    duplicates = sorted(
        module
        for module in {row.module for row in rows}
        if sum(r.module == module for r in rows) > 1
    )
    findings.extend(
        f"duplicate capability inventory row: {name}" for name in duplicates
    )

    package_root = root / "src" / "easyicu" / "research_agent"
    for row in rows:
        if row.status not in ALLOWED_STATUSES:
            findings.append(f"unknown capability status for {row.module}: {row.status}")
        if not row.owner or not row.activation:
            findings.append(f"capability row lacks owner/precondition: {row.module}")
        if row.module.endswith("/*"):
            exists = (package_root / row.module[:-2]).is_dir()
        elif row.module.endswith("/"):
            exists = (package_root / row.module).is_dir()
        else:
            exists = (package_root / row.module).is_file()
        if not exists:
            findings.append(
                f"capability inventory points to missing path: {row.module}"
            )
        if row.status == "production_reachable":
            if "→" not in row.activation:
                findings.append(
                    "production capability lacks a public-API-to-executor route: "
                    f"{row.module}"
                )
            test_finding = _production_test_reference(root, row.tests)
            if test_finding is not None:
                findings.append(
                    "production capability lacks a valid reachability integration "
                    f"test: {row.module} ({test_finding})"
                )
            else:
                proof_finding = _production_route_proof(
                    root,
                    raw_reference=row.tests,
                    raw_proof=row.proof,
                )
                if proof_finding is not None:
                    findings.append(
                        "production capability reachability test does not prove "
                        f"its declared route: {row.module} ({proof_finding})"
                    )
        if DATE_PATTERN.fullmatch(row.review):
            review_date = date.fromisoformat(row.review)
            if review_date < current_date:
                findings.append(
                    f"capability review is overdue: {row.module} ({row.review})"
                )
        elif row.status != "disabled":
            findings.append(f"capability review date is invalid: {row.module}")

    current_graph = graph if graph is not None else _current_graph(root)
    for relative_path in zero_inbound_leaf_paths(current_graph):
        if not any(row.covers(relative_path) for row in rows):
            findings.append(
                f"zero-inbound module lacks an explicit disposition: {relative_path}"
            )

    return tuple(dict.fromkeys(findings))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    findings = audit_capability_inventory(args.repo_root)
    if findings:
        for finding in findings:
            print(f"ERROR: {finding}")
        return 1
    print("research-agent capability inventory: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
