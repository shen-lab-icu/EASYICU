"""Explicit V4-only reuse of unchanged, hash-pinned historical contracts.

The original modules remain untouched and are run at the pinned historical
commit.  This projector changes only evaluator namespace and exact authority
coordinate literals; it does not patch legacy modules, remove assertions,
change parametrizations or select tests according to outcomes.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "benchmarks/figure2_canonical9/figure2_versioned_ci_v4.json"
_COUPLED = {"paper_rubric_v3", "scoring", "scoring_inputs", "safety_issuer"}


def policy():
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


class AuthorityProjection(ast.NodeTransformer):
    def __init__(self, declaration):
        self.declaration = declaration
        self.old = declaration["legacy_namespace"]
        self.new = declaration["current_namespace"]
        self.literals = declaration["authority_coordinates"]

    def visit_ImportFrom(self, node):
        if node.level:
            return node
        if node.module == self.old:
            result = []
            for alias in node.names:
                if alias.name in _COUPLED:
                    name = "paper_rubric_v4" if alias.name == "paper_rubric_v3" else alias.name
                    alias = ast.alias(name=name, asname=alias.asname or alias.name)
                    module = self.new
                else:
                    module = self.old
                result.append(ast.copy_location(ast.ImportFrom(module=module, names=[alias], level=0), node))
            return result
        for name in _COUPLED:
            if node.module == self.old + "." + name:
                target = "paper_rubric_v4" if name == "paper_rubric_v3" else name
                node.module = self.new + "." + target
                return node
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in self.literals:
            return ast.copy_location(ast.Constant(value=self.literals[node.value]), node)
        return node

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        node.name = node.name.replace("_v3", "_v4")
        return node

    def visit_Name(self, node):
        node.id = node.id.replace("_v3", "_v4")
        return node


def projected_ast(source, declaration=None):
    return ast.fix_missing_locations(AuthorityProjection(declaration or policy()).visit(ast.parse(source)))


def export_contracts(namespace, original):
    declaration = policy()
    entry = declaration["test_sources"][original]
    source = ROOT / original
    if source.is_symlink() or not source.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError("Figure2 projected contract source is not local regular source")
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise RuntimeError("Figure2 original contract source changed; mapping must be reviewed")
    if Path(namespace["__file__"]).resolve() != (ROOT / entry["successor_test"]).resolve():
        raise RuntimeError("Figure2 projected contract namespace/path is not the declared successor")
    tree = projected_ast(payload.decode("utf-8"), declaration)
    namespace["__figure2_contract_origin__"] = original
    namespace["__figure2_contract_origin_sha256__"] = entry["sha256"]
    namespace["__figure2_contract_projection__"] = "current-v4-authority-only"
    # co_filename identifies the original unchanged contract body. __module__
    # and pytest/JUnit node paths identify the new explicit V4 test namespace.
    exec(compile(tree, str(source), "exec"), namespace)
