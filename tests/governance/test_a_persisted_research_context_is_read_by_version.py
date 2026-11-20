"""A persisted research context is read only through its versioned owner (S1752).

Runs write typed contexts (V2/V3, with ``materialized_inputs``); the plain V1
model refuses them.  Three rounds of the same defect (B34 append seal, B34b
append authority, the multi-database and replication readers) each parsed a
saved context with ``ResearchContext.model_validate*``.  Every reader now calls
``parse_research_context`` / ``parse_research_context_json`` in
``research_context/typed.py``; this static check refuses a new direct parse,
including through an import alias or a ``TypeAdapter``.  In-memory
constructions of one fixed version are allow-listed by file and model.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_SCANNED = ("src/easyicu", "tools", "benchmarks")
_OWNER = "src/easyicu/research_agent/research_context/typed.py"
_MODELS = frozenset({"ResearchContext", "ResearchContextV2", "ResearchContextV3"})
_PARSERS = frozenset({"model_validate", "model_validate_json", "model_validate_strings",
                      "parse_obj", "parse_raw", "parse_file"})
#: (file, model): a builder that constructs one known version in memory.
_IN_MEMORY = frozenset({
    ("src/easyicu/research_agent/research_context/builder.py", "ResearchContextV3"),
})


def _model_names(tree: ast.AST) -> dict[str, str]:
    names = {model: model for model in _MODELS}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _MODELS and alias.asname:
                    names[alias.asname] = alias.name
    return names


def _model(node: ast.AST, names: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.Attribute) and node.attr in _MODELS:
        return node.attr
    return None


def direct_context_parses(path: Path, text: str) -> list[str]:
    tree = ast.parse(text, filename=str(path))
    names = _model_names(tree)
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        model = None
        if isinstance(func, ast.Attribute) and func.attr in _PARSERS:
            model = _model(func.value, names)
        elif isinstance(func, ast.Name) and func.id == "TypeAdapter" and node.args:
            model = _model(node.args[0], names)
        if model is not None:
            hits.append(f"{model}:{node.lineno}")
    return hits


def _violations() -> list[str]:
    found = []
    for root in _SCANNED:
        for path in sorted((ROOT / root).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if relative == _OWNER or "__pycache__" in path.parts or "node_modules" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            if "ResearchContext" not in text:
                continue
            for hit in direct_context_parses(path, text):
                if (relative, hit.split(":")[0]) not in _IN_MEMORY:
                    found.append(f"{relative}:{hit}")
    return found


def test_no_reader_parses_a_saved_context_outside_the_versioned_owner() -> None:
    assert _violations() == []


def test_the_check_sees_direct_aliased_and_adapter_parses() -> None:
    sample = (
        "from easyicu.research_agent.schema import ResearchContext as Context\n"
        "from pydantic import TypeAdapter\n"
        "a = ResearchContext.model_validate_json(raw)\n"
        "b = schema.ResearchContext.model_validate(payload)\n"
        "c = Context.model_validate(payload)\n"
        "d = TypeAdapter(ResearchContext).validate_python(payload)\n"
        "e = parse_research_context_json(raw)\n"
    )
    assert direct_context_parses(Path("sample.py"), sample) == [
        "ResearchContext:3", "ResearchContext:4", "ResearchContext:5", "ResearchContext:6",
    ]
