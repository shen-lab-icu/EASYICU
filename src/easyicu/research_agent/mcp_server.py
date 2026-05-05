"""Minimal MCP-compatible server stub (M4-inspired).

M4 [1] exposes its clinical-research toolkit through Model Context
Protocol (MCP), which lets Claude Desktop, Continue and other MCP
clients invoke server-side tools as if they were local functions.

This module provides a *stub* MCP server that exposes three tools:

* ``research_agent.run`` — start a pipeline run from a JSON request;
* ``research_agent.list_skills`` — enumerate the registered
  :class:`ClinicalSkill` objects;
* ``research_agent.read_manifest`` — return a past run's manifest by
  ``run_id``.

The stub is intentionally protocol-agnostic: the tool dispatch table
is a plain dict of callables. Wiring it up to a specific MCP SDK
(``mcp`` python SDK, FastMCP, etc.) is one ``main()`` away — and we
do not pin those SDKs as a dependency of EasyICU.

References
----------
[1] M4: Infrastructure for AI-Assisted Clinical Research.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .pipeline import ResearchAgentPipeline
from .skills import list_skills


def _tool_run(args: Dict[str, Any]) -> Dict[str, Any]:
    workdir = args.pop("workdir", "./research_output")
    pipeline = ResearchAgentPipeline(workdir=workdir)
    cohort = args.pop("cohort_path", None)
    if cohort is None:
        return {"error": "cohort_path is required"}
    result = pipeline.run(cohort=cohort, **args)
    return result.model_dump()


def _tool_list_skills(args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "skills": [
            {
                "key": s.key,
                "name": s.name,
                "description": s.description,
                "target_outcome": s.target_outcome,
                "primary_predictor": s.primary_predictor,
                "expected_variables": s.expected_variables,
            }
            for s in list_skills()
        ]
    }


def _tool_read_manifest(args: Dict[str, Any]) -> Dict[str, Any]:
    workdir = Path(args.get("workdir", "./research_output"))
    run_id = args.get("run_id")
    if not run_id:
        return {"error": "run_id is required"}
    path = workdir / run_id / "manifest.json"
    if not path.exists():
        return {"error": f"manifest not found at {path}"}
    return json.loads(path.read_text(encoding="utf-8"))


# Public dispatch table — map tool names to handlers.
TOOLS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "research_agent.run": _tool_run,
    "research_agent.list_skills": _tool_list_skills,
    "research_agent.read_manifest": _tool_read_manifest,
}


# Tool schemas — mirrored after MCP's tool descriptor format. A real
# MCP server adapter walks this dict and registers each entry.
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "research_agent.run",
        "description": "Run an ICU-aware research-agent pipeline against a cohort parquet.",
        "inputSchema": {
            "type": "object",
            "required": ["question", "cohort_path"],
            "properties": {
                "question": {"type": "string"},
                "cohort_path": {"type": "string"},
                "workdir": {"type": "string"},
                "cohort_name": {"type": "string"},
                "database": {"type": "string"},
                "target_outcome": {"type": "string"},
                "cross_database_validation": {"type": "array", "items": {"type": "string"}},
                "inclusion_criteria": {"type": "array", "items": {"type": "string"}},
                "exclusion_criteria": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "research_agent.list_skills",
        "description": "Enumerate the registered ClinicalSkill recipes.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "research_agent.read_manifest",
        "description": "Read a past run's manifest by run_id.",
        "inputSchema": {
            "type": "object",
            "required": ["run_id"],
            "properties": {
                "run_id": {"type": "string"},
                "workdir": {"type": "string"},
            },
        },
    },
]


def dispatch(tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run a tool by name. Returns either the tool's result dict or an error."""
    arguments = dict(arguments or {})
    fn = TOOLS.get(tool_name)
    if fn is None:
        return {"error": f"unknown tool '{tool_name}'", "known": list(TOOLS)}
    try:
        return fn(arguments)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> int:  # pragma: no cover
    """Tiny stdin/stdout JSON-RPC loop, useful for ad-hoc testing.

    A real MCP adapter would replace this with the official SDK's
    server bootstrap. The body is intentionally trivial.
    """
    import sys
    print(json.dumps({"tools": [t["name"] for t in TOOL_SCHEMAS]}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"error": "invalid JSON"}), flush=True)
            continue
        result = dispatch(req.get("tool", ""), req.get("arguments"))
        print(json.dumps(result, default=str), flush=True)
    return 0


__all__ = ["TOOLS", "TOOL_SCHEMAS", "dispatch", "main"]
