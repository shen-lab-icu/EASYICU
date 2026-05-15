"""MCP-compatible tool server.

Model Context Protocol (MCP) lets Claude Desktop, Continue and other
clients invoke server-side tools as if they were local functions.

This module exposes EasyICU research-agent tools at two granularities:

* ``research_agent.run`` — start a pipeline run from a JSON request;
* ``research_agent.list_skills`` — enumerate the registered
  :class:`ClinicalSkill` objects;
* ``research_agent.read_manifest`` — return a past run's manifest by
  ``run_id``.
* ``research_agent.build_context`` / ``list_concepts`` /
  ``describe_concept`` / ``load_concepts`` / ``extract_concept`` /
  ``audit_cohort`` / ``run_validator`` — atomic ICU extraction, concept and validator
  surfaces for external coding agents that do not want the whole
  end-to-end manuscript pipeline.
* ``research_agent.cross_database_concept_availability`` — standardized
  extraction support matrix across EasyICU's public ICU database layer.
* ``research_agent.bind_evidence`` — register an external artefact in the
  EasyICU EvidenceStore so downstream manuscript claims can cite it by
  evidence id.

The dispatch table remains a plain dict for tests and Python callers,
but ``main()`` now speaks the MCP JSON-RPC methods used by desktop
clients: ``initialize``, ``tools/list`` and ``tools/call`` over stdio.
An optional no-dependency SSE transport is provided for clients that
still expect the older MCP SSE shape.
"""

from __future__ import annotations

import argparse
import http.server
import json
import queue
import sys
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .concept_availability import cross_database_concept_availability
from .context import build_research_context
from .evidence import EvidenceStore
from .pipeline import ResearchAgentPipeline
from .skills import list_skills
from .audits.validators import CohortAuditor, ConceptUsageAuditor


PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "easyicu-research-agent", "version": "0.1.0"}


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


def _build_context_from_args(args: Dict[str, Any]):
    cohort = args.get("cohort_path")
    if cohort is None:
        raise ValueError("cohort_path is required")
    return build_research_context(
        research_question=args.get("question") or "Inspect this ICU cohort.",
        cohort=cohort,
        cohort_name=args.get("cohort_name") or "cohort",
        database=args.get("database") or "miiv",
        target_outcome=args.get("target_outcome"),
        cross_database_validation=args.get("cross_database_validation"),
        inclusion_criteria=args.get("inclusion_criteria"),
        exclusion_criteria=args.get("exclusion_criteria"),
        id_columns=args.get("id_columns"),
        time_columns=args.get("time_columns"),
        outcome_columns=args.get("outcome_columns"),
        concept_descriptions=args.get("concept_descriptions"),
    )


def _concept_payload(variable) -> Dict[str, Any]:
    return {
        "name": variable.name,
        "description": variable.description,
        "role": variable.role.value,
        "dtype": variable.dtype,
        "unit": variable.unit,
        "valid_range": variable.valid_range,
        "allowed_aggregations": [a.value for a in variable.allowed_aggregations],
        "aggregation_default": (
            variable.aggregation_default.value
            if variable.aggregation_default is not None
            else None
        ),
        "is_ordinal": variable.is_ordinal,
        "analysis_window": variable.analysis_window,
        "source_concept": variable.source_concept,
        "source_tables": variable.source_tables,
        "pitfalls": variable.pitfalls,
        "clinical_caveats": variable.clinical_caveats,
        "missingness": (
            variable.missingness.model_dump(mode="json")
            if variable.missingness is not None
            else None
        ),
    }


def _tool_build_context(args: Dict[str, Any]) -> Dict[str, Any]:
    ctx = _build_context_from_args(args)
    return ctx.model_dump(mode="json")


def _tool_list_concepts(args: Dict[str, Any]) -> Dict[str, Any]:
    ctx = _build_context_from_args(args)
    return {
        "cohort": ctx.cohort.model_dump(mode="json"),
        "target_outcome": ctx.target_outcome,
        "concepts": [_concept_payload(v) for v in ctx.variables],
    }


def _tool_describe_concept(args: Dict[str, Any]) -> Dict[str, Any]:
    name = args.get("concept_name") or args.get("name")
    if not name:
        return {"error": "concept_name is required"}
    ctx = _build_context_from_args(args)
    variable = ctx.variable(str(name))
    if variable is None:
        return {
            "error": f"concept '{name}' not found",
            "known_concepts": [v.name for v in ctx.variables],
        }
    return {"concept": _concept_payload(variable)}


def _tool_audit_cohort(args: Dict[str, Any]) -> Dict[str, Any]:
    ctx = _build_context_from_args(args)
    findings = CohortAuditor().audit(
        context=ctx,
        cohort_path=Path(args["cohort_path"]),
    )
    return {"findings": [f.model_dump(mode="json") for f in findings]}


def _tool_run_validator(args: Dict[str, Any]) -> Dict[str, Any]:
    validator = str(args.get("validator") or "cohort_auditor")
    ctx = _build_context_from_args(args)
    if validator in {"cohort", "cohort_auditor"}:
        findings = CohortAuditor().audit(
            context=ctx,
            cohort_path=Path(args["cohort_path"]),
        )
    elif validator in {"concept_usage", "concept_usage_auditor"}:
        script_text = args.get("script_text")
        if not isinstance(script_text, str) or not script_text.strip():
            return {"error": "script_text is required for concept_usage_auditor"}
        findings = ConceptUsageAuditor().audit(
            context=ctx,
            script_text=script_text,
        )
    else:
        return {
            "error": f"unknown validator '{validator}'",
            "known": ["cohort_auditor", "concept_usage_auditor"],
        }
    return {
        "validator": validator,
        "findings": [f.model_dump(mode="json") for f in findings],
    }


def _tool_cross_database_concept_availability(args: Dict[str, Any]) -> Dict[str, Any]:
    concepts = args.get("concepts") or args.get("concept_names")
    if isinstance(concepts, str):
        concepts = [concepts]
    if not isinstance(concepts, list) or not concepts:
        return {"error": "concepts must be a non-empty string or string array"}
    databases = args.get("databases")
    if isinstance(databases, str):
        databases = [databases]
    if databases is not None and not isinstance(databases, list):
        return {"error": "databases must be a string array when provided"}
    return {
        "availability": cross_database_concept_availability(
            concepts=[str(c) for c in concepts],
            databases=[str(d) for d in databases] if databases else None,
        )
    }


def _tool_load_concepts(args: Dict[str, Any]) -> Dict[str, Any]:
    """Extract standardized EasyICU concept data through easyicu.load_concepts."""
    concepts = args.get("concepts") or args.get("concept_names")
    if isinstance(concepts, str):
        concepts = [concepts]
    if not isinstance(concepts, list) or not concepts:
        return {"error": "concepts must be a non-empty string or string array"}

    import easyicu as easyicu_pkg

    easyicu_load_concepts = getattr(easyicu_pkg, "load_concepts", None)
    if not callable(easyicu_load_concepts):
        from easyicu.api import load_concepts as easyicu_load_concepts

    kwargs: Dict[str, Any] = {"concepts": [str(c) for c in concepts]}
    passthrough_keys = {
        "patient_ids",
        "database",
        "data_path",
        "interval",
        "win_length",
        "aggregate",
        "keep_components",
        "use_sofa2",
        "merge",
        "r_compatible",
        "dict_path",
        "chunk_size",
        "progress",
        "parallel_workers",
        "concept_workers",
        "parallel_backend",
        "max_patients",
        "limit",
        "sample_strategy",
        "batch_size",
        "memory_efficient",
    }
    for key in sorted(passthrough_keys):
        if key in args:
            kwargs[key] = args[key]

    result = easyicu_load_concepts(**kwargs)
    preview_rows = int(args.get("preview_rows") or 5)
    output_paths = _write_concept_result_if_requested(
        result=result,
        args=args,
        concepts=[str(c) for c in concepts],
    )
    evidence_records = _register_concept_outputs_if_requested(
        output_paths=output_paths,
        args=args,
        concepts=[str(c) for c in concepts],
    )
    return {
        "api": "easyicu.load_concepts",
        "concepts": [str(c) for c in concepts],
        "database": args.get("database"),
        "data_path": args.get("data_path"),
        "summary": _summarise_concept_result(result, preview_rows=preview_rows),
        "output_paths": [str(path) for path in output_paths],
        "evidence": evidence_records,
    }


def _tool_extract_concept(args: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a concept and bind the result to evidence by default."""

    forwarded = dict(args)
    if not (forwarded.get("concepts") or forwarded.get("concept_names")):
        concept = forwarded.get("concept") or forwarded.get("concept_name")
        if concept is not None:
            forwarded["concepts"] = [str(concept)]
    forwarded.setdefault("register_evidence", True)
    return _tool_load_concepts(forwarded)


def _tool_bind_evidence(args: Dict[str, Any]) -> Dict[str, Any]:
    """Register an external artefact without exposing database internals."""
    workdir = Path(args.get("workdir") or "./research_output")
    kind = str(args.get("kind") or "log")
    if kind not in {"table", "figure", "statistic", "log", "code"}:
        return {"error": "kind must be one of table, figure, statistic, log, code"}
    description = str(
        args.get("description")
        or "External artifact registered through the EasyICU MCP evidence tool."
    )
    evidence_id = args.get("evidence_id")
    source_path = args.get("source_path")
    text = args.get("text")
    json_payload = args.get("json_payload")
    payload_modes = sum(
        value is not None for value in (source_path, text, json_payload)
    )
    if payload_modes != 1:
        return {"error": "provide exactly one of source_path, text, or json_payload"}

    aliases = _optional_string_list(args.get("aliases"))
    inputs = _optional_string_list(args.get("inputs"))
    metadata = args.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return {"error": "metadata must be an object when provided"}

    store = EvidenceStore(workdir)
    common = {
        "kind": kind,
        "description": description,
        "produced_by_step": args.get("produced_by_step"),
        "inputs": inputs,
        "evidence_id": str(evidence_id) if evidence_id else None,
        "aliases": aliases,
        "producer": str(args.get("producer") or "mcp_external_agent"),
        "generation_mode": str(args.get("generation_mode") or "external"),
        "metadata": dict(metadata or {}),
    }
    if source_path is not None:
        record = store.register_file(
            source_path=Path(str(source_path)),
            script_evidence_id=args.get("script_evidence_id"),
            prompt_pack_version=args.get("prompt_pack_version"),
            **common,
        )
    elif json_payload is not None:
        record = store.register_json(
            payload=json_payload,
            filename=str(args.get("filename") or "external_artifact.json"),
            prompt_pack_version=args.get("prompt_pack_version"),
            **common,
        )
    else:
        if not isinstance(text, str):
            return {"error": "text must be a string"}
        record = store.register_text(
            text=text,
            filename=str(args.get("filename") or "external_artifact.txt"),
            script_evidence_id=args.get("script_evidence_id"),
            prompt_pack_version=args.get("prompt_pack_version"),
            **common,
        )
    return {"evidence": record.model_dump(mode="json")}


def _summarise_concept_result(result: Any, *, preview_rows: int) -> Dict[str, Any]:
    if isinstance(result, dict):
        return {
            "result_type": "dict",
            "concepts": {
                str(name): _summarise_frame(frame, preview_rows=preview_rows)
                for name, frame in result.items()
            },
        }
    return {
        "result_type": type(result).__name__,
        "frame": _summarise_frame(result, preview_rows=preview_rows),
    }


def _summarise_frame(frame: Any, *, preview_rows: int) -> Dict[str, Any]:
    if not hasattr(frame, "shape") or not hasattr(frame, "columns"):
        return {"type": type(frame).__name__, "repr": repr(frame)[:500]}
    preview = frame.head(max(preview_rows, 0)).to_dict(orient="records")
    return {
        "type": type(frame).__name__,
        "rows": int(frame.shape[0]),
        "columns": [str(c) for c in frame.columns],
        "dtypes": {str(c): str(t) for c, t in frame.dtypes.items()},
        "preview": _jsonable(preview),
    }


def _write_concept_result_if_requested(
    *,
    result: Any,
    args: Dict[str, Any],
    concepts: List[str],
) -> List[Path]:
    should_write = bool(args.get("output_path") or args.get("register_evidence"))
    if not should_write:
        return []
    output_path = args.get("output_path")
    if output_path:
        base = Path(str(output_path))
    else:
        workdir = Path(args.get("workdir") or "./research_output")
        stem = _safe_filename("_".join(concepts[:6])) or "concepts"
        base = workdir / "mcp_concept_extracts" / f"{stem}_{uuid.uuid4().hex[:8]}.parquet"

    if isinstance(result, dict):
        out_dir = base if not base.suffix else base.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        for name, frame in result.items():
            path = out_dir / f"{_safe_filename(str(name)) or 'concept'}.parquet"
            _write_frame(frame, path)
            paths.append(path)
        return paths

    path = base if base.suffix else base.with_suffix(".parquet")
    _write_frame(result, path)
    return [path]


def _register_concept_outputs_if_requested(
    *,
    output_paths: List[Path],
    args: Dict[str, Any],
    concepts: List[str],
) -> List[Dict[str, Any]]:
    if not args.get("register_evidence"):
        return []
    if not output_paths:
        return [{"error": "register_evidence requires writable concept output"}]
    workdir = Path(args.get("workdir") or "./research_output")
    store = EvidenceStore(workdir)
    records = []
    for index, path in enumerate(output_paths):
        evidence_id = args.get("evidence_id")
        if evidence_id and len(output_paths) > 1:
            evidence_id = f"{evidence_id}_{index + 1}"
        record = store.register_file(
            kind="table",
            description=str(
                args.get("description")
                or "Standardized concept table extracted through easyicu.load_concepts."
            ),
            source_path=path,
            evidence_id=str(evidence_id) if evidence_id else None,
            aliases=_optional_string_list(args.get("aliases")),
            producer="easyicu.load_concepts",
            generation_mode="deterministic_extraction",
            metadata={
                "concepts": concepts,
                "database": args.get("database"),
                "data_path": args.get("data_path"),
                "interval": args.get("interval"),
                "win_length": args.get("win_length"),
                "aggregate": args.get("aggregate"),
                **dict(args.get("metadata") or {}),
            },
        )
        records.append(record.model_dump(mode="json"))
    return records


def _write_frame(frame: Any, path: Path) -> None:
    if not hasattr(frame, "to_parquet"):
        raise TypeError(f"Cannot write non-DataFrame concept result: {type(frame).__name__}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _optional_string_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return None


# Public dispatch table — map tool names to handlers.
TOOLS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "research_agent.run": _tool_run,
    "research_agent.list_skills": _tool_list_skills,
    "research_agent.read_manifest": _tool_read_manifest,
    "research_agent.build_context": _tool_build_context,
    "research_agent.list_concepts": _tool_list_concepts,
    "research_agent.describe_concept": _tool_describe_concept,
    "research_agent.audit_cohort": _tool_audit_cohort,
    "research_agent.run_validator": _tool_run_validator,
    "research_agent.load_concepts": _tool_load_concepts,
    "research_agent.extract_concept": _tool_extract_concept,
    "research_agent.cross_database_concept_availability": (
        _tool_cross_database_concept_availability
    ),
    "research_agent.bind_evidence": _tool_bind_evidence,
}


# Tool schemas — mirrored after MCP's tool descriptor format.
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
    {
        "name": "research_agent.build_context",
        "description": (
            "Build a typed ResearchContext for a cohort without running "
            "the full pipeline."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["cohort_path"],
            "properties": {
                "cohort_path": {"type": "string"},
                "question": {"type": "string"},
                "cohort_name": {"type": "string"},
                "database": {"type": "string"},
                "target_outcome": {"type": "string"},
                "inclusion_criteria": {"type": "array", "items": {"type": "string"}},
                "exclusion_criteria": {"type": "array", "items": {"type": "string"}},
                "id_columns": {"type": "array", "items": {"type": "string"}},
                "time_columns": {"type": "array", "items": {"type": "string"}},
                "outcome_columns": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "research_agent.list_concepts",
        "description": (
            "List ICU-aware concept descriptors, roles, missingness, "
            "and pitfalls for a cohort."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["cohort_path"],
            "properties": {
                "cohort_path": {"type": "string"},
                "question": {"type": "string"},
                "cohort_name": {"type": "string"},
                "database": {"type": "string"},
                "target_outcome": {"type": "string"},
            },
        },
    },
    {
        "name": "research_agent.describe_concept",
        "description": (
            "Describe one cohort concept with role, allowed aggregations, "
            "missingness, and ICU caveats."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["cohort_path", "concept_name"],
            "properties": {
                "cohort_path": {"type": "string"},
                "concept_name": {"type": "string"},
                "question": {"type": "string"},
                "cohort_name": {"type": "string"},
                "database": {"type": "string"},
                "target_outcome": {"type": "string"},
            },
        },
    },
    {
        "name": "research_agent.audit_cohort",
        "description": (
            "Run the cohort auditor only, returning validation findings "
            "without executing analysis code."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["cohort_path"],
            "properties": {
                "cohort_path": {"type": "string"},
                "question": {"type": "string"},
                "cohort_name": {"type": "string"},
                "database": {"type": "string"},
                "target_outcome": {"type": "string"},
            },
        },
    },
    {
        "name": "research_agent.run_validator",
        "description": "Run an atomic validator: cohort_auditor or concept_usage_auditor.",
        "inputSchema": {
            "type": "object",
            "required": ["cohort_path", "validator"],
            "properties": {
                "cohort_path": {"type": "string"},
                "validator": {
                    "type": "string",
                    "enum": ["cohort_auditor", "concept_usage_auditor"],
                },
                "script_text": {"type": "string"},
                "question": {"type": "string"},
                "cohort_name": {"type": "string"},
                "database": {"type": "string"},
                "target_outcome": {"type": "string"},
            },
        },
    },
    {
        "name": "research_agent.load_concepts",
        "description": (
            "Call EasyICU's standardized load_concepts extraction API for "
            "one or more concepts, optionally writing the extracted table "
            "and registering it in the EvidenceStore."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["concepts"],
            "properties": {
                "concepts": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
                "patient_ids": {
                    "oneOf": [
                        {"type": "array"},
                        {"type": "object"},
                    ]
                },
                "database": {"type": "string"},
                "data_path": {"type": "string"},
                "interval": {"type": ["string", "null"]},
                "win_length": {"type": ["string", "null"]},
                "aggregate": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "object"},
                        {"type": "null"},
                    ]
                },
                "keep_components": {"type": "boolean"},
                "use_sofa2": {"type": "boolean"},
                "merge": {"type": "boolean"},
                "r_compatible": {"type": "boolean"},
                "dict_path": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
                "chunk_size": {"type": "integer"},
                "progress": {"type": "boolean"},
                "parallel_workers": {"type": "integer"},
                "concept_workers": {"type": "integer"},
                "parallel_backend": {"type": "string"},
                "max_patients": {"type": "integer"},
                "limit": {"type": "integer"},
                "sample_strategy": {"type": "string"},
                "batch_size": {"type": "integer"},
                "memory_efficient": {"type": "boolean"},
                "preview_rows": {"type": "integer"},
                "output_path": {"type": "string"},
                "register_evidence": {"type": "boolean"},
                "workdir": {"type": "string"},
                "evidence_id": {"type": "string"},
                "aliases": {"type": "array", "items": {"type": "string"}},
                "description": {"type": "string"},
                "metadata": {"type": "object"},
            },
        },
    },
    {
        "name": "research_agent.extract_concept",
        "description": (
            "Extract one EasyICU standardized concept through load_concepts "
            "and register the result in EvidenceStore by default."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["concept"],
            "properties": {
                "concept": {"type": "string"},
                "concepts": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
                "patient_ids": {
                    "oneOf": [
                        {"type": "array"},
                        {"type": "object"},
                    ]
                },
                "database": {"type": "string"},
                "data_path": {"type": "string"},
                "interval": {"type": ["string", "null"]},
                "win_length": {"type": ["string", "null"]},
                "aggregate": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "object"},
                        {"type": "null"},
                    ]
                },
                "keep_components": {"type": "boolean"},
                "use_sofa2": {"type": "boolean"},
                "merge": {"type": "boolean"},
                "preview_rows": {"type": "integer"},
                "output_path": {"type": "string"},
                "register_evidence": {"type": "boolean"},
                "workdir": {"type": "string"},
                "evidence_id": {"type": "string"},
                "aliases": {"type": "array", "items": {"type": "string"}},
                "description": {"type": "string"},
                "metadata": {"type": "object"},
            },
        },
    },
    {
        "name": "research_agent.cross_database_concept_availability",
        "description": (
            "Check whether EasyICU's standardized load_concepts extraction "
            "layer can derive requested concepts across public ICU databases "
            "without exposing raw SQL or database schemas."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["concepts"],
            "properties": {
                "concepts": {"type": "array", "items": {"type": "string"}},
                "databases": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "research_agent.bind_evidence",
        "description": (
            "Register a file, text blob, or JSON payload into an EasyICU "
            "EvidenceStore with SHA-256 provenance for external agent outputs."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["workdir", "kind"],
            "properties": {
                "workdir": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["table", "figure", "statistic", "log", "code"],
                },
                "description": {"type": "string"},
                "source_path": {"type": "string"},
                "text": {"type": "string"},
                "json_payload": {"type": "object"},
                "filename": {"type": "string"},
                "evidence_id": {"type": "string"},
                "aliases": {"type": "array", "items": {"type": "string"}},
                "inputs": {"type": "array", "items": {"type": "string"}},
                "produced_by_step": {"type": "string"},
                "producer": {"type": "string"},
                "generation_mode": {"type": "string"},
                "script_evidence_id": {"type": "string"},
                "prompt_pack_version": {"type": "string"},
                "metadata": {"type": "object"},
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


# ---------------------------------------------------------------------------
# MCP JSON-RPC
# ---------------------------------------------------------------------------


def _response(req_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id: Any, code: int, message: str, data: Optional[Any] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }
    if data is not None:
        payload["error"]["data"] = data
    return payload


def _tool_result_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    is_error = "error" in result
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(result, indent=2, ensure_ascii=False, default=str),
            }
        ],
        "isError": is_error,
    }


def _handle_single_jsonrpc(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handle one MCP JSON-RPC request.

    Notifications have no ``id`` and therefore intentionally return
    ``None``. This matters for MCP clients: replying to
    ``notifications/initialized`` is a protocol violation.
    """
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}
    is_notification = "id" not in request

    # Backwards-compatible ad-hoc shape used by the original stub:
    # {"tool": "research_agent.list_skills", "arguments": {...}}
    if method is None and request.get("tool"):
        result = dispatch(str(request.get("tool")), request.get("arguments"))
        if is_notification:
            return result
        return _response(req_id, result)

    if method == "initialize":
        return _response(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        })

    if method in {"notifications/initialized", "initialized"}:
        return None if is_notification else _response(req_id, {})

    if method == "ping":
        return None if is_notification else _response(req_id, {})

    if method == "tools/list":
        return _response(req_id, {"tools": TOOL_SCHEMAS})

    if method == "tools/call":
        if not isinstance(params, dict):
            return _error(req_id, -32602, "Invalid params: expected object")
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not name:
            return _error(req_id, -32602, "Invalid params: tools/call requires string 'name'")
        if not isinstance(arguments, dict):
            return _error(req_id, -32602, "Invalid params: 'arguments' must be an object")
        return _response(req_id, _tool_result_payload(dispatch(name, arguments)))

    # These are optional MCP surfaces. Returning empty lists lets clients
    # complete capability discovery without assuming resources/prompts exist.
    if method == "resources/list":
        return _response(req_id, {"resources": []})
    if method == "prompts/list":
        return _response(req_id, {"prompts": []})

    if is_notification:
        return None
    return _error(req_id, -32601, f"Method not found: {method!r}")


def handle_jsonrpc(payload: Any) -> Optional[Any]:
    """Handle a JSON-RPC request or batch payload.

    Returns a JSON-serialisable response, a list of responses for a
    batch, or ``None`` for notifications.
    """
    if isinstance(payload, list):
        responses = [r for item in payload if (r := handle_jsonrpc(item)) is not None]
        return responses or None
    if not isinstance(payload, dict):
        return _error(None, -32600, "Invalid Request: expected JSON object")
    return _handle_single_jsonrpc(payload)


def _run_stdio() -> int:  # pragma: no cover - covered through handle_jsonrpc
    """Run MCP JSON-RPC over stdin/stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps(_error(None, -32700, "Parse error")), flush=True)
            continue
        response = handle_jsonrpc(req)
        if response is not None:
            print(json.dumps(response, ensure_ascii=False, default=str), flush=True)
    return 0


# ---------------------------------------------------------------------------
# Minimal SSE transport
# ---------------------------------------------------------------------------


class _SSESession:
    def __init__(self) -> None:
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()


_SSE_SESSIONS: Dict[str, _SSESession] = {}


def _make_sse_handler() -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "EasyICUMCP/0.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # Keep stdout clean for JSON-RPC clients; diagnostics go to stderr.
            print(format % args, file=sys.stderr)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path not in {"/sse", "/events"}:
                self.send_error(404, "not found")
                return
            session_id = uuid.uuid4().hex
            session = _SSESession()
            _SSE_SESSIONS[session_id] = session

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            endpoint = f"/messages?session_id={session_id}"
            self._write_sse("endpoint", endpoint)
            try:
                while True:
                    item = session.queue.get()
                    self._write_sse("message", json.dumps(item, ensure_ascii=False, default=str))
            except (BrokenPipeError, ConnectionError):
                pass
            finally:
                _SSE_SESSIONS.pop(session_id, None)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length else b""
            try:
                req = json.loads(body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._write_json(_error(None, -32700, "Parse error"), status=400)
                return

            response = handle_jsonrpc(req)
            if parsed.path in {"/messages", "/message"}:
                session_id = (qs.get("session_id") or qs.get("sessionId") or [""])[0]
                session = _SSE_SESSIONS.get(session_id)
                if session is None:
                    self._write_json({"error": "unknown SSE session"}, status=404)
                    return
                if response is not None:
                    session.queue.put(response)
                self._write_json({"accepted": True})
                return

            if parsed.path in {"/jsonrpc", "/rpc"}:
                self._write_json(response or {})
                return

            self.send_error(404, "not found")

        def _write_sse(self, event: str, data: str) -> None:
            payload = f"event: {event}\ndata: {data}\n\n".encode("utf-8")
            self.wfile.write(payload)
            self.wfile.flush()

        def _write_json(self, payload: Any, *, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _run_sse(host: str, port: int) -> int:  # pragma: no cover
    server = http.server.ThreadingHTTPServer((host, port), _make_sse_handler())
    print(f"easyicu research-agent MCP SSE listening on http://{host}:{port}/sse", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover
    """Run the MCP server.

    Default transport is stdio, which is what Claude Desktop, Continue
    and Cursor typically mount for local MCP servers. ``--transport
    sse`` starts a tiny stdlib HTTP/SSE bridge for clients that still
    use the legacy SSE transport.
    """
    parser = argparse.ArgumentParser(description="EasyICU research-agent MCP server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.transport == "sse":
        return _run_sse(args.host, args.port)
    return _run_stdio()


__all__ = [
    "TOOLS",
    "TOOL_SCHEMAS",
    "dispatch",
    "handle_jsonrpc",
    "main",
]
