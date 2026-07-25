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
import ipaddress
import json
import os
import queue
import re
import secrets
import sys
import threading
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .concept_availability import (
    concept_database_availability_from_load_record,
    cross_database_concept_availability,
)
from .research_context.builder import build_research_context
from .research_context.outbound import (
    outbound_safe_context_payload,
    project_outbound_records,
)
from .authority.evidence_store import EvidenceStore
from .gates.figure_egress import (
    RESERVED_PRIVACY_METADATA_KEYS,
    TRUSTED_FIGURE_PRODUCERS,
)
from .providers.llm import OpenAIClient
from .pipeline import ResearchAgentPipeline
from .providers import ProviderConfigurationError, build_provider_client
from .skills import list_skills
from .audits.validators import CohortAuditor, ConceptUsageAuditor
from .mcp_policy import (
    MCP_AUDIT_ROOT_ENV,
    MCP_PATIENT_DATA_TOKEN_ENV,
    MIN_NON_MISSING_FOR_COLUMN_STATS,
    SCOPE_BIND_EVIDENCE,
    SCOPE_METADATA,
    SCOPE_READ_INTERNAL_CONTEXT,
    SCOPE_READ_PATIENT_DATA,
    SCOPE_RUN_PIPELINE,
    SCOPE_WRITE_ARTIFACTS,
    DisclosurePolicy,
    MCPAuthorizationError,
    MCPPathError,
    granted_scopes,
    patient_data_audit_payload,
    patient_data_audit_root,
    process_scopes,
    scope_override,
    path_digest as _path_digest,
    require_scope,
    resolve_within_roots,
    summarise_frame,
)

PROTOCOL_VERSION = "2024-11-05"
MCP_BEARER_TOKEN_ENV = "EASYICU_MCP_BEARER_TOKEN"
DEFAULT_HTTP_MAX_BODY_BYTES = 1024 * 1024


def _server_version() -> str:
    """Report the installed package version rather than a frozen literal."""

    try:
        from importlib.metadata import version

        return version("easyicu")
    except Exception:  # pragma: no cover - source checkout without metadata
        return "0+unknown"


SERVER_INFO = {"name": "easyicu-research-agent", "version": _server_version()}


def _provider_configuration_error_payload(
    error: ProviderConfigurationError,
) -> Dict[str, str]:
    """Render the canonical provider failure in the MCP response contract."""

    if error.issue == "missing_openrouter_key":
        message = "OPENROUTER_API_KEY is required for provider=openrouter"
        error_code = "llm_configuration_required"
    elif error.issue == "missing_openai_key":
        message = (
            "OPENAI_API_KEY is required for provider=openai unless base_url "
            "is a local OpenAI-compatible server"
        )
        error_code = "llm_configuration_required"
    elif error.issue in {
        "invalid_openai_base_url_override",
        "openrouter_base_url_override",
    }:
        message = str(error)
        error_code = "llm_configuration_invalid"
    elif error.issue == "external_llm_not_authorized":
        message = str(error)
        error_code = "llm_external_transport_not_authorized"
    else:
        message = f"unsupported provider {error.provider!r}"
        error_code = "llm_configuration_required"
    return {
        "error": f"configuration_error: {message}",
        "error_code": error_code,
    }


def _build_run_llm(args: Dict[str, Any]):
    """Build the explicitly requested MCP run client or return an error payload."""

    provider = str(args.pop("provider", "openai") or "openai").strip().lower()
    model = str(args.pop("model", "") or "").strip()
    base_url_arg = args.pop("base_url", None)
    request_timeout = float(args.pop("request_timeout", 120.0))
    if not model:
        return None, {
            "error": (
                "configuration_error: research_agent.run requires an explicit "
                "model and configured provider credentials"
            ),
            "error_code": "llm_configuration_required",
        }
    build_kwargs: Dict[str, Any] = {
        "provider": provider,
        "model": model,
        "request_timeout": request_timeout,
        "title": "EasyICU research-agent MCP",
        "client_cls": OpenAIClient,
    }
    if base_url_arg is not None:
        build_kwargs["base_url_override"] = base_url_arg
    try:
        return build_provider_client(**build_kwargs), None
    except ProviderConfigurationError as exc:
        return None, _provider_configuration_error_payload(exc)


def _safe_run_id(value: Any) -> str:
    run_id = str(value or "")
    if (
        not run_id
        or run_id in {".", ".."}
        or "\x00" in run_id
        or "/" in run_id
        or "\\" in run_id
        or Path(run_id).is_absolute()
        or Path(run_id).name != run_id
    ):
        raise ValueError("run_id must be a single safe path component")
    return run_id


def _tool_run(args: Dict[str, Any]) -> Dict[str, Any]:
    require_scope(SCOPE_RUN_PIPELINE, tool="research_agent.run")
    workdir = resolve_within_roots(
        args.pop("workdir", None) or "./research_output", field="workdir"
    )
    cohort = args.pop("cohort_path", None)
    if cohort is None:
        return {"error": "cohort_path is required"}
    cohort = str(resolve_within_roots(cohort, field="cohort_path"))
    try:
        llm, config_error = _build_run_llm(args)
    except Exception as exc:
        return {
            "error": (
                "configuration_error: could not construct the requested LLM "
                f"client: {type(exc).__name__}: {exc}"
            ),
            "error_code": "llm_configuration_invalid",
        }
    if config_error is not None:
        return config_error
    try:
        pipeline = ResearchAgentPipeline(workdir=workdir, llm=llm)
    except Exception as exc:
        return {
            "error": (
                "configuration_error: could not initialise the requested LLM "
                f"client: {type(exc).__name__}: {exc}"
            ),
            "error_code": "llm_configuration_invalid",
        }
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
    workdir = resolve_within_roots(
        args.get("workdir") or "./research_output", field="workdir"
    )
    run_id = args.get("run_id")
    if not run_id:
        return {"error": "run_id is required"}
    safe_run_id = _safe_run_id(run_id)
    path = (workdir / safe_run_id / "manifest.json").resolve()
    try:
        path.relative_to(workdir)
    except ValueError as exc:
        raise ValueError("manifest path escapes workdir") from exc
    if not path.exists():
        return {"error": f"manifest not found at {path}"}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if _internal_context_granted():
        return manifest
    return _safe_manifest_payload(manifest)


#: Manifest keys that are host-generated run provenance with no patient-derived
#: content: identity, timing, and the content-addressed authority envelopes.
_SAFE_MANIFEST_KEYS = (
    "schema_version",
    "checkpoint_sequence",
    "run_id",
    "research_question",
    "started_at",
    "finished_at",
    "cost_records",
    "reproducibility",
    "provider_authorization",
    "execution_identity",
    "code_version",
    "submission_profile_name",
    "submission_profile_version",
)


def _safe_manifest_payload(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Project a run manifest down to identity, status and evidence index.

    A manifest embeds host paths (``context_path``, ``plan_path``), validator
    prose that can quote cohort values, and full evidence descriptions. An
    external client needs to know *which* artefacts exist and whether they
    verify — not what they say.
    """

    payload: Dict[str, Any] = {
        key: manifest[key] for key in _SAFE_MANIFEST_KEYS if key in manifest
    }
    payload["context_path_sha256"] = (
        _path_digest(manifest["context_path"]) if manifest.get("context_path") else None
    )
    payload["plan_path_sha256"] = (
        _path_digest(manifest["plan_path"]) if manifest.get("plan_path") else None
    )
    evidence = manifest.get("evidence")
    if isinstance(evidence, list):
        payload["evidence"] = [
            {
                key: record.get(key)
                for key in (
                    "evidence_id",
                    "kind",
                    "sha256",
                    "produced_by_step",
                    "script_evidence_id",
                    "producer",
                    "generation_mode",
                    "prompt_pack_version",
                    "finding_severity",
                    "created_at",
                )
                if key in record
            }
            for record in evidence
            if isinstance(record, Mapping)
        ]
    findings = manifest.get("findings")
    if isinstance(findings, list):
        payload["findings"] = [
            _safe_finding_payload(finding)
            for finding in findings
            if isinstance(finding, Mapping)
        ]
    records = manifest.get("per_step_records")
    if isinstance(records, list):
        payload["per_step_records"] = project_outbound_records(
            [record for record in records if isinstance(record, Mapping)]
        )
    payload["projection"] = _projection_note()
    return payload


#: Detail keys that describe *which rule fired*, never *what the data said*.
#: A validator detail bucket is free-form and routinely carries column names,
#: value ranges, outliers, small-cell sizes and host paths, so it is projected
#: by allow-list rather than filtered by deny-list.
_SAFE_FINDING_DETAIL_KEYS = (
    "code",
    # Column names are schema, not patient data: the outbound projection
    # already discloses variable names and definitions, and the caller wrote
    # these ones in the script it submitted.
    "column",
    "columns",
    "duplicate_count",
    "fallback",
    "function",
    "human_review_required",
    "kind",
    "reason",
    "rule",
    "step_id",
    "validator",
)


#: Keys whose integer value is a cohort/cell count. Below the column-stats
#: floor these are reported as a bound, exactly as column stats are — an exact
#: small count is the same disclosure whichever field carries it.
_COUNT_DETAIL_KEYS = frozenset({"duplicate_count"})

#: A run of digits long enough to be a record identifier rather than a count.
_ID_TOKEN_RE = re.compile(r"(?<![\d.])\d{6,}(?!\d)")

_MAX_DETAIL_STRING = 300


def _sanitize_detail_string(value: str) -> str:
    """Strip the two things a free-text detail value reliably carries.

    A key allow-list decides *which* fields cross the boundary; it says nothing
    about what is inside them. ``reason`` and ``fallback`` are written by
    validators as free text and have carried absolute paths (disclosing the
    host layout and the operator's directory names) and identifier-shaped
    tokens. Both are stripped here rather than trusted per-validator.
    """

    text = str(value)
    # Absolute and home-relative paths: keep the leaf so the message still
    # reads, drop the tree above it.
    text = re.sub(r"(?:/|~/)[^\s'\"]{2,}", lambda m: Path(m.group(0)).name, text)
    text = _ID_TOKEN_RE.sub("<id>", text)
    if len(text) > _MAX_DETAIL_STRING:
        text = text[:_MAX_DETAIL_STRING] + "…"
    return text


def _sanitize_detail_value(key: str, value: Any) -> Any:
    """Bound what a single allow-listed detail value may disclose."""

    if isinstance(value, bool) or value is None:
        return value
    if key in _COUNT_DETAIL_KEYS and isinstance(value, int):
        if value < MIN_NON_MISSING_FOR_COLUMN_STATS:
            return f"<{MIN_NON_MISSING_FOR_COLUMN_STATS}"
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return _sanitize_detail_string(value)
    if isinstance(value, (list, tuple)):
        return [_sanitize_detail_value(key, item) for item in value[:50]]
    # A nested mapping is an unbounded shape the allow-list never reviewed.
    return "<withheld>"


def _safe_finding_payload(finding: Any) -> Dict[str, Any]:
    """Reduce one validation finding to its stable, PHI-free identity.

    ``message`` is dropped: it is an interpolated sentence and the place a
    concrete value ("only 3 stays have lactate") most often ends up. The keys
    that survive are additionally value-checked — an allow-listed key is not a
    guarantee about what a validator wrote into it.
    """

    raw = finding if isinstance(finding, Mapping) else finding.model_dump(mode="json")
    payload: Dict[str, Any] = {
        key: raw.get(key) for key in ("validator", "severity") if key in raw
    }
    detail = raw.get("detail")
    if isinstance(detail, Mapping):
        projected = {
            key: _sanitize_detail_value(key, detail[key])
            for key in _SAFE_FINDING_DETAIL_KEYS
            if key in detail
        }
        if projected:
            payload["detail"] = projected
        withheld = sorted(set(detail) - set(_SAFE_FINDING_DETAIL_KEYS))
        if withheld:
            payload["detail_withheld_keys"] = withheld
    return payload


def _project_findings(findings: Sequence[Any]) -> List[Dict[str, Any]]:
    """Project auditor findings unless the caller holds the internal scope."""

    if _internal_context_granted():
        return [f.model_dump(mode="json") for f in findings]
    return [_safe_finding_payload(f) for f in findings]


def _cohort_path_from_args(args: Dict[str, Any]) -> Path:
    cohort = args.get("cohort_path")
    if cohort is None:
        raise ValueError("cohort_path is required")
    return resolve_within_roots(cohort, field="cohort_path")


def _build_context_from_args(args: Dict[str, Any]):
    cohort = _cohort_path_from_args(args)
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


def _internal_context_granted() -> bool:
    return SCOPE_READ_INTERNAL_CONTEXT in granted_scopes()


def _projection_note() -> str:
    return (
        "outbound-safe projection (the same one the Planner prompt receives); "
        f"grant the {SCOPE_READ_INTERNAL_CONTEXT!r} scope for the internal "
        "ResearchContext shape"
    )


def _tool_build_context(args: Dict[str, Any]) -> Dict[str, Any]:
    """Return the study context an external client is allowed to see.

    The internal :class:`ResearchContext` carries the cohort parquet path,
    free-text user notes, cohort/parser provenance and per-variable source
    tables and item ids. Every *other* outbound path in the agent — Planner,
    Coder, Replanner — projects that through
    :func:`outbound_safe_context_payload` first; MCP returned the raw dump.
    """

    ctx = _build_context_from_args(args)
    if _internal_context_granted():
        return ctx.model_dump(mode="json")
    return {**outbound_safe_context_payload(ctx), "projection": _projection_note()}


def _tool_list_concepts(args: Dict[str, Any]) -> Dict[str, Any]:
    ctx = _build_context_from_args(args)
    if _internal_context_granted():
        return {
            "cohort": ctx.cohort.model_dump(mode="json"),
            "target_outcome": ctx.target_outcome,
            "concepts": [_concept_payload(v) for v in ctx.variables],
        }
    payload = outbound_safe_context_payload(ctx)
    return {
        "cohort": payload.get("cohort", {}),
        "target_outcome": payload.get("target_outcome"),
        "concepts": payload.get("variables", []),
        "projection": _projection_note(),
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
    if _internal_context_granted():
        return {"concept": _concept_payload(variable)}
    projected = outbound_safe_context_payload(ctx, variable_names=[variable.name]).get(
        "variables", []
    )
    return {
        "concept": projected[0] if projected else {"name": variable.name},
        "projection": _projection_note(),
    }


def _tool_audit_cohort(args: Dict[str, Any]) -> Dict[str, Any]:
    ctx = _build_context_from_args(args)
    findings = CohortAuditor().audit(
        context=ctx,
        cohort_path=_cohort_path_from_args(args),
    )
    return {
        "findings": _project_findings(findings),
        "projection": _projection_note(),
    }


def _tool_run_validator(args: Dict[str, Any]) -> Dict[str, Any]:
    validator = str(args.get("validator") or "cohort_auditor")
    ctx = _build_context_from_args(args)
    if validator in {"cohort", "cohort_auditor"}:
        findings = CohortAuditor().audit(
            context=ctx,
            cohort_path=_cohort_path_from_args(args),
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
        "findings": _project_findings(findings),
        "projection": _projection_note(),
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

    if "data_path" in kwargs:
        # A caller-supplied database path is a read of the host filesystem, so
        # it is confined to the roots the operator configured at startup.
        kwargs["data_path"] = str(
            resolve_within_roots(kwargs["data_path"], field="data_path")
        )

    availability_sink: Dict[str, Any] = {}
    kwargs["availability_sink"] = availability_sink
    result = easyicu_load_concepts(**kwargs)
    policy = DisclosurePolicy.current(args.get("preview_rows") or 5)
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
    summary = _summarise_concept_result(result, policy=policy)
    _record_patient_data_access(
        tool="research_agent.load_concepts",
        args=args,
        concepts=[str(c) for c in concepts],
        summary=summary,
        output_paths=output_paths,
        policy=policy,
    )
    return {
        "api": "easyicu.load_concepts",
        "concepts": [str(c) for c in concepts],
        "database": args.get("database"),
        "data_path_sha256": (
            _path_digest(kwargs.get("data_path")) if kwargs.get("data_path") else None
        ),
        "summary": summary,
        "availability": {
            str(concept): concept_database_availability_from_load_record(
                record,
                requested_concept=str(concept),
            ).model_dump(mode="json")
            for concept, record in availability_sink.items()
        },
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
    require_scope(SCOPE_BIND_EVIDENCE, tool="research_agent.bind_evidence")
    workdir = resolve_within_roots(
        args.get("workdir") or "./research_output", field="workdir"
    )
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

    # The figure-egress gate reads its authorization out of evidence metadata.
    # An external caller that can write those keys can clear its own image to
    # leave the host, which is the whole gate. Same for impersonating the
    # in-process producer the gate trusts.
    reserved = sorted(set(metadata or {}) & RESERVED_PRIVACY_METADATA_KEYS)
    if reserved:
        return {
            "error": (
                "metadata keys "
                + ", ".join(reserved)
                + " are owned by the host privacy audit and cannot be set by an "
                "external caller"
            )
        }
    producer = str(args.get("producer") or "mcp_external_agent")
    if producer in TRUSTED_FIGURE_PRODUCERS:
        return {
            "error": (
                f"producer {producer!r} is an in-process host producer and "
                "cannot be claimed through the MCP evidence tool"
            )
        }

    store = EvidenceStore(workdir)
    common = {
        "kind": kind,
        "description": description,
        "produced_by_step": args.get("produced_by_step"),
        "inputs": inputs,
        "evidence_id": str(evidence_id) if evidence_id else None,
        "aliases": aliases,
        "producer": producer,
        "generation_mode": str(args.get("generation_mode") or "external"),
        "metadata": dict(metadata or {}),
    }
    if source_path is not None:
        record = store.register_file(
            source_path=resolve_within_roots(source_path, field="source_path"),
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


def _summarise_concept_result(
    result: Any, *, policy: DisclosurePolicy
) -> Dict[str, Any]:
    if isinstance(result, dict):
        return {
            "result_type": "dict",
            "concepts": {
                str(name): _summarise_frame(frame, policy=policy)
                for name, frame in result.items()
            },
        }
    return {
        "result_type": type(result).__name__,
        "frame": _summarise_frame(result, policy=policy),
    }


def _summarise_frame(frame: Any, *, policy: DisclosurePolicy) -> Dict[str, Any]:
    """Project one frame through the MCP disclosure policy."""

    summary = summarise_frame(frame, policy=policy)
    if "preview" in summary:
        summary["preview"] = _jsonable(summary["preview"])
    return summary


def _frame_summaries(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten ``_summarise_concept_result`` output to name -> frame summary."""

    concepts = summary.get("concepts")
    if isinstance(concepts, Mapping):
        return {str(name): value for name, value in concepts.items()}
    frame = summary.get("frame")
    return {"frame": frame} if isinstance(frame, Mapping) else {}


def _record_patient_data_access(
    *,
    tool: str,
    args: Dict[str, Any],
    concepts: List[str],
    summary: Mapping[str, Any],
    output_paths: List[Path],
    policy: DisclosurePolicy,
) -> None:
    """Register a PHI-free audit record for one extraction call.

    Fail-closed, and written to a **server-owned** root. The earlier version
    keyed off the caller's ``workdir`` and swallowed registration errors, so a
    client could receive patient rows with no trail simply by omitting
    ``workdir`` — the two conditions that make an audit worth having were both
    under the audited party's control.

    Calls that disclose no rows still record the access, but a failure there is
    logged rather than raised: there is nothing to withhold.
    """

    payload = patient_data_audit_payload(
        tool=tool,
        concepts=concepts,
        database=args.get("database"),
        data_path=args.get("data_path"),
        patient_ids=args.get("patient_ids"),
        frame_summaries=_frame_summaries(summary),
        output_paths=output_paths,
        policy=policy,
    )
    disclosed = int(payload.get("disclosed_patient_rows") or 0)
    try:
        root = patient_data_audit_root()
        root.mkdir(parents=True, exist_ok=True)
        EvidenceStore(root).register_json(
            kind="log",
            description="Audit record for one MCP concept extraction call.",
            payload=payload,
            filename="mcp_patient_data_access.json",
            producer="mcp_server",
            generation_mode="system",
        )
    except Exception as exc:
        print(
            f"easyicu-mcp: could not record the patient-data access audit: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        if disclosed > 0:
            raise MCPAuthorizationError(
                "patient-level rows are withheld because the access audit could "
                "not be written; set "
                f"{MCP_AUDIT_ROOT_ENV} to a writable directory"
            ) from exc


def _write_concept_result_if_requested(
    *,
    result: Any,
    args: Dict[str, Any],
    concepts: List[str],
) -> List[Path]:
    should_write = bool(args.get("output_path") or args.get("register_evidence"))
    if not should_write:
        return []
    require_scope(SCOPE_WRITE_ARTIFACTS, tool="research_agent.load_concepts")
    output_path = args.get("output_path")
    if output_path:
        base = resolve_within_roots(output_path, field="output_path")
    else:
        workdir = resolve_within_roots(
            args.get("workdir") or "./research_output", field="workdir"
        )
        stem = _safe_filename("_".join(concepts[:6])) or "concepts"
        base = (
            workdir / "mcp_concept_extracts" / f"{stem}_{uuid.uuid4().hex[:8]}.parquet"
        )

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
    require_scope(SCOPE_BIND_EVIDENCE, tool="research_agent.load_concepts")
    if not output_paths:
        return [{"error": "register_evidence requires writable concept output"}]
    workdir = resolve_within_roots(
        args.get("workdir") or "./research_output", field="workdir"
    )
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
                "data_path_sha256": _path_digest(args.get("data_path")),
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
        raise TypeError(
            f"Cannot write non-DataFrame concept result: {type(frame).__name__}"
        )
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
            "required": ["question", "cohort_path", "model"],
            "properties": {
                "question": {"type": "string"},
                "cohort_path": {"type": "string"},
                "workdir": {"type": "string"},
                "provider": {
                    "type": "string",
                    "enum": ["openai", "openrouter"],
                },
                "model": {"type": "string"},
                "request_timeout": {"type": "number", "minimum": 1},
                "cohort_name": {"type": "string"},
                "database": {"type": "string"},
                "target_outcome": {"type": "string"},
                "cross_database_validation": {
                    "type": "array",
                    "items": {"type": "string"},
                },
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
                "preview_rows": {
                    "type": "integer",
                    "maximum": 20,
                    "description": (
                        "Patient-level rows to preview. Ignored unless the "
                        "server grants the read_patient_data scope; capped at 20."
                    ),
                },
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
                "preview_rows": {
                    "type": "integer",
                    "maximum": 20,
                    "description": (
                        "Patient-level rows to preview. Ignored unless the "
                        "server grants the read_patient_data scope; capped at 20."
                    ),
                },
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


#: Minimum scope each tool needs before it may run at all. Patient-level
#: disclosure is deliberately *not* listed here: extraction still runs without
#: it (the parquet stays server-side and the evidence record is still written);
#: what the missing scope removes is the row preview in the response.
TOOL_SCOPES: Dict[str, str] = {
    "research_agent.run": SCOPE_RUN_PIPELINE,
    "research_agent.list_skills": SCOPE_METADATA,
    "research_agent.read_manifest": SCOPE_METADATA,
    "research_agent.build_context": SCOPE_METADATA,
    "research_agent.list_concepts": SCOPE_METADATA,
    "research_agent.describe_concept": SCOPE_METADATA,
    "research_agent.audit_cohort": SCOPE_METADATA,
    "research_agent.run_validator": SCOPE_METADATA,
    "research_agent.load_concepts": SCOPE_METADATA,
    "research_agent.extract_concept": SCOPE_METADATA,
    "research_agent.cross_database_concept_availability": SCOPE_METADATA,
    "research_agent.bind_evidence": SCOPE_BIND_EVIDENCE,
}


def dispatch(
    tool_name: str, arguments: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run a tool by name. Returns either the tool's result dict or an error.

    Errors are returned as a stable ``error_code`` plus a message written for
    an external caller. Raw exception text is not echoed back: it can carry
    absolute paths, table and column names, database layout and fragments of
    generated SQL. The full detail goes to the server's stderr log instead.
    """

    arguments = dict(arguments or {})
    fn = TOOLS.get(tool_name)
    if fn is None:
        return {
            "error": f"unknown tool '{tool_name}'",
            "error_code": "unknown_tool",
            "known": list(TOOLS),
        }
    try:
        required = TOOL_SCOPES.get(tool_name)
        if required is not None:
            require_scope(required, tool=tool_name)
        return fn(arguments)
    except MCPAuthorizationError as exc:
        return {"error": str(exc), "error_code": "scope_not_granted"}
    except MCPPathError as exc:
        return {"error": str(exc), "error_code": "path_not_allowed"}
    except ValueError as exc:
        # Argument-shape problems are the caller's to fix, so the message is
        # useful to them and does not describe server internals.
        return {"error": str(exc), "error_code": "invalid_argument"}
    except Exception as exc:
        print(
            f"easyicu-mcp: tool {tool_name} failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return {
            "error": (f"tool {tool_name!r} failed; see the MCP server log for detail"),
            "error_code": "tool_failed",
            "error_type": type(exc).__name__,
        }


# ---------------------------------------------------------------------------
# MCP JSON-RPC
# ---------------------------------------------------------------------------


def _response(req_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(
    req_id: Any, code: int, message: str, data: Optional[Any] = None
) -> Dict[str, Any]:
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
        return _response(
            req_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        )

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
            return _error(
                req_id, -32602, "Invalid params: tools/call requires string 'name'"
            )
        if not isinstance(arguments, dict):
            return _error(
                req_id, -32602, "Invalid params: 'arguments' must be an object"
            )
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


#: Ceilings for the SSE transport. Unbounded sessions and unbounded per-session
#: queues let any authenticated client (or any local process on a loopback
#: bind) hold every ThreadingHTTPServer worker open and grow server memory
#: without limit, which starves real MCP traffic.
MAX_SSE_SESSIONS = 32
SSE_QUEUE_MAXSIZE = 256
SSE_IDLE_TIMEOUT_SECONDS = 300.0
SSE_HEARTBEAT_SECONDS = 15.0

#: Host-wide ceiling on tool calls executing at once, across every HTTP
#: endpoint and session. Session and queue caps bound *delivery*; this bounds
#: *work*, which is what actually consumes memory, CPU and Docker slots.
MAX_CONCURRENT_TOOL_CALLS = 4
#: How long an over-limit request waits for a slot before being turned away.
#: Short on purpose: a client should retry, not hold a server worker.
TOOL_ADMISSION_TIMEOUT_SECONDS = 5.0

_TOOL_EXECUTION_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_TOOL_CALLS)

#: What an unauthenticated caller gets over the HTTP transport. The stdio
#: transport keeps the full process scopes: a desktop client mounting the
#: server is an explicit operator action, whereas a loopback socket with no
#: token is reachable by anything running as the same user.
LOOPBACK_ANONYMOUS_SCOPES = frozenset({SCOPE_METADATA})


class _SSESession:
    def __init__(self) -> None:
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(
            maxsize=SSE_QUEUE_MAXSIZE
        )


_SSE_SESSIONS: Dict[str, _SSESession] = {}
#: ThreadingHTTPServer runs each request on its own thread, so the session
#: table is shared mutable state and needs a lock, not just a bound.
_SSE_SESSIONS_LOCK = threading.Lock()


def _open_sse_session() -> Optional[tuple[str, _SSESession]]:
    """Register a new SSE session, or None when the server is at capacity."""

    session_id = uuid.uuid4().hex
    session = _SSESession()
    with _SSE_SESSIONS_LOCK:
        if len(_SSE_SESSIONS) >= MAX_SSE_SESSIONS:
            return None
        _SSE_SESSIONS[session_id] = session
    return session_id, session


def _close_sse_session(session_id: str) -> None:
    with _SSE_SESSIONS_LOCK:
        _SSE_SESSIONS.pop(session_id, None)


def _get_sse_session(session_id: str) -> Optional[_SSESession]:
    with _SSE_SESSIONS_LOCK:
        return _SSE_SESSIONS.get(session_id)


def _normalise_http_host(value: str) -> Optional[str]:
    """Return the hostname from an HTTP Host header, or ``None`` if invalid."""

    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = urllib.parse.urlsplit("//" + raw)
        if parsed.username is not None or parsed.password is not None:
            return None
        _ = parsed.port
    except ValueError:
        return None
    if not parsed.hostname:
        return None
    return parsed.hostname.rstrip(".").lower()


def _normalise_origin(value: str) -> Optional[str]:
    """Normalise a browser Origin header without accepting URL decorations."""

    raw = str(value or "").strip()
    if not raw or raw.lower() == "null":
        return None
    try:
        parsed = urllib.parse.urlsplit(raw)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            return None
        port = parsed.port
    except ValueError:
        return None
    host = parsed.hostname.rstrip(".").lower()
    default_port = 80 if parsed.scheme == "http" else 443
    port_suffix = "" if port in {None, default_port} else f":{port}"
    if ":" in host:
        host = f"[{host}]"
    return f"{parsed.scheme}://{host}{port_suffix}"


def _is_loopback_bind_host(host: str) -> bool:
    hostname = str(host or "").strip().rstrip(".").lower()
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _default_allowed_hosts(bind_host: str) -> set[str]:
    hostname = str(bind_host or "").strip().rstrip(".").lower()
    if _is_loopback_bind_host(hostname):
        return {"localhost", "127.0.0.1", "::1"}
    return {hostname} if hostname else set()


def _default_allowed_origins(allowed_hosts: set[str], port: int) -> set[str]:
    origins: set[str] = set()
    for host in allowed_hosts:
        rendered = f"[{host}]" if ":" in host else host
        for scheme in ("http", "https"):
            default_port = 80 if scheme == "http" else 443
            suffix = "" if port == default_port else f":{port}"
            origin = _normalise_origin(f"{scheme}://{rendered}{suffix}")
            if origin:
                origins.add(origin)
    return origins


def _validate_sse_server_config(
    host: str, bearer_token: Optional[str]
) -> Optional[str]:
    """Validate bind/auth configuration before opening a listening socket."""

    token = str(bearer_token or "").strip() or None
    if not _is_loopback_bind_host(host) and token is None:
        raise ValueError(
            "non-loopback MCP HTTP binding requires an independent bearer token "
            f"in {MCP_BEARER_TOKEN_ENV}"
        )
    if token is not None:
        for secret_name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
            provider_secret = os.environ.get(secret_name)
            if provider_secret and secrets.compare_digest(token, provider_secret):
                raise ValueError(f"{MCP_BEARER_TOKEN_ENV} must not reuse {secret_name}")
    return token


def _make_sse_handler(
    *,
    bind_host: str = "127.0.0.1",
    port: int = 8765,
    bearer_token: Optional[str] = None,
    allowed_hosts: Optional[List[str]] = None,
    allowed_origins: Optional[List[str]] = None,
    max_body_bytes: int = DEFAULT_HTTP_MAX_BODY_BYTES,
) -> type[http.server.BaseHTTPRequestHandler]:
    configured_hosts = _default_allowed_hosts(bind_host)
    for value in allowed_hosts or []:
        normalised = _normalise_http_host(value)
        if normalised is None:
            raise ValueError(f"invalid allowed MCP Host value: {value!r}")
        configured_hosts.add(normalised)
    configured_origins = _default_allowed_origins(configured_hosts, int(port))
    for value in allowed_origins or []:
        normalised = _normalise_origin(value)
        if normalised is None:
            raise ValueError(f"invalid allowed MCP Origin value: {value!r}")
        configured_origins.add(normalised)
    if int(max_body_bytes) <= 0:
        raise ValueError("max_body_bytes must be positive")

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "EasyICUMCP/0.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # Keep stdout clean for JSON-RPC clients; diagnostics go to stderr.
            print(format % args, file=sys.stderr)

        def do_GET(self) -> None:  # noqa: N802
            if not self._request_authorized():
                return
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path not in {"/sse", "/events"}:
                self.send_error(404, "not found")
                return
            opened = _open_sse_session()
            if opened is None:
                self._write_json(
                    {
                        "error": (
                            "the MCP SSE server already has "
                            f"{MAX_SSE_SESSIONS} open sessions"
                        ),
                        "error_code": "sse_session_limit",
                    },
                    status=503,
                )
                return
            session_id, session = opened

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            endpoint = f"/messages?session_id={session_id}"
            self._write_sse("endpoint", endpoint)
            idle = 0.0
            try:
                while True:
                    try:
                        item = session.queue.get(timeout=SSE_HEARTBEAT_SECONDS)
                    except queue.Empty:
                        idle += SSE_HEARTBEAT_SECONDS
                        if idle >= SSE_IDLE_TIMEOUT_SECONDS:
                            # An idle session is holding a server worker
                            # thread; reclaim it instead of blocking forever.
                            break
                        self._write_sse("ping", "{}")
                        continue
                    idle = 0.0
                    self._write_sse(
                        "message", json.dumps(item, ensure_ascii=False, default=str)
                    )
            except (BrokenPipeError, ConnectionError):
                pass
            finally:
                _close_sse_session(session_id)

        def do_POST(self) -> None:  # noqa: N802
            if not self._request_authorized():
                return
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            media_type = self.headers.get("Content-Type", "").split(";", 1)[0]
            if media_type.strip().lower() != "application/json":
                self._write_json(
                    {"error": "Content-Type must be application/json"},
                    status=415,
                )
                return
            if self.headers.get("Transfer-Encoding"):
                self._write_json(
                    {"error": "chunked request bodies are not accepted"},
                    status=400,
                )
                return
            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length) if raw_length is not None else -1
            except ValueError:
                length = -1
            if length < 0:
                self._write_json(
                    {"error": "a valid Content-Length header is required"},
                    status=411,
                )
                return
            if length > int(max_body_bytes):
                self.close_connection = True
                self._write_json(
                    {
                        "error": (
                            "request body exceeds the configured MCP HTTP limit "
                            f"of {int(max_body_bytes)} bytes"
                        )
                    },
                    status=413,
                )
                return
            body = self.rfile.read(length)
            try:
                req = json.loads(body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._write_json(_error(None, -32700, "Parse error"), status=400)
                return

            # Admission control runs BEFORE the tool does. A request naming a
            # session that does not exist, or whose delivery queue is already
            # full, produces a result nobody can receive — running a full
            # extraction first and discarding it afterwards is free work for an
            # unauthenticated local caller.
            session = None
            session_id = ""
            if parsed.path in {"/messages", "/message"}:
                session_id = (qs.get("session_id") or qs.get("sessionId") or [""])[0]
                session = _get_sse_session(session_id)
                if session is None:
                    self._write_json({"error": "unknown SSE session"}, status=404)
                    return
                if session.queue.full():
                    self._write_json(
                        {
                            "error": (
                                "SSE session delivery queue is full; the client "
                                "is not draining /sse fast enough"
                            )
                        },
                        status=429,
                    )
                    return

            # One host-wide ceiling on tools actually executing. Without it a
            # local caller can start unbounded concurrent pipeline runs and
            # extractions through /jsonrpc, each holding memory and a Docker
            # slot, regardless of the session and queue caps above.
            if not _TOOL_EXECUTION_SLOTS.acquire(
                timeout=TOOL_ADMISSION_TIMEOUT_SECONDS
            ):
                self._write_json(
                    {
                        "error": (
                            "server is already running the maximum of "
                            f"{MAX_CONCURRENT_TOOL_CALLS} concurrent tool calls"
                        )
                    },
                    status=429,
                )
                return
            try:
                # Patient-level disclosure needs its own credential even on a
                # loopback bind, where the general bearer token is optional: any
                # local process could otherwise read patient rows through MCP.
                request_scopes = process_scopes()
                if not self._patient_data_authorized():
                    request_scopes = request_scopes - {SCOPE_READ_PATIENT_DATA}
                if bearer_token is None:
                    # An unauthenticated loopback bind is reachable by every
                    # process on the machine. Read-only metadata is a fair
                    # default there; starting expensive agent runs, writing
                    # files under the allowed roots and registering evidence are
                    # not. Configure EASYICU_MCP_BEARER_TOKEN to get them back.
                    request_scopes = request_scopes & LOOPBACK_ANONYMOUS_SCOPES
                with scope_override(request_scopes):
                    response = handle_jsonrpc(req)
            finally:
                _TOOL_EXECUTION_SLOTS.release()

            if session is not None:
                if response is not None:
                    try:
                        session.queue.put_nowait(response)
                    except queue.Full:
                        self._write_json(
                            {
                                "error": (
                                    "the SSE session queue is full; the client is "
                                    "not reading its event stream"
                                ),
                                "error_code": "sse_queue_full",
                            },
                            status=429,
                        )
                        return
                self._write_json({"accepted": True})
                return

            if parsed.path in {"/jsonrpc", "/rpc"}:
                self._write_json(response or {})
                return

            self.send_error(404, "not found")

        def _patient_data_authorized(self) -> bool:
            """Check the separate patient-data credential for this request.

            Returns False when no patient-data token is configured, so an
            operator who only set the general bearer token never accidentally
            exposes rows.
            """

            expected = str(os.environ.get(MCP_PATIENT_DATA_TOKEN_ENV, "") or "").strip()
            if not expected:
                return False
            supplied = str(self.headers.get("X-EasyICU-Patient-Data", "") or "").strip()
            if not supplied:
                return False
            return secrets.compare_digest(supplied, expected)

        def _request_authorized(self) -> bool:
            host_header = self.headers.get("Host", "")
            request_host = _normalise_http_host(host_header)
            if request_host not in configured_hosts:
                self._write_json({"error": "invalid Host header"}, status=400)
                return False
            try:
                host_port = urllib.parse.urlsplit("//" + host_header).port
            except ValueError:
                host_port = None
            expected_port = int(port) if int(port) > 0 else int(self.server.server_port)
            if host_port is not None and host_port != expected_port:
                self._write_json({"error": "invalid Host port"}, status=400)
                return False

            if bearer_token is not None:
                supplied = self.headers.get("Authorization", "")
                prefix = "Bearer "
                candidate = (
                    supplied[len(prefix) :] if supplied.startswith(prefix) else ""
                )
                if not candidate or not secrets.compare_digest(candidate, bearer_token):
                    self._write_json(
                        {"error": "missing or invalid bearer token"},
                        status=401,
                        extra_headers={"WWW-Authenticate": "Bearer"},
                    )
                    return False

            # Origin is optional for trusted non-browser MCP clients.  When a
            # browser sends it, however, it must be a same-origin/default-local
            # value or an explicitly configured allowlist entry.  No CORS
            # response headers are emitted, so cross-origin browser reads stay
            # blocked even after authentication.
            origin_header = self.headers.get("Origin")
            if origin_header is not None:
                origin = _normalise_origin(origin_header)
                if origin is None or origin not in configured_origins:
                    self._write_json({"error": "invalid Origin header"}, status=403)
                    return False
            return True

        def _write_sse(self, event: str, data: str) -> None:
            payload = f"event: {event}\ndata: {data}\n\n".encode("utf-8")
            self.wfile.write(payload)
            self.wfile.flush()

        def _write_json(
            self,
            payload: Any,
            *,
            status: int = 200,
            extra_headers: Optional[Dict[str, str]] = None,
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _run_sse(
    host: str,
    port: int,
    *,
    bearer_token: Optional[str] = None,
    allowed_hosts: Optional[List[str]] = None,
    allowed_origins: Optional[List[str]] = None,
    max_body_bytes: int = DEFAULT_HTTP_MAX_BODY_BYTES,
) -> int:  # pragma: no cover
    token = _validate_sse_server_config(host, bearer_token)
    handler = _make_sse_handler(
        bind_host=host,
        port=port,
        bearer_token=token,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
        max_body_bytes=max_body_bytes,
    )
    server = http.server.ThreadingHTTPServer((host, port), handler)
    print(
        f"easyicu research-agent MCP SSE listening on http://{host}:{port}/sse",
        file=sys.stderr,
    )
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
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help=(
            "Additional exact Host header accepted by the SSE transport. "
            "Required for practical wildcard/LAN binds."
        ),
    )
    parser.add_argument(
        "--allowed-origin",
        action="append",
        default=[],
        help="Additional exact browser Origin accepted by the SSE transport.",
    )
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=DEFAULT_HTTP_MAX_BODY_BYTES,
    )
    args = parser.parse_args(argv)
    if args.transport == "sse":
        try:
            return _run_sse(
                args.host,
                args.port,
                bearer_token=os.environ.get(MCP_BEARER_TOKEN_ENV),
                allowed_hosts=args.allowed_host,
                allowed_origins=args.allowed_origin,
                max_body_bytes=args.max_request_bytes,
            )
        except ValueError as exc:
            parser.error(str(exc))
    return _run_stdio()


__all__ = [
    "TOOLS",
    "TOOL_SCHEMAS",
    "dispatch",
    "handle_jsonrpc",
    "main",
]
