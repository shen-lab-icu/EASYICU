"""Offline study replay from the existing Web run and conversation records.

No provider, database, runner or transcript access. Only fixed projection files
and the selected, digest-pinned evidence closure are read. The return value is
a deterministic filename -> bytes bundle, ready for a download or a static site.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
import math
from pathlib import Path
from typing import Any

from easyicu.research_agent.gates.figure_privacy import MIN_DISCLOSED_GROUP_SIZE
from easyicu.research_agent.gates.publication_disclosure import source_disclosure
from .demo_source_contracts import SOURCE_BY_ID
from .study_replay_evidence import ReplayEvidence
from .study_replay_io import (
    MAX_BUNDLE_BYTES,
    ReplayFiles,
    digest,
    identity,
    write_bundle,
)
from .study_replay_policy import (
    ReplayExportError,
    canonical_json,
    project_tree,
    public_text,
    scrub_secret_values,
    secret_values,
)

REPLAY_SCHEMA = "easyicu.study-replay/1"


def _mapping(value: Any) -> dict:
    if not isinstance(value, dict):
        raise ReplayExportError("replay_record_invalid")
    return value


def _rows(value: Any) -> list[dict]:
    if (
        not isinstance(value, list)
        or len(value) > 1000
        or not all(isinstance(row, dict) for row in value)
    ):
        raise ReplayExportError("replay_record_invalid")
    return value


def _source(value: Any) -> dict:
    source = _mapping(value)
    database = identity(source.get("database"))
    if source_disclosure(database).profile is None:
        raise ReplayExportError("replay_source_unsupported")
    demo_id = source.get("demo_source_id")
    if demo_id is not None:
        demo = SOURCE_BY_ID.get(demo_id)
        if demo is None or demo.database != database:
            raise ReplayExportError("replay_source_identity_mismatch")
        kind = "public_demo"
    elif database in {"synthetic", "test", "mock", "fixture"}:
        kind = "synthetic"
    elif database in {"mimic_demo", "eicu_demo"}:
        kind = "public_demo"
    else:
        kind = "credentialed"
    return {
        "database": database,
        "kind": kind,
        "label": public_text(source.get("label", database)),
        **({"demo_source_id": demo_id} if demo_id else {}),
    }


def _trace(rows: Any) -> list[dict]:
    result, stamps = [], {}
    for row in _rows(rows):
        clean = project_tree(
            {
                key: row[key]
                for key in ("step_id", "phase", "action", "event", "status")
                if key in row
            }
        )
        elapsed = row.get("elapsed_seconds")
        if elapsed is not None and (
            type(elapsed) not in {int, float}
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ReplayExportError("replay_record_invalid")
        clean["elapsed_seconds"] = elapsed
        timestamp = row.get("timestamp")
        key = row.get("step_id") or row.get("phase")
        if timestamp is not None:
            try:
                moment = datetime.fromisoformat(timestamp)
                if moment.tzinfo is None:
                    raise ValueError("timezone required")
            except (TypeError, ValueError):
                raise ReplayExportError("replay_record_invalid") from None
            if key in stamps and moment < stamps[key]:
                raise ReplayExportError("replay_record_invalid")
            if elapsed is None and key in stamps:
                clean["elapsed_seconds"] = (moment - stamps[key]).total_seconds()
                clean["duration_basis"] = "since_first_step_event"
            stamps.setdefault(key, moment)
        result.append(clean)
    return result


def _source_databases(*values: Any) -> list[str]:
    """Keep declared validation/evidence sources alongside the primary source."""
    databases, pending = set(), [("", value) for value in values]
    keys = {
        "database",
        "source_database",
        "validation_database",
        "external_database",
        "cross_database_validation",
    }
    while pending:
        key, value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.items())
        elif isinstance(value, list):
            pending.extend((key, item) for item in value)
        elif key in keys and value is not None:
            database = identity(value)
            if source_disclosure(database).profile is None:
                raise ReplayExportError("replay_source_unsupported")
            databases.add(database)
    return sorted(databases)


def _decisions(conversation: dict | None, run_id: str) -> list[dict]:
    if conversation is None:
        return []
    if (
        _mapping(conversation).get("schema_version")
        != "easyicu.pi-conversation-replay/1"
    ):
        raise ReplayExportError("replay_conversation_invalid")
    jobs = {
        row.get("job_id", row.get("id"))
        for row in _rows(conversation.get("child_jobs", []))
        if isinstance(row.get("result"), dict)
        and row["result"].get("run_id") == run_id
        and isinstance(row.get("job_id", row.get("id")), str)
        and row.get("job_id", row.get("id"))
    }
    return [
        project_tree({key: row[key] for key in ("action_code", "status") if key in row})
        for row in _rows(conversation.get("turns", []))
        if row.get("kind") == "host_action" and row.get("child_job_id") in jobs
    ]


def _html(replay: dict) -> bytes:
    def section(title, value):
        return f"<section><h2>{title}</h2><pre>{escape(canonical_json(value).decode())}</pre></section>"

    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "img-src 'self'; base-uri 'none'; form-action 'none'\">",
        "<title>Study replay</title><body><h1>Study replay</h1>",
        f"<p>{escape(replay['question'])}</p>",
    ]
    for title, key in [
        ("Source", "source"),
        ("Recorded plan", "reviewed_plan"),
        ("Trace", "trace_steps"),
        ("Decisions", "decisions"),
        ("Result", "result_answer"),
        ("Scientific review", "scientific_review"),
        ("Plan review", "plan_review"),
    ]:
        parts.append(section(title, replay[key]))
    for table in replay["aggregate_tables"]:
        parts.append(
            "<table><thead><tr>"
            + "".join(f"<th>{escape(x)}</th>" for x in table["headers"])
            + "</tr></thead><tbody>"
        )
        parts.extend(
            "<tr>" + "".join(f"<td>{escape(x)}</td>" for x in row) + "</tr>"
            for row in table["rows"]
        )
        parts.append("</tbody></table>")
    for figure in replay["figures"]:
        parts.append(
            f'<figure><img src="{escape(figure["path"], quote=True)}" '
            f'alt="{escape(figure["label"], quote=True)}"><figcaption>'
            f"{escape(figure['caption'])}</figcaption></figure>"
        )
    parts.append(
        '<p><a href="replay.json">Replay JSON</a> <a href="manifest.json">Manifest</a></p></body></html>'
    )
    return "\n".join(parts).encode("utf-8")


def _build_study_replay(
    run_dir: Path, *, conversation: dict | None = None
) -> dict[str, bytes]:
    """Build a shareable snapshot or raise ReplayExportError(code), without writes.

    ``run_dir`` is a Web wrapper with run_context.json, agent_plan.json,
    source_run_manifest.json, manuscript_draft.json, result_tables.json and
    figure_gallery.json. Evidence can be local or in pipeline/<run_id>/.
    Optional scientific_readiness.json and scientific_plan_review.json retain
    their recorded authority. Conversation input is Pi's lifecycle projection,
    never its private JSONL; only host decisions bound to this run survive.
    """
    files = ReplayFiles(run_dir)
    if conversation is None:
        conversation = files.json("conversation_replay.json", optional=True)
    else:
        files.secrets.update(secret_values(conversation))
    context = _mapping(files.json("run_context.json"))
    manifest = _mapping(files.json("source_run_manifest.json"))
    run_id = identity(context.get("run_id"))
    if manifest.get("run_id") != run_id:
        raise ReplayExportError("replay_source_identity_mismatch")
    source = _source(context.get("source"))
    if _source(manifest.get("source")) != source:
        raise ReplayExportError("replay_source_identity_mismatch")
    plan = _mapping(files.json("agent_plan.json"))
    answer = _mapping(files.json("manuscript_draft.json"))
    review = files.json("scientific_readiness.json", optional=True)
    plan_review = files.json("scientific_plan_review.json", optional=True)
    for payload in (plan, answer, review, plan_review):
        if payload is not None and _mapping(payload).get("run_id", run_id) != run_id:
            raise ReplayExportError("replay_source_identity_mismatch")
    evidence = ReplayEvidence(files, run_id)
    tables = []
    for row in _rows(_mapping(files.json("result_tables.json")).get("tables")):
        if not isinstance(row.get("sha256"), str):
            raise ReplayExportError("replay_digest_mismatch")
        tables.append(
            {
                **evidence.table(row.get("evidence_id"), expected=row["sha256"]),
                "label": public_text(row.get("label", "Aggregate table")),
            }
        )
    figures, bundle = [], {}
    for row in _rows(_mapping(files.json("figure_gallery.json")).get("figures")):
        figure, raw = evidence.figure(row)
        if figure["path"] in bundle:
            raise ReplayExportError("replay_evidence_ambiguous")
        figures.append(figure)
        bundle[figure["path"]] = raw
    decisions = _decisions(conversation, run_id)
    replay = {
        "schema_version": REPLAY_SCHEMA,
        "run_id": run_id,
        "source": source,
        "question": public_text(context.get("question")),
        "status": public_text(manifest.get("status", "unknown")),
        "reviewed_plan": project_tree(plan),
        "trace_steps": _trace(manifest.get("trace_timings", [])),
        "decisions": decisions,
        "result_answer": project_tree(
            {
                key: answer[key]
                for key in ("status", "markdown_preview", "claims", "sentences")
                if key in answer
            }
        ),
        "scientific_review": project_tree(review),
        "plan_review": project_tree(plan_review),
        "aggregate_tables": tables,
        "figures": figures,
    }
    coordinates = [run_id, *bundle, *evidence.receipts]
    if scrub_secret_values(coordinates, files.secrets) != coordinates:
        raise ReplayExportError("replay_credential_in_identity")
    replay = scrub_secret_values(replay, files.secrets)
    bundle["replay.json"] = canonical_json(replay)
    bundle["index.html"] = _html(replay)
    bundle["manifest.json"] = canonical_json(
        {
            "schema_version": "easyicu.study-replay-manifest/1",
            "run_id": run_id,
            "source": replay["source"],
            "source_databases": _source_databases(
                context,
                plan,
                *[
                    evidence.records[ident].get("metadata", {})
                    for ident in evidence.receipts
                ],
            ),
            "policy": {
                "aggregate_only": True,
                "min_group_size": MIN_DISCLOSED_GROUP_SIZE,
                "small_cells": "refuse_entire_bundle",
                "original_png_metadata": "removed",
            },
            "inputs": [
                {"record": name, "sha256": sha}
                for name, sha in sorted(files.inputs.items())
            ],
            "decisions_sha256": digest(canonical_json(decisions)),
            "evidence": sorted(
                evidence.receipts.values(), key=lambda row: row["evidence_id"]
            ),
            "files": [
                {"path": name, "sha256": digest(raw), "bytes": len(raw)}
                for name, raw in sorted(bundle.items())
            ],
        }
    )
    if sum(map(len, bundle.values())) > MAX_BUNDLE_BYTES:
        raise ReplayExportError("replay_size_limit")
    return dict(sorted(bundle.items()))


def build_study_replay(
    run_dir: Path, *, conversation: dict | None = None
) -> dict[str, bytes]:
    """Build from Web wrapper projections; every invalid input has a safe code."""
    try:
        return _build_study_replay(run_dir, conversation=conversation)
    except ReplayExportError:
        raise
    except (
        TypeError,
        ValueError,
        KeyError,
        UnicodeError,
        RecursionError,
        OverflowError,
    ):
        raise ReplayExportError("replay_record_invalid") from None


def export_study_replay(
    run_dir: Path, destination: Path, *, conversation: dict | None = None
) -> Path:
    """Validate all inputs before atomically publishing a new directory."""
    if Path(destination).resolve().is_relative_to(Path(run_dir).resolve()):
        raise ReplayExportError("replay_destination_forbidden")
    write_bundle(destination, build_study_replay(run_dir, conversation=conversation))
    return Path(destination)


__all__ = [
    "REPLAY_SCHEMA",
    "ReplayExportError",
    "build_study_replay",
    "export_study_replay",
]
