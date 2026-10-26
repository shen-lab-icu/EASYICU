"""Host-owned mention constraints on study reading and run provenance."""

from __future__ import annotations

import hashlib

from easyicu.extensions.contracts import ExtensionRegistryError

from .mention_objects import resolved_payload


def mentioned_concepts(study) -> tuple[str, ...]:
    return tuple(row["reference"]["id"] for row in study.get("mentioned_objects", [])
                 if row["reference"]["kind"] == "concept")


def apply_mention_setup(study) -> dict:
    """An explicit role survives subsequent model writes; candidates stay typed."""
    rows = resolved_payload(study.get("mentioned_objects") or [])
    patch = {}
    covariates = list(study.get("covariates") or [])
    execution = dict(study.get("execution_concepts") or {})
    for row in rows:
        ref, obj = row["reference"], row["object"]
        if ref["kind"] == "database":
            database = (study.get("data_source") or {}).get("database")
            if database:
                from easyicu.databases.profiles import get_database_profile
                if get_database_profile(database).key != obj["database"]:
                    raise ExtensionRegistryError("mention_database_conflict", "mention_database_conflict")
            else:
                patch["data_source"] = {"database": obj["database"], "label": obj["label"], "path": ""}
            if obj.get("source_id"):
                path = str((study.get("data_source") or {}).get("path") or "")
                if hashlib.sha256(path.encode()).hexdigest() != obj["path_sha256"]:
                    raise ExtensionRegistryError("mention_source_conflict", "mention_source_conflict")
        if ref["kind"] != "concept":
            continue
        role, concept = ref["role"], ref["id"]
        if role in {"exposure", "outcome"}:
            field = "primary_exposure" if role == "exposure" else "outcome"
            patch[field] = concept
            # This is a chosen catalogue concept, not a guessed export column.
            # The existing materialization owner still resolves its column.
            execution[field] = concept
        elif role == "covariate" and concept not in covariates:
            covariates.append(concept)
        if role == "covariate" and "covariates" in execution:
            execution["covariates"] = list(dict.fromkeys([*execution["covariates"], concept]))
    if covariates != list(study.get("covariates") or []):
        patch["covariates"] = covariates
    if execution != dict(study.get("execution_concepts") or {}):
        patch["execution_concepts"] = execution
    return patch


def mention_planning_constraints(study) -> dict:
    rows = resolved_payload(study.get("mentioned_objects") or [])
    # Planner transport has an existing hard budget. Carry exact identities;
    # the full owner snapshots remain in the study and turn receipts.
    return {"mentions": [row["reference"] for row in rows]} if rows else {}
