"""Exact registry selection and per-source egress grants for a batch."""
from __future__ import annotations

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.providers.client_trust import require_data_handling, source_requires_clearance
from . import sources
from .demo_source_storage import egress_source_database, public_demo_egress_scope
from .research_multi_contracts import EgressGrant, MultiDatabaseError


def select_source(target):
    rows = [row for row in sources.load_registry().get("sources", []) if row.get("ok")
            and row.get("database") == target.database
            and (target.source_id is None or row.get("id") == target.source_id)]
    if len(rows) != 1:
        raise MultiDatabaseError("multi_database_source_ambiguous" if rows else "multi_database_source_unavailable")
    row = rows[0]
    identity = sources.registered_source_identity(row["path"])
    if identity is None or identity.registration_id != row["id"] or identity.database != target.database:
        raise MultiDatabaseError("multi_database_source_changed")
    # The registration identity has no path. Bind the private path separately
    # so relocating an export cannot keep an old approval alive.
    digest = canonical_sha256({"identity": identity.to_dict(), "path": row["path"],
                               "source_kind": row.get("source_kind")})
    return row, digest


def egress_grant(source, digest, *, provider_sha256, data_handling):
    # Scope is entered once per source, never around a batch.
    with public_demo_egress_scope(source["path"]):
        database = egress_source_database(source["path"], source["database"])
        require_data_handling(database, data_handling)
        clearance = source_requires_clearance(database)
    return EgressGrant(source_id=source["id"], source_sha256=digest, database=source["database"],
        provider_sha256=provider_sha256, data_handling=data_handling,
        clearance="provider_required" if clearance else "verified_public_or_synthetic",
        grants=("prepare_source", "review_plan", "execute_plan", "provider_egress"))
