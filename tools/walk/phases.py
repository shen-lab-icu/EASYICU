"""Read-only environment, dictionary and catalogue walks, plus synthetic ideas."""

from __future__ import annotations

from collections import Counter
import re

from .contracts import Status, WalkStop
from .privacy import token


def _owner_link(row):
    action = row.get("use_action") or {}
    entry = action.get("entry")
    api_entry = (isinstance(entry, str) and len(entry) <= 160 and ".." not in entry
                 and re.fullmatch(r"/api/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", entry))
    return (action.get("kind") == "open_owner"
            and token(action.get("owner")) != "not_reported"
            and bool(token(entry) != "not_reported" or api_entry))


def _registered_method(row):
    # Registered method-suite components can have no deterministic kernel;
    # their family/key binding still declares an owner for agent execution.
    family, key = token(row.get("method_family")), token(row.get("method_key"))
    return ("not_reported" not in (family, key) and row.get("id") == f"{family}.{key}"
            and row.get("execution_mode") in {"deterministic_host", "agent_coded_with_host_gates"})


def _kernel_binding(row, components):
    if row.get("kernel_modules") is not None or "primary_runner" in row:
        return bool(row.get("kernel_modules")) or token(row.get("primary_runner")) != "not_reported"
    bindings = [
        bool(component.get("kernel_modules")) or token(component.get("primary_runner")) != "not_reported"
        for identity in row.get("action_ids", []) if (component := components.get(identity)) is not None
        and ("kernel_modules" in component or "primary_runner" in component)
    ]
    if bindings:
        return any(bindings)
    return False if _owner_link(row) else None


class MetadataWalk:
    def _public_database(self):
        """Select public metadata by the registered database key, never its label.

        Demo loader keys declare their parent family with the ``_demo`` suffix.
        Confirm that parent against the server's public profiles before querying.
        This is packaged family metadata, not observed coverage of demo rows.
        """
        database = self.source()["database"]
        profiles = self.request("GET", "/api/data-map", params={"concept_limit": 1, "source_limit": 1})
        public = {row.get("database") for row in profiles.get("databases", [])}
        if database in public:
            return database
        parent = database.removesuffix("_demo")
        if parent != database and parent in public:
            return parent
        raise WalkStop(Status.SKIPPED, "data_map_public_parent_not_reported", {"database": database})

    def environment_items(self):
        def health():
            payload = self.request("GET", "/api/health")
            if payload.get("status") != "ok":
                raise WalkStop(Status.FAILED, "server_unhealthy")
            return {"status": "ok"}

        def version():
            about = self.request("GET", "/api/settings").get("about", {})
            fields = {key: token(about.get(key)) for key in ("version", "commit", "runner_image_tag")}
            fields["unavailable"] = [f"{key}_not_exposed" for key, value in fields.items() if value == "not_reported"]
            return fields

        def provider():
            runtime = self.request("GET", "/api/copilot/pi/status").get("runtime", {})
            configuration = runtime.get("configuration") or {}
            return {"provider_status": token(runtime.get("status")),
                    "shell_ready": runtime.get("shell_ready") is True,
                    "connection_verified": configuration.get("connection_verified"),
                    "refusal_codes": [token(code) for code in runtime.get("blockers", [])]}

        def capabilities():
            source = self.source()
            database = self._public_database()
            payload = self.request("GET", "/api/data-map", params={
                "database": database, "concept_limit": 1, "source_limit": 1})
            rows = payload.get("databases") or []
            if len(rows) != 1 or rows[0].get("database") != database:
                raise WalkStop(Status.FAILED, "source_capability_summary_missing")
            return {"database": source["database"], "profile_database": database, "basis": payload.get("basis"),
                    "concept_count": payload.get("concept_total"), "source_count": rows[0].get("source_total"),
                    "families": [{"id": row.get("family"), "status": row.get("status")}
                                 for row in rows[0].get("studies", [])]}

        self.item("environment", "health", health)
        self.item("environment", "build-identity", version)
        self.item("environment", "provider-connection", provider)
        self.item("environment", "source-capabilities", capabilities)

    def mapping(self):
        source = self.source()
        database = self._public_database()
        catalog = self.request("GET", "/api/catalog")
        statuses, offset, visited = {}, 0, set()
        while True:
            if offset in visited:
                raise WalkStop(Status.FAILED, "dictionary_pagination_cycle")
            visited.add(offset)
            payload = self.request("GET", "/api/data-map", params={
                "database": database, "concept_offset": offset,
                "concept_limit": 100, "source_limit": 1})
            rows = payload.get("databases") or []
            if len(rows) != 1 or rows[0].get("database") != database:
                raise WalkStop(Status.FAILED, "dictionary_database_mismatch")
            for concept in rows[0].get("concepts", []):
                status = concept.get("status")
                if status not in {"available", "substituted", "missing"}:
                    raise WalkStop(Status.FAILED, "dictionary_status_unknown")
                statuses[concept["concept"]] = status
            following = payload.get("concept_next_offset")
            if following is None:
                break
            if type(following) is not int or following <= offset:
                raise WalkStop(Status.FAILED, "dictionary_pagination_invalid")
            offset = following
        families = []
        for family, members in catalog.get("groupConcepts", {}).items():
            counts = Counter(statuses.get(name, "missing") for name in members)
            families.append({"id": token(family), "mapped": counts["available"],
                             "missing": counts["missing"], "changed_meaning": counts["substituted"]})
        if not statuses or not families:
            raise WalkStop(Status.FAILED, "dictionary_coverage_missing")
        counts = Counter(statuses.values())
        return {"basis": payload.get("basis"), "database": source["database"], "profile_database": database,
                "concept_count": len(statuses), "families": families,
                "mapped": counts["available"], "missing": counts["missing"],
                "changed_meaning": counts["substituted"],
                "unavailable": ["source_observed_coverage_not_exposed"]}

    def extraction_smoke(self):
        database = self._public_database()
        projection = self.request("GET", "/api/capabilities/method-skills/six-database-cohort-extraction/projection",
                                  params={"database": database})
        rows = projection.get("databases") or []
        modules = next((row.get("modules") or {} for row in rows
                        if row.get("database") == database), {})
        # A projection/prefill is not an executed 19-module extraction receipt.
        raise WalkStop(Status.SKIPPED, "extraction_smoke_api_not_exposed", {
            "module_count": len(modules), "smoke_executed": False,
        })

    def catalogue_inventory(self):
        catalogue, workflows = self.catalogues()
        workflow_by_id = {row["id"]: row for row in workflows}
        components = {row["id"]: row for row in catalogue.get("components", [])}
        rows = []
        planned = []
        unreported = 0
        for row in [*catalogue.get("items", []), *catalogue.get("components", [])]:
            identity = token(row.get("id"))
            kernel = _kernel_binding(row, components)
            declared = (any(row.get(key) for key in (
                "action_ids", "capability_id", "included_module_ids", "owner", "primary_runner"))
                or _owner_link(row) or _registered_method(row))
            owner = row.get("owner_status") or ("declared" if declared else None)
            if identity in workflow_by_id and workflow_by_id[identity].get("steps"):
                owner = owner or "declared"
            if not owner and kernel:
                owner = "kernel_bound"
            unreported += not bool(owner)
            implementation = row.get("implementation") or row.get("execution_mode")
            if implementation == "planned" or row.get("status") == "planned":
                planned.append(identity)
            rows.append({"id": identity, "layer": row.get("layer", "method_component"),
                         "implementation": implementation, "owner_status": owner or "not_reported",
                         "kernel_binding": kernel})
        planned_count = catalogue.get("planned_method_count")
        metrics = {"item_count": len(rows), "catalogue_items": rows, "planned_items": planned,
                   "planned_count": planned_count, "unreported_owner_count": unreported,
                   "workflow_count": len(workflows), "method_count": len(catalogue.get("components", [])),
                   "layer_counts": dict(Counter(row["layer"] for row in rows)),
                   "unavailable": ["planned_item_identities_not_exposed"] if planned_count and not planned else []}
        if unreported:
            raise WalkStop(Status.FAILED, "catalogue_owner_binding_not_reported", metrics)
        return metrics

    def ideas(self):
        payload = self.request("POST", "/api/ideas/mine", {
            "source_type": "manual", "title": "Synthetic ICU study planning exercise",
            "excerpt": ("This fictional abstract proposes describing early ICU lactate and blood pressure "
                        "measurements and examining their association with hospital mortality. "
                        "It contains no observations or effect estimates. Feasibility and evidence require review."),
            "metadata_only": True, "allow_network": False,
        })
        ledger = payload.get("idea_ledger")
        if not isinstance(ledger, list):
            raise WalkStop(Status.FAILED, "idea_candidates_missing")
        evidence = Counter()
        feasibility = Counter()
        for row in ledger:
            evidence[token((row.get("prior_art") or {}).get("status"))] += 1
            verdict = row.get("feasibility") or {}
            feasibility[token(verdict.get("tier") if isinstance(verdict, dict) else verdict)] += 1
        return {"candidate_count": len(ledger), "evidence_verdicts": dict(evidence),
                "feasibility_verdicts": dict(feasibility),
                "unavailable": ["metadata_only_source_feasibility_not_tested", "evidence_independent_adjudication_not_exposed"]}
