"""Read-only environment, dictionary and catalogue walks, plus synthetic ideas."""

from __future__ import annotations

from collections import Counter

from .contracts import Status, WalkStop
from .privacy import token


class MetadataWalk:
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
            payload = self.request("GET", "/api/data-map", params={
                "database": source["database"], "concept_limit": 1, "source_limit": 1})
            rows = payload.get("databases") or []
            if len(rows) != 1:
                raise WalkStop(Status.FAILED, "source_capability_summary_missing")
            return {"database": source["database"], "basis": payload.get("basis"),
                    "concept_count": payload.get("concept_total"), "source_count": rows[0].get("source_total"),
                    "families": [{"id": row.get("family"), "status": row.get("status")}
                                 for row in rows[0].get("studies", [])]}

        self.item("environment", "health", health)
        self.item("environment", "build-identity", version)
        self.item("environment", "provider-connection", provider)
        self.item("environment", "source-capabilities", capabilities)

    def mapping(self):
        source = self.source()
        catalog = self.request("GET", "/api/catalog")
        statuses, offset, visited = {}, 0, set()
        while True:
            if offset in visited:
                raise WalkStop(Status.FAILED, "dictionary_pagination_cycle")
            visited.add(offset)
            payload = self.request("GET", "/api/data-map", params={
                "database": source["database"], "concept_offset": offset,
                "concept_limit": 100, "source_limit": 1})
            rows = payload.get("databases") or []
            if len(rows) != 1 or rows[0].get("database") != source["database"]:
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
        return {"basis": payload.get("basis"), "database": source["database"],
                "concept_count": len(statuses), "families": families,
                "mapped": counts["available"], "missing": counts["missing"],
                "changed_meaning": counts["substituted"],
                "unavailable": ["source_observed_coverage_not_exposed"]}

    def extraction_smoke(self):
        projection = self.request("GET", "/api/capabilities/method-skills/six-database-cohort-extraction/projection",
                                  params={"database": self.source()["database"]})
        rows = projection.get("databases") or []
        modules = next((row.get("modules") or {} for row in rows
                        if row.get("database") == self.source()["database"]), {})
        # A projection/prefill is not an executed 19-module extraction receipt.
        raise WalkStop(Status.SKIPPED, "extraction_smoke_api_not_exposed", {
            "module_count": len(modules), "smoke_executed": False,
        })

    def catalogue_inventory(self):
        catalogue, workflows = self.catalogues()
        workflow_by_id = {row["id"]: row for row in workflows}
        rows = []
        planned = []
        unreported = 0
        for row in [*catalogue.get("items", []), *catalogue.get("components", [])]:
            identity = token(row.get("id"))
            kernel = row.get("kernel_modules")
            owner = row.get("owner_status") or ("declared" if any(row.get(key) for key in (
                "action_ids", "capability_id", "included_module_ids", "owner")) else None)
            if identity in workflow_by_id and workflow_by_id[identity].get("steps"):
                owner = owner or "declared"
            if not owner and kernel:
                owner = "kernel_bound"
            unreported += not bool(owner)
            implementation = row.get("implementation")
            if implementation == "planned" or row.get("status") == "planned":
                planned.append(identity)
            rows.append({"id": identity, "layer": row.get("layer", "method_component"),
                         "implementation": implementation, "owner_status": owner or "not_reported",
                         "kernel_binding": bool(kernel) if kernel is not None else None})
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
