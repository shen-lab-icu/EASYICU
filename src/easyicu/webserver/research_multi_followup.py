"""fu1 adapter: registered-source proposals, compiled again at batch intake."""
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.schema import CohortDescriptor
from . import sources
from .research_followup_contracts import DatabaseDecision
from .research_multi_contracts import DatabaseTarget, MultiDatabaseError
from .research_multi_plan import compile_same_study, target_decision
from .research_multi_sources import select_source


class RegisteredDatabaseOwner:
    def registered_databases(self):
        return tuple(row["database"] for row in sources.load_registry().get("sources", ()) if row.get("ok"))

    def decide(self, source, *, database):
        select_source(DatabaseTarget(database=database))
        decision = target_decision(source, database)
        return DatabaseDecision(database=database, source_context_sha256=canonical_sha256(source.context.model_dump(mode="json")),
            status=decision.status, reason_codes=tuple(dict.fromkeys(
                [*(row.reason_code for row in decision.substitutions), *(row.code for row in decision.missing)])))

    def dry_compile(self, source, *, decision):
        if self.decide(source, database=decision.database) != decision:
            raise MultiDatabaseError("multi_database_followup_changed")
        # This is a design-only proposal, carrying declared levels, not target
        # observations. Real target intake recompiles before a card is issued.
        from easyicu.research_agent.schema import ConceptDescriptor

        variables = [ConceptDescriptor.model_validate({key: value for key, value in row.model_dump().items()
            if key in {"name", "role", "dtype", "unit", "source_concept", "observed_domain", "is_ordinal", "ordinal_levels"}})
            for row in source.context.variables]
        proposal = target_decision(source, decision.database)
        for change in proposal.substitutions:
            if change.capability == "outcome_meaning" and change.substitute not in {row.name for row in variables}:
                # Only a binary endpoint's declared domain transfers. There is
                # no assumed death/censor time or target observation here.
                if source.context.endpoint is None or source.context.endpoint.kind != "binary":
                    raise MultiDatabaseError("multi_database_design_not_recompilable")
                variables.append(ConceptDescriptor(name=change.substitute, source_concept=change.substitute,
                    role="outcome", dtype="bool", observed_domain={"levels": [False, True], "n_unique": 2, "is_binary": True}))
        cohort = CohortDescriptor(cohort_name="multi_database_design_proposal", database=decision.database,
                                  n_stays=0, id_columns=["stay_id"], outcome_columns=[], provenance={})
        context = source.context.model_copy(deep=True, update={"cohort": cohort, "variables": variables})
        context, plan, _ = compile_same_study(source, context)
        return context, plan
