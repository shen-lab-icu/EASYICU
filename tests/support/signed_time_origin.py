"""Synthetic, non-clinical signed-origin planning fixture."""
import json

from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.databases.research_substitutions import bind_source_substitutions, time_origin_selection
from easyicu.databases.time_origin import SIC_CARE_UNIT_ORIGIN
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.schema import TimeWindow
from easyicu.webserver.time_origin_source_authority import RegisteredExportTimeOrigin
from tests.support.baseline_association import baseline_context, baseline_plan_attempt, baseline_runtime


def signed_origin_case():
    context = baseline_context("sex", outcome="icu_death")
    context.cohort.database = "sic"
    attempt, _ = baseline_plan_attempt(context)
    plan = attempt.output
    runtime = baseline_runtime(context, plan).seal_for_plan(plan)
    plan, _ = runtime.bind_plan(plan)
    identity = {"database": "sic", "registration_id": "src_example", "version": "1.0.6"}
    constraints = json.loads(context.user_preferences.data_constraints or "{}")
    constraints["source_identity"] = identity
    constraints["database_time_origin_selection"] = {
        "requested_origin": "icu_admission", "selected_origin": SIC_CARE_UNIT_ORIGIN,
        "source_field": research_capability_profile("sic").care_unit_admission_proxy.model_dump(mode="json"),
        "source_identity_sha256": canonical_sha256(identity),
    }
    context.user_preferences.data_constraints = json.dumps(constraints)
    context = context.model_copy(update={
        "time_windows": [TimeWindow(name="first_day", anchor=SIC_CARE_UNIT_ORIGIN, start_hours=0, end_hours=24)],
        "variables": [v.model_copy(update={"analysis_window": "first_day", "analysis_window_role": "exposure_definition"})
                      if v.name == "sex" else v for v in context.variables],
    })
    context = bind_source_substitutions(context)
    selection = time_origin_selection(context)
    receipt = RegisteredExportTimeOrigin(selection.source_identity_sha256, "a" * 64, "b" * 64, selection.selected_origin)

    def attestor(_context, _selection):
        return selection, receipt

    return context, plan, runtime.current_case, receipt, attestor
