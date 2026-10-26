"""Approach, materials and limitations projected from existing run records."""

from .pi_copilot.source_adjustments import project_source_adjustments
from .scientific_review_items import ReviewSource, mapping, pointer, rows


def review_sections(payloads, items):
    approach, materials, limitations = [], [], []
    plan = mapping(payloads.get("agent_plan.json"))
    context = mapping(payloads.get("run_context.json"))
    manifest = mapping(payloads.get("source_run_manifest.json"))
    metadata = mapping(manifest.get("scientific_review"))
    review = mapping(payloads.get("scientific_plan_review.json"))
    facts = mapping(review.get("facts"))

    def add(target, code, value, artifact, location="", step_id=""):
        target.append(items.item(code, value, ReviewSource(artifact, location, step_id=step_id)))

    for field, code in (("cohort", "POPULATION"), ("endpoint", "OUTCOME")):
        if plan.get(field) is not None:
            add(approach, "SR_APPROACH_" + code, plan[field], "agent_plan.json", pointer(field))
    endpoint = mapping(plan.get("endpoint"))
    if endpoint.get("time_origin"):
        add(approach, "SR_APPROACH_TIME_ZERO", endpoint["time_origin"], "agent_plan.json", "/endpoint/time_origin")
    for index, candidate in enumerate(rows(mapping(plan.get("design_selection")).get("candidates"))):
        candidate = mapping(candidate)
        if candidate.get("disposition") == "selected":
            for field in ("time_zero", "estimand", "observation_window"):
                if candidate.get(field):
                    add(approach, "SR_APPROACH_" + field.upper(), candidate[field], "agent_plan.json",
                        pointer("design_selection", "candidates", index, field))
    for index, raw in enumerate(rows(plan.get("steps"))):
        step = mapping(raw)
        step_id = step.get("step_id", "")
        for mi, model in enumerate(rows(step.get("model_requirements"))):
            for field in ("exposure_source", "diagnostics"):
                if mapping(model).get(field):
                    add(approach, "SR_APPROACH_" + ("EXPOSURE" if field == "exposure_source" else "DIAGNOSTICS"), model[field],
                        "agent_plan.json", pointer("steps", index, "model_requirements", mi, field), step_id)
        group_by = mapping(step.get("table_one_spec")).get("group_by")
        if group_by:
            add(approach, "SR_APPROACH_GROUPING", group_by, "agent_plan.json", pointer("steps", index, "table_one_spec", "group_by"), step_id)
        for field, code in (("cohort_definition_spec", "POPULATION"), ("exposure_outcome_distribution_spec", "GROUPING"),
                            ("model_requirements", "MODEL"), ("family_primary_result_requirement", "MODEL"),
                            ("measurement_audit_spec", "DIAGNOSTICS")):
            if step.get(field):
                add(approach, "SR_APPROACH_" + code, step[field], "agent_plan.json", pointer("steps", index, field), step_id)
        if step.get("method"):
            add(approach, "SR_APPROACH_STEP", step["method"], "agent_plan.json", pointer("steps", index, "method"), step_id)
            add(materials, "SR_METHOD", step["method"], "agent_plan.json", pointer("steps", index, "method"), step_id)
    if context.get("exposure_groups"):
        add(approach, "SR_APPROACH_GROUPING", context["exposure_groups"], "run_context.json", "/exposure_groups")
    for index, row in enumerate(approach, 1):
        row["order"] = index

    source = mapping(metadata.get("source")) or mapping(context.get("source"))
    artifact, prefix = ("source_run_manifest.json", "/scientific_review/source") if metadata.get("source") else ("run_context.json", "/source")
    for key in ("database", "version", "export_id"):
        add(materials, "SR_DATA_" + key.upper(), source.get(key), artifact, prefix + "/" + key)
        materials[-1]["status"] = "recorded" if source.get(key) else "not_recorded"
    cohort = mapping(payloads.get("cohort_summary.json"))
    for key in ("summary", "cohort", "population_preparation"):
        if cohort.get(key):
            add(materials, "SR_COHORT", cohort[key], "cohort_summary.json", pointer(key))
    for index, raw in enumerate(rows(mapping(payloads.get("result_tables.json")).get("tables"))):
        table = mapping(raw)
        if {"n_before", "n_excluded", "n_remaining"}.intersection(rows(table.get("headers"))):
            headers = rows(table.get("headers"))
            for ri, row in enumerate(rows(table.get("rows"))):
                values = {str(key): value for key, value in zip(headers, rows(row))}
                materials.append(items.item("SR_COHORT_FLOW", values, ReviewSource(
                    "result_tables.json", pointer("tables", index, "rows", ri),
                    table.get("evidence_id", ""), table.get("produced_by_step", ""), table.get("sha256", ""))))
        for key in ("descriptive_summary", "stratified_summary"):
            definitions = mapping(mapping(table.get(key)).get("variable_display"))
            for variable, definition in definitions.items():
                add(materials, "SR_CONCEPT_DEFINITION", {"concept": variable, "definition": definition}, "result_tables.json",
                    pointer("tables", index, key, "variable_display", variable), table.get("produced_by_step", ""))
    selected = mapping(context.get("summary")).get("selected_concepts")
    for index, concept in enumerate(rows(selected)):
        add(materials, "SR_CONCEPT_USED", concept, "run_context.json", pointer("summary", "selected_concepts", index))
    for key in ("clinical_definitions", "covariate_rationales", "covariate_temporal_roles"):
        if facts.get(key):
            add(materials, "SR_CONCEPT_DEFINITION", facts[key], "scientific_plan_review.json", pointer("facts", key))
    capability_review = mapping(facts.get("database_capability_review"))
    decision = mapping(capability_review.get("decision"))
    for index, raw in enumerate(rows(decision.get("substitutions"))[:12]):
        # Preserve the original source index when the existing owner rejects
        # a malformed preceding row. Normalization must not move source links.
        adjustments = project_source_adjustments({**capability_review, "decision": {**decision, "substitutions": [raw]}})
        if adjustments and adjustments["decision"]["substitutions"]:
            sub = adjustments["decision"]["substitutions"][0]
            value = {**sub, "applied": sub["capability"] not in rows(capability_review.get("unapplied_substitutions"))}
            location = pointer("facts", "database_capability_review", "decision", "substitutions", index)
            add(materials, "SR_SOURCE_SUBSTITUTION", value, "scientific_plan_review.json", location)
            add(limitations, "SR_SOURCE_SUBSTITUTION", value, "scientific_plan_review.json", location)
    for index, raw in enumerate(rows(metadata.get("runtime"))):
        row = mapping(raw)
        field = row.get("field")
        if field not in {"package_versions", "python_version", "data_property_notes", "limitations", "diagnostics", "claim_ceiling"}:
            continue
        target = materials if field in {"package_versions", "python_version"} else limitations
        add(target, "SR_RECORDED_" + str(field).upper(), row.get("value"), "source_run_manifest.json",
            pointer("scientific_review", "runtime", index), row.get("step_id", ""))
    if not any(row["code"] == "SR_RECORDED_PACKAGE_VERSIONS" for row in materials):
        add(materials, "SR_PACKAGE_VERSIONS_NOT_RECORDED", None, "source_run_manifest.json", "/scientific_review")
    if metadata.get("truncated"):
        add(limitations, "SR_METADATA_INCOMPLETE", True, "source_run_manifest.json", "/scientific_review/truncated")
    gate = mapping(mapping(payloads.get("quality_gate.json")).get("gate"))
    for index, check in enumerate(rows(gate.get("checks"))):
        if mapping(check).get("passed") is not True:
            add(limitations, "SR_GATE", check, "quality_gate.json", pointer("gate", "checks", index))
    for name in ("scientific_readiness.json", "scientific_plan_review.json"):
        payload = mapping(payloads.get(name))
        if payload.get("claim_ceiling"):
            add(limitations, "SR_CLAIM_CEILING", payload["claim_ceiling"], name, "/claim_ceiling")
        for index, finding in enumerate(rows(payload.get("findings"))):
            # Codes and recorded details retain the original finding identity.
            add(limitations, "SR_FINDING", finding, name, pointer("findings", index))
    return {"approach": approach, "materials": materials, "limitations": limitations}
