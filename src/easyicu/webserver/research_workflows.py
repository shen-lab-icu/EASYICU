"""Read-only workflow hub projection; no execution or patient-data access.

Only identity bindings live here. Inputs, products, topology and database
decisions are projected from public package, family and capability owners.
An owner without a fixed DAG exposes its declared composition, with no
invented ordering edges. Canonical shapes are examples, never runnable plans.
"""

from __future__ import annotations

from easyicu.databases.research_requirements import family_requirements
from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.research_agent.method_skills import (
    method_skill_catalog,
    method_skill_package,
    method_skill_workflow_metadata,
)
from easyicu.research_agent.planning.progressive_contract import PROGRESSIVE_HOST_COMPILED_OUTPUTS
from easyicu.research_agent.planning.analysis_types import required_endpoint_kind_for_family

from .research_workflow_shapes import canonical_workflow_shape
from .raw_source_authority import source_identity_capability_profiles


EXAMPLES = {
    "repeated-measure-trajectory": "example-repeated-measure-change",
    "adjusted-exposure-outcome-study": "example-lactate-death",
    "ordinal-dose-response-study": "example-sofa-trend",
    "fixed-landmark-association-study": "example-landmark-lactate",
    "time-varying-exposure-survival-study": "example-ventilation-time-varying",
    "survival-time-to-event": "example-aki-survival",
    "clinical-risk-prediction": "example-predict-death",
    "dynamic-landmark-prediction": "example-dynamic-prediction",
    "target-trial-emulation": "example-early-norepinephrine",
    "cross-sectional-phenotyping": "example-first-day-phenotypes",
    "trajectory-phenotyping": "example-pressure-trajectories",
}


def _template_graph(draft, optional, labels):
    steps, edges, outputs = [], {}, {}
    for materialization in draft.materializations:
        step = materialization.step
        products = dict(PROGRESSIVE_HOST_COMPILED_OUTPUTS.get(step.module_id, ()))
        products.update((item.product_id, item.semantic_role) for item in step.outputs)
        kind = {"cohort_definition": "cohort", "measurement_audit": "preflight", "table_one": "diagnostic",
                "robustness_replay": "sensitivity", "visualization": "figure", "report": "report"}.get(
                    step.module_id, "model" if step.planned_analysis_role == "primary" else "diagnostic")
        names = labels.get(step.scientific_action_id, {})
        steps.append(dict(id=step.step_id, label=names.get("title", step.objective),
                          **({"label_zh": names["title_zh"]} if names.get("title_zh") else {}),
                          kind=kind, module_id=step.module_id,
                          action_id=step.scientific_action_id, optional=step.step_id in optional,
                          outputs=list(products)))
        for product, role in products.items():
            outputs[product] = dict(id=product, kind=product.split(":", 1)[0], role=role,
                                    producer=step.step_id)
        for parent in step.depends_on:
            edges.setdefault((parent, step.step_id), [])
        for ref in step.product_inputs:
            edges.setdefault((ref.producer_step_id, step.step_id), []).append(ref.product_id)
    return steps, [dict(source=source, target=target, products=products)
                   for (source, target), products in edges.items()], list(outputs.values())


def _workflow(row, labels):
    metadata = method_skill_workflow_metadata(row["id"], enabled=row["enabled"])
    package = method_skill_package(row["id"], enabled=row["enabled"])
    requirements = metadata["requirements"]
    inputs = [dict(role=role, kind=role) for role in family_requirements(requirements["family"])]
    if requirements["outcome"]:
        outcome = dict(role="outcome", concept=requirements["outcome"],
                       **metadata["input_labels"].get(requirements["outcome"], {}))
        endpoint_kind = required_endpoint_kind_for_family(requirements["family"])
        if endpoint_kind is not None:
            outcome["kind"] = endpoint_kind
        inputs.append(outcome)
    inputs.extend(dict(role="concept", concept=value, **metadata["input_labels"].get(value, {}))
                  for value in requirements["concepts"])
    shape = canonical_workflow_shape(row["id"])
    if shape:
        request, draft, optional = shape
        steps, edges, outputs = _template_graph(draft, optional, labels)
        for item in inputs:
            if item["role"] == "outcome":
                item["kind"] = "time_to_event" if request.survival_suite else "binary"
        if request.exposure_kind != "none":
            inputs.append(dict(role="exposure", kind=request.exposure_kind, ordered=request.exposure_is_ordered))
        if request.observation_window_hours is not None:
            inputs.append(dict(role="observation_window", unit="hours"))
        if request.cohort_time_zero_hours is not None:
            inputs.append(dict(role="time_zero", anchor="icu_admission"))
        if request.observation_duration_column is not None:
            inputs.append(dict(role="followup", unit=request.observation_duration_unit))
        if request.survival_suite is not None:
            inputs.append(dict(role="followup"))
        graph_source = request.family_id
    else:
        steps = [dict(**step, optional=False) for step in metadata["steps"]]
        outputs = [dict(id=value, kind="file", planned=metadata["planned_filenames"])
                   for value in metadata["output_files"]]
        graph_source = "package_composition"
    databases = []
    for note in package["database_notes"]:
        decision = note["decision"]
        status = {"exact": "supported", "substitution": "degraded", "refusal": "unavailable"}[decision["status"]]
        if status == "supported" and decision["unassessed_concepts"]:
            status = "degraded"
        databases.append(dict(id=decision["database"], label=note["display_name"], status=status,
                              decision=decision, real_data_acceptance=note["real_data_acceptance"],
                              analysis_units=list(research_capability_profile(decision["database"]).analysis_units),
                              identity_profiles=source_identity_capability_profiles(decision["database"])))
    roles = {}
    for item in inputs:
        roles.setdefault((item["role"], item.get("concept")), {}).update(item)
    result = {key: row[key] for key in ("id", "title", "title_zh", "description", "description_zh", "version", "enabled")}
    result.update(inputs=list(roles.values()), outputs=outputs, steps=steps, databases=databases,
                  graph_source=graph_source, canonical=bool(shape), read_only=True,
                  references=[item["path"] for item in package["files"] if item["path"].startswith("references/")],
                  package_url=f"/api/capabilities/method-skills/{row['id']}/package")
    if shape:
        result["edges"] = edges
    if row["id"] in EXAMPLES:
        result["example"] = EXAMPLES[row["id"]]
    return result


def research_workflow_catalog(*, enabled: bool) -> dict:
    catalog = method_skill_catalog(enabled=enabled)
    labels = {row["id"]: row for row in catalog["components"]}
    return {"items": [_workflow(row, labels) for row in catalog["items"]
                      if row["layer"] == "research_workflow"]}
