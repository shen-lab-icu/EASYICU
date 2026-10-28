"""Read-only repeated-outcome workflow graphs, with no ASGI transport."""

from easyicu.webserver.research_workflow_shapes import canonical_workflow_shape
from easyicu.webserver.research_workflows import research_workflow_catalog


def test_repeated_measure_workflow_has_time_model_and_dropout_steps():
    shape = canonical_workflow_shape("repeated-measure-trajectory")
    assert shape is not None
    request, draft, optional = shape
    assert request.family_id == "repeated_measure_trajectory"
    assert optional == set()
    ids = [item.step.step_id for item in draft.materializations]
    assert ids == ["cohort_accounting", "measurement_audit", "repeated_model", "report"]
    model = draft.materializations[2].step
    assert model.repeated_measure_spec.time_origin == "icu_admission"
    assert "death" in model.objective.lower()
    row = next(item for item in research_workflow_catalog(enabled=True)["items"] if item["id"] == "repeated-measure-trajectory")
    assert row["canonical"] and row["graph_source"] == "repeated_measure_trajectory"
    assert row["edges"]
