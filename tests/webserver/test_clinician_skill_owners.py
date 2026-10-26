"""Exercise the clinician skill adapters without an ASGI transport or patient data."""

import math

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError
import pytest


def test_skill_routes_are_registered_on_the_app():
    from easyicu.webserver.app import app

    registered = [
        route
        for branch in app.routes
        for route in (
            branch.effective_route_contexts()
            if hasattr(branch, "effective_route_contexts") else [branch]
        )
    ]
    paths = {(method, route.path) for route in registered for method in getattr(route, "methods", ())}
    prefix = "/api/capabilities/method-skills"
    assert {
        ("GET", f"{prefix}/clinical-concept-definitions/projection"),
        ("GET", f"{prefix}/six-database-cohort-extraction/projection"),
        ("POST", f"{prefix}/sample-size-and-power/calculate"),
    } <= paths


@pytest.mark.parametrize("database", [None, "miiv", "eicu_demo", "MIMIC-IV"])
def test_definition_projection_is_the_unchanged_owner_result(database):
    from easyicu.research_agent import concept_catalog
    from easyicu.webserver.routes import clinician_skills as routes

    concepts = ["sofa2", "sep3", "kdigo_aki", "sofa2"]
    assert routes.get_clinical_definitions(concepts, database) == (
        concept_catalog.clinical_definitions(concepts, database=database)
    )


def test_definition_default_still_exposes_all_registered_composites():
    from easyicu.research_agent import concept_catalog
    from easyicu.webserver.routes import clinician_skills as routes

    assert routes.get_clinical_definitions(database="miiv") == (
        concept_catalog.clinical_definitions(database="miiv")
    )


@pytest.mark.parametrize("database", [None, "eicu_demo"])
def test_extraction_projection_is_the_unchanged_owner_result(database):
    from easyicu.research_agent import concept_catalog
    from easyicu.webserver.routes import clinician_skills as routes

    result = routes.get_extraction_projection(database)
    assert result == concept_catalog.extraction_projection(database)
    assert all(row["use_action"]["auto_submit"] is False for row in result["databases"])


@pytest.mark.parametrize("concepts,database,code", [
    ([], None, "definition.concepts_required"),
    (["unregistered_composite"], None, "definition.concept_unknown"),
    (["sofa2", "unregistered_composite"], None, "definition.concept_unknown"),
    (["sofa2"], "unregistered_database", "definition.database_unknown"),
    (["sofa2"], "", "definition.database_unknown"),
    (["sofa2"] * 257, None, "definition.concepts_required"),
])
def test_definition_refusals_keep_sk4_codes(concepts, database, code):
    from easyicu.webserver.routes import clinician_skills as routes

    with pytest.raises(HTTPException) as caught:
        routes.get_clinical_definitions(concepts, database)
    assert caught.value.status_code == 422
    assert caught.value.detail == {"error": code}


def test_extraction_refuses_unknown_database():
    from easyicu.webserver.routes import clinician_skills as routes

    with pytest.raises(HTTPException) as caught:
        routes.get_extraction_projection("unregistered_database")
    assert caught.value.status_code == 422
    assert caught.value.detail == {"error": "definition.database_unknown"}


def test_definition_and_extraction_adapters_reach_projection_owners(monkeypatch):
    from easyicu.research_agent import concept_catalog
    from easyicu.webserver.routes import clinician_skills as routes

    calls = []
    definitions = {"schema_version": "easyicu.clinical_definition_projection/1", "items": []}
    extraction = {"schema_version": "easyicu.extraction_skill_projection/1", "databases": []}

    def clinical_owner(concepts, *, database):
        calls.append(("clinical_definitions", concepts, database))
        return definitions

    def extraction_owner(database):
        calls.append(("extraction_projection", database))
        return extraction

    monkeypatch.setattr(concept_catalog, "clinical_definitions", clinical_owner)
    monkeypatch.setattr(concept_catalog, "extraction_projection", extraction_owner)
    assert routes.get_clinical_definitions(["sofa2"], "miiv") is definitions
    assert routes.get_extraction_projection("eicu_demo") is extraction
    assert calls == [("clinical_definitions", ["sofa2"], "miiv"), ("extraction_projection", "eicu_demo")]


def _request(payload):
    from easyicu.webserver.skill_planning import PlanningRequest

    return TypeAdapter(PlanningRequest).validate_python(payload)


EXAMPLES = [
    ("two_means", {"difference": 5, "standard_deviation": 10},
     {"group_sizes": [64, 64], "total_n": 128}),
    ("two_proportions", {"p1": .5, "p2": .65},
     {"group_sizes": [170, 170], "total_n": 340}),
    ("survival_events", {"hazard_ratio": .7, "event_fraction": .4},
     {"required_events": 247, "total_n": 618}),
    ("prediction_binary", {"parameters": 24, "cox_snell_r2": .051, "prevalence": .174},
     {"minimum_n": 4115, "criteria_n": {
         "shrinkage": 4115, "r2_optimism": 763, "intercept_precision": 221,
     }}),
]


@pytest.mark.parametrize("calculation,inputs,expected", EXAMPLES)
def test_planning_adapter_reaches_scalar_owner(calculation, inputs, expected, monkeypatch):
    from easyicu.research_agent import skill_packages
    from easyicu.webserver.routes import clinician_skills as routes

    # Sources for these sk4 numerical regression examples:
    # Cohen (1988), Statistical Power Analysis, 2nd ed., chapters 2 and 6:
    # common-SD d=.5 and arcsine h=2*asin(sqrt(.65))-2*asin(sqrt(.5)).
    # https://doi.org/10.4324/9780203771587
    # Schoenfeld (1983), Biometrics 39:499-503, required-event equation:
    # https://doi.org/10.2307/2531021 (HR=.7, q=.5, alpha=.05, power=.8).
    # Riley et al. (2019), Stat Med 38:1276-1296, binary worked example:
    # https://doi.org/10.1002/sim.7992 (24 parameters, R2=.051, prevalence=.174).
    owner = getattr(skill_packages, calculation)
    owner_result = owner(**inputs)
    calls = []

    def traced(**kwargs):
        calls.append(kwargs)
        return owner(**kwargs)

    monkeypatch.setattr(skill_packages, calculation, traced)
    result = routes.post_sample_size_calculation(_request({"calculation": calculation, **inputs}))
    assert len(calls) == 1
    assert result == owner_result
    assert all(result[key] == value for key, value in expected.items())
    assert result["claim_ceiling"] == "analysis_only"
    assert result["patient_data_read"] is False
    assert result["manuscript_findings"] is False
    assert result["basis"] == "user_assumptions"
    assert result["aggregate_receipt"] is None
    if calculation in {"two_means", "two_proportions"}:
        from statsmodels import __version__

        assert result["implementation"] == f"statsmodels/{__version__}"
        assert result["achieved_power"] >= .8
    else:
        assert result["implementation"] == {
            "survival_events": "Schoenfeld-1983-equation-port/1",
            "prediction_binary": "Riley-2019-binary-equation-port/1",
        }[calculation]


@pytest.mark.parametrize("patch", [
    {"calculation": "public_demo"}, {"calculation": "__import__"},
    {"difference": True}, {"difference": "5"}, {"difference": math.inf},
    {"difference": math.nan}, {"difference": 1_000_001}, {"difference": -1_000_001},
    {"standard_deviation": 0}, {"standard_deviation": 1e-7},
    {"standard_deviation": 1_000_001},
    {"alpha": 0}, {"alpha": 1}, {"power": 1e-7}, {"power": .9999999},
    {"allocation_ratio": .009}, {"allocation_ratio": 101},
    {"basis": "governed_aggregate"}, {"aggregate_receipt": "unverified"},
    {"path": "synthetic.parquet"}, {"patient_rows": []},
    {"method": "two_means"}, {"hazard_ratio": .7},
])
def test_planning_transport_refuses_untyped_unbounded_or_extra_inputs(patch):
    with pytest.raises(ValidationError):
        _request({"calculation": "two_means", "difference": 5, "standard_deviation": 10, **patch})


@pytest.mark.parametrize("calculation,inputs,field,bad", [
    ("two_proportions", {"p1": .5, "p2": .65}, "p1", True),
    ("two_proportions", {"p1": .5, "p2": .65}, "p2", 1),
    ("survival_events", {"hazard_ratio": .7}, "hazard_ratio", .0009),
    ("survival_events", {"hazard_ratio": .7}, "hazard_ratio", 1001),
    ("survival_events", {"hazard_ratio": .7}, "allocation_fraction", 0),
    ("survival_events", {"hazard_ratio": .7}, "event_fraction", 1),
    ("prediction_binary", {"parameters": 24, "cox_snell_r2": .051, "prevalence": .174}, "parameters", True),
    ("prediction_binary", {"parameters": 24, "cox_snell_r2": .051, "prevalence": .174}, "parameters", 2.5),
    ("prediction_binary", {"parameters": 24, "cox_snell_r2": .051, "prevalence": .174}, "parameters", 0),
    ("prediction_binary", {"parameters": 24, "cox_snell_r2": .051, "prevalence": .174}, "parameters", 10_001),
])
def test_method_specific_inputs_are_bounded(calculation, inputs, field, bad):
    with pytest.raises(ValidationError):
        _request({"calculation": calculation, **inputs, field: bad})


@pytest.mark.parametrize("field", [
    "cox_snell_r2", "prevalence", "shrinkage", "r2_optimism", "prevalence_margin", "confidence",
])
@pytest.mark.parametrize("value", [0, 1, "0.1", False, math.nan])
def test_every_prediction_probability_is_bounded(field, value):
    with pytest.raises(ValidationError):
        _request({"calculation": "prediction_binary", "parameters": 24,
                  "cox_snell_r2": .051, "prevalence": .174, field: value})


@pytest.mark.parametrize("payload", [
    {}, {"calculation": "two_means", "difference": 5},
    {"calculation": "two_proportions", "p1": .5},
    {"calculation": "survival_events"},
    {"calculation": "prediction_binary", "parameters": 24, "prevalence": .174},
    [], None,
])
def test_calculation_and_required_scalars_cannot_be_omitted(payload):
    with pytest.raises(ValidationError):
        _request(payload)


def test_transport_endpoints_are_inclusive_and_survival_event_fraction_is_optional():
    from easyicu.webserver.routes import clinician_skills as routes

    model = _request({"calculation": "two_means", "difference": -1_000_000,
                      "standard_deviation": 1e-6, "alpha": 1e-6, "power": 1 - 1e-6,
                      "allocation_ratio": .01})
    assert model.difference == -1_000_000
    other = _request({"calculation": "two_means", "difference": 1_000_000,
                      "standard_deviation": 1_000_000, "allocation_ratio": 100})
    assert other.standard_deviation == 1_000_000
    for hazard_ratio in (.001, 1000):
        assert _request({"calculation": "survival_events", "hazard_ratio": hazard_ratio}).hazard_ratio == hazard_ratio
    for parameters in (1, 10_000):
        assert _request({"calculation": "prediction_binary", "parameters": parameters,
                         "cox_snell_r2": .051, "prevalence": .174}).parameters == parameters
    result = routes.post_sample_size_calculation(_request({"calculation": "survival_events", "hazard_ratio": .7}))
    assert result["total_n"] is None
    assert result["required_events"] == 247


def test_planning_adapter_does_not_read_or_write_files(monkeypatch):
    import builtins
    import io
    from pathlib import Path

    import pandas as pd
    from easyicu.webserver.routes import clinician_skills as routes

    # Import numerical dependencies before the IO tripwire; all inputs are scalars.
    requests = [_request({"calculation": calculation, **inputs}) for calculation, inputs, _ in EXAMPLES]
    for request in requests:
        routes.post_sample_size_calculation(request)

    def forbidden(*args, **kwargs):
        pytest.fail("scalar planning attempted file IO")

    for owner, name in [(builtins, "open"), (io, "open"), (Path, "open"),
                        (pd, "read_csv"), (pd, "read_parquet")]:
        monkeypatch.setattr(owner, name, forbidden)
    for request in requests:
        assert routes.post_sample_size_calculation(request)["patient_data_read"] is False


def test_http_schema_exposes_one_discriminated_scalar_body():
    from easyicu.webserver.app import app

    schema = app.openapi()
    route = schema["paths"]["/api/capabilities/method-skills/sample-size-and-power/calculate"]["post"]
    body = route["requestBody"]["content"]["application/json"]["schema"]
    assert body["discriminator"]["propertyName"] == "calculation"
    assert set(body["discriminator"]["mapping"]) == {
        "two_means", "two_proportions", "survival_events", "prediction_binary",
    }
    for alternative in body["oneOf"]:
        model = schema["components"]["schemas"][alternative["$ref"].rsplit("/", 1)[1]]
        assert model["additionalProperties"] is False


@pytest.mark.parametrize("patch", [
    {"difference": math.nan}, {"difference": math.inf},
    {"patient_rows": {"synthetic": math.nan}},
])
def test_transport_validation_errors_are_json_safe(patch, monkeypatch):
    import asyncio
    import json

    from fastapi.exceptions import RequestValidationError
    from fastapi.routing import APIRoute
    from easyicu.webserver.routes import clinician_skills as routes

    with pytest.raises(ValidationError) as invalid:
        _request({"calculation": "two_means", "difference": 5, "standard_deviation": 10, **patch})

    async def rejected(_request):
        raise RequestValidationError(invalid.value.errors())

    monkeypatch.setattr(APIRoute, "get_route_handler", lambda _route: rejected)
    route = next(route for route in routes.router.routes if "POST" in route.methods)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(route.get_route_handler()(None))
    assert caught.value.status_code == 422
    assert caught.value.detail
    assert all(set(issue) == {"loc", "type", "msg"} for issue in caught.value.detail)
    json.dumps(caught.value.detail, allow_nan=False)


def test_overlong_concept_selection_is_refused_before_projection(monkeypatch):
    from easyicu.research_agent import concept_catalog
    from easyicu.webserver.routes import clinician_skills as routes

    calls = []
    monkeypatch.setattr(concept_catalog, "clinical_definitions", lambda *a, **kw: calls.append((a, kw)))
    assert routes.get_clinical_definitions(["sofa2"] * 256, "miiv") is None
    with pytest.raises(HTTPException, match="definition.concepts_required"):
        routes.get_clinical_definitions(["sofa2"] * 257, "miiv")
    assert len(calls) == 1


@pytest.mark.parametrize("calculation,inputs,code", [
    ("two_means", {"difference": 0, "standard_deviation": 10}, "planning.null_effect"),
    ("two_proportions", {"p1": .5, "p2": .5}, "planning.null_effect"),
    ("survival_events", {"hazard_ratio": 1}, "planning.null_effect"),
    ("survival_events", {"hazard_ratio": .7, "alpha": .8, "power": .8}, "planning.power_not_above_alpha"),
    ("prediction_binary", {"parameters": 5, "cox_snell_r2": .7, "prevalence": .01}, "planning.incompatible_r2"),
])
def test_planning_semantic_refusals_keep_sk4_codes(calculation, inputs, code):
    from easyicu.webserver.routes import clinician_skills as routes

    with pytest.raises(HTTPException) as caught:
        routes.post_sample_size_calculation(_request({"calculation": calculation, **inputs}))
    assert caught.value.status_code == 422
    assert caught.value.detail == {"error": code}
