"""Strict Planner transport-schema ownership and representation regressions."""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents import plan_payload as payload_owner
from tests.support.schema_names import authority_schema_json as _schema_json
from easyicu.research_agent.agents.plan_payload import (
    PlannerHostOwnedFieldError,
    PlannerStructuredOutputSchemaError,
    decode_planner_transport_payload,
    planner_structured_output_request,
)


from easyicu.research_agent.contracts.capability_ids import CAPABILITY_FAMILIES
from easyicu.research_agent.contracts.product_identity import (
    CANONICAL_TYPED_PRODUCT_TOKEN_PATTERN,
)
from easyicu.research_agent.agents.core import PlannerAgent
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    PlannedModelRequirement,
    ResearchContext,
    TrajectoryStabilitySpec,
)


def _walk_schema(node):
    if not isinstance(node, dict):
        return
    properties = node.get("properties")
    if isinstance(properties, dict):
        yield node
        for child in properties.values():
            yield from _walk_schema(child)
    definitions = node.get("$defs")
    if isinstance(definitions, dict):
        for child in definitions.values():
            yield from _walk_schema(child)
    for key in ("items", "not", "if", "then", "else"):
        child = node.get(key)
        if isinstance(child, dict):
            yield from _walk_schema(child)
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        children = node.get(key)
        if isinstance(children, list):
            for child in children:
                yield from _walk_schema(child)


def test_planner_transport_schema_is_closed_compact_and_deterministic():
    request = planner_structured_output_request()
    schema = _schema_json(request.schema_json)

    assert request.name == "easyicu_analysis_plan_v1"
    assert request.strict is True
    assert request.payload_bytes < 30_000
    assert len(request.authority_sha256) == 64
    for object_schema in _walk_schema(schema):
        properties = object_schema["properties"]
        assert set(object_schema["required"]) == set(properties)
        assert object_schema["additionalProperties"] is False

    # A schema property named like stripped metadata must remain a property.
    criterion = schema["$defs"]["CohortEligibilityCriterion"]
    assert criterion["properties"]["description"]["type"] == "string"
    assert "description" not in criterion["properties"]["description"]

    capability = schema["$defs"]["AnalysisStep"]["properties"][
        "scientific_capability"
    ]
    assert capability == {
        "anyOf": [
            {"type": "string", "enum": sorted(CAPABILITY_FAMILIES)},
            {"type": "null"},
        ]
    }

    consumption = schema["$defs"]["ArtifactConsumptionContract"]
    branches = {
        mode: branch
        for branch in consumption["anyOf"]
        for mode in (
            branch["properties"]["mode"].get("enum")
            or [branch["properties"]["mode"]["const"]]
        )
    }
    assert set(branches) == {"all_rows", "single_row", "one_per_role"}
    for mode in ("all_rows", "single_row"):
        properties = branches[mode]["properties"]
        assert (
            properties["input_key"]["pattern"]
            == CANONICAL_TYPED_PRODUCT_TOKEN_PATTERN
        )
        assert properties["role_column"] == {"type": "null"}
        assert properties["expected_roles"]["maxItems"] == 0
    role_properties = branches["one_per_role"]["properties"]
    assert role_properties["role_column"]["minLength"] == 1
    assert role_properties["expected_roles"]["minItems"] == 1
    assert "uniqueItems" not in role_properties["expected_roles"]

    # Callers receive a fresh wire mapping; mutation cannot alter authority.
    first = request.to_openai_response_format()
    first["json_schema"]["schema"]["properties"].clear()
    second = request.to_openai_response_format()
    assert second["json_schema"]["schema"]["properties"]


def test_shared_transport_shapes_preserve_every_scientific_constraint(monkeypatch):
    shared = payload_owner._planner_transport_schema(("strobe_2007",))
    monkeypatch.setattr(payload_owner, "_share_planner_transport_shapes", lambda value: value)
    expanded_source = payload_owner._planner_transport_schema(("strobe_2007",))

    def expand(node, definitions):
        if isinstance(node, dict):
            if "$ref" in node:
                assert set(node) == {"$ref"}
                return expand(definitions[node["$ref"].removeprefix("#/$defs/")], definitions)
            return {
                key: expand(value, definitions)
                for key, value in node.items()
                if key != "$defs"
            }
        if isinstance(node, list):
            return [expand(value, definitions) for value in node]
        return node

    assert expand(shared, shared["$defs"]) == expand(
        expanded_source, expanded_source["$defs"]
    )
    assert "presentation" in shared["$defs"]["PlannedFigurePanelSpec"]["properties"]
    assert len(json.dumps(shared)) < len(json.dumps(expanded_source))


def test_compact_definition_names_preserve_the_authority_schema():
    request = planner_structured_output_request()
    assert _schema_json(request.schema_json) == payload_owner._planner_transport_schema()


def _wire_model_requirement():
    return {
        "requirement_id": "age_adjusted",
        "outcome": "death",
        "outcome_type": "binary",
        "method_family": "statsmodels_logit_mle",
        "exposure_source": "stage",
        "analysis_role": "primary",
        "analysis_set": "source_aware",
        "covariates": ["age"],
        "covariate_rationales": [
            {"key": "age", "value": "Baseline age may influence exposure and outcome."}
        ],
        "covariate_temporal_roles": [{"key": "age", "value": "baseline_static"}],
    }


def test_covariate_decisions_use_closed_wire_rows_and_preserve_authority():
    schema = _schema_json(planner_structured_output_request().schema_json)
    properties = schema["$defs"]["PlannedModelRequirement"]["properties"]
    for field in ("covariate_rationales", "covariate_temporal_roles"):
        assert properties[field]["type"] == "array"
        row = properties[field]["items"]
        assert set(row["required"]) == {"key", "value"}
        assert row["additionalProperties"] is False
    assert properties["covariate_temporal_roles"]["items"]["properties"]["value"][
        "enum"
    ] == ["baseline_static", "at_or_before_time_zero"]

    source = {"steps": [{"model_requirements": [_wire_model_requirement()]}]}
    decoded = decode_planner_transport_payload(source)
    requirement = PlannedModelRequirement.model_validate(
        decoded["steps"][0]["model_requirements"][0]
    )
    assert requirement.covariate_rationales == {
        "age": "Baseline age may influence exposure and outcome."
    }
    assert requirement.covariate_temporal_roles == {"age": "baseline_static"}
    assert isinstance(source["steps"][0]["model_requirements"][0]["covariate_rationales"], list)


@pytest.mark.parametrize("field", ["covariate_rationales", "covariate_temporal_roles"])
@pytest.mark.parametrize(
    "rows",
    [
        [{"key": "age", "value": "first"}, {"key": " age ", "value": "second"}],
        [{"key": "age", "value": "first", "override": True}],
        [{"key": "", "value": "first"}],
        [{"key": ["age"], "value": "first"}],
    ],
)
def test_covariate_wire_rows_reject_ambiguous_or_malformed_decisions(field, rows):
    requirement = _wire_model_requirement()
    requirement[field] = rows
    with pytest.raises(PlannerStructuredOutputSchemaError):
        decode_planner_transport_payload({"steps": [{"model_requirements": [requirement]}]})


@pytest.mark.parametrize(
    ("field", "rows"),
    [
        ("covariate_rationales", [{"key": "age", "value": "short"}]),
        ("covariate_temporal_roles", [{"key": "age", "value": "after_time_zero"}]),
        ("covariate_temporal_roles", [{"key": "sex", "value": "baseline_static"}]),
    ],
)
def test_decoded_covariate_decisions_still_require_the_scientific_contract(field, rows):
    requirement = _wire_model_requirement()
    requirement[field] = rows
    decoded = decode_planner_transport_payload(
        {"steps": [{"model_requirements": [requirement]}]}
    )
    with pytest.raises(ValueError):
        PlannedModelRequirement.model_validate(decoded["steps"][0]["model_requirements"][0])


def test_run_bound_literature_schema_closes_source_element_pairs():
    request = planner_structured_output_request(
        ["strobe_2007", "singer_sepsis3_2016", "strobe_2007"]
    )
    schema = _schema_json(request.schema_json)
    step = schema["$defs"]["AnalysisStep"]["properties"]

    assert step["literature_citation_keys"]["items"]["enum"] == [
        "strobe_2007",
        "singer_sepsis3_2016",
    ]
    branches = schema["$defs"]["LiteratureDesignBinding"]["anyOf"]
    by_source = {
        source: branch
        for branch in branches
        for source in branch["properties"]["citation_key"]["enum"]
    }
    assert set(by_source) == {"strobe_2007", "singer_sepsis3_2016"}
    assert by_source["strobe_2007"]["properties"]["design_elements"]["items"][
        "enum"
    ] == ["dependence", "estimand", "outcome", "reporting"]
    assert "population" in by_source["singer_sepsis3_2016"]["properties"][
        "design_elements"
    ]["items"]["enum"]


def test_run_bound_literature_schema_groups_only_identical_authority_sets():
    schema = _schema_json(
        planner_structured_output_request(
            [
                "suissa_immortal_time_2008",
                "anderson_landmark_1983",
                "singer_sepsis3_2016",
                "johnson_mimiciv_2023",
                "strobe_2007",
            ]
        ).schema_json
    )
    branches = schema["$defs"]["LiteratureDesignBinding"]["anyOf"]
    groups = {
        tuple(branch["properties"]["citation_key"]["enum"]): tuple(
            branch["properties"]["design_elements"]["items"]["enum"]
        )
        for branch in branches
    }

    assert groups[
        ("suissa_immortal_time_2008", "anderson_landmark_1983")
    ] == ("estimand", "exposure", "time_zero")
    assert groups[("singer_sepsis3_2016", "johnson_mimiciv_2023")]
    assert ("strobe_2007",) in groups
    assert len(groups) == 3


def test_ten_source_run_bound_literature_schema_keeps_retry_headroom():
    request = planner_structured_output_request(
        [
            "vincent_sofa_1996",
            "singer_sepsis3_2016",
            "strobe_2007",
            "record_2015",
            "suissa_immortal_time_2008",
            "anderson_landmark_1983",
            "durrleman_splines_1989",
            "sterne_missing_data_2009",
            "ricu_2023",
            "johnson_mimiciv_2023",
        ]
    )
    schema = _schema_json(request.schema_json)

    # Ratchet (reviewed 2026-09-17): 30_000 -> 30_100 to pay for the deliberate
    # planner-settable ``AnalysisStep.allow_fallback_as_primary`` permission
    # flag (C-F9: fallback-as-primary requires explicit plan permission).
    # Any further growth must be justified here, not absorbed silently.
    assert request.payload_bytes < 30_100
    assert "CandidateLiteratureDesignDecision" not in request.schema_json
    assert len(schema["$defs"]["LiteratureDesignBinding"]["anyOf"]) == 6


@pytest.mark.parametrize("runtime_suffix", [False, True])
def test_descriptive_interval_limitations_are_not_planner_transport_fields(runtime_suffix):
    request = (
        payload_owner.runtime_suffix_structured_output_request(
            allowed_literature_citation_keys=[],
            replace_from_step_id="distribution",
            planned_analysis_role="secondary",
            allowed_inputs=["dataset:cohort"],
            expected_outputs=["table:distribution"],
            scientific_action_ids=[],
        )
        if runtime_suffix else planner_structured_output_request()
    )
    definitions = _schema_json(request.schema_json)["$defs"]
    distribution = definitions["ExposureOutcomeDistributionSpec"]

    assert "interval_limitation" not in distribution["properties"]
    assert "interval_limitation" not in distribution["required"]
    assert "DescriptiveIntervalLimitation" not in definitions


@pytest.mark.parametrize("transport", ["plan", "replan", "runtime_suffix"])
def test_planner_cannot_write_a_host_compiled_interval_limitation(transport):
    limitation = {
        "declared_variance_estimator": "model_based",
        "reason": "repeated_stays_possible_without_verified_patient_grouping",
    }
    step = {
        "step_id": "distribution",
        "intent": "Report the declared exposure and outcome distribution.",
        "inputs": ["exposure", "outcome"],
        "expected_outputs": ["table:exposure_outcome_distribution"],
        "exposure_outcome_distribution_spec": {
            "schema_version": "easyicu.exposure_outcome_distribution/3",
            "exposure": "exposure",
            "exposure_levels": [0, 1],
            "outcome": "outcome",
            "outcome_levels": [0, 1],
            "outcome_positive_value": 1,
            "level_match_policy": "exact_typed",
            "denominator_policy": "observed_outcome_rows",
            "missing_outcome_policy": "exclude_from_denominator",
            "interval_method": "none_counts_only",
            "repeated_unit_interval_method": None,
            "confidence_level": None,
            "interval_limitation": limitation,
        },
    }
    payload = {"steps": [step]}
    assert AnalysisStep.model_validate(
        step,
    ).exposure_outcome_distribution_spec.interval_limitation is not None

    with pytest.raises(PlannerHostOwnedFieldError) as caught:
        if transport == "runtime_suffix":
            payload_owner.parse_runtime_plan_suffix(json.dumps({
                "replace_from_step_id": "distribution",
                "replacement_step": step,
                "rationale": "Keep the declared distribution population.",
            }))
        else:
            decode_planner_transport_payload(
                payload, host_owned_plan=payload if transport == "replan" else None,
            )

    assert caught.value.issue_code == "planner_host_owned_field_written"
    assert caught.value.path == "steps[0].exposure_outcome_distribution_spec"
    assert caught.value.fields == ("interval_limitation",)
    assert step["exposure_outcome_distribution_spec"]["interval_limitation"] == limitation


def test_planner_null_interval_limitation_leaves_the_host_to_compile_it():
    spec = {"interval_limitation": None, "interval_method": "wilson"}
    payload = {"steps": [{"exposure_outcome_distribution_spec": spec}]}

    decoded = decode_planner_transport_payload(payload)

    assert decoded["steps"][0]["exposure_outcome_distribution_spec"] == {
        "interval_method": "wilson",
    }
    assert spec == {"interval_limitation": None, "interval_method": "wilson"}


def test_single_version_tags_are_host_defaults_but_version_choices_remain():
    definitions = _schema_json(planner_structured_output_request().schema_json)["$defs"]

    for name in (
        "CohortDefinitionSpec", "DescriptiveClaimContract",
        "ExposureOutcomeRiskDifferenceContrast", "FunctionalFormSpec",
        "MeasurementAuditSpec", "PhenotypeComparisonSpec",
        "PlannedDependenceRequirement", "PlannedFigurePanelSpec",
        "RobustnessReplaySpec",
    ):
        assert "schema_version" not in definitions[name]["properties"]
        assert "schema_version" not in definitions[name]["required"]

    assert definitions["ExposureOutcomeDistributionSpec"]["properties"][
        "schema_version"
    ]["enum"] == [
        "easyicu.exposure_outcome_distribution/2",
        "easyicu.exposure_outcome_distribution/3",
    ]
    assert definitions["TableOneSpec"]["properties"]["schema_version"]["enum"] == [
        "easyicu.table_one/1", "easyicu.table_one/2", "easyicu.table_one/3",
    ]
    # These literal declarations are scientific assumptions, not version tags.
    assumptions = definitions["PatientWeightedIntervalAssumptions"]
    assert set(assumptions["required"]) == {"target", "independence", "weights"}
    assert assumptions["properties"]["weights"]["const"] == (
        "fixed_observed_patient_stay_weights"
    )


@pytest.mark.parametrize(
    ("model_name", "payload"),
    [
        ("ExposureOutcomeRiskDifferenceContrast", {
            "reference_exposure_level": 0, "comparison_exposure_level": 1,
        }),
        ("FunctionalFormSpec", {
            "target_column": "lactate", "knot_quantiles": [0.1, 0.5, 0.9],
        }),
    ],
)
def test_the_host_restores_a_projected_version_without_changing_the_design(
    model_name, payload,
):
    from easyicu.research_agent import schema as authority

    model = getattr(authority, model_name)
    restored = model.model_validate(payload)
    explicit = model.model_validate({**payload, "schema_version": restored.schema_version})
    assert restored.model_dump(mode="json") == explicit.model_dump(mode="json")
    assert restored.schema_version.startswith("easyicu.")
    with pytest.raises(ValueError):
        model.model_validate({**payload, "schema_version": "invented/99"})


def test_trajectory_stability_transport_keeps_only_planner_decisions():
    request = planner_structured_output_request()
    schema = _schema_json(request.schema_json)
    properties = schema["$defs"]["TrajectoryStabilitySpec"]["properties"]

    assert set(properties) == {
        "n_resamples",
        "sample_fraction",
        "sample_size",
        "minimum_mean_stability",
        "base_seed",
        "refit_max_iter",
        "refit_tolerance",
        "refit_regularization",
    }
    assert "resampling_method" not in properties
    assert "minimum_successful_resamples" not in properties
    assert "decision_mode" not in properties

    decoded = decode_planner_transport_payload(
        {
            "steps": [
                {
                    "trajectory_stability_spec": {
                        "n_resamples": 25,
                        "sample_fraction": 0.8,
                        "sample_size": None,
                        "minimum_mean_stability": None,
                        "base_seed": 123,
                        "refit_max_iter": None,
                        "refit_tolerance": None,
                        "refit_regularization": None,
                    }
                }
            ]
        }
    )
    raw_spec = decoded["steps"][0]["trajectory_stability_spec"]
    assert raw_spec == {
        "n_resamples": 25,
        "sample_fraction": 0.8,
        "sample_size": None,
        "minimum_mean_stability": None,
        "base_seed": 123,
    }
    spec = TrajectoryStabilitySpec.model_validate(raw_spec)
    assert spec.base_seed == 123
    assert spec.minimum_successful_resamples == 25
    assert spec.decision_mode == "report_only"


def test_empty_run_bound_literature_schema_requires_empty_arrays():
    schema = _schema_json(planner_structured_output_request([]).schema_json)
    step = schema["$defs"]["AnalysisStep"]["properties"]

    assert step["literature_citation_keys"]["maxItems"] == 0
    assert step["literature_design_bindings"]["maxItems"] == 0


def test_transport_projection_only_decodes_labels_and_null_placeholders():
    source = {
        "research_question": "q",
        "display_labels": [
            {"key": "x", "value": "Exposure"},
            {"key": "y", "value": "Outcome"},
        ],
        "robustness_specs": [
            {
                "spec_id": "cc",
                "missing_override": {
                    "strategy": "complete_case",
                    "variables": ["x", "y"],
                    "audit_flags": None,
                },
                "outcome_override": {
                    "column": None,
                    "concept_id": None,
                    "target": None,
                    "event_time_column": None,
                    "time_column": None,
                    "aggregation": None,
                },
            }
        ],
        "steps": [
            {
                "literature_citation_keys": ["strobe_2007"],
                "literature_design_bindings": [
                    {"citation_key": "sterne_missing_data_2009"}
                ],
            }
        ],
    }

    decoded = decode_planner_transport_payload(source)

    assert decoded["display_labels"] == {"x": "Exposure", "y": "Outcome"}
    assert decoded["robustness_specs"][0]["missing_override"] == {
        "strategy": "complete_case",
        "variables": ["x", "y"],
    }
    assert decoded["robustness_specs"][0]["outcome_override"] is None
    assert decoded["steps"][0]["literature_citation_keys"] == [
        "strobe_2007",
        "sterne_missing_data_2009",
    ]
    assert isinstance(source["display_labels"], list), "input must not be mutated"
    assert source["steps"][0]["literature_citation_keys"] == ["strobe_2007"]


def test_transport_projection_rejects_duplicate_display_label_keys():
    with pytest.raises(PlannerStructuredOutputSchemaError, match="repeats key"):
        decode_planner_transport_payload(
            {
                "display_labels": [
                    {"key": "x", "value": "first"},
                    {"key": "x", "value": "second"},
                ]
            }
        )


def test_strict_planner_prompt_does_not_duplicate_the_wire_schema_example():
    context = ResearchContext(
        research_question="Describe the ICU cohort.",
        cohort=CohortDescriptor(
            cohort_name="strict-probe",
            database="synthetic",
            n_patients=10,
            n_stays=10,
            id_columns=["stay_id"],
        ),
        variables=[],
    )

    ordinary = PlannerAgent.request_messages(context)[1].content
    strict = PlannerAgent.request_messages(
        context, strict_transport_schema=True
    )[1].content

    assert "Required JSON shape (truncated example)" in ordinary
    assert "Required JSON shape (truncated example)" not in strict
    assert "HOST-ENFORCED STRICT JSON SCHEMA" in strict
    assert "RESEARCH CONTEXT:" in strict
    assert "schema_version='easyicu.exposure_outcome_distribution/2'" in strict
    assert "interval_method='wilson'" in strict
    assert "repeated_unit_interval_method='patient_cluster_robust_wald'" in strict
    assert "even while dependence is null before host binding" in strict
    assert len(strict.encode("utf-8")) < len(ordinary.encode("utf-8"))


def test_planner_schema_capability_and_request_survive_runtime_wrapper_chain(tmp_path):
    """Exercise the production envelope -> hard-stop -> meter composition."""

    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopLedger,
        ProviderHardStopLimits,
    )
    from easyicu.research_agent.providers.capabilities import (
        llm_supports_strict_json_schema,
    )
    from easyicu.research_agent.providers.cost import CostMeter, MeteredClient
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.research_agent.replication.envelope import (
        ReproEnvelope,
        ReproRecordingClient,
    )

    observed = {}

    class _StrictLeaf:
        name = "strict-leaf"
        _model = "strict-model"
        supports_strict_json_schema = True

        def complete_with_usage(
            self,
            messages,
            *,
            max_tokens=2048,
            temperature=0.2,
            structured_output=None,
        ):
            observed["structured_output"] = structured_output
            return "{}", {
                "prompt_tokens": 11,
                "completion_tokens": 1,
                "actual_model": self._model,
            }

    limits = ProviderHardStopLimits(
        max_provider_attempts_per_run=2,
        max_provider_attempts_per_batch=2,
        max_total_tokens_per_run=1_000_000,
        max_total_tokens_per_batch=1_000_000,
        max_estimated_cost_usd_per_batch=10.0,
        max_wall_clock_seconds_per_task=60.0,
        input_cost_usd_per_million_tokens=1.0,
        output_cost_usd_per_million_tokens=2.0,
    )
    ledger = ProviderHardStopLedger(
        path=tmp_path / "provider_progress.json",
        task_ids=("E1",),
        limits=limits,
        batch_id="wrapper-chain",
    )
    envelope = ReproEnvelope(run_id="wrapper-chain")
    recorded = ReproRecordingClient(
        _StrictLeaf(), role="planner", envelope=envelope
    )
    stopped = HardStopClient(
        recorded, role="planner", task=ledger.start_task("E1")
    )
    client = MeteredClient(
        stopped,
        role="planner",
        meter=CostMeter(runtime_dir=tmp_path / "cost-runtime"),
    )
    request = planner_structured_output_request()

    assert llm_supports_strict_json_schema(client) is True
    response, usage = client.complete_with_usage(
        [LLMMessage(role="user", content="return an object")],
        max_tokens=8,
        temperature=0.0,
        structured_output=request,
    )

    assert response == "{}"
    assert usage["total_tokens"] == 12
    assert observed["structured_output"] is request
    assert len(envelope.calls) == 1


@pytest.mark.parametrize(
    "payload,field",
    [
        ({"adjustment_proposal": {"covariates": ["age"]}}, "adjustment_proposal"),
        ({"cohort": {"unapplied_population_criteria": ["adult"]}}, "unapplied_population_criteria"),
        ({"steps": [{"figure_panels": [{"policy_alternative_chart_types": ["bar_chart"]}]}]}, "policy_alternative_chart_types"),
    ],
)
def test_non_strict_transport_refuses_host_owned_plan_fields(payload: dict, field: str) -> None:
    with pytest.raises(PlannerHostOwnedFieldError, match=field):
        decode_planner_transport_payload(payload)


def test_replan_inherits_host_roster_without_accepting_model_revisions() -> None:
    prior = {"adjustment_proposal": {"source_step_id": "primary", "covariates": ["age", "sex"]}}
    assert decode_planner_transport_payload({}, host_owned_plan=prior) == prior
    assert decode_planner_transport_payload(prior, host_owned_plan=prior) == prior
    with pytest.raises(PlannerHostOwnedFieldError, match="adjustment_proposal"):
        decode_planner_transport_payload(
            {"adjustment_proposal": {"source_step_id": "primary", "covariates": ["age"]}},
            host_owned_plan=prior,
        )
    assert prior["adjustment_proposal"]["covariates"] == ["age", "sex"]


def test_replan_chart_policy_requires_same_host_panel_binding() -> None:
    panel = {"panel_id": "A", "figure_output": "figure:primary", "article_role": "primary_result",
             "chart_type": "forest_plot", "source_products": ["table:effect"],
             "policy_alternative_chart_types": ["bar_chart"]}
    prior = {"steps": [{"step_id": "plot", "figure_panels": [panel]}]}
    assert decode_planner_transport_payload(prior, host_owned_plan=prior) == prior
    changed = {"steps": [{"step_id": "plot", "figure_panels": [{**panel, "source_products": ["table:other"]}]}]}
    with pytest.raises(PlannerHostOwnedFieldError, match="policy_alternative_chart_types"):
        decode_planner_transport_payload(changed, host_owned_plan=prior)
