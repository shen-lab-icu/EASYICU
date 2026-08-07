"""Contract tests for optional scientific-library adapter POCs.

These are deliberately adapter-unit tests.  They prove the library boundary,
receipts, and fail-closed behaviour without pretending a local developer
environment has approved a new execution image or publication capability.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from easyicu.research_agent.scientific_adapters.runtime import ExternalAdapterRuntime


def _available(adapter_id: str, version: str) -> ExternalAdapterRuntime:
    spec = {
        "pandera_dataframe_contract_v1": ("pandera", "pandera.pandas"),
        "dowhy_identification_v1": ("dowhy", "dowhy"),
        "sksurv_competing_risks_cif_v1": ("scikit-survival", "sksurv"),
    }[adapter_id]
    return ExternalAdapterRuntime(
        adapter_id=adapter_id,
        package_name=spec[0],
        import_name=spec[1],
        status="available",
        installed_version=version,
        issue_code=None,
    )


def test_adapter_specs_create_existing_non_installable_capability_requests() -> None:
    from easyicu.research_agent.scientific_adapters.runtime import (
        EXTERNAL_ADAPTER_SPECS,
        build_external_adapter_request,
    )

    assert {spec.adapter_id for spec in EXTERNAL_ADAPTER_SPECS} == {
        "pandera_dataframe_contract_v1",
        "dowhy_identification_v1",
        "sksurv_competing_risks_cif_v1",
    }
    request = build_external_adapter_request(
        adapter_id="sksurv_competing_risks_cif_v1",
        requested_by="planner:adapter-poc",
        requested_at="2026-08-07T00:00:00Z",
        runtime_import_names=("pandas", "numpy", "statsmodels"),
    )
    assert request.import_name == "sksurv"
    assert request.runtime_install_allowed is False
    assert request.produced_output_roles == ("cumulative_incidence_curve",)


def test_unavailable_adapter_is_observed_without_becoming_a_runtime_capability(
    monkeypatch,
) -> None:
    from easyicu.research_agent.scientific_adapters import runtime

    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _name: None)
    receipt = runtime.probe_external_adapter("pandera_dataframe_contract_v1")

    assert receipt.status == "unavailable"
    assert receipt.issue_code == "external_adapter_dependency_unavailable"
    assert receipt.to_dict()["schema_version"] == "easyicu.external_adapter_runtime/1"


def test_pandera_adapter_builds_a_strict_non_coercing_schema(monkeypatch) -> None:
    from easyicu.research_agent.scientific_adapters import pandera

    captured: dict[str, object] = {}

    class SchemaError(Exception):
        pass

    class Check:
        @staticmethod
        def isin(values):
            return ("isin", tuple(values))

        @staticmethod
        def ge(value):
            return ("ge", value)

        @staticmethod
        def le(value):
            return ("le", value)

    def column(dtype, **kwargs):
        return {"dtype": dtype, **kwargs}

    class DataFrameSchema:
        def __init__(self, columns, **kwargs):
            captured["columns"] = columns
            captured["schema_kwargs"] = kwargs

        def validate(self, dataframe, *, lazy):
            captured["dataframe"] = dataframe
            captured["lazy"] = lazy
            return dataframe

    fake_pa = SimpleNamespace(
        Check=Check,
        Column=column,
        DataFrameSchema=DataFrameSchema,
    )
    monkeypatch.setattr(
        pandera,
        "probe_external_adapter",
        lambda _adapter_id: _available("pandera_dataframe_contract_v1", "0.32.0"),
    )
    monkeypatch.setattr(
        pandera.importlib,
        "import_module",
        lambda name: fake_pa if name == "pandera.pandas" else SimpleNamespace(
            SchemaError=SchemaError, SchemaErrors=SchemaError
        ),
    )
    contract = pandera.PanderaDataFrameContract(
        contract_id="generic_binary_input",
        columns=(
            pandera.PanderaColumnContract(
                name="outcome",
                dtype="int",
                nullable=False,
                allowed_values=(0, 1),
            ),
            pandera.PanderaColumnContract(
                name="age",
                dtype="float",
                minimum=18.0,
                maximum=120.0,
            ),
        ),
    )

    receipt = pandera.validate_dataframe_contract(pd.DataFrame({"outcome": [0]}), contract)

    assert receipt.status == "validated"
    assert receipt.adapter_version == "0.32.0"
    assert captured["lazy"] is True
    assert captured["schema_kwargs"] == {
        "strict": True,
        "coerce": False,
        "name": "generic_binary_input",
    }
    outcome = captured["columns"]["outcome"]
    assert outcome["coerce"] is False
    assert outcome["checks"] == [("isin", (0, 1))]


def test_pandera_validation_failure_never_echoes_source_values(monkeypatch) -> None:
    from easyicu.research_agent.scientific_adapters import pandera

    class SchemaErrors(Exception):
        pass

    class DataFrameSchema:
        def __init__(self, *_args, **_kwargs):
            pass

        def validate(self, *_args, **_kwargs):
            raise SchemaErrors("raw patient value=secret")

    fake_pa = SimpleNamespace(
        Check=SimpleNamespace(isin=lambda _values: object(), ge=lambda _value: object(), le=lambda _value: object()),
        Column=lambda *_args, **_kwargs: object(),
        DataFrameSchema=DataFrameSchema,
    )
    monkeypatch.setattr(
        pandera,
        "probe_external_adapter",
        lambda _adapter_id: _available("pandera_dataframe_contract_v1", "0.32.0"),
    )
    monkeypatch.setattr(
        pandera.importlib,
        "import_module",
        lambda name: fake_pa if name == "pandera.pandas" else SimpleNamespace(
            SchemaError=SchemaErrors, SchemaErrors=SchemaErrors
        ),
    )

    receipt = pandera.validate_dataframe_contract(
        pd.DataFrame({"outcome": [0]}),
        pandera.PanderaDataFrameContract(
            contract_id="private_values",
            columns=(pandera.PanderaColumnContract(name="outcome", dtype="int"),),
        ),
    )

    assert receipt.status == "invalid"
    assert receipt.issue_code == "dataframe_contract_invalid"
    assert "secret" not in str(receipt.to_dict())


def test_dowhy_adapter_only_records_identification(monkeypatch) -> None:
    from easyicu.research_agent.scientific_adapters import dowhy

    captured: dict[str, object] = {}

    class CausalModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def identify_effect(self, **kwargs):
            captured["identify_kwargs"] = kwargs
            return object()

    monkeypatch.setattr(
        dowhy,
        "probe_external_adapter",
        lambda _adapter_id: _available("dowhy_identification_v1", "0.13.0"),
    )
    monkeypatch.setattr(
        dowhy.importlib,
        "import_module",
        lambda _name: SimpleNamespace(CausalModel=CausalModel),
    )
    contract = dowhy.DoWhyIdentificationContract(
        treatment="treated",
        outcome="death",
        causal_graph="digraph { treated -> death; age -> treated; age -> death; }",
        observed_common_causes=("age",),
        identifier="id-algorithm",
    )

    receipt = dowhy.identify_declared_causal_effect(
        pd.DataFrame({"treated": [0, 1], "death": [0, 1], "age": [60, 70]}),
        contract,
    )

    assert receipt.status == "identified"
    assert receipt.to_dict()["effect_estimate_present"] is False
    assert captured["common_causes"] == ["age"]
    assert captured["identify_kwargs"] == {"method_name": "id-algorithm"}


def test_sksurv_adapter_preserves_declared_event_codes(monkeypatch) -> None:
    from easyicu.research_agent.scientific_adapters import sksurv

    captured: dict[str, object] = {}

    def cumulative_incidence_competing_risks(event, time_exit, **kwargs):
        captured["event"] = event
        captured["time"] = time_exit
        captured["kwargs"] = kwargs
        return np.asarray([1.0, 3.0]), np.asarray(
            [[0.0, 0.1], [0.0, 0.15], [0.0, 0.25]]
        )

    monkeypatch.setattr(
        sksurv,
        "probe_external_adapter",
        lambda _adapter_id: _available("sksurv_competing_risks_cif_v1", "0.28.0"),
    )
    monkeypatch.setattr(
        sksurv.importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            cumulative_incidence_competing_risks=cumulative_incidence_competing_risks
        ),
    )
    contract = sksurv.CompetingRisksCIFContract(
        time_column="days",
        event_column="event_type",
        event_of_interest=8,
        competing_event_codes=(2,),
    )

    result = sksurv.estimate_declared_cumulative_incidence(
        pd.DataFrame({"days": [1.0, 3.0, 2.0], "event_type": [0, 8, 2]}),
        contract,
    )

    assert result.status == "estimated"
    assert result.event_code_mapping == ((2, 1), (8, 2))
    assert result.cumulative_incidence == (0.0, 0.25)
    assert captured["event"].tolist() == [0, 2, 1]
    assert captured["kwargs"] == {"conf_type": None}


def test_sksurv_adapter_never_converts_an_invalid_event_code_to_censoring() -> None:
    from easyicu.research_agent.scientific_adapters import sksurv

    result = sksurv.estimate_declared_cumulative_incidence(
        pd.DataFrame({"days": [1.0, 2.0, 3.0], "event_type": [-1, 8, 2]}),
        sksurv.CompetingRisksCIFContract(
            time_column="days",
            event_column="event_type",
            event_of_interest=8,
            competing_event_codes=(2,),
        ),
    )

    assert result.status == "input_contract_failed"
    assert result.issue_code == "competing_risk_event_codes_invalid"


def test_adapters_do_not_upgrade_current_scientific_capabilities() -> None:
    from easyicu.research_agent.planning.capability_registry import get_capability
    from easyicu.research_agent.execution.method_capabilities import (
        coder_method_capability_block,
    )

    causal = get_capability("causal_emulation")
    assert causal is not None
    assert causal.scientific_validation == "analysis_only"
    prompt = coder_method_capability_block(snapshot=("pandas", "numpy"))
    assert "pandera.pandas" not in prompt
    assert "dowhy" not in prompt
    assert "sksurv" not in prompt
