"""Converter-schema regressions exposed by synthetic sources, with tiny records."""

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept import ConceptResolver, ConceptSource, _apply_callback
from easyicu.table import ICUTable


@pytest.mark.parametrize("origin,hour", [(0, 1), (1800, 1), (21600, 1), (18000000, 1), (21600, 6000)])
def test_sic_rate_uses_the_icu_clock_once(origin, hour):
    cases = pd.DataFrame({"CaseID": [17], "ICUOffset": [origin], "TimeOfStay": [origin + (hour + 72) * 3600]})
    data_source = SimpleNamespace(config=SimpleNamespace(name="sic"), load_table=lambda *a, **kw: cases)
    source = ConceptSource.from_mapping({"table": "medication", "val_var": "AmountPerMinute",
                                        "index_var": "Offset", "stop_var": "OffsetDrugEnd",
                                        "callback": "sic_rate_kg"})
    result = _apply_callback(pd.DataFrame({"CaseID": [17], "Offset": [origin + hour * 3600],
                                          "OffsetDrugEnd": [origin + (hour + 1) * 3600], "weight": [80.],
                                          "norepi_rate": [.000001]}), source,
                             concept_name="norepi_rate", data_source=data_source)
    aligned = ConceptResolver.__new__(ConceptResolver)._align_time_to_admission(
        result, data_source, ["CaseID"], "Offset")
    assert aligned.Offset.tolist() == [hour, hour + 1]
    assert aligned.norepi_rate.tolist() == pytest.approx([.0125, .0125])


def test_aumc_fluid_totals_do_not_inherit_the_drug_dose_unit():
    source = ConceptSource.from_mapping({"table": "drugitems", "val_var": "fluidin",
                                        "index_var": "start", "end_var": "stop",
                                        "alternate_value_var": "solutionadministered",
                                        "callback": "distribute_volume_hourly"})
    data_source = SimpleNamespace(config=SimpleNamespace(name="aumc"), load_table=lambda *a, **kw: ICUTable(
        data=pd.DataFrame({"admissionid": [17], "admittedat": [1000.]}),
        id_columns=["admissionid"], index_column="admittedat"))
    frame = pd.DataFrame({"admissionid": [17], "start": [1000.], "stop": [1060.],
                          "total_input_ml": [50.], "solutionadministered": [60.], "doseunit": ["mg"]})
    result = _apply_callback(frame, source, concept_name="total_input_ml", data_source=data_source, unit_column="doseunit")
    assert result.total_input_ml.tolist() == [60.]


def test_sic_rate_refuses_an_unknown_icu_origin():
    cases = pd.DataFrame({"CaseID": [17], "ICUOffset": [float("nan")]})
    data_source = SimpleNamespace(config=SimpleNamespace(name="sic"), load_table=lambda *a, **kw: cases)
    source = ConceptSource.from_mapping({"table": "medication", "val_var": "AmountPerMinute",
                                        "index_var": "Offset", "stop_var": "OffsetDrugEnd", "callback": "sic_rate_kg"})
    with pytest.raises(ValueError, match="ICUOffset"):
        _apply_callback(pd.DataFrame({"CaseID": [17], "Offset": [3600], "OffsetDrugEnd": [7200],
                                      "weight": [80.], "norepi_rate": [.000001]}), source,
                        concept_name="norepi_rate", data_source=data_source)
