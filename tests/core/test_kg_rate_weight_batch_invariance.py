"""A per-kg rate's weight is resolved per stay, whatever stays share the frame.

lead-r13 aumcwt: AUMC weight concatenates admissions rows (weightgroup, which
carry ``admittedat``) with measured numericitems weights (which do not).  The
admission alignment took the origin from the frame whenever the column was
present, so a stay without a weightgroup had no origin unless another stay of
the same frame contributed an admissions row.  In a 50-stay batch whose only
epinephrine stay was such a stay, the one-stay weight load raised, the
``aumc_rate_kg`` dispatch swallowed it and every rate of that stay became NA;
the unbatched run, whose frame held other stays, kept them.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptResolver, ConceptSource, _apply_callback

MEASURED_ONLY = 7  # the stay without a weightgroup
STAYS = list(range(1, 51))
ADMITTED = {stay: 600.0 * stay for stay in STAYS}  # absolute source minutes


def _aumc(admission_ids=STAYS):
    admissions = pd.DataFrame(
        {
            "admissionid": list(admission_ids),
            "admittedat": [ADMITTED[s] for s in admission_ids],
            "dischargedat": [ADMITTED[s] + 6000.0 for s in admission_ids],
        }
    )
    return SimpleNamespace(
        config=SimpleNamespace(name="aumc"),
        load_table=lambda *_args, **_kwargs: admissions,
    )


def _weight_sources(stays):
    """The two AUMC weight sources for ``stays``, concatenated as the loader does."""
    grouped = [s for s in stays if s != MEASURED_ONLY]
    admissions = pd.DataFrame(
        {
            "admissionid": grouped,
            "weight": [75.0] * len(grouped),
            "admittedat": [ADMITTED[s] for s in grouped],
            "dischargedat": [ADMITTED[s] + 6000.0 for s in grouped],
        },
        columns=["admissionid", "weight", "admittedat", "dischargedat"],
    )
    measured = [s for s in stays if s == MEASURED_ONLY]
    numeric = pd.DataFrame(
        {
            "admissionid": measured * 3,
            "weight": [79.0, 80.0, 81.0][: 3 * len(measured)],
            "measuredat": [ADMITTED[MEASURED_ONLY] + m for m in (-60.0, 60.0, 600.0)][: 3 * len(measured)],
        },
        columns=["admissionid", "weight", "measuredat"],
    )
    return pd.concat([admissions, numeric], ignore_index=True)


class _WeightResolver:
    """Loads AUMC weight for the requested stays through the alignment owner."""

    def __init__(self, source):
        self.source = source
        self.owner = ConceptResolver.__new__(ConceptResolver)
        self.owner._aumc_admissions_cache = None

    def _load_single_concept(self, concept_name, data_source, *, patient_ids, **_kwargs):
        assert concept_name == "weight"
        stays = list(patient_ids["admissionid"])
        aligned = self.owner._align_time_to_admission(
            _weight_sources(stays), data_source, ["admissionid"], "measuredat",
            clip_icu_episode=False,
        )
        return SimpleNamespace(data=aligned)


def _epi_rows(stays):
    return pd.DataFrame(
        {
            "admissionid": stays,
            "start": [ADMITTED[s] + 120.0 for s in stays],
            "stop": [ADMITTED[s] + 240.0 for s in stays],
            "dose": [8.0] * len(stays),
            "doseunit": ["µg"] * len(stays),
            "doserateunit": ["min"] * len(stays),
            "doserateperkg": [False] * len(stays),
        }
    )


def _epi_rate(stays, source=None):
    source = source or _aumc()
    return _apply_callback(
        _epi_rows(stays),
        ConceptSource(
            callback="aumc_rate_kg", table="drugitems", value_var="dose",
            unit_var="doseunit", index_var="start",
            params={"rel_weight": "doserateperkg", "rate_uom": "doserateunit", "stop_var": "stop"},
        ),
        concept_name="epi_rate",
        resolver=_WeightResolver(source),
        data_source=source,
    )


def _rows_of(result, stay):
    rows = result.loc[result["admissionid"] == stay]
    return rows.drop(columns=["admissionid"]).reset_index(drop=True)


def test_measured_weight_stay_alone_in_its_frame_keeps_its_per_kg_rate():
    alone = _epi_rate([MEASURED_ONLY])
    shared = _epi_rate(STAYS)

    assert not _rows_of(alone, MEASURED_ONLY).empty
    assert _rows_of(alone, MEASURED_ONLY)["epi_rate"].notna().all()
    assert _rows_of(alone, MEASURED_ONLY)["epi_rate"].to_numpy() == pytest.approx(8.0 / 80.0)
    pd.testing.assert_frame_equal(_rows_of(alone, MEASURED_ONLY), _rows_of(shared, MEASURED_ONLY))


def test_admission_origin_comes_from_the_admissions_table_for_rows_without_one():
    owner = ConceptResolver.__new__(ConceptResolver)
    owner._aumc_admissions_cache = None
    frame = _weight_sources([MEASURED_ONLY])  # admittedat column present, all NA

    out = owner._align_time_to_admission(
        frame, _aumc(), ["admissionid"], "measuredat", clip_icu_episode=False,
    )

    assert out["measuredat"].tolist() == pytest.approx([-1.0, 1.0, 10.0])
    assert "admittedat" not in out.columns and "dischargedat" not in out.columns


def test_empty_weight_input_aligns_to_an_empty_frame():
    owner = ConceptResolver.__new__(ConceptResolver)
    owner._aumc_admissions_cache = None
    empty = _weight_sources([]).iloc[0:0]

    out = owner._align_time_to_admission(
        empty, _aumc(), ["admissionid"], "measuredat", clip_icu_episode=False,
    )

    assert out.empty
    assert "admittedat" not in out.columns


def test_alignment_failure_in_the_weight_load_raises_instead_of_na_rates():
    # The admissions table has no row for the stay: a real alignment failure.
    source = _aumc(admission_ids=[1, 2])

    with pytest.raises(RuntimeError, match="aumc_rate_kg_weight_load_failed") as info:
        _epi_rate([MEASURED_ONLY], source=source)
    assert isinstance(info.value.__cause__, ValueError)
    assert "admittedat" in str(info.value.__cause__)


def test_aumc_rate_kg_weight_load_failure_raises():
    class _Failing:
        def _load_single_concept(self, *_args, **_kwargs):
            raise OSError("read failed")

    with pytest.raises(RuntimeError, match="aumc_rate_kg_weight_load_failed"):
        _apply_callback(
            _epi_rows([1]),
            ConceptSource(
                callback="aumc_rate_kg", table="drugitems", value_var="dose",
                unit_var="doseunit", index_var="start",
                params={"rel_weight": "doserateperkg", "rate_uom": "doserateunit", "stop_var": "stop"},
            ),
            concept_name="epi_rate",
            resolver=_Failing(),
            data_source=_aumc(),
        )


def test_legacy_loader_weight_load_failure_raises():
    from easyicu.load_concepts import ConceptLoader

    loader = ConceptLoader.__new__(ConceptLoader)
    loader._src_name = "aumc"

    def _fail(**_kwargs):
        raise OSError("read failed")

    loader._load_one_concept = _fail
    frame = pd.DataFrame({"admissionid": [1], "dose": [np.float64(1.0)]})
    with pytest.raises(RuntimeError, match="kg_rate_weight_load_failed"):
        loader._get_patient_weights(frame, "admissionid", "icustay")


# --- Legacy ConceptLoader: a read failure is never "no data" (delta13w D13W-LEGACY-CATCH)


def _legacy_loader():
    from easyicu.load_concepts import ConceptLoader

    loader = ConceptLoader.__new__(ConceptLoader)
    loader._src_name = "aumc"
    loader._columns_for_source = lambda *_args, **_kwargs: None
    loader._preload_tables = lambda *_args, **_kwargs: None
    return loader


def test_legacy_source_read_failure_raises():
    loader = _legacy_loader()

    def _fail(*_args, **_kwargs):
        raise OSError("read failed")

    loader._safe_load_table = _fail
    with pytest.raises(OSError, match="read failed"):
        loader._load_concept_source(
            source=ConceptSource(table="numericitems", sub_var="itemid", ids=[1]),
            concept_name="weight", patient_ids=None, id_type="icustay", interval=None,
        )


def test_legacy_missing_table_source_is_empty():
    from easyicu.datasource import MissingTableSourceError

    loader = _legacy_loader()

    def _absent(*_args, **_kwargs):
        raise MissingTableSourceError("no table source registered")

    loader._safe_load_table = _absent
    out = loader._load_concept_source(
        source=ConceptSource(table="numericitems", sub_var="itemid", ids=[1]),
        concept_name="weight", patient_ids=None, id_type="icustay", interval=None,
    )
    assert out.empty


@pytest.mark.parametrize("workers", [1, 2])
def test_legacy_load_concepts_refuses_partial_results_in_both_modes(workers):
    from easyicu.concept import load_dictionary

    dictionary = load_dictionary("miiv")
    concepts = [dictionary["hr"], dictionary["sbp"], dictionary["map"]]
    loader = _legacy_loader()

    def _one(*, concept, **_kwargs):
        if concept.name == "sbp":
            raise OSError("injected read failure")
        return pd.DataFrame({"stay_id": [1], "charttime": [0.0], concept.name: [1.0]})

    loader._load_one_concept = _one
    with pytest.raises(OSError, match="injected read failure"):
        loader.load_concepts(concepts, verbose=False, concept_workers=workers)


def test_legacy_weightless_stay_gives_no_weight_without_raising():
    loader = _legacy_loader()
    loader._load_one_concept = lambda **_kwargs: pd.DataFrame()
    frame = pd.DataFrame({"admissionid": [1], "dose": [1.0]})

    assert loader._get_patient_weights(frame, "admissionid", "icustay") is None
