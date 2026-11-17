"""sofa_resp scores only hours that have a P/F ratio or ventilation.

ricu's pafi() drops a PaO2 or FiO2 that has no partner, so its sofa_resp has
no row there. EasyICU's internal pafi table keeps such rows with P/F missing
(missing_fio2 provenance); scoring them 0 published respiratory hours that no
P/F supports (paper decomposition 2026-11-16, B1).
"""

import pandas as pd
import pytest

from easyicu.concept.callbacks import ConceptCallbackContext, _callback_sofa_resp
from easyicu.table import ICUTable


def _ctx():
    return ConceptCallbackContext(concept_name="sofa_resp", target=None, interval=pd.Timedelta(hours=1),
                                  resolver=object(), data_source=None, patient_ids=None)


def _run(pafi_rows, vent_rows):
    pafi = ICUTable(pd.DataFrame(pafi_rows, columns=["stay_id", "charttime", "pafi"]),
                    id_columns=["stay_id"], index_column="charttime", value_column="pafi")
    vent = ICUTable(pd.DataFrame(vent_rows, columns=["stay_id", "charttime", "vent_ind"]),
                    id_columns=["stay_id"], index_column="charttime", value_column="vent_ind")
    out = _callback_sofa_resp({"pafi": pafi, "vent_ind": vent}, _ctx()).data
    return {(int(r.stay_id), float(r.charttime)): int(r.sofa_resp) for r in out.itertuples()}


def test_an_unmatched_gas_hour_gets_no_respiratory_score():
    got = _run([(1, 0.0, 150.0), (1, 1.0, None), (1, 2.0, 350.0)], [(1, 0.0, True)])
    assert got == {(1, 0.0): 3, (1, 2.0): 1}


def test_a_ventilated_hour_without_pf_still_scores_zero_as_ricu_does():
    got = _run([(1, 3.0, None)], [(1, 3.0, True)])
    assert got == {(1, 3.0): 0}


def test_scores_with_a_ratio_are_unchanged():
    got = _run([(2, 0.0, 90.0), (2, 1.0, 180.0), (2, 2.0, 450.0)], [])
    assert got == {(2, 0.0): 2, (2, 1.0): 2, (2, 2.0): 0}


def _table(name, hours, values):
    return ICUTable(pd.DataFrame({'stay_id': 1, 'charttime': hours, name: values}),
                    id_columns=['stay_id'], index_column='charttime', value_column=name)


def test_full_score_anchors_do_not_invent_respiratory_evidence():
    ctx = _ctx()
    ctx.kwargs['_preserve_sofa_score_grid'] = True
    out = _callback_sofa_resp({
        'pafi': _table('pafi', [0., 1., 2., 3.], [None, None, None, 150.]),
        'vent_ind': _table('vent_ind', [1., 2.], [False, True]),
    }, ctx).data.set_index('charttime').sofa_resp
    assert out.loc[[0., 1.]].isna().all()
    assert out.loc[2.] == 0
    assert out.loc[3.] == 2


@pytest.mark.parametrize('cached_grid_mode', [False, True])
def test_respiratory_raw_cache_cannot_cross_full_score_grid_modes(monkeypatch, cached_grid_mode):
    from types import SimpleNamespace

    from easyicu.concept import ConceptResolver, _compute_patient_ids_hash
    from easyicu.resources import load_dictionary

    resolver = ConceptResolver(load_dictionary())
    interval = pd.Timedelta(hours=1)
    tables = {'pafi': _table('pafi', [2., 26.], [None, None]),
              'vent_ind': _table('vent_ind', [], [])}

    def respiratory(mode):
        ctx = _ctx()
        ctx.kwargs['_preserve_sofa_score_grid'] = mode
        return _callback_sofa_resp(tables, ctx)

    resolver._store_raw_concept_cache(
        'sofa_resp', _compute_patient_ids_hash(None), respiratory(cached_grid_mode),
        aggregator=False, store_legacy=False, grid=interval,
        cache_scope=resolver._raw_cache_scope({'_preserve_sofa_score_grid': cached_grid_mode}))
    calls = []

    def cold_load(*args, **kwargs):
        calls.append(args[0])
        return respiratory(kwargs.get('_preserve_sofa_score_grid', False))

    monkeypatch.setattr(resolver, '_load_single_concept', cold_load)
    result = resolver._ensure_concept_loaded(
        'sofa_resp', SimpleNamespace(config=SimpleNamespace(name='miiv')),
        aggregators={'sofa_resp': False}, patient_ids=None, verbose=False,
        interval=interval, align_to_admission=False,
        kwargs={'_preserve_sofa_score_grid': not cached_grid_mode}).data
    assert calls.count('sofa_resp') == 1
    if cached_grid_mode:
        assert result.empty
    else:
        assert result.charttime.tolist() == [2., 26.]
        assert result.sofa_resp.isna().all()


def test_sofa2_keeps_unmatched_pf_grid_and_expires_other_organ_evidence():
    from easyicu.concept.callbacks import _callback_sofa_component, _callback_sofa2_score
    from easyicu.scores.sofa2 import sofa2_resp

    ctx = _ctx()
    ctx.concept_name = 'sofa2_resp'
    respiratory = _callback_sofa_component(sofa2_resp)(
        {'pafi': _table('pafi', [2., 26.], [None, None])}, ctx)
    assert respiratory.data.charttime.tolist() == [2., 26.]
    assert respiratory.data.sofa2_resp_available.eq(0).all()
    ctx.concept_name = 'sofa2'
    ctx.kwargs['keep_components'] = True
    result = _callback_sofa2_score({
        'sofa2_resp': respiratory,
        'sofa2_cardio': _table('sofa2_cardio', [0.], [3.]),
    }, ctx).data.set_index('charttime')
    assert result.index.tolist() == list(range(27))
    assert result.loc[2., 'sofa2_cardio'] == 3.
    assert pd.isna(result.loc[24., 'sofa2_cardio'])
    assert result.sofa2_resp_available.fillna(0).eq(0).all()
