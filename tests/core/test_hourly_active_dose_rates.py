"""S732TW: hand-integrated active-rate means on each source clock."""
import pandas as pd
import pytest

from easyicu.utils.callback_utils import (
    aumc_rate_kg, eicu_rate_kg_callback, mimic_rate_cv, mimic_rate_mv,
)
from tests.support.eicu_source import _eicu_source


CASES = [
    ([0, 15, 60], [.1, .3, 0], .25),
    ([0, 15, 60], [.3, .1, 0], .15),
    ([0, 10, 40, 60], [.1, .4, .2, 0], 17 / 60),
    ([15, 45], [.2, 0], .2),
]


def point_rate(path, minutes, rates):
    if path == "cv":
        frame = pd.DataFrame({"icustay_id": 1, "charttime": [m / 60 for m in minutes],
                              "linkorderid": 10, "norepi_rate": rates})
        return mimic_rate_cv(frame, val_col="norepi_rate", grp_var="linkorderid", id_cols=["icustay_id"])
    if path == "eicu":
        frame = pd.DataFrame({"patientunitstayid": 1, "infusionoffset": minutes,
                              "drugrate": rates, "drugname": "Norepinephrine (mcg/kg/min)",
                              "patientweight": 70.})
        out = eicu_rate_kg_callback(float("nan"))(frame, val_var="drugrate", sub_var="drugname",
                                                     weight_var="patientweight", concept_name="norepi_rate")
        out["infusionoffset"] /= 60
        return out.rename(columns={"infusionoffset": "charttime"})
    frame = pd.DataFrame({"stay_id": 1, "starttime": [m / 60 for m in minutes[:-1]],
                          "endtime": [m / 60 for m in minutes[1:]], "norepi_rate": rates[:-1]})
    if path == "mv":
        out = mimic_rate_mv(frame, val_col="norepi_rate", stop_var="endtime", id_cols=["stay_id"])
        return out.rename(columns={"starttime": "charttime"})
    frame = frame.rename(columns={"stay_id": "admissionid", "starttime": "start", "endtime": "stop"})
    frame[["start", "stop"]] *= 60
    frame = frame.assign(doseunit="mcg", doserateunit="min", doserateperkg=True)
    out = aumc_rate_kg(frame, concept_name="norepi_rate", val_col="norepi_rate", unit_col="doseunit",
                       rel_weight_col="doserateperkg", rate_unit_col="doserateunit", index_col="start", stop_col="stop")
    out["start"] /= 60
    return out.rename(columns={"start": "charttime"})


@pytest.mark.parametrize("path", ["cv", "eicu", "mv", "aumc"])
@pytest.mark.parametrize("minutes,rates,expected", CASES, ids=["up", "down", "multi", "partial"])
def test_hand_integrated_hour(path, minutes, rates, expected):
    out = point_rate(path, minutes, rates)
    assert out.charttime.tolist() == [0.]
    assert out.norepi_rate.tolist() == pytest.approx([expected])


@pytest.mark.parametrize('path', ['cv', 'eicu', 'mv', 'aumc'])
def test_concurrent_infusions_sum_before_mean_and_max(path):
    from easyicu.concept.dose_rate import collapse_spans, source_rate_scope

    if path in {'cv', 'eicu'}:
        ident = 'icustay_id' if path == 'cv' else 'patientunitstayid'
        index = 'charttime' if path == 'cv' else 'infusionoffset'
        frame = pd.DataFrame({ident: 1, index: [0., 60., 30., 45.],
                              'linkorderid': [10, 10, 20, 20], 'norepi_rate': [.1, 0, .2, 0]})
        if path == 'cv':
            frame[index] /= 60.
            def producer():
                return mimic_rate_cv(frame, val_col='norepi_rate', grp_var='linkorderid', id_cols=[ident])
        else:
            frame = frame.assign(drugrate=frame.norepi_rate, drugname='Norepinephrine (mcg/kg/min)', patientweight=70.)
            def producer():
                return eicu_rate_kg_callback(float('nan'))(
                            frame, val_var='drugrate', sub_var='drugname', weight_var='patientweight', concept_name='norepi_rate')
    else:
        ident, index = ('stay_id', 'starttime') if path == 'mv' else ('admissionid', 'start')
        frame = pd.DataFrame({ident: 1, index: [0., .5], 'endtime': [1., .75],
                              'orderid': [10, 20], 'linkorderid': 99, 'norepi_rate': [.1, .2]})
        if path == 'mv':
            def producer():
                return mimic_rate_mv(frame, val_col='norepi_rate', stop_var='endtime', id_cols=[ident])
        else:
            frame[[index, 'endtime']] *= 60.
            frame = frame.assign(doseunit='mcg', doserateunit='min', doserateperkg=True)
            def producer():
                return aumc_rate_kg(
                            frame, concept_name='norepi_rate', val_col='norepi_rate', unit_col='doseunit',
                            rel_weight_col='doserateperkg', rate_unit_col='doserateunit', index_col=index, stop_col='endtime')
    with source_rate_scope('norepi_rate', '1h'):
        spans = producer()
    mean = collapse_spans(spans, ids=[ident], index=index, value='norepi_rate')
    maximum = collapse_spans(spans.rename(columns={'norepi_rate': 'norepi_rate_max'}),
                             ids=[ident], index=index, value='norepi_rate_max')
    assert mean[index].tolist() == [0.]
    assert mean.norepi_rate.tolist() == pytest.approx([.15])
    assert maximum.norepi_rate_max.tolist() == pytest.approx([.3])


@pytest.mark.parametrize('path', ['cv', 'eicu'])
def test_five_hour_boundary_keeps_terminal_overhang(path):
    exact = point_rate(path, [0, 300], [.1, .2])
    assert exact.charttime.tolist() == [0., 1., 2., 3., 4., 5.]
    assert exact.norepi_rate.tolist() == pytest.approx([.1] * 5 + [.2])
    assert point_rate(path, [15, 30], [0., 0.]).empty


@pytest.mark.parametrize('path', ['cv', 'eicu'])
@pytest.mark.parametrize('minutes,rates,hours,expected', [
    ([0], [.2], [0.], [.2]),
    ([15], [.2], [0., 1.], [.2, .2]),
    # 45 minutes at .1, then 15 at .3; the final .3 also covers 60..105.
    ([0, 45], [.1, .3], [0., 1.], [.15, .3]),
    # CareVue episode gaps retain a one-hour tail; eICU keeps dev9 carry.
    ([0, 301], [.1, .2], [0., 5., 6.], [.1, .2, .2]),
], ids=['singleton', 'singleton_mid_hour', 'last_mid_hour', 'episode_gap'])
def test_point_episode_ends_have_one_hour_overhang(path, minutes, rates, hours, expected):
    if path == 'eicu' and minutes == [0, 301]:
        hours = [0., 1., 2., 3., 4., 5., 6.]
        expected = [.1] * 5 + [(1 / 60 * .1 + 59 / 60 * .2), .2]
    out = point_rate(path, minutes, rates)
    assert out.charttime.tolist() == hours
    assert out.norepi_rate.tolist() == pytest.approx(expected)


@pytest.mark.parametrize('start', [0, 15])
@pytest.mark.parametrize('closing', [.3, 0., float('nan')])
def test_eicu_long_gap_preserves_dev9_carry_and_inclusive_five_hour_boundary(start, closing):
    out = point_rate('eicu', [start, 600], [.2, closing])
    carry_hours = list(range(6 if start == 0 else 7))
    expected_hours = carry_hours + ([10.] if closing > 0 else [])
    assert out.charttime.tolist() == expected_hours
    assert out.norepi_rate.tolist() == pytest.approx([.2] * len(carry_hours) + ([.3] if closing > 0 else []))


def test_eicu_long_gap_carry_ends_at_the_next_setting_and_integrates_both_rates():
    out = point_rate('eicu', [0, 330, 360], [.1, .3, 0.])
    assert out.charttime.tolist() == list(range(6))
    assert out.norepi_rate.tolist() == pytest.approx([.1] * 5 + [.2])


@pytest.mark.parametrize('score', ['sofa_cardio', 'sofa2_cardio'])
@pytest.mark.parametrize('maximum_state', ['empty', 'partial', 'absent'])
def test_sofa_missing_maximum_preserves_existing_qualification_and_score(score, maximum_state):
    from types import SimpleNamespace
    from easyicu.concept.callbacks import CALLBACK_REGISTRY, ConceptCallbackContext
    from easyicu.concept.dose_rate_sofa import cardio_max_tables
    from easyicu.table import ICUTable

    def table(name, hours, values):
        return ICUTable(pd.DataFrame({'stay_id': [1] * len(hours), 'charttime': hours, name: values}),
                        id_columns=['stay_id'], index_column='charttime', value_column=name)

    tables = {'map': table('map', [0., 1., 2., 3.], [80.] * 4),
              'norepi60': table('norepi60', [0., 1., 2., 3.], [.2, 0., -.1, .05])}
    if maximum_state != 'absent':
        hours, values = ([1., 2., 3.], [.3, .3, .2]) if maximum_state == 'partial' else ([], [])
        tables['norepi_rate_max'] = table('norepi_rate_max', hours, values)
    result = cardio_max_tables(tables, SimpleNamespace(concept_name=score))
    expected = [.2, 0., -.1, .2 if maximum_state == 'partial' else .05]
    assert result['norepi60'].data.norepi60.tolist() == pytest.approx(expected)
    # The adapter must not mutate the duration-qualified dependency.
    assert tables['norepi60'].data.norepi60.tolist() == [.2, 0., -.1, .05]
    ctx = ConceptCallbackContext(concept_name=score, target=None, interval=pd.Timedelta(hours=1),
                                 resolver=None, data_source=None, patient_ids=None)
    score_tables = dict(tables)
    score_tables['norepi60'] = table('norepi60', [0., 1., 3.], [.2, 0., .05])
    scored = CALLBACK_REGISTRY[score](score_tables, ctx).data
    assert scored[score].notna().all()
    assert scored.loc[scored.charttime == 0., score].item() == (4. if score == 'sofa_cardio' else 2.)


@pytest.mark.parametrize('path', ['cv', 'eicu'])
@pytest.mark.parametrize('closing', [0., -1.])
def test_nonpositive_terminal_setting_closes_without_overhang(path, closing):
    out = point_rate(path, [45, 75], [.2, closing])
    assert out.charttime.tolist() == [0., 1.]
    assert out.norepi_rate.tolist() == pytest.approx([.2, .2])


def test_invalid_eicu_terminal_setting_closes_without_overhang():
    out = point_rate('eicu', [45, 75], [.2, float('nan')])
    assert out.charttime.tolist() == [0., 1.]
    assert out.norepi_rate.tolist() == pytest.approx([.2, .2])


def test_sources_are_combined_before_point_intervals_and_concurrency():
    from easyicu.concept.dose_rate import collapse_spans, source_rate_scope

    with source_rate_scope('norepi_rate', '1h'):
        # A source-label/unit change still belongs to the same infusion.
        a = pd.DataFrame({'icustay_id': [1], 'charttime': [0.], 'linkorderid': [10], 'norepi_rate': [.3]})
        b = pd.DataFrame({'icustay_id': [1, 1], 'charttime': [.25, 1.],
                          'linkorderid': [10, 10], 'norepi_rate': [.1, 0.]})
        frames = [mimic_rate_cv(f, val_col='norepi_rate', grp_var='linkorderid', id_cols=['icustay_id']) for f in (a, b)]
    out = collapse_spans(pd.concat(frames), ids=['icustay_id'], index='charttime', value='norepi_rate')
    assert out.norepi_rate.tolist() == pytest.approx([.15])


def test_sofa_reads_max_companions_and_keeps_the_sixty_minute_gate():
    import json
    from pathlib import Path
    from easyicu.concept.callbacks import CALLBACK_REGISTRY, ConceptCallbackContext
    from easyicu.table import ICUTable

    concepts = {'map': [80., 80.], 'norepi60': [.05, 0.], 'norepi_rate_max': [.2, .3],
                'norepi_rate': [.05, .05]}
    tables = {name: ICUTable(pd.DataFrame({'stay_id': [1, 1], 'charttime': [0., 1.], name: vals}),
                             id_columns=['stay_id'], index_column='charttime', value_column=name)
              for name, vals in concepts.items() if name != 'norepi_rate'}
    ctx = ConceptCallbackContext(concept_name='sofa_cardio', target=None, interval=pd.Timedelta(hours=1),
                                 resolver=None, data_source=None, patient_ids=None)
    result = CALLBACK_REGISTRY['sofa_cardio'](tables, ctx)
    assert result.data.sofa_cardio.tolist() == [4., 0.]
    root = Path(__file__).resolve().parents[2] / 'src/easyicu/data'
    for file, key in [('concept-dict.json', 'sofa_cardio'), ('sofa2-dict.json', 'sofa2_cardio')]:
        definition = json.loads((root / file).read_text())[key]
        assert all(f'{drug}_rate_max' in definition['concepts'] for drug in ('norepi', 'epi', 'dopa', 'dobu'))
        assert all(f'{drug}60' in definition['concepts'] for drug in ('norepi', 'epi', 'dopa', 'dobu'))


@pytest.mark.parametrize('database', ['aumc', 'hirid', 'sic'])
def test_public_resolver_returns_one_mean_and_one_max_per_hour(tmp_path, database):
    from easyicu import load_concepts
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / 'source'
    generate_source(root, database, stays=3)
    try:
        out = load_concepts(concepts=['norepi_rate', 'norepi_rate_max'], database=database,
                            data_path=str(root), merge=True, verbose=False)
    finally:
        clear_global_loader()
    assert out.norepi_rate.notna().any()
    joined = out.dropna(subset=['norepi_rate', 'norepi_rate_max'])
    assert not joined.empty
    assert (joined.norepi_rate <= joined.norepi_rate_max + 1e-12).all()
    assert not out.duplicated([out.columns[0], 'charttime']).any()
    assert not any(c.startswith('_dose_') for c in out)


def test_inventory_and_hidden_companions_are_dictionary_declared():
    import json
    import re
    from pathlib import Path
    from easyicu.concept.catalog import HIDDEN_DICTIONARY_CONCEPTS, CONCEPT_GROUPS_INTERNAL
    from easyicu.concept.dose_rate import DOSE_RATE_CONCEPTS, RATE_MAX_CONCEPTS

    path = Path(__file__).resolve().parents[2] / 'src/easyicu/data/concept-dict.json'
    dictionary = json.loads(path.read_text())
    callbacks = re.compile(r'mimic_rate_[cm]v|eicu_rate|aumc_rate|hirid_rate|sic_rate|distribute_amount|sicdb_amount_rate|miiv_angiotensin_rate')
    users = {name for name, entry in dictionary.items()
             if any(callbacks.search(source.get('callback', ''))
                    for sources in entry.get('sources', {}).values() for source in sources)}
    assert users - set(RATE_MAX_CONCEPTS) == set(DOSE_RATE_CONCEPTS) | {'ins'}
    assert all(dictionary[name]['_hourly_rule'] == 'time_weighted_mean_active_rate' for name in DOSE_RATE_CONCEPTS)
    assert set(RATE_MAX_CONCEPTS) <= HIDDEN_DICTIONARY_CONCEPTS
    assert not set(RATE_MAX_CONCEPTS) & {c for members in CONCEPT_GROUPS_INTERNAL.values() for c in members}
    for name in RATE_MAX_CONCEPTS:
        assert dictionary[name]['sources'] == dictionary[name[:-4]]['sources']
        assert dictionary[name]['_hourly_rule'] == 'maximum_active_rate'


@pytest.mark.parametrize('database', ['mimic', 'eicu', 'miiv'])
@pytest.mark.parametrize('minutes,rates,expected', CASES, ids=['up', 'down', 'multi', 'partial'])
def test_hand_integrated_public_source_path(tmp_path, database, minutes, rates, expected):
    from easyicu.concept import ConceptResolver
    from easyicu.resources import load_dictionary
    from tests.support.mimic_synthetic_source import mimic_synthetic_source

    root = tmp_path / 'source'
    root.mkdir()
    origin = pd.Timestamp('2150-01-01')
    if database == 'mimic':
        frame = pd.DataFrame({'icustay_id': 1, 'itemid': 30120, 'rate': rates,
                              'charttime': origin + pd.to_timedelta(minutes, unit='m'),
                              'rateuom': 'mcgkgmin', 'linkorderid': 10})
        source = mimic_synthetic_source(root, database, inputevents_cv=frame)
        ident = 'icustay_id'
    elif database == 'miiv':
        frame = pd.DataFrame({'stay_id': 1, 'itemid': 221906, 'rate': rates[:-1],
                              'starttime': origin + pd.to_timedelta(minutes[:-1], unit='m'),
                              'endtime': origin + pd.to_timedelta(minutes[1:], unit='m'),
                              'rateuom': 'mcg/kg/min', 'statusdescription': 'FinishedRunning', 'orderid': 10})
        frame = frame.assign(hadm_id=21, amount=0., amountuom='mg', linkorderid=10)
        source = mimic_synthetic_source(root, database, inputevents=frame)
        ident = 'stay_id'
    else:
        frame = pd.DataFrame({'infusionoffset': minutes, 'drugrate': rates,
                              'drugname': 'Norepinephrine (mcg/kg/min)', 'patientweight': 70.})
        source = _eicu_source(root, database, infusiondrug=frame)
        ident = 'patientunitstayid'
    out = ConceptResolver(load_dictionary()).load_concepts(
        ['norepi_rate', 'norepi_rate_max', 'norepi_equiv'], source, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={ident: [1]}, concept_workers=1, verbose=False)
    assert out['norepi_rate'].data.norepi_rate.tolist() == pytest.approx([expected])
    assert out['norepi_rate_max'].data.norepi_rate_max.tolist() == pytest.approx([max(rates)])
    assert out['norepi_equiv'].data.norepi_equiv.tolist() == pytest.approx([expected])


def test_point_ties_and_source_order_do_not_select_different_doses():
    frame = pd.DataFrame({'icustay_id': 1, 'charttime': [0., .5, .5, 1.],
                          'linkorderid': 10, 'norepi_rate': [.1, .2, .3, 0.]})
    result = mimic_rate_cv(frame, val_col='norepi_rate', grp_var='linkorderid', id_cols=['icustay_id'])
    shuffled = mimic_rate_cv(frame.iloc[::-1], val_col='norepi_rate', grp_var='linkorderid', id_cols=['icustay_id'])
    assert result.norepi_rate.tolist() == pytest.approx([.2])
    pd.testing.assert_frame_equal(result, shuffled)
    frame.loc[2, 'norepi_rate'] = 0.
    stopped = mimic_rate_cv(frame, val_col='norepi_rate', grp_var='linkorderid', id_cols=['icustay_id'])
    assert stopped.norepi_rate.tolist() == pytest.approx([.1])


@pytest.mark.parametrize('database', ['aumc', 'hirid'])
def test_raw_public_request_never_exposes_span_metadata(tmp_path, database):
    from easyicu import load_concepts
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / 'source'
    generate_source(root, database, stays=3)
    try:
        out = load_concepts(['norepi_rate'], database=database, data_path=str(root), interval=None, verbose=False)
    finally:
        clear_global_loader()
    assert out.norepi_rate.notna().any()
    assert not any(c.startswith('_dose_') for c in out)


@pytest.mark.parametrize('path', ['mv', 'aumc'])
def test_explicit_six_hour_interval_is_not_a_point_gap(path):
    out = point_rate(path, [0, 360], [.1, 0.])
    assert out.charttime.tolist() == [0., 1., 2., 3., 4., 5.]
    assert out.norepi_rate.tolist() == pytest.approx([.1] * 6)


def test_same_order_rate_change_supersedes_overlapping_recorded_end():
    frame = pd.DataFrame({'stay_id': 1, 'starttime': [0., .25], 'endtime': [1., .75],
                          'orderid': 10, 'norepi_rate': [.1, .3]})
    out = mimic_rate_mv(frame, val_col='norepi_rate', stop_var='endtime', id_cols=['stay_id'])
    assert out.norepi_rate.tolist() == pytest.approx([7 / 30])


def test_same_infusion_setting_across_interval_sources_is_not_double_counted():
    from easyicu.concept.dose_rate import collapse_spans, source_rate_scope

    first = pd.DataFrame({'stay_id': [1], 'starttime': [0.], 'endtime': [1.], 'orderid': [10], 'norepi_rate': [.1]})
    second = first.assign(starttime=.25, endtime=.75, norepi_rate=.3, orderid=10.)
    with source_rate_scope('norepi_rate', '1h'):
        spans = pd.concat([mimic_rate_mv(f, val_col='norepi_rate', stop_var='endtime', id_cols=['stay_id'])
                           for f in (first, second)], ignore_index=True)
    out = collapse_spans(spans, ids=['stay_id'], index='starttime', value='norepi_rate')
    assert out.norepi_rate.tolist() == pytest.approx([7 / 30])


@pytest.mark.parametrize('database', ['mimic', 'miiv', 'eicu'])
def test_summary_precedes_episode_clipping_without_losing_stop_boundary(tmp_path, database):
    from easyicu.concept import ConceptResolver
    from easyicu.resources import load_dictionary
    from tests.support.mimic_synthetic_source import mimic_synthetic_source

    root = tmp_path / 'source'
    root.mkdir()
    origin = pd.Timestamp('2150-01-01')
    if database == 'miiv':
        frame = pd.DataFrame({'stay_id': [1], 'itemid': [221906], 'rate': [.1],
                              'starttime': [origin + pd.Timedelta(hours=95)],
                              'endtime': [origin + pd.Timedelta(hours=99)],
                              'rateuom': ['mcg/kg/min'], 'statusdescription': ['FinishedRunning']})
        frame = frame.assign(hadm_id=21, amount=0., amountuom='mg', linkorderid=10)
        source = mimic_synthetic_source(root, database, inputevents=frame)
        ident, expected = 'stay_id', [95., 96.]
    elif database == 'mimic':
        frame = pd.DataFrame({'icustay_id': 1, 'itemid': 30120, 'rate': [.1, 0.],
                              'charttime': origin + pd.to_timedelta([95., 97.], unit='h'),
                              'rateuom': 'mcgkgmin', 'linkorderid': 10})
        source = mimic_synthetic_source(root, database, inputevents_cv=frame)
        ident, expected = 'icustay_id', [95., 96.]
    else:
        frame = pd.DataFrame({'infusionoffset': [71 * 60, 73 * 60], 'drugrate': [.1, 0.],
                              'drugname': 'Norepinephrine (mcg/kg/min)', 'patientweight': 70.})
        source = _eicu_source(root, database, infusiondrug=frame)
        ident, expected = 'patientunitstayid', [71., 72.]
    result = ConceptResolver(load_dictionary()).load_concepts(
        ['norepi_rate'], source, merge=False, r_compatible=False, interval=pd.Timedelta(hours=1),
        patient_ids={ident: [1]}, concept_workers=1, verbose=False)['norepi_rate']
    assert result.data[result.index_column].tolist() == expected
    assert result.data.norepi_rate.tolist() == pytest.approx([.1] * len(expected))


@pytest.mark.parametrize('database,last_hour', [('mimic', 96.5), ('eicu', 72.5)])
def test_terminal_overhang_obeys_existing_episode_end(tmp_path, database, last_hour):
    from easyicu.concept import ConceptResolver
    from easyicu.resources import load_dictionary
    from tests.support.mimic_synthetic_source import mimic_synthetic_source

    if database == 'mimic':
        frame = pd.DataFrame({'icustay_id': [1], 'itemid': [30120], 'rate': [.2],
                              'charttime': [pd.Timestamp('2150-01-01') + pd.Timedelta(hours=last_hour)],
                              'rateuom': ['mcgkgmin'], 'linkorderid': [10]})
        source = mimic_synthetic_source(tmp_path, database, inputevents_cv=frame)
        ident = 'icustay_id'
    else:
        frame = pd.DataFrame({'infusionoffset': [last_hour * 60], 'drugrate': [.2],
                              'drugname': ['Norepinephrine (mcg/kg/min)'], 'patientweight': [70.]})
        source = _eicu_source(tmp_path, database, infusiondrug=frame)
        ident = 'patientunitstayid'
    out = ConceptResolver(load_dictionary()).load_concepts(
        ['norepi_rate', 'norepi_rate_max'], source, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={ident: [1]}, concept_workers=1, verbose=False)
    for name in ('norepi_rate', 'norepi_rate_max'):
        result = out[name]
        assert result.data[result.index_column].tolist() == [last_hour // 1]
        assert result.data[name].tolist() == pytest.approx([.2])


def test_carevue_restart_closes_a_setting_without_creating_a_new_one():
    frame = pd.DataFrame({'icustay_id': 1, 'charttime': [0., .5, 1., 1.5], 'linkorderid': 10,
                          'norepi_rate': [.1, float('nan'), .2, 0.],
                          'stopped': [None, 'Restart', None, 'Stopped']})
    out = mimic_rate_cv(frame, val_col='norepi_rate', id_cols=['icustay_id'])
    assert out.charttime.tolist() == [0., 1.]
    assert out.norepi_rate.tolist() == pytest.approx([.1, .2])


def test_public_miiv_concurrent_orders_add_even_when_linked(tmp_path):
    from easyicu.concept import ConceptResolver
    from easyicu.resources import load_dictionary
    from tests.support.mimic_synthetic_source import mimic_synthetic_source

    origin = pd.Timestamp('2150-01-01')
    frame = pd.DataFrame({'stay_id': 1, 'hadm_id': 21, 'itemid': 221906, 'rate': [.1, .2],
                          'starttime': origin + pd.to_timedelta([0, 30], unit='m'),
                          'endtime': origin + pd.to_timedelta([60, 45], unit='m'),
                          'orderid': [10, 20], 'linkorderid': 99, 'rateuom': 'mcg/kg/min',
                          'amount': 0., 'amountuom': 'mg', 'statusdescription': 'FinishedRunning'})
    source = mimic_synthetic_source(tmp_path, 'miiv', inputevents=frame)
    out = ConceptResolver(load_dictionary()).load_concepts(
        ['norepi_rate', 'norepi_rate_max'], source, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={'stay_id': [1]}, concept_workers=1, verbose=False)
    assert out['norepi_rate'].data.norepi_rate.tolist() == pytest.approx([.15])
    assert out['norepi_rate_max'].data.norepi_rate_max.tolist() == pytest.approx([.3])


def test_legacy_minute_rate_dependencies_keep_their_original_projection():
    from easyicu.concept.dose_rate import source_rate_scope

    frame = pd.DataFrame({'stay_id': 1, 'starttime': [0., .5], 'endtime': [1., .75],
                          'orderid': [10, 20], 'linkorderid': 99, 'norepi_rate': [.1, .2]})
    with source_rate_scope('norepi_rate', '1min'):
        out = mimic_rate_mv(frame, val_col='norepi_rate', stop_var='endtime', id_cols=['stay_id'])
    assert out.norepi_rate.tolist() == [.1, .2]
    assert set(out) == {'stay_id', 'starttime', 'norepi_rate'}


def test_sic_propofol_hourly_summary_retains_its_grid_and_converted_dose():
    from easyicu.concept.callback_apply import _apply_callback
    from easyicu.resources import load_dictionary

    source = load_dictionary()['propofol_rate'].sources['sic'][0]
    frame = pd.DataFrame({'CaseID': [1, 1, 1], 'Offset': [0., 3600., 10800.],
                          'propofol_rate': [.000006, 0., .000012]})
    out = _apply_callback(frame, source, 'propofol_rate', interval=pd.Timedelta(hours=1))
    assert out.Offset.tolist() == frame.Offset.tolist()
    assert out.propofol_rate.tolist() == pytest.approx(frame.propofol_rate * 16666.6667)
    assert not any(c.startswith('_dose_') for c in out)


def test_hirid_legacy_hourly_bounds_preserve_coverage_before_active_time_mean(tmp_path):
    from easyicu.resources import load_dictionary
    from easyicu.source_bounds import collect_bounds_drops
    from tests.support.hirid_source import hirid_load, hirid_source_root

    definition = load_dictionary()['norepi_rate']
    ceiling = definition.maximum
    item = definition.sources['hirid'][0].ids[0]
    # A quarter-hour dose is valid on dev9's hourly denominator but its active
    # mean exceeds that hourly ceiling. A separate invalid legacy hour stays out.
    rows = [{'patientid': 1, 'givenat': t, 'pharmaid': item, 'givendose': dose,
             'doseunit': 'µg', 'route': 'iv-inf', 'infusionid': infusion, 'recordstatus': 0}
            for infusion, begin, dose in [(7, 0., ceiling * .5 * 60 * 80),
                                           (8, 2., ceiling * 2 * 60 * 80)]
            for t, dose in [(begin, 0.), (begin + .25, dose)]]
    root = hirid_source_root(tmp_path, 'pharma', rows)
    weight = pd.read_parquet(root / 'observations/1.parquet').iloc[[0]].copy()
    weight = weight.assign(patientid=1, variableid=10000400, value=80.)
    weight['datetime'] = pd.read_parquet(root / 'general.parquet').iloc[0].admissiontime
    weight.to_parquet(root / 'observations/1.parquet', index=False)
    with collect_bounds_drops() as drops:
        result = hirid_load(root, ['norepi_rate', 'norepi_rate_max', 'norepi_equiv'], patient_ids={'patientid': [1]})
    assert drops['norepi_rate']['pharma']['dropped_values'] == 1
    for name, data in result.items():
        values = data.dropna(subset=[name])
        assert not values.empty
        clock = next(c for c in ('charttime', 'givenat', 'datetime') if c in values)
        assert values[clock].tolist() == [0.]
        assert values[name].tolist() == pytest.approx([ceiling * 2])
        assert not any(c.startswith('_dose_') for c in data)


def test_eicu_units_legacy_hourly_median_bounds_do_not_erase_active_dose(tmp_path):
    from easyicu.concept import ConceptResolver
    from easyicu.resources import load_dictionary

    dictionary = load_dictionary()
    ceiling = dictionary['adh_rate'].maximum
    frame = pd.DataFrame({'infusionoffset': [0., 30., 120., 150.],
                          'drugrate': [2 * ceiling, 0., 4 * ceiling, 0.],
                          'drugname': 'Vasopressin (units/min)'})
    source = _eicu_source(tmp_path, 'eicu', infusiondrug=frame)
    result = ConceptResolver(dictionary).load_concepts(
        ['adh_rate', 'adh_rate_max', 'norepi_equiv'], source, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={'patientunitstayid': [1]},
        concept_workers=1, verbose=False)
    # dev9's medians are ceiling and 2*ceiling: only the first hour is valid.
    # The first hour's active dose is 2*ceiling, with no active time after stop.
    for name in ['adh_rate', 'adh_rate_max']:
        data = result[name].data.dropna(subset=[name])
        assert data.charttime.tolist() == [0.]
        assert data[name].tolist() == pytest.approx([2 * ceiling])
        assert not any(c.startswith('_dose_') for c in data)
    assert result['norepi_equiv'].data.norepi_equiv.tolist() == pytest.approx([2 * ceiling / .4])


@pytest.mark.parametrize('minutes', [7, 13, 17, 23, 31, 49, 59])
def test_constant_active_rate_at_ceiling_survives_fractional_hour_roundoff(minutes):
    from easyicu.concept.dose_rate import collapse_spans, START, END, POINT, CHANNEL

    frame = pd.DataFrame({'stay_id': [1], 'charttime': [0.], 'propofol_rate': [200.],
                          START: [0.], END: [minutes / 60.], POINT: [False], CHANNEL: ['']})
    out = collapse_spans(frame, ids=['stay_id'], index='charttime', value='propofol_rate', minimum=0, maximum=200)
    assert out.propofol_rate.between(0., 200.).all()
    assert out.propofol_rate.tolist() == pytest.approx([200.])


def test_sofa_keeps_cardio_window_grid_without_scoring_unmatched_pf(monkeypatch):
    from types import SimpleNamespace
    from easyicu.concept import ConceptResolver
    from easyicu.concept.callbacks import ConceptCallbackContext, _callback_sofa_resp
    from easyicu.resources import load_dictionary
    from easyicu.table import ICUTable

    def table(name, hours, values):
        return ICUTable(pd.DataFrame({'stay_id': 1, 'charttime': hours, name: values}),
                        id_columns=['stay_id'], index_column='charttime', value_column=name)

    pafi = table('pafi', [2., 26.], [float('nan'), float('nan')])
    vent = table('vent_ind', [], [])
    resolver = ConceptResolver(load_dictionary())
    source = SimpleNamespace(config=SimpleNamespace(name='miiv', class_prefix=['miiv']))

    def children(*args, **kwargs):
        ctx = ConceptCallbackContext(concept_name='sofa_resp', target=None, interval=pd.Timedelta(hours=1),
                                     resolver=resolver, data_source=source, patient_ids=None, kwargs=kwargs)
        respiratory = _callback_sofa_resp({'pafi': pafi, 'vent_ind': vent}, ctx)
        # The full SOFA grid may use these keys, but they carry no respiratory score.
        assert respiratory.data.charttime.tolist() == [2., 26.]
        assert respiratory.data.sofa_resp.isna().all()
        return {'sofa_resp': respiratory, 'sofa_cardio': table('sofa_cardio', [0.], [3.])}

    monkeypatch.setattr(resolver, 'load_concepts', children)
    result = resolver._load_recursive_concept(
        'sofa', load_dictionary()['sofa'], source, aggregator=None, patient_ids=None,
        interval=pd.Timedelta(hours=1), verbose=False, keep_components=True).data
    assert result.charttime.tolist() == list(range(27))
    assert result.loc[result.charttime == 2., 'sofa_cardio'].item() == 3.
    assert result.loc[result.charttime == 24., 'sofa_cardio'].isna().all()
    assert result.sofa_resp.isna().all()
