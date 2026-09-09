"""Independent, comparable model results are distinct from documentation rows."""
from dataclasses import replace

import pytest

from easyicu.research_agent.robustness.panel import (
    RobustnessPanel, RobustnessPanelRow, RobustnessPlanError,
    _row_matches_summary_payload, numeric_digest_for_panel, worst_rows_by_axis,
)


def _row(spec, **changes):
    row = RobustnessPanelRow(
        spec, 'primary' if spec == 'primary' else 'missing', 123,
        1.2, 0.9, 1.5, None, 'source_' + spec, True,
        effect_scale='OR', estimand_id='mortality_association',
        contrast_id='per_one_unit', effect_unit='source_unit',
    )
    return replace(row, **changes)


def test_documentation_keeps_its_count_but_is_not_a_variant_or_range_extreme():
    rows = [_row('primary'), _row('sensitivity'),
            _row('same_analysis_set', independent_variant=False,
                 point_estimate=9., ci_low=8., ci_high=10.)]
    panel = RobustnessPanel.from_rows(rows)
    restored = RobustnessPanel.from_dict(panel.to_dict())
    assert restored.n_variants == 1
    assert restored.rows[2].n == 123
    assert restored.rows[2].independent_variant is False
    assert (restored.range_low, restored.range_high) == (.9, 1.5)
    assert worst_rows_by_axis(restored)['missing'].spec_id == 'sensitivity'


@pytest.mark.parametrize('changes', [
    {'contrast_id': 'high_vs_low'}, {'estimand_id': 'other_outcome'},
    {'effect_unit': 'other_unit'}, {'effect_scale': 'HR'}, {'contrast_id': ''},
])
def test_distinct_or_missing_identity_cannot_produce_a_common_range(changes):
    panel = RobustnessPanel.from_rows([_row('primary'), _row('variant', **changes)])
    assert panel.n_variants == 1
    assert panel.range_low is None and panel.range_high is None
    assert worst_rows_by_axis(panel) == {}
    digest = numeric_digest_for_panel(panel)
    assert 'range_low' not in digest and 'range_high' not in digest
    assert 'primary_point_estimate' in digest


def test_only_documentation_means_zero_independent_variants():
    panel = RobustnessPanel.from_rows([_row('document', independent_variant=False)])
    assert panel.n_variants == 0
    assert panel.range_low is None
    assert len(panel.rows) == 1


def test_independence_and_contrast_cannot_be_changed_relative_to_source():
    source = _row('variant', independent_variant=False)
    payload = {'robustness_rows': [source.to_dict()]}
    assert _row_matches_summary_payload(source, payload)
    assert not _row_matches_summary_payload(replace(source, independent_variant=True), payload)
    assert not _row_matches_summary_payload(replace(source, contrast_id='other'), payload)
    with pytest.raises(RobustnessPlanError, match='JSON boolean'):
        RobustnessPanelRow.from_dict({**source.to_dict(), 'independent_variant':'false'})


def test_primary_comparability_identity_needs_matching_source_metadata():
    row = _row('primary')
    payload = dict(primary_or=1.2, primary_ci_low=.9, primary_ci_high=1.5, sample_size=123)
    assert not _row_matches_summary_payload(row, payload)
    payload.update(effect_scale='OR', estimand_id='mortality_association',
                   contrast_id='per_one_unit', effect_unit='source_unit')
    assert _row_matches_summary_payload(row, payload)
