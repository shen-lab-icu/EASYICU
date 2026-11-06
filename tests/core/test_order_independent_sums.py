"""Hourly sums do not depend on row order, batch split or chunk layout.

p9b found fluid totals that differed in the last bit between batch sizes,
thread counts and the perf/no-perf builds: the same rows were summed in a
different order. Sum and mean aggregates are now exact (correctly rounded),
so any order or split of the same rows publishes the same bytes.
"""

from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pandas as pd
import pytest

from easyicu.utils.callback_utils import distribute_volume_hourly
from easyicu.utils.exact_sum import (
    exact_group_totals,
    exact_groupby_agg,
    exact_groupby_sum,
    exact_terms_to_values,
)
from tests.support.mimic_synthetic_source import (
    mimic_synthetic_load,
    mimic_synthetic_source,
)

# Each multiset gives two different plain float sums depending on order.
ORDER_SENSITIVE = {
    1: [0.1, 0.7, 0.2, 0.3],
    2: [100.1, 0.3, 0.7, 33.3],
    3: [250.3, 0.1, 0.001, 77.7, 0.07],
}
FLUIDS = ["total_input_ml", "urine", "fluid_balance", "fluid_balance_cumulative"]


def test_the_fixture_values_are_order_sensitive_for_a_plain_sum():
    for values in ORDER_SENSITIVE.values():
        sums = {
            pd.Series(list(order)).groupby([0] * len(order)).sum().iloc[0]
            for order in itertools.permutations(values)
        }
        assert len(sums) > 1


def test_exact_group_totals_are_the_correctly_rounded_sum_for_any_order():
    rng = random.Random(3)
    for _ in range(20):
        rows = [(key, value) for key, values in ORDER_SENSITIVE.items() for value in values]
        rows += [(4, np.nan), (5, 2.5), (5, 0.25)]
        rng.shuffle(rows)
        codes = np.array([key for key, _ in rows]) - 1
        values = np.array([value for _, value in rows])
        totals, counts = exact_group_totals(codes, values, 5)
        expected = [math.fsum(ORDER_SENSITIVE[key]) for key in (1, 2, 3)] + [0.0, 2.75]
        assert totals.tolist() == expected
        assert counts.tolist() == [4, 4, 5, 0, 2]


def test_grouped_sum_and_mean_keep_pandas_shape_and_empty_group_rules():
    frame = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3],
            "t": [0.0, 0.0, 0.0, 0.0, 1.0],
            "v": [0.1, 0.2, np.nan, np.nan, 4.0],
        }
    )
    summed = exact_groupby_agg(frame, ["id", "t"], {"v": "sum"})
    plain = frame.groupby(["id", "t"], sort=False, as_index=False).agg({"v": "sum"})
    pd.testing.assert_frame_equal(summed, plain)
    mean = exact_groupby_agg(frame, ["id", "t"], {"v": "mean"}, sort=True)
    assert mean["v"].tolist()[:1] == [0.15000000000000002] and np.isnan(mean["v"].iloc[1])
    min_one = exact_groupby_sum(frame, ["id", "t"], "v", min_count=1)
    assert np.isnan(min_one["v"].iloc[1]) and min_one["v"].iloc[2] == 4.0


def test_sql_term_lists_collapse_to_the_same_exact_values():
    terms = pd.Series([ORDER_SENSITIVE[1], list(reversed(ORDER_SENSITIVE[1])), None, [], [5.0]])
    summed = exact_terms_to_values(terms, "sum")
    assert summed.iloc[0] == summed.iloc[1] == math.fsum(ORDER_SENSITIVE[1])
    assert np.isnan(summed.iloc[2]) and np.isnan(summed.iloc[3]) and summed.iloc[4] == 5.0
    assert exact_terms_to_values(terms, "mean").iloc[0] == math.fsum(ORDER_SENSITIVE[1]) / 4


def _intervals(seed: int) -> pd.DataFrame:
    rows = []
    for stay, values in ORDER_SENSITIVE.items():
        for position, value in enumerate(values):
            # Overlapping fractional intervals: several pieces land in each hour.
            start = 0.1 * position
            rows.append((stay, start, start + 2.7, value * 3.0))
    frame = pd.DataFrame(rows, columns=["stay_id", "starttime", "endtime", "amount"])
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def test_volume_allocation_is_independent_of_row_order_and_chunk_layout():
    results = []
    for seed in range(6):
        for chunk in (1, 2, 5, 100_000):
            result = distribute_volume_hourly(
                _intervals(seed), val_col="amount", end_col="endtime",
                index_col="starttime", row_chunk_size=chunk,
            )
            results.append(
                result.sort_values(["stay_id", "starttime"]).reset_index(drop=True)
            )
    for other in results[1:]:
        pd.testing.assert_frame_equal(results[0], other, check_exact=True)


def _miiv_tables(order: list[int]) -> dict[str, pd.DataFrame]:
    start = pd.Timestamp("2150-01-01")
    rows = [
        (stay, value) for stay, values in ORDER_SENSITIVE.items() for value in values
    ]
    rows = [rows[i] for i in order]
    # Same-minute boluses: the source row order is the only order they have.
    inputs = pd.DataFrame(
        {
            "subject_id": [10 + stay for stay, _ in rows],
            "hadm_id": [20 + stay for stay, _ in rows],
            "stay_id": [stay for stay, _ in rows],
            "orderid": [100 + i for i in order],
            "linkorderid": [100 + i for i in order],
            "itemid": 220949,
            "starttime": start + pd.Timedelta(minutes=10),
            "endtime": start + pd.Timedelta(minutes=10),
            "amount": [value for _, value in rows],
            "amountuom": "mL",
            "statusdescription": "FinishedRunning",
            "rate": 0.0,
            "rateuom": "mL/hour",
        }
    )
    # Same-minute urine records take the SQL (DuckDB) sum path.
    urine = pd.DataFrame(
        {
            "subject_id": [10 + stay for stay, _ in rows],
            "hadm_id": [20 + stay for stay, _ in rows],
            "stay_id": [stay for stay, _ in rows],
            "itemid": 226559,
            "charttime": start + pd.Timedelta(minutes=30),
            "value": [value / 10.0 for _, value in rows],
            "valueuom": "mL",
        }
    )
    return {"inputevents": inputs, "outputevents": urine}


def _fluid_bytes(tmp_path, name, order, batches):
    root = tmp_path / name
    root.mkdir()
    source = mimic_synthetic_source(root, "miiv", **_miiv_tables(order))
    frames = {fluid: [] for fluid in FLUIDS}
    for batch in batches:
        loaded = _load(source, batch)
        for fluid in FLUIDS:
            frames[fluid].append(loaded[fluid])
    out = {}
    for fluid, parts in frames.items():
        frame = pd.concat(parts, ignore_index=True)
        keys = [column for column in frame.columns if column != fluid]
        frame = frame.sort_values(keys, kind="mergesort").reset_index(drop=True)
        out[fluid] = frame[fluid].to_numpy(dtype=np.float64).tobytes()
    return out


def _load(source, stays):
    from easyicu.concept import ConceptResolver
    from easyicu.resources import load_dictionary

    resolver = ConceptResolver(load_dictionary(include_sofa2=True))
    result = resolver.load_concepts(
        FLUIDS, source, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={"stay_id": stays},
        concept_workers=1, verbose=False,
    )
    return {name: table.data for name, table in result.items()}


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_public_loader_publishes_the_same_fluid_bytes_for_shuffled_and_split_rows(
    tmp_path, seed
):
    rows = sum(len(values) for values in ORDER_SENSITIVE.values())
    forward = list(range(rows))
    shuffled = forward[:]
    random.Random(seed).shuffle(shuffled)
    reference = _fluid_bytes(tmp_path, "forward", forward, [[1, 2, 3]])
    variants = {
        "reversed": (forward[::-1], [[1, 2, 3]]),
        "shuffled": (shuffled, [[1, 2, 3]]),
        "shuffled_split": (shuffled, [[3], [1, 2]]),
        "reversed_split": (forward[::-1], [[2], [3, 1]]),
    }
    for name, (order, batches) in variants.items():
        assert _fluid_bytes(tmp_path, name, order, batches) == reference, name
    total = mimic_synthetic_load(
        mimic_synthetic_source(_mkdir(tmp_path / "check"), "miiv", **_miiv_tables(forward)),
        ["total_input_ml"],
    )["total_input_ml"].data
    by_stay = total.groupby("stay_id")["total_input_ml"].sum()
    assert by_stay.to_dict() == {
        stay: math.fsum(values) for stay, values in ORDER_SENSITIVE.items()
    }


def _mkdir(path):
    path.mkdir()
    return path
