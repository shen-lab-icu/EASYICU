"""Regression tests for extract_database module grouping (shared table reads).

``extract_database`` runs each module in an isolated subprocess. Previously
every subprocess re-read its source tables from disk, so modules that share
chartevents/labevents re-scanned the same 30GB table N times. The fix groups
modules that share source-table families into one subprocess and reuses the
raw/table cache across them via ``keep_cache``.

These tests pin the pure grouping/partitioning logic (no data, no subprocess):

1. every requested normal module appears exactly once across the groups;
2. grouping strictly reduces the subprocess count vs. one-per-module;
3. the Sepsis-3 special modules attach to the SOFA scoring group so
   susp_inf/sofa/sofa2 hit the shared cache instead of being recomputed;
4. ``group_modules=False`` reproduces the legacy one-subprocess-per-module
   shape;
5. the sofa2 trigger detection stays consistent with ``load_concepts``.

Run without ``--run-real``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.api.extraction import (
    EXTRACT_MODULES,
    EXTRACT_MODULE_ORDER,
    _SPECIAL_CONCEPT_MODULES,
    _get_extraction_mp_context,
    _group_modules_for_extraction,
    _normalise_module_frame_for_parquet,
    _resolve_extraction_grouping,
)
from easyicu.api.concepts import _concepts_need_sofa2


def _split(modules):
    normal = [m for m in modules if m not in _SPECIAL_CONCEPT_MODULES]
    special = [m for m in modules if m in _SPECIAL_CONCEPT_MODULES]
    return normal, special


def test_every_module_grouped_exactly_once():
    normal, special = _split(EXTRACT_MODULE_ORDER)
    groups = _group_modules_for_extraction(normal, special, True)

    flat = [m for g in groups for m in g["modules"]]
    assert sorted(flat) == sorted(normal), (sorted(flat), sorted(normal))
    # no duplicates
    assert len(flat) == len(set(flat))


def test_grouping_reduces_subprocess_count():
    normal, special = _split(EXTRACT_MODULE_ORDER)
    grouped = _group_modules_for_extraction(normal, special, True)
    ungrouped = _group_modules_for_extraction(normal, special, False)

    assert len(grouped) < len(ungrouped)
    # ungrouped == one subprocess per normal module + one for the special set
    assert len(ungrouped) == len(normal) + 1
    # each ungrouped normal group holds exactly one module
    normal_ungrouped = [g for g in ungrouped if g["modules"]]
    assert all(len(g["modules"]) == 1 for g in normal_ungrouped)


def test_low_memory_host_automatically_isolates_modules():
    enabled, reason = _resolve_extraction_grouping(
        True,
        False,
        environment={},
        total_memory_mb=16 * 1024,
    )

    assert enabled is False
    assert reason == "low_memory_host"


def test_constrained_cache_budget_isolates_modules_on_large_server():
    enabled, reason = _resolve_extraction_grouping(
        True,
        False,
        environment={"EASYICU_CACHE_BUDGET_MB": "2048"},
        total_memory_mb=1024 * 1024,
    )

    assert enabled is False
    assert reason == "constrained_cache_budget"


def test_large_server_keeps_grouped_speed_path():
    enabled, reason = _resolve_extraction_grouping(
        True,
        False,
        environment={},
        total_memory_mb=128 * 1024,
    )

    assert enabled is True
    assert reason == "shared_cache_speed_path"


def test_expert_grouping_override_has_priority_over_host_size():
    enabled, reason = _resolve_extraction_grouping(
        True,
        False,
        environment={"EASYICU_EXTRACT_GROUPING": "1"},
        total_memory_mb=16 * 1024,
    )

    assert enabled is True
    assert reason == "forced_by_environment"


def test_special_modules_attach_to_scoring_group():
    normal, special = _split(EXTRACT_MODULE_ORDER)
    assert special, "fixture expects Sepsis-3 special modules present"
    groups = _group_modules_for_extraction(normal, special, True)

    carriers = [g for g in groups if g["special"]]
    assert len(carriers) == 1, "special modules must live in exactly one group"
    carrier = carriers[0]
    assert set(carrier["special"]) == set(special)
    # the carrier group must also hold a SOFA scoring module so susp_inf/sofa
    # are already cached in-process when sep3 is computed
    assert any(
        m in ("sofa1_score", "sofa2_score", "sepsis_shared")
        for m in carrier["modules"]
    )


def test_special_only_request_still_runs():
    # Requesting only the Sepsis-3 modules (no scoring group present) must
    # still produce a runnable group rather than silently dropping them.
    groups = _group_modules_for_extraction([], list(_SPECIAL_CONCEPT_MODULES), True)
    assert len(groups) == 1
    assert set(groups[0]["special"]) == set(_SPECIAL_CONCEPT_MODULES)
    assert groups[0]["modules"] == []


def test_subset_request_groups_only_requested():
    groups = _group_modules_for_extraction(
        ["vitals", "sofa2_score"], ["sepsis3_sofa2"], True
    )
    flat = [m for g in groups for m in g["modules"]]
    assert sorted(flat) == ["sofa2_score", "vitals"]
    # sofa2_score group carries the sep3_sofa2 special module
    carrier = next(g for g in groups if g["special"])
    assert "sofa2_score" in carrier["modules"]


def test_unknown_module_forms_its_own_group():
    groups = _group_modules_for_extraction(["vitals", "brand_new_module"], [], True)
    flat = [m for g in groups for m in g["modules"]]
    assert sorted(flat) == ["brand_new_module", "vitals"]


def test_sofa2_trigger_detection_matches_loader():
    assert _concepts_need_sofa2(["sofa2"]) is True
    assert _concepts_need_sofa2(["sofa2_cardio"]) is True
    assert _concepts_need_sofa2(["rrt_criteria"]) is True
    # substring match on 'sofa2' (e.g. a derived name) also triggers
    assert _concepts_need_sofa2(["my_sofa2_delta"]) is True
    # plain SOFA-1 concepts must NOT pull in the sofa2 dictionary
    assert _concepts_need_sofa2(["sofa", "hr", "map"]) is False


class _FakeMultiprocessing:
    def __init__(self, methods=("fork", "spawn")):
        self.methods = list(methods)
        self.requested = None

    def get_all_start_methods(self):
        return list(self.methods)

    def get_context(self, method):
        self.requested = method
        return SimpleNamespace(method=method)


@pytest.mark.parametrize(
    "platform_name",
    ["posix", "nt"],
)
def test_extraction_process_default_is_spawn_on_all_platforms(
    monkeypatch, platform_name
):
    monkeypatch.delenv("EASYICU_MP_START_METHOD", raising=False)
    fake_mp = _FakeMultiprocessing()

    context = _get_extraction_mp_context(
        fake_mp, platform_name=platform_name
    )

    assert context.method == "spawn"
    assert fake_mp.requested == "spawn"


def test_expert_can_explicitly_override_spawn_default(monkeypatch):
    monkeypatch.setenv("EASYICU_MP_START_METHOD", "fork")
    fake_mp = _FakeMultiprocessing()

    context = _get_extraction_mp_context(fake_mp, platform_name="posix")

    assert context.method == "fork"


def test_invalid_extraction_process_method_fails_clearly(monkeypatch):
    monkeypatch.setenv("EASYICU_MP_START_METHOD", "not-a-method")
    fake_mp = _FakeMultiprocessing()

    with pytest.raises(ValueError, match="not-a-method"):
        _get_extraction_mp_context(fake_mp, platform_name="posix")


def test_module_parquet_columns_follow_catalog_order_across_hash_seeds():
    frame = pd.DataFrame(
        {
            "patientunitstayid": [1],
            "charttime": [0.0],
            "circ_event": [0],
            "circ_failure": [1],
        }
    )

    normalised = _normalise_module_frame_for_parquet(
        frame,
        ["circ_failure", "circ_event"],
    )

    assert list(normalised.columns) == [
        "patientunitstayid",
        "charttime",
        "circ_failure",
        "circ_event",
    ]


def test_module_parquet_normalisation_reuses_canonical_numeric_frame():
    """Do not duplicate a full dense module at the Arrow write boundary."""
    frame = pd.DataFrame(
        {
            "patientunitstayid": [1, 2],
            "charttime": [0.0, 1.0],
            "glu": [90.0, 100.0],
        }
    )

    normalised = _normalise_module_frame_for_parquet(frame, ["glu"])

    assert normalised is frame


def test_renal_module_exports_reference_native_and_evidence_receipts():
    """Current renal exports separate phenotype layers from data quality."""
    renal = set(EXTRACT_MODULES["renal"])
    assert {
        "aki_reference",
        "aki_stage_reference",
        "aki_severe_reference",
        "aki_source_native",
        "aki_stage_source_native",
        "aki_severe_source_native",
        "aki_reference_profile",
        "aki_source_native_profile",
        "kidney_observation_window_coverage",
        "creatinine_evidence_status",
        "creatinine_evidence_reason",
        "urine_evidence_status",
        "rrt_evidence_status",
        "creat_baseline_n_48h",
        "creat_baseline_n_7d",
        "creat_baseline_source",
        "creat_pre_icu_history_observed",
    }.issubset(renal)
    assert {
        "aki",
        "aki_stage",
        "aki_assessable",
        "aki_ascertainment",
    }.isdisjoint(renal)
