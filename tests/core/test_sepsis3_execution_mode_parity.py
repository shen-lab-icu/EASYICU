"""Sepsis-3 (SOFA-2) does not depend on how the extraction was scheduled.

lead-r13 p9b: on the same MIMIC-III CareVue cohort, sepsis3_sofa2 had 929
rows when it streamed from the published score modules and 928 when it ran
in-process in the scoring group, before those modules were published. The
in-process branch recomputed ``sofa2`` and left an hour without any scored
component NA. The published total counts such an hour 0 (ricu ``sofa_score``:
fill_gaps, rowSums(na.rm = TRUE)), so the delta baseline, and with it the
onset, moved. Every mode now derives from the score module outputs.
"""

from __future__ import annotations

import contextlib
import json

import pandas as pd
import pytest

import easyicu
import easyicu.api as easyicu_api
from easyicu.api import extraction as ex
from easyicu.scores.sofa2_aggregate import SOFA2_COMPONENT_NAMES

HOURS = list(range(0, 11))
SI_HOUR = 5.0


def _sepsis_shared() -> pd.DataFrame:
    return pd.DataFrame({
        "stay_id": [1] * len(HOURS),
        "charttime": [float(h) for h in HOURS],
        "susp_inf": [h == SI_HOUR for h in HOURS],
    })


def _sofa2_score() -> pd.DataFrame:
    """Hours 0-2 have no scored component; respiration 2 from hour 3, 4 from hour 7."""
    rows = {"stay_id": [1] * len(HOURS), "charttime": [float(h) for h in HOURS]}
    for component in SOFA2_COMPONENT_NAMES:
        values, available = [], []
        for h in HOURS:
            if h < 3:
                values.append(None)
                available.append(False)
            elif component == "sofa2_resp":
                values.append(2.0 if h < 7 else 4.0)
                available.append(True)
            else:
                values.append(0.0)
                available.append(True)
        rows[component] = values
        rows[f"{component}_available"] = available
    return pd.DataFrame(rows)


def _recomputed_total() -> pd.DataFrame:
    """What the former in-process branch saw: the concept total, unscored hours NA."""
    shared = _sepsis_shared()
    total = [None if h < 3 else (2.0 if h < 7 else 4.0) for h in HOURS]
    return shared.assign(sofa2=total)


def _publish(directory, module: str, frame: pd.DataFrame) -> None:
    frame.to_parquet(directory / f"{module}.parquet", index=False)


@pytest.fixture
def published(tmp_path):
    directory = tmp_path / "published"
    directory.mkdir()
    _publish(directory, "sepsis_shared", _sepsis_shared())
    _publish(directory, "sofa2_score", _sofa2_score())
    return directory


def _streamed(tmp_path, published) -> pd.DataFrame:
    out = tmp_path / "streamed"
    out.mkdir()
    ex._run_special_extraction(
        ["sepsis3_sofa2"], "miiv", str(tmp_path / "unused-data"),
        {"stay_id": [1]}, 1, str(out), use_sofa2=True,
        stream_output_batches=True, published_output_dir=str(published),
    )
    return pd.read_parquet(out / "sep3_sofa2.parquet")


def _fake_module_extraction(module_name, concepts, database, data_path, patient_ids_filter,
                            batch_size, output_dir, use_sofa2=False, **_kwargs):
    from pathlib import Path

    out = Path(output_dir)
    frame = {"sepsis_shared": _sepsis_shared(), "sofa2_score": _sofa2_score()}[module_name]
    path = out / f"{module_name}.parquet"
    frame.to_parquet(path, index=False)
    (out / "_manifest.json").write_text(json.dumps({
        "module": module_name,
        "saved": {module_name: {"path": str(path), "rows": len(frame)}},
        "errors": [],
    }))


def _in_process_group(tmp_path, monkeypatch) -> pd.DataFrame:
    monkeypatch.setattr(ex, "_run_module_extraction", _fake_module_extraction)
    monkeypatch.setattr(ex, "_extract_worker_env_setup", lambda *a, **k: None)
    monkeypatch.setattr(
        easyicu_api, "keep_cache", lambda **_k: contextlib.nullcontext(None)
    )
    # The former in-process branch would recompute the score through
    # load_concepts; give it exactly what that recomputation returned.
    monkeypatch.setattr(easyicu, "load_concepts", lambda **_k: _recomputed_total())
    root = tmp_path / "group"
    root.mkdir()
    no_published = tmp_path / "nothing-published-yet"
    no_published.mkdir()
    ex._extract_module_group_worker(
        [("sepsis_shared", ex.EXTRACT_MODULES["sepsis_shared"]),
         ("sofa2_score", ex.EXTRACT_MODULES["sofa2_score"])],
        ["sepsis3_sofa2"], "miiv", str(tmp_path / "unused-data"), {"stay_id": [1]},
        None, str(root), True, False, str(no_published),
    )
    return pd.read_parquet(root / ex._SPECIAL_OUTPUT_DIRNAME / "sep3_sofa2.parquet")


def test_in_process_scoring_group_matches_the_streamed_derivation(tmp_path, published, monkeypatch):
    streamed = _streamed(tmp_path, published)
    in_group = _in_process_group(tmp_path, monkeypatch)
    # The unscored hours count 0, so the rise to 2 at hour 3 is the onset.
    assert streamed.loc[streamed["sep3_sofa2"].eq(1), "charttime"].tolist() == [3.0]
    pd.testing.assert_frame_equal(
        in_group.sort_values(["stay_id", "charttime"]).reset_index(drop=True),
        streamed.sort_values(["stay_id", "charttime"]).reset_index(drop=True),
    )


def test_a_lone_sepsis_request_extracts_its_dependencies_with_the_module_owner(
    tmp_path, published, monkeypatch
):
    monkeypatch.setattr(ex, "_run_module_extraction", _fake_module_extraction)
    calls = []
    monkeypatch.setattr(easyicu, "load_concepts", lambda **_k: calls.append(1))
    out = tmp_path / "alone"
    out.mkdir()
    ex._run_special_extraction(
        ["sepsis3_sofa2"], "miiv", str(tmp_path / "unused-data"), {"stay_id": [1]},
        None, str(out), use_sofa2=True,
    )
    assert calls == []  # no separate recomputation path
    alone = pd.read_parquet(out / "sep3_sofa2.parquet")
    pd.testing.assert_frame_equal(
        alone.sort_values(["stay_id", "charttime"]).reset_index(drop=True),
        _streamed(tmp_path, published).sort_values(["stay_id", "charttime"]).reset_index(drop=True),
    )
    assert not (out / ex._SPECIAL_DEPENDENCY_DIRNAME).exists()


def test_a_failed_dependency_module_refuses_sepsis(tmp_path, monkeypatch):
    def failed(module_name, concepts, database, data_path, patient_ids_filter, batch_size,
               output_dir, use_sofa2=False, **_kwargs):
        from pathlib import Path

        (Path(output_dir) / "_manifest.json").write_text(json.dumps({
            "module": module_name, "saved": {}, "errors": ["source read failed"],
        }))

    monkeypatch.setattr(ex, "_run_module_extraction", failed)
    out = tmp_path / "failed"
    out.mkdir()
    with pytest.raises(ex.SpecialDependencyError, match="sepsis_dependency_module_failed"):
        ex._run_special_extraction(
            ["sepsis3_sofa2"], "miiv", str(tmp_path / "unused-data"), {"stay_id": [1]},
            None, str(out), use_sofa2=True,
        )


def test_one_sofa2_delta_total_rule_for_every_path():
    from easyicu.scores.sepsis_sofa2 import sofa2_total_for_sepsis_delta

    supported = sofa2_total_for_sepsis_delta([None, 2.0], "miiv")
    assert supported.tolist() == [0.0, 2.0]
    # SICdb has no structural SOFA-2 total: no hour is imputed, so no Sepsis-3.
    unsupported = sofa2_total_for_sepsis_delta([None, 2.0], "sic")
    assert unsupported.isna().all()


# --- A lone Sepsis-3 request without a batch size stages with a bounded batch.
# lead-r13: the owner staged sofa2_score one-shot and was OOM-killed at 24 GiB.

COHORT = 2500


def _cohort_frames():
    import numpy as np

    hours = 30
    ids = np.repeat(np.arange(1, COHORT + 1), hours)
    times = np.tile(np.arange(hours, dtype=float), COHORT)
    shared = pd.DataFrame({"stay_id": ids, "charttime": times, "susp_inf": times == 12.0})
    rng = np.random.default_rng(20261106)
    score = {"stay_id": ids, "charttime": times}
    for component in SOFA2_COMPONENT_NAMES:
        available = rng.random(len(ids)) > 0.2
        score[component] = np.where(available, rng.integers(0, 3, len(ids)).astype(float), np.nan)
        score[f"{component}_available"] = available
    return {"sepsis_shared": shared, "sofa2_score": pd.DataFrame(score)}


def _recording_owner(frames, received):
    def owner(module_name, concepts, database, data_path, ids, batch_size, output_dir,
              use_sofa2=False, stream_output_batches=False, **_kwargs):
        from pathlib import Path

        received[module_name] = (batch_size, stream_output_batches)
        target = Path(output_dir) / f"{module_name}.parquet"
        frames[module_name].to_parquet(target, index=False)
        (Path(output_dir) / "_manifest.json").write_text(json.dumps({
            "module": module_name,
            "saved": {module_name: {"path": str(target)}},
            "errors": [],
        }))

    return owner


def _lone_request(tmp_path, name, batch_size):
    out = tmp_path / name
    out.mkdir()
    ex._run_special_extraction(
        ["sepsis3_sofa2"], "mimic", str(tmp_path / "unused-data"),
        {"icustay_id": list(range(1, COHORT + 1))}, batch_size, str(out), use_sofa2=True,
    )
    return out


def test_a_lone_request_without_batch_stages_bounded_and_matches_batch_1000(tmp_path, monkeypatch):
    frames = _cohort_frames()
    received = {}
    monkeypatch.setattr(ex, "_run_module_extraction", _recording_owner(frames, received))
    monkeypatch.setattr(ex, "_available_memory_mb", lambda: 4096.0)  # a constrained laptop

    auto = _lone_request(tmp_path, "auto", None)
    staging = json.loads((auto / "_manifest.json").read_text())["dependency_staging"]
    plan = ex.plan_extraction_resources(
        "mimic", ["sepsis_shared", "sofa2_score"], COHORT, available_memory_mb=4096.0
    )
    # The owner stages through its streamed (bounded-batch) path, never an
    # unbatched one-shot load.  The planner's batch is the upper bound (here
    # the whole cohort), capped at the special batch of 1000.
    assert plan.batch_size == COHORT
    assert staging["staging_batch_size"] == 1000
    assert set(received.values()) == {(1000, True)}
    assert staging["staging_reason_code"] == "special_staging_batch_cap"
    assert staging["staging_resource_plan"]["batch_size"] == plan.batch_size
    assert staging["staging_resource_plan"]["reason_code"] == plan.reason_code
    assert staging["derivation_batch_size"] == 1000
    assert staging["derivation_reason_code"] == "special_derivation_fixed_batch"
    assert staging["dependency_sources"] == {
        "sepsis_shared": "produced", "sofa2_score": "produced",
    }

    received.clear()
    explicit = _lone_request(tmp_path, "explicit", 1000)
    explicit_staging = json.loads((explicit / "_manifest.json").read_text())["dependency_staging"]
    assert set(received.values()) == {(1000, True)}
    assert explicit_staging["staging_reason_code"] == "explicit_batch_size"
    assert (auto / "sep3_sofa2.parquet").read_bytes() == (
        explicit / "sep3_sofa2.parquet"
    ).read_bytes()
    assert len(pd.read_parquet(auto / "sep3_sofa2.parquet")) > 0


def test_staging_uses_the_fixed_batch_when_the_planner_cannot_plan(tmp_path, monkeypatch):
    frames = _cohort_frames()
    received = {}
    monkeypatch.setattr(ex, "_run_module_extraction", _recording_owner(frames, received))

    def no_plan(*_args, **_kwargs):
        raise ValueError("no plan")

    monkeypatch.setattr(ex, "plan_extraction_resources", no_plan)
    out = _lone_request(tmp_path, "fixed", None)
    staging = json.loads((out / "_manifest.json").read_text())["dependency_staging"]
    assert set(received.values()) == {(1000, True)}
    assert staging["staging_reason_code"] == "special_staging_fixed_batch"
    assert staging["staging_planner_error"] == "ValueError"


@pytest.mark.parametrize("cohort,expected,reason", [
    (30000, 1000, "special_staging_batch_cap"),
    (600, 600, None),  # a cohort below the cap stages whole, with the planner's reason
])
def test_the_staging_batch_is_the_planner_bound_capped_at_1000(monkeypatch, cohort, expected, reason):
    monkeypatch.setattr(ex, "_available_memory_mb", lambda: 4096.0)
    policy = ex._special_dependency_batch_policy(
        "mimic", ["sepsis_shared", "sofa2_score"], cohort, None
    )
    assert policy["staging_batch_size"] == expected
    assert policy["staging_reason_code"] == (
        reason or policy["staging_resource_plan"]["reason_code"]
    )
    assert policy["derivation_batch_size"] == min(cohort, 1000)
