"""The runner wrote one product name; the registry advertised another.

The capability registry advertises this owner to the Planner under the name
``absolute_risk_context``, so plans promise ``table:absolute_risk_context``.
``run_absolute_risk_context`` wrote ``exposure_outcome_summary.csv`` and
registered ``output_files = {"exposure_outcome_summary": ...}``, unconditionally.

So the step ran, produced a correct table, and ``declared_product_contract``
refused it: ``declared_product_missing``, ``missing_products =
['table:absolute_risk_context']`` against ``registered_products`` containing
``table:exposure_outcome_summary``.  The host then spent both LLM repairs, and
the second one introduced ``pd.isfinite`` -- which does not exist -- so the
container died with AttributeError on host-owned work.

This went unseen for as long as the owner never claimed a step (0 of 89, the
method allowlist).  e2's first real claim hit all three incompatibilities in
sequence: the allowlist, the plausibility receipt, and this name.

Only the four names the ownership predicate admits are honoured, so a plan
cannot rename the product into something this runner does not compute.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from easyicu.research_agent.execution.runners.deterministic_descriptive import (
    _SUPPORTED_PRODUCTS,
    _declared_product,
)
from easyicu.research_agent.execution.phase import (
    _absolute_risk_context_runner_owns_step as owns_step,
)


def test_the_advertised_name_is_what_gets_registered():
    """The exact case that failed in e2/verify19."""

    assert (
        _declared_product({"expected_outputs": ["table:absolute_risk_context"]})
        == "absolute_risk_context"
    )


def test_the_historical_name_still_works():
    assert (
        _declared_product({"expected_outputs": ["table:exposure_outcome_summary"]})
        == "exposure_outcome_summary"
    )


def test_nothing_declared_keeps_the_historical_name():
    assert _declared_product({}) == "exposure_outcome_summary"
    assert _declared_product({"expected_outputs": []}) == "exposure_outcome_summary"


def test_a_product_this_runner_does_not_compute_is_refused():
    """A plan may choose the NAME, never the science.

    Honouring an arbitrary declared name would let a plan promising an adjusted
    odds ratio receive a descriptive table under that name.
    """

    for outputs in (
        ["table:adjusted_odds_ratio"],
        ["table:table_one"],
        ["table:robustness_matrix"],
        ["figure:absolute_risk_context"],
    ):
        assert _declared_product({"expected_outputs": outputs}) == (
            "exposure_outcome_summary"
        )


def test_every_honoured_name_is_one_this_owner_would_claim():
    """The two lists must not drift apart.

    A name honoured here but refused by the ownership predicate would register
    a product for a step this runner never runs; the reverse would refuse a
    plan the owner did claim -- which is exactly the defect being fixed.
    """

    for product in _SUPPORTED_PRODUCTS:
        assert owns_step("descriptive", "04_context", [f"table:{product}"]), product


def test_the_registration_carries_a_kind_prefix(tmp_path):
    """The canonical envelope refuses a bare name.

    verify20: this step was the only one of ten registering a bare name, and
    the bounded-metric shadow blocked it with ``invalid_product_identity`` --
    "a registered product did not use a valid kind:name identity". Every
    sibling registers ``table:``/``figure:``/``log:``.
    """

    summary = _run_against(tmp_path, "table:absolute_risk_context")

    for identity in summary["output_files"]:
        kind, separator, name = identity.partition(":")
        assert separator, identity
        assert kind == "table", identity
        assert name in _SUPPORTED_PRODUCTS, identity


def test_the_kind_prefix_is_required():
    """A bare filename is not a typed product declaration."""

    assert (
        _declared_product({"expected_outputs": ["absolute_risk_context"]})
        == "exposure_outcome_summary"
    )


def _run_against(tmp_path: pathlib.Path, declared: str) -> dict:
    """Execute the real runner over a tiny cohort and return its summary.

    Two mutations survived a resolver-only test file: hard-coding the product
    back into the write, and registering a name different from the file
    written. Both leave ``_declared_product`` perfect and the runner broken, so
    the runner itself has to be executed.
    """

    import os

    import pandas as pd

    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / "04_absolute_risk_context" / "outputs"
    out_dir.mkdir(parents=True)
    cohort = pd.DataFrame(
        {
            "stay_id": range(40),
            "lact_max": [1.0 + (index % 7) for index in range(40)],
            "death": [index % 4 == 0 for index in range(40)],
        }
    )
    cohort_path = run_dir / "cohort_analysis.parquet"
    cohort.to_parquet(cohort_path)
    (run_dir / "analysis_plan.json").write_text(
        json.dumps(
            {
                "research_question": "q",
                "rationale": "r",
                "steps": [
                    {
                        "step_id": "04_absolute_risk_context",
                        "intent": "Descriptive absolute risk.",
                        "method": "descriptive",
                        "planned_analysis_role": "auxiliary",
                        "inputs": ["artifact:analysis_cohort", "death", "lact_max"],
                        "expected_outputs": [declared],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "research_context.json").write_text(
        json.dumps({"outcome": "death", "variables": []}), encoding="utf-8"
    )

    from easyicu.research_agent.execution.runners.deterministic_descriptive import (
        run_absolute_risk_context,
    )

    previous = {
        key: os.environ.get(key)
        for key in (
            "STEP_OUT_DIR",
            "COHORT_PARQUET",
            "EASYICU_RUN_DIR",
            "EASYICU_STEP_ID",
            "OUTCOME_COL",
        )
    }
    os.environ.update(
        {
            "STEP_OUT_DIR": str(out_dir),
            "COHORT_PARQUET": str(cohort_path),
            "EASYICU_RUN_DIR": str(run_dir),
            "EASYICU_STEP_ID": "04_absolute_risk_context",
            "OUTCOME_COL": "death",
        }
    )
    try:
        run_absolute_risk_context()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    return json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))


def test_the_runner_registers_the_declared_name_end_to_end(tmp_path):
    """The case that died: the plan promises the registry's advertised name."""

    summary = _run_against(tmp_path, "table:absolute_risk_context")

    assert summary["status"] == "ok", summary
    # ``kind:name``, as every other runner registers. A bare name is refused by
    # the canonical envelope with ``invalid_product_identity`` -- measured in
    # verify20, where this step was the ONLY one of ten registering a bare name.
    assert list(summary["output_files"]) == ["table:absolute_risk_context"], summary[
        "output_files"
    ]


def test_the_registered_name_always_matches_the_file_written(tmp_path):
    """A registration pointing at a differently-named file is unresolvable."""

    for declared in ("table:absolute_risk_context", "table:exposure_outcome_summary"):
        summary = _run_against(tmp_path / declared.replace(":", "_"), declared)
        ((identity, filename),) = summary["output_files"].items()
        kind, separator, product = identity.partition(":")
        assert separator and kind == "table", identity
        assert filename == f"{product}.csv", summary["output_files"]
        assert declared == identity


def test_the_recorded_failure_is_what_this_repairs():
    """Anchors the fix in the run that exposed it."""

    run = pathlib.Path(
        "/Volumes/外置硬盘/easyicu_data/canonical9_runs"
        "/batch_20260804_luna_miiv_FULL_83a5e66_verify19"
    )
    if not run.exists():
        pytest.skip("the verify19 run is not mounted")

    manifests = list(run.glob("*/aware/run_*/manifest.json"))
    if not manifests:
        pytest.skip("the recorded run carries no manifest")
    manifest = json.loads(manifests[0].read_text())

    record = next(
        (
            item
            for item in manifest.get("per_step_records", [])
            if str(item.get("step_id")) == "04_absolute_risk_context"
        ),
        None,
    )
    if record is None:
        pytest.skip("that run's plan had no absolute-risk context step")

    triggers = record.get("contract_repair_triggers") or []
    detail = next(
        (
            finding.get("detail") or {}
            for batch in triggers
            for finding in batch
            if str(finding.get("validator")) == "declared_product_contract"
        ),
        None,
    )
    if detail is None:
        pytest.skip("no declared-product trigger was recorded")

    assert detail.get("missing_products") == ["table:absolute_risk_context"]
    assert "table:exposure_outcome_summary" in (detail.get("registered_products") or [])
    # And the plan's declaration is exactly what the resolver now honours.
    plan = json.loads((manifests[0].parent / "analysis_plan.json").read_text())
    step = next(
        item for item in plan["steps"] if item["step_id"] == "04_absolute_risk_context"
    )
    assert _declared_product(step) == "absolute_risk_context"
