"""``cohort:x`` and ``dataset:x`` are one product, so one check must say so.

The binding registry canonicalises the Planner's cohort alias on purpose --
``contracts.declared_product._canonical_kind`` maps ``cohort`` to ``dataset``
so plan DAG construction, declared-output validation and runtime binding share
one identity.  ``load_typed_input`` then compared that canonical
``declared_kind`` against the *raw* prefix of the input key, which can never
agree for a plan that spelled its cohort ``cohort:analysis_set`` -- one of the
four spellings the planner prompt lists as legal.

Measured on fresh24: four steps died this way in one plan, and the repair loop
was a trap.  The refusal says "kind 'dataset' does not match the input key
'cohort:analysis_set'", so the model rewrote the key to ``dataset:analysis_set``
-- the canonical spelling -- and bindings are filed under the plan's spelling,
so that attempt failed as ``binding_absent``.  Both spellings failed, for
opposite reasons, and each step spent its whole repair budget finding out there
was no third one.

``test_the_real_fresh24_capsule_loads`` is the load-bearing one; the rest hold
the line that this is an alias, not a hole.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.typed_input_binding import (
    TypedInputBindingError,
    load_typed_input,
)

STEP_ID = "06_primary_adjusted_association"


def _capsule(tmp_path: Path, *, input_key: str, declared_kind: str, product: str):
    """The shape the host actually wrote: canonical kind, planner-spelled key."""

    run_dir = tmp_path / "run"
    (run_dir / "evidence").mkdir(parents=True)
    table = run_dir / "evidence" / "analysis_set.csv"
    pd.DataFrame({"age": [61, 74, 55], "death": [0, 1, 0]}).to_csv(table, index=False)
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    manifest = {
        "step_id": STEP_ID,
        "inputs": {
            input_key: {
                "relative_path": "evidence/analysis_set.csv",
                "sha256": digest,
                "declared_kind": declared_kind,
                "evidence_kind": "table",
                "product": product,
                "evidence_id": "ev-cohort",
                "identity_row": {
                    "input_key": input_key,
                    "declared_kind": declared_kind,
                    "product": product,
                    "evidence_id": "ev-cohort",
                    "sha256": digest,
                },
                "product_contract": {
                    "columns": ["age", "death"],
                    "row_count": 3,
                },
                "consumption_contract": {
                    "input_key": input_key,
                    "mode": "all_rows",
                    "artifact_sha256": digest,
                },
            }
        },
    }
    return run_dir, manifest


def _load(run_dir: Path, manifest: dict, *, input_key: str):
    return load_typed_input(
        input_key=input_key,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=STEP_ID,
        expected_evidence_kind="table",
        expected_columns=("age", "death"),
        exclusive=True,
        require_consumption_contract=True,
    )


def test_the_real_fresh24_capsule_loads(tmp_path: Path) -> None:
    """Planner spelled it ``cohort:``; the registry recorded ``dataset``."""

    run_dir, manifest = _capsule(
        tmp_path,
        input_key="cohort:analysis_set",
        declared_kind="dataset",
        product="analysis_set",
    )

    bound = _load(run_dir, manifest, input_key="cohort:analysis_set")

    assert bound.input_key == "cohort:analysis_set"
    assert bound.declared_kind == "dataset"
    assert bound.row_count == 3


def test_the_canonical_spelling_of_the_same_capsule_also_loads(tmp_path: Path) -> None:
    """Whichever spelling the plan used, one product is one product."""

    run_dir, manifest = _capsule(
        tmp_path,
        input_key="dataset:analysis_set",
        declared_kind="dataset",
        product="analysis_set",
    )

    bound = _load(run_dir, manifest, input_key="dataset:analysis_set")

    assert bound.declared_kind == "dataset"


def test_a_real_kind_disagreement_is_still_refused(tmp_path: Path) -> None:
    """An alias is not a hole: ``table`` and ``dataset`` stay distinct."""

    run_dir, manifest = _capsule(
        tmp_path,
        input_key="dataset:analysis_set",
        declared_kind="table",
        product="analysis_set",
    )

    with pytest.raises(TypedInputBindingError) as excinfo:
        _load(run_dir, manifest, input_key="dataset:analysis_set")

    assert excinfo.value.reason_code == "product_identity_mismatch"


def test_a_real_product_disagreement_is_still_refused(tmp_path: Path) -> None:
    run_dir, manifest = _capsule(
        tmp_path,
        input_key="cohort:analysis_set",
        declared_kind="dataset",
        product="some_other_table",
    )

    with pytest.raises(TypedInputBindingError) as excinfo:
        _load(run_dir, manifest, input_key="cohort:analysis_set")

    assert excinfo.value.reason_code == "product_identity_mismatch"


def test_the_alias_comes_from_the_shared_owner_not_a_second_copy() -> None:
    """If this module ever grows its own map, the two can drift apart again."""

    from easyicu.research_agent.contracts.declared_product import typed_product

    assert typed_product("cohort:analysis_set") == typed_product("dataset:analysis_set")
    assert typed_product("table:analysis_set") != typed_product("dataset:analysis_set")

    source = Path(
        "src/easyicu/research_agent/execution/runners/typed_input_binding.py"
    ).read_text(encoding="utf-8")
    assert "declared_product import typed_product" in source
    assert '"cohort": "dataset"' not in source
