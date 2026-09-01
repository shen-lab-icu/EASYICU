"""Naming a column without naming what may appear in it is half a contract.

e2 in the 2026-08-02 sweep ran 7 steps and lost one:
``06_lactate_mortality_association_figure`` died in 1.1 s with::

    RuntimeError: No lact_max outcome_risk rows are available

The host had told that step the table has a ``summary_type`` column, and had
told it the column's dtype -- but not one word about what may be IN it.  The
real values were ``continuous_distribution`` and
``quartile_mortality_distribution``; the model wrote ``.eq("outcome_risk")``,
selected zero rows, and failed closed.  The rows it wanted were right there
with ``event_rate``, ``event_n`` and ``denominator`` all populated.  The figure
was one string away from working, and that one death is what turned the whole
e2 draft into a fail-closed placeholder.

Measured before writing this, over the recorded corpus: 222 distinct typed
tables compile a receipt, 221 of them gain a published value set, none loses
its dtype facts, none exceeds the 64 KB receipt cap (median 1,431 bytes, max
2,932), and the largest real vocabulary is 19 values -- inside the bound of 24,
so the bound is not truncating anything real.

Honest scope: only ONE recorded step demonstrably died of this.  The argument
for publishing anyway is not frequency, it is that this is a fact the host
already holds, it costs ~1.4 KB, and unlike a gate it has no false-block mode.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from easyicu.research_agent.contracts.typed_schema import (
    _MAX_TYPED_TABLE_CATEGORY_CARDINALITY,
    _MAX_TYPED_TABLE_CATEGORY_VALUE_CHARS,
    _MAX_TYPED_TABLE_RECEIPT_BYTES,
    merge_host_table_contract,
    typed_product_prompt_facts,
    typed_product_schema_receipt,
)


def _seal(frame: pd.DataFrame, path: Path) -> dict:
    """Write the table and compile its receipt the way the host does."""

    frame.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return typed_product_schema_receipt(
        artifact_path=path,
        expected_sha256=digest,
    )


def test_a_category_column_publishes_the_values_it_can_hold(tmp_path: Path):
    """The exact shape e2 needed and did not get."""

    receipt = _seal(
        pd.DataFrame(
            {
                "variable": ["lact_max"] * 4,
                "summary_type": [
                    "continuous_distribution",
                    "quartile_mortality_distribution",
                    "quartile_mortality_distribution",
                    "quartile_mortality_distribution",
                ],
                "event_rate": [None, 0.075592, 0.095419, 0.281122],
            }
        ),
        tmp_path / "absolute_risk_context.csv",
    )

    assert receipt["categorical_values"]["summary_type"] == [
        "continuous_distribution",
        "quartile_mortality_distribution",
    ]
    # A consumer that has to select rows can now check its own literal.
    assert "outcome_risk" not in receipt["categorical_values"]["summary_type"]
    # The numeric column is described by its dtype, not by a value set.
    assert "event_rate" not in receipt["categorical_values"]


def test_a_high_cardinality_column_publishes_nothing(tmp_path: Path):
    """Above the bound the column is not a vocabulary, so say nothing.

    Silence is the pre-existing state and claims nothing; a truncated list
    would claim a closed set that is not closed.
    """

    over = _MAX_TYPED_TABLE_CATEGORY_CARDINALITY + 1
    receipt = _seal(
        pd.DataFrame({"note": [f"free text {index}" for index in range(over)]}),
        tmp_path / "wide.csv",
    )
    assert "note" not in receipt.get("categorical_values", {})


def test_a_long_valued_column_publishes_nothing(tmp_path: Path):
    """A long value is prose or an identifier, not a category label."""

    too_long = "x" * (_MAX_TYPED_TABLE_CATEGORY_VALUE_CHARS + 1)
    receipt = _seal(
        pd.DataFrame({"blob": [too_long, "short"]}),
        tmp_path / "long.csv",
    )
    assert "blob" not in receipt.get("categorical_values", {})


def test_an_all_null_column_publishes_nothing(tmp_path: Path):
    """No observed value is not the same claim as an empty vocabulary."""

    receipt = _seal(
        pd.DataFrame({"label": [None, None], "keep": ["a", "b"]}),
        tmp_path / "null.csv",
    )
    values = receipt.get("categorical_values", {})
    assert "label" not in values
    assert values["keep"] == ["a", "b"]


def test_the_coder_is_told_the_values_for_the_columns_it_was_given(
    tmp_path: Path,
):
    """The receipt only helps if it reaches the prompt, scoped like the rest."""

    receipt = _seal(
        pd.DataFrame(
            {
                "summary_type": ["a", "b"],
                "other_label": ["p", "q"],
                "event_rate": [0.1, 0.2],
            }
        ),
        tmp_path / "facts.csv",
    )
    contract = merge_host_table_contract({"product": "x"}, receipt)

    facts = typed_product_prompt_facts(contract, ["summary_type", "event_rate"])
    assert facts["categorical_values"] == {"summary_type": ["a", "b"]}
    assert (
        "other_label" not in facts["categorical_values"]
    ), "a column the step was not given must not leak into its prompt"


def test_a_producer_cannot_write_its_own_value_set(tmp_path: Path):
    """Host-observed facts are the host's; a producer claim is not evidence.

    The case that matters is the one where the host publishes NOTHING for a
    column -- above the cardinality bound, deliberately silent.  If the
    producer's key is not reserved, the host's silence is exactly where an
    invented vocabulary survives, because there is no host value to overwrite
    it with.  A first version of this test used a small table instead and
    passed even with the reservation removed: the receipt overwrote the forged
    key on its own, so the test was watching the wrong half.
    """

    over = _MAX_TYPED_TABLE_CATEGORY_CARDINALITY + 1
    receipt = _seal(
        pd.DataFrame({"note": [f"free text {index}" for index in range(over)]}),
        tmp_path / "forge.csv",
    )
    assert "categorical_values" not in receipt, "the host must be silent here"

    contract = merge_host_table_contract(
        {
            "product": "x",
            "categorical_values": {"note": ["invented_value"]},
        },
        receipt,
    )
    assert (
        "categorical_values" not in contract
    ), "the producer's invented vocabulary survived the host's silence"


def test_oversize_value_sets_cost_only_themselves(tmp_path: Path):
    """Dropping the optional part must not drop what consumers already use."""

    label = "y" * _MAX_TYPED_TABLE_CATEGORY_VALUE_CHARS
    frame = pd.DataFrame(
        {
            f"cat_{index:03d}": [
                f"{label[:-4]}{value:04d}"
                for value in range(_MAX_TYPED_TABLE_CATEGORY_CARDINALITY)
            ]
            for index in range(80)
        }
    )
    receipt = _seal(frame, tmp_path / "huge.csv")

    assert receipt is not None, "the receipt must survive, only shrink"
    assert "categorical_values" not in receipt
    assert "numeric_columns" in receipt and "column_dtypes" in receipt
    assert len(json.dumps(receipt).encode()) <= _MAX_TYPED_TABLE_RECEIPT_BYTES
