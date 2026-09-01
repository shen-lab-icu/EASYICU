"""The Planner was asked to transcribe a constant, and a step died when it didn't.

m1 in the 2026-08-02 sweep ran 7 steps and lost two.  One of them,
``06_adjusted_association_figure``, died in 1.3 s inside the container::

    RuntimeError: Typed input is missing consumption_contract:
    table:adjusted_association_estimates

The comparison that settles it: e1's ``11_missingness_figure`` is the same
shape of step -- one figure, one typed table -- at the same plan revision.
e1's step declared ``input_consumption_contracts`` and ran.  m1's declared
``[]``, so ``_attach_verified_consumption_contract`` returned the binding
untouched and the executor, which requires the receipt, failed closed.

Measured over the recorded corpus before changing anything: 235 consumption
contracts are declared and **all 235 have the identical shape** -- mode
``all_rows``, no role column, no expected roles.  There is no variation to
learn from, because the field carries no choice in the case that matters.
466 typed inputs were handed to steps with no contract at all, and every one
of those 466 already carried the row count, digest and file the receipt is
made of.

``all_rows`` is the *absence* of a selection -- ``artifact_consumption``'s own
rule is that a consumer with no explicit role selection must preserve every
row.  So the host compiling it asserts nothing new.  ``single_row`` and
``one_per_role`` do assert something, and the host never compiles those.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.typed_binding import (
    _attach_verified_consumption_contract,
)
from easyicu.research_agent.contracts.artifact_consumption import (
    ArtifactConsumptionError,
)
from easyicu.research_agent.schema import (
    AnalysisStep,
    ArtifactConsumptionContract,
)

INPUT_KEY = "table:adjusted_association_estimates"


def _table(path: Path, rows: int = 3) -> dict:
    frame = pd.DataFrame(
        {"term": [f"level_{index}" for index in range(rows)], "estimate": range(rows)}
    )
    frame.to_csv(path, index=False)
    return {
        "absolute_path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "identity_row": {"input_key": INPUT_KEY},
        "product_contract": {
            "schema_version": "easyicu.host_typed_product.v4",
            "tabular_format": "csv",
            "column_count": len(frame.columns),
            "columns": list(frame.columns),
            "row_count": len(frame),
        },
    }


def _step(contracts: list[ArtifactConsumptionContract] | None = None) -> AnalysisStep:
    return AnalysisStep(
        step_id="06_adjusted_association_figure",
        intent="Render the adjusted association estimates.",
        inputs=[INPUT_KEY],
        expected_outputs=["figure:adjusted_association"],
        method="visualization",
        input_consumption_contracts=contracts or [],
    )


def test_an_undeclared_typed_input_still_gets_a_verified_receipt(tmp_path: Path):
    """The exact m1 step, which previously reached its executor with nothing."""

    binding = _attach_verified_consumption_contract(
        step=_step(),
        input_name=INPUT_KEY,
        binding=_table(tmp_path / "estimates.csv"),
    )

    receipt = binding["consumption_contract"]
    assert receipt["input_key"] == INPUT_KEY
    assert receipt["mode"] == "all_rows"
    assert receipt["verified_row_count"] == 3
    # The receipt is bound to bytes, not asserted: it carries the same digest
    # the host verified, so a consumer can check it against what it opens.
    assert receipt["artifact_sha256"] == binding["sha256"]


def test_a_declared_contract_still_wins(tmp_path: Path):
    """Compiling a default must not overwrite a Planner decision."""

    binding = _attach_verified_consumption_contract(
        step=_step(
            [
                ArtifactConsumptionContract(
                    input_key=INPUT_KEY,
                    mode="one_per_role",
                    role_column="term",
                    expected_roles=["level_0", "level_1", "level_2"],
                )
            ]
        ),
        input_name=INPUT_KEY,
        binding=_table(tmp_path / "estimates.csv"),
    )

    assert binding["consumption_contract"]["mode"] == "one_per_role"


def test_the_host_never_compiles_a_mode_that_claims_something(tmp_path: Path):
    """A singleton claim is a scientific claim, so only the Planner may make it.

    The table here has three rows.  Had the host inferred ``single_row`` from
    anything, this would have raised; had it inferred nothing, the executor
    would die as m1's did.  ``all_rows`` is the answer that adds no claim, and
    a consumer that genuinely needs a singleton still sees a mode mismatch
    rather than a fabricated guarantee.
    """

    binding = _attach_verified_consumption_contract(
        step=_step(),
        input_name=INPUT_KEY,
        binding=_table(tmp_path / "estimates.csv", rows=3),
    )
    assert binding["consumption_contract"]["mode"] == "all_rows"


@pytest.mark.parametrize(
    "mutate, why",
    [
        (lambda b: b.pop("product_contract"), "no host product contract"),
        (lambda b: b["product_contract"].pop("row_count"), "no verified row count"),
        (lambda b: b.update(sha256="not-a-digest"), "no usable digest"),
        (lambda b: b.update(absolute_path="/nonexistent/file.csv"), "file is gone"),
    ],
)
def test_a_binding_without_the_facts_is_left_exactly_as_it_was(
    tmp_path: Path, mutate, why: str
):
    """Compiling must be strictly additive.

    The receipt is made of the row count, the digest and the bytes.  When the
    binding does not already carry those, the host stays silent -- which is
    what it did before this change, so nothing that works today can begin to
    fail here.
    """

    binding = _table(tmp_path / "estimates.csv")
    mutate(binding)

    result = _attach_verified_consumption_contract(
        step=_step(),
        input_name=INPUT_KEY,
        binding=binding,
    )
    assert "consumption_contract" not in result, why


def test_an_untyped_input_name_is_not_given_a_receipt(tmp_path: Path):
    """The receipt names one canonical ``kind:product``; anything else is not one."""

    binding = _table(tmp_path / "estimates.csv")
    binding["identity_row"] = {"input_key": "lactate_max"}
    result = _attach_verified_consumption_contract(
        step=_step(),
        input_name="lactate_max",
        binding=binding,
    )
    assert "consumption_contract" not in result


def test_two_contracts_for_one_input_still_fail_closed(tmp_path: Path):
    """The ambiguity refusal must survive the clause being reordered.

    ``AnalysisStep`` already rejects duplicate ``input_key`` values, so this
    cannot arrive through a validated plan and the raise is a second line of
    defence -- which is why the step is built here without validation.  It is
    locked because this change reordered the clauses around it: the old code
    returned early on an empty list and only then counted, and the count now
    has to come first or a duplicate pair would be silently resolved by
    picking the first.
    """

    step = AnalysisStep.model_construct(
        step_id="06_adjusted_association_figure",
        intent="Render the adjusted association estimates.",
        inputs=[INPUT_KEY],
        expected_outputs=["figure:adjusted_association"],
        method="visualization",
        input_consumption_contracts=[
            ArtifactConsumptionContract(input_key=INPUT_KEY, mode="all_rows"),
            ArtifactConsumptionContract(input_key=INPUT_KEY, mode="single_row"),
        ],
    )
    with pytest.raises(ArtifactConsumptionError):
        _attach_verified_consumption_contract(
            step=step,
            input_name=INPUT_KEY,
            binding=_table(tmp_path / "estimates.csv"),
        )
