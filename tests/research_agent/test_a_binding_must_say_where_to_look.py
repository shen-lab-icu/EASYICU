"""A typed input the host cannot describe must not be published as readable.

The typed-input plane hands a consumer a path and a digest.  For a table it
also compiles a schema receipt -- columns, dtypes, row count -- so generated
code knows where the values are.  For a serialization it cannot read, it
compiles nothing, and the consumer is left to guess.

MEASURED over the recorded resolved inputs on 2026-08-03 (1,071 bindings across
the then-current canonical-9 corpus):

    992  resolve to physical tables       -> full column/dtype/row receipt
     76  are self-describing JSON values  -> parsed directly
      3  are NEITHER

All three of the third group are one pickle bound as
``artifact:trained_prediction_model`` in
``batch_20260802_luna_miiv_FULL_44d5d5c_verify01/m2_mortality_prediction``.  Its
whole contract was ``{schema_version, identity_row}`` -- a sha256 and nothing
else.  Each of its three consumers invented its own guess and died::

    06_held_out_discrimination  ValueError: Prediction artifact contract must
                                declare id_column and prediction_column
    08_held_out_calibration     RuntimeError: The trained_prediction_model
                                artifact does not contain a supported held-out
                                prediction table or aligned prediction vectors
    10_clinical_utility         RuntimeError: table:validation_design lacks
                                consumption_contract

``08_held_out_calibration_figure`` and ``11_robustness_replay_figure`` then died
as dependency collateral.  One unreadable binding cost five of thirteen steps --
the largest single failure cluster in that task.

Later runs added structured JSON artifacts. Their suffix makes the bytes
parseable, but consumers still need a host-sealed value-free ``json_structure``
receipt to locate nested paths without guessing. That receipt is compiled by
the production binder; this predicate remains solely the serialization-level
readability boundary.

The rule is keyed on the SERIALIZATION, deliberately.
``RUNTIME_TYPED_INPUT_EVIDENCE_KINDS`` maps ``artifact``, ``log``, ``manifest``
and ``model`` all onto ``log`` evidence, so the registry label does not say
whether the bytes can be read.  Whether a consumer needs coordinates does.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from easyicu.research_agent.authority.typed_binding import (  # noqa: E402
    _SELF_DESCRIBING_TYPED_INPUT_SUFFIXES,
    _binding_is_readable_without_a_schema_receipt as readable,
)

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")

#: The exact run in which one unreadable binding cost five of thirteen steps.
_M2_RUN = (
    _CORPUS
    / "batch_20260802_luna_miiv_FULL_44d5d5c_verify01"
    / "m2_mortality_prediction"
    / "aware"
    / "run_20260803T022017_18735e"
)


# --------------------------------------------------------------------------
# The rule itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "suffix",
    sorted(_SELF_DESCRIBING_TYPED_INPUT_SUFFIXES),
)
def test_a_self_describing_file_needs_no_coordinates(suffix: str) -> None:
    """The recorded statistic path: 76 of 76 bindings, contract-free, working."""

    assert readable(pathlib.Path(f"evidence/x{suffix}"), None) is True


@pytest.mark.parametrize(
    "suffix", [".pkl", ".pickle", ".joblib", ".npy", ".npz", ".bin", ""]
)
def test_an_opaque_file_with_no_producer_contract_is_refused(suffix: str) -> None:
    """Nothing in the binding says what is inside, and nothing can."""

    assert readable(pathlib.Path(f"evidence/x{suffix}"), None) is False
    assert readable(pathlib.Path(f"evidence/x{suffix}"), {}) is False


def test_a_bare_schema_version_is_not_a_description() -> None:
    """The exact contract the dead pickle carried -- a version and an identity.

    ``identity_row`` is stripped before this predicate sees the contract, so
    what remains for the failing case is ``{"schema_version": ...}``.  A version
    number tells a consumer nothing about where a value lives.
    """

    assert (
        readable(
            pathlib.Path("evidence/trained_prediction_model.pkl"),
            {"schema_version": "easyicu.host_typed_product.v1"},
        )
        is False
    )


def test_a_producer_that_declared_coordinates_keeps_its_binding() -> None:
    """The refusal is for what the host cannot describe, not for pickles.

    A producer that states where its values are has satisfied the requirement
    this rule exists to enforce, so the serialization stops mattering.
    """

    assert (
        readable(
            pathlib.Path("evidence/trained_prediction_model.pkl"),
            {
                "schema_version": "easyicu.host_typed_product.v1",
                "id_column": "patient_stay_id",
                "prediction_column": "predicted_probability",
            },
        )
        is True
    )


# --------------------------------------------------------------------------
# The rule against the recorded corpus, not against restated intent
# --------------------------------------------------------------------------


def _recorded_non_table_bindings():
    """Every recorded binding that does NOT take the schema-receipt branch."""

    for path in _CORPUS.rglob("resolved_inputs/*.json"):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for input_key, binding in (document.get("inputs") or {}).items():
            if not isinstance(binding, dict):
                continue
            if binding.get("evidence_kind") == "table":
                continue
            contract = {
                key: value
                for key, value in (binding.get("product_contract") or {}).items()
                if key != "identity_row"
            }
            yield (
                str(input_key),
                pathlib.Path(str(binding.get("relative_path") or "")),
                contract,
                str(binding.get("product") or ""),
            )


def test_the_rule_keeps_parseable_bindings_and_refuses_recorded_opaque_failure() -> None:
    """Replay the serialization boundary over an append-only run corpus.

    New parseable products may be appended over time, so the working set is not
    frozen to three historical statistic names. The exact opaque failure stays
    locked by the focused production-binder test below.
    """

    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    kept: list[str] = []
    refused: list[tuple[str, str]] = []
    for input_key, relative_path, contract, product in _recorded_non_table_bindings():
        if readable(relative_path, contract):
            kept.append(input_key)
        else:
            refused.append((product, relative_path.suffix.lower()))

    if not kept and not refused:
        pytest.skip("no recorded non-table typed binding is on disk")

    # The working side grows as fresh runs add parseable JSON products.
    assert kept, "the corpus must still contain the bindings this rule keeps"
    assert {
        "statistic:complete_case_n",
        "statistic:primary_or",
        "statistic:robustness_summary",
    } <= set(kept)

    # The historical opaque failure remains refused. Future opaque products may
    # add further entries without making this append-only corpus test stale.
    assert refused, "the defect must still be present in the corpus to be meaningful"
    assert ("trained_prediction_model", ".pkl") in set(refused)


def test_the_production_binder_refuses_the_pickle_and_says_why() -> None:
    """Drives ``_resolved_typed_input_binding``, not the predicate behind it.

    Every other test in this file asks the rule.  This one asks the function
    production calls, with the real sealed evidence of the run that died, and
    checks both halves of the fix: no binding is published, and the refusal
    names the serialization and tells the Planner what to declare instead.

    Without this, deleting the gate at the CALL SITE leaves the whole file
    green -- a load-bearing test that drove a helper.  That is exactly how the
    previous fix in this series first failed its own mutation check.
    """

    run_dir = _M2_RUN
    if not run_dir.exists():
        pytest.skip("the m2 run that recorded this failure is not on disk")

    from easyicu.research_agent.authority.typed_binding import (
        _resolved_typed_input_binding,
    )
    from easyicu.research_agent.schema import EvidenceRef

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    evidence_records = manifest["evidence"]
    step_records = manifest["per_step_records"]
    pickle_record = next(
        record
        for record in evidence_records
        if str(record.get("relative_path", "")).endswith(
            "__trained_prediction_model.pkl"
        )
    )

    refusals: list[dict] = []
    binding = _resolved_typed_input_binding(
        input_name="artifact:trained_prediction_model",
        evidence_ref=EvidenceRef(
            evidence_id=str(pickle_record["evidence_id"]),
            kind=str(pickle_record["kind"]),
            relative_path=str(pickle_record["relative_path"]),
        ),
        evidence_records=evidence_records,
        run_dir=run_dir,
        producer_step_records=step_records,
        refusals=refusals,
    )

    assert binding is None, "an undescribable product must not be published"
    assert len(refusals) == 1, refusals
    refusal = refusals[0]
    assert refusal["reason"] == "typed_input_serialization_is_unreadable"
    assert refusal["input"] == "artifact:trained_prediction_model"
    assert refusal["serialization"] == ".pkl"
    assert refusal["produced_by_step"] == "05_fit_prediction_model"
    # The refusal has to be at least as useful as the three invented errors it
    # replaces, or this fix only moves the confusion earlier.
    assert "table" in refusal["message"]


def test_the_production_binder_still_publishes_a_readable_sibling() -> None:
    """The same run, the same call, the input that always worked.

    ``table:validation_design`` is bound by all three of the steps that died,
    carries a full schema receipt, and must be untouched.  A rule that refuses
    the pickle by refusing everything is not a fix.
    """

    run_dir = _M2_RUN
    if not run_dir.exists():
        pytest.skip("the m2 run that recorded this failure is not on disk")

    from easyicu.research_agent.authority.typed_binding import (
        _resolved_typed_input_binding,
    )
    from easyicu.research_agent.schema import EvidenceRef

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    evidence_records = manifest["evidence"]
    table_record = next(
        record
        for record in evidence_records
        if str(record.get("relative_path", "")).endswith("__validation_design.csv")
    )

    refusals: list[dict] = []
    binding = _resolved_typed_input_binding(
        input_name="table:validation_design",
        evidence_ref=EvidenceRef(
            evidence_id=str(table_record["evidence_id"]),
            kind=str(table_record["kind"]),
            relative_path=str(table_record["relative_path"]),
        ),
        evidence_records=evidence_records,
        run_dir=run_dir,
        producer_step_records=manifest["per_step_records"],
        refusals=refusals,
    )

    assert refusals == []
    assert binding is not None
    assert binding["product_contract"]["columns"]


def test_no_recorded_table_binding_is_touched_by_this_rule() -> None:
    """The 992 schema-receipt bindings never reach the predicate.

    Stated as its own test because the cheapest way to break this fix is to
    move the check above the table branch, where it would refuse every parquet
    cohort in the corpus for having no producer-declared coordinates.
    """

    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    tables = 0
    for path in _CORPUS.rglob("resolved_inputs/*.json"):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for binding in (document.get("inputs") or {}).values():
            if isinstance(binding, dict) and binding.get("evidence_kind") == "table":
                tables += 1
    if not tables:
        pytest.skip("no recorded table binding is on disk")
    assert tables >= 900, tables
