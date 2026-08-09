"""The one owner of "which bytes may this step read".

This module exists because the same question was previously answered by a copy
in each runner, and each copy checked a slightly different subset. So these
tests are written against the *owner*, not through a consumer: every refusal
path is exercised directly and pinned to its stable ``reason_code``, which is
the part callers and audits are allowed to branch on.

The reason codes matter more than the messages. A message may be reworded; a
code that silently changes meaning would break an audit that reads it.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.typed_input_binding import (
    BINDING_REASON_CODES,
    TypedInputBindingError,
    contained_regular_file,
    load_typed_input,
)

INPUT_KEY = "table:some_product"
STEP_ID = "07_a_step"


def _run(tmp_path: Path, *, rows: int = 3) -> tuple[Path, dict]:
    """A complete, honest capsule -- every negative test starts from this."""

    run_dir = tmp_path / "run"
    (run_dir / "inputs").mkdir(parents=True)
    table = run_dir / "inputs" / "some_product.csv"
    pd.DataFrame({"a": range(rows), "b": range(rows)}).to_csv(table, index=False)
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    manifest = {
        "step_id": STEP_ID,
        "inputs": {
            INPUT_KEY: {
                "relative_path": "inputs/some_product.csv",
                "sha256": digest,
                "declared_kind": "table",
                "evidence_kind": "table",
                "product": "some_product",
                "evidence_id": "ev-1",
                "identity_row": {
                    "input_key": INPUT_KEY,
                    "declared_kind": "table",
                    "product": "some_product",
                    "evidence_id": "ev-1",
                    "sha256": digest,
                },
                "product_contract": {"columns": ["a", "b"], "row_count": rows},
                "consumption_contract": {
                    "input_key": INPUT_KEY,
                    "mode": "all_rows",
                    "artifact_sha256": digest,
                },
            }
        },
    }
    return run_dir, manifest


def _load(run_dir: Path, manifest: dict, **kwargs):
    options = {
        "input_key": INPUT_KEY,
        "run_dir": run_dir,
        "resolved_inputs": manifest,
        "step_id": STEP_ID,
        "expected_declared_kind": "table",
        "expected_evidence_kind": "table",
        "expected_columns": ("a", "b"),
        "exclusive": True,
        "require_consumption_contract": True,
    }
    options.update(kwargs)
    return load_typed_input(**options)


def _refusal(run_dir: Path, manifest: dict, **kwargs) -> str:
    with pytest.raises(TypedInputBindingError) as excinfo:
        _load(run_dir, manifest, **kwargs)
    assert excinfo.value.reason_code in BINDING_REASON_CODES
    return excinfo.value.reason_code


# --------------------------------------------------------------------------


def test_a_complete_capsule_loads(tmp_path: Path) -> None:
    run_dir, manifest = _run(tmp_path)
    bound = _load(run_dir, manifest)
    assert bound.input_key == INPUT_KEY
    assert bound.columns == ("a", "b")
    assert bound.row_count == 3
    assert bound.declared_kind == "table"
    assert bound.product == "some_product"
    assert bound.evidence_id == "ev-1"
    assert len(bound.frame) == 3
    assert bound.path.is_file()


def test_every_refusal_carries_a_declared_reason_code(tmp_path: Path) -> None:
    """The table of refusals, each pinned to the code callers may branch on.

    Written as one table rather than one test per row because the property
    being pinned is the mapping itself: a capsule broken in exactly one way
    produces exactly one code.
    """

    def _break(mutate, **kwargs) -> str:
        run_dir, manifest = _run(tmp_path / str(id(mutate)))
        mutate(manifest, run_dir)
        return _refusal(run_dir, manifest, **kwargs)

    def _set(field, value):
        def mutate(manifest, _run_dir):
            manifest["inputs"][INPUT_KEY][field] = value

        return mutate

    assert _break(lambda m, _: m.__setitem__("step_id", "other")) == (
        "manifest_step_mismatch"
    )
    assert _break(lambda m, _: m.pop("inputs")) == "binding_absent"
    assert (
        _break(
            lambda m, _: m["inputs"].__setitem__(
                "table:other", dict(m["inputs"][INPUT_KEY])
            )
        )
        == "binding_widened"
    )
    assert _break(lambda m, _: m["inputs"].pop(INPUT_KEY)) == "binding_absent"
    assert _break(_set("relative_path", "")) == "binding_incomplete"
    assert _break(_set("sha256", "nothex")) == "binding_incomplete"
    assert _break(_set("product_contract", "not-a-contract")) == "binding_incomplete"
    assert _break(_set("declared_kind", "artifact")) == "declared_kind_mismatch"
    assert _break(_set("evidence_kind", "figure")) == "evidence_kind_mismatch"
    assert _break(_set("product", "another_product")) == "product_identity_mismatch"
    assert _break(_set("consumption_contract", {"input_key": INPUT_KEY})) == (
        "consumption_contract_mismatch"
    )
    assert _break(lambda m, _: m["inputs"][INPUT_KEY].pop("consumption_contract")) == (
        "consumption_contract_mismatch"
    )
    assert _break(_set("relative_path", "../escape.csv")) == "path_not_contained"
    assert _break(_set("relative_path", "inputs/missing.csv")) == "path_not_contained"


def test_a_capsule_that_disagrees_with_its_own_identity_row_is_refused(
    tmp_path: Path,
) -> None:
    """Four fields near each other are not the same thing as one record."""

    for field, value in (
        ("sha256", "0" * 64),
        ("product", "something_else"),
        ("evidence_id", "ev-other"),
        ("input_key", "table:other"),
    ):
        run_dir, manifest = _run(tmp_path / field)
        manifest["inputs"][INPUT_KEY]["identity_row"][field] = value
        assert _refusal(run_dir, manifest) == "product_identity_mismatch"


def test_edited_bytes_are_refused_even_when_the_contract_still_fits(
    tmp_path: Path,
) -> None:
    """The digest is what makes 'the same shape' mean 'the same table'."""

    run_dir, manifest = _run(tmp_path)
    table = run_dir / "inputs" / "some_product.csv"
    pd.DataFrame({"a": [9, 9, 9], "b": [9, 9, 9]}).to_csv(table, index=False)
    assert _refusal(run_dir, manifest) == "digest_mismatch"


def test_a_contract_that_does_not_describe_the_bytes_is_refused(
    tmp_path: Path,
) -> None:
    """A digest proves the file is unchanged, not that it is what was promised."""

    run_dir, manifest = _run(tmp_path / "cols")
    manifest["inputs"][INPUT_KEY]["product_contract"]["columns"] = ["a", "z"]
    assert _refusal(run_dir, manifest, expected_columns=None) == (
        "contract_columns_mismatch"
    )

    run_dir, manifest = _run(tmp_path / "rows")
    manifest["inputs"][INPUT_KEY]["product_contract"]["row_count"] = 99
    assert _refusal(run_dir, manifest) == "contract_row_count_mismatch"

    run_dir, manifest = _run(tmp_path / "expected")
    assert _refusal(run_dir, manifest, expected_columns=("a", "b", "c")) == (
        "contract_columns_mismatch"
    )

    run_dir, manifest = _run(tmp_path / "floor")
    assert _refusal(run_dir, manifest, minimum_row_count=10) == (
        "product_contract_incomplete"
    )


def test_an_unreadable_manifest_is_refused(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path)
    bad = run_dir / "not_json.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(TypedInputBindingError) as excinfo:
        load_typed_input(
            input_key=INPUT_KEY, run_dir=run_dir, resolved_inputs=bad, step_id=STEP_ID
        )
    assert excinfo.value.reason_code == "manifest_unreadable"


def test_a_manifest_on_disk_loads_the_same_way(tmp_path: Path) -> None:
    """Path and mapping are the same contract, not two code paths."""

    run_dir, manifest = _run(tmp_path)
    path = run_dir / "resolved_inputs.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _load(run_dir, path).sha256 == _load(run_dir, manifest).sha256


def test_an_unsupported_table_format_is_refused(tmp_path: Path) -> None:
    run_dir, manifest = _run(tmp_path)
    source = run_dir / "inputs" / "some_product.csv"
    target = run_dir / "inputs" / "some_product.txt"
    target.write_bytes(source.read_bytes())
    binding = manifest["inputs"][INPUT_KEY]
    binding["relative_path"] = "inputs/some_product.txt"
    assert _refusal(run_dir, manifest) == "unsupported_format"


# --------------------------------------------------------------------------
# Containment
# --------------------------------------------------------------------------


def test_a_symlink_on_any_segment_is_refused(tmp_path: Path) -> None:
    """A link that lands back inside the run is still a link out of it.

    Both the pre- and post-resolution checks matter: this one resolves to a
    real file, so only the per-segment symlink walk catches it.
    """

    root = tmp_path / "root"
    (root / "real").mkdir(parents=True)
    target = root / "real" / "table.csv"
    target.write_text("a,b\n1,2\n", encoding="utf-8")
    link = root / "link"
    os.symlink(root / "real", link)

    assert contained_regular_file(root / "real" / "table.csv", root) is not None
    assert contained_regular_file(link / "table.csv", root) is None


def test_a_path_leaving_the_run_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("a,b\n1,2\n", encoding="utf-8")
    assert contained_regular_file(outside, root) is None
    assert contained_regular_file(root / ".." / "outside.csv", root) is None


def test_a_directory_is_not_a_regular_file(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "subdir").mkdir(parents=True)
    assert contained_regular_file(root / "subdir", root) is None


def test_the_owner_carries_no_case_specific_branch() -> None:
    import easyicu.research_agent.execution.runners.typed_input_binding as module

    source = Path(module.__file__).read_text().lower()
    for token in ("sepsis", "sep3", "e1_", "mortality", "icu_readmission"):
        assert token not in source, f"case-specific token in production: {token}"
