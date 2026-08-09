"""The host must lend the Coder this helper, not paste its body into the prompt.

Measured on 2026-07-30 over the 408 recorded generated ``analysis.py`` files:
236 (58%) hand-write a JSON coercion helper and 242 (59%) hand-write
``sha256_file``.  They were following orders -- ``providers/prompts/v1/
coder.txt`` pasted the body of ``to_jsonable`` under "Define this helper near
the top of EVERY script".  Twenty lines below, the same prompt tells the model
to *import* ``strict_numeric_input`` from ``methods.descriptive_inputs``, and
239 scripts duly import it.  So the container could always import; the pasted
body was an instruction, not a capability limit.

Both recorded module-level ``NameError`` deaths landed in that copied plumbing
-- ``hashlib`` (fresh22) and ``manifest`` (the 2026-07-30 H1 canary) -- not in
any analysis.  Every line the model does not write is a line it cannot get
wrong.

The pasted helper was also, on its own terms, broken: see
``test_the_pasted_helper_wrote_invalid_json_for_a_model_that_did_not_converge``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.script_runtime import to_jsonable, write_json

PROMPT = (
    Path(__file__).resolve().parents[2]
    / "src/easyicu/research_agent/providers/prompts/v1/coder.txt"
)


def _strict_loads(text: str):
    """Reject the non-standard tokens Python's json accepts by default."""

    def _refuse(constant: str):
        raise ValueError(f"not JSON: {constant}")

    return json.loads(text, parse_constant=_refuse)


def _pasted_helper(x):
    """The exact body the prompt used to ship, kept as the thing we replaced."""

    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return str(x)


def test_the_prompt_no_longer_ships_a_function_body_to_copy() -> None:
    """A pasted body is an instruction to hand-write; an import is not."""

    text = PROMPT.read_text(encoding="utf-8")

    assert "def to_jsonable" not in text
    assert "easyicu.research_agent.script_runtime import to_jsonable" in text
    assert "authority.provenance import sha256_file" in text


def test_the_pasted_helper_wrote_invalid_json_for_a_model_that_did_not_converge(
    tmp_path: Path,
) -> None:
    """``default=`` never sees a numpy float, because it is a Python float.

    ``json.dump`` serialises ``float`` subclasses itself, so the hook is not
    consulted and a non-converged estimate is written as the bare token ``NaN``.
    This is the case the helper existed for.
    """

    payload = {"primary_or": np.float64("nan"), "ci": [np.float64("inf"), 1.2]}
    destination = tmp_path / "old.json"
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, default=_pasted_helper)

    written = destination.read_text(encoding="utf-8")
    assert "NaN" in written and "Infinity" in written
    with pytest.raises(ValueError):
        _strict_loads(written)


def test_the_imported_writer_emits_null_for_every_non_finite_number(
    tmp_path: Path,
) -> None:
    """The same payload, through the host's own writer."""

    destination = write_json(
        tmp_path / "nested" / "new.json",
        {"primary_or": np.float64("nan"), "ci": [np.float64("inf"), 1.2]},
    )

    parsed = _strict_loads(destination.read_text(encoding="utf-8"))

    assert parsed == {"primary_or": None, "ci": [None, 1.2]}


def test_non_finite_floats_are_replaced_at_every_depth(tmp_path: Path) -> None:
    """A summary nests; sanitising only the top level would still emit NaN."""

    destination = write_json(
        tmp_path / "deep.json",
        {"models": [{"fit": {"or": float("nan"), "se": float("-inf")}}]},
    )

    parsed = _strict_loads(destination.read_text(encoding="utf-8"))

    assert parsed == {"models": [{"fit": {"or": None, "se": None}}]}


def test_the_writer_creates_the_directory_the_step_writes_into(
    tmp_path: Path,
) -> None:
    """Scripts write into a step output dir that may not exist yet."""

    destination = write_json(tmp_path / "a" / "b" / "s.json", {"n": 1})

    assert destination.is_file()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.int64(3), 3),
        (np.float64(1.5), 1.5),
        (np.float64("nan"), None),
        (np.bool_(True), True),
        (np.array([1, 2]), [1, 2]),
        (pd.NaT, None),
    ],
)
def test_the_hook_still_matches_the_pasted_helper_value_for_value(
    value, expected
) -> None:
    """Behaviour the scripts already depend on must not shift under them."""

    assert to_jsonable(value) == expected


def test_a_value_the_hook_cannot_place_becomes_text_rather_than_raising() -> None:
    """A summary that is one odd cell short should still be written."""

    class Odd:
        def __repr__(self) -> str:
            return "<odd>"

    assert to_jsonable(Odd()) == "<odd>"


def test_the_helper_does_not_re_export_a_function_that_already_has_an_owner() -> None:
    """``sha256_file`` lives in provenance; a second name is how copies start."""

    import easyicu.research_agent.script_runtime as module

    assert not hasattr(module, "sha256_file")
    assert sorted(module.__all__) == ["to_jsonable", "write_json"]
