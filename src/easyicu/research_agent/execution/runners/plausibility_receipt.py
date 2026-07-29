"""Render the flag-only plausibility receipt used by standard executors.

The authority layer decides *which* exact raw columns owe a receipt.  This
module owns only the mechanical executor-side implementation: read those
columns' bounds from the sealed resolved-input manifest, retain every value,
count observations outside each bound, and expose the canonical
``plausibility_audit`` object for ``step_summary.json``.

The comparisons are deliberately rendered into the sandbox source instead of
hidden behind an imported runtime helper.  That lets the pre-execution static
gate verify the exact code that will run.

Each record also carries ``compared_n``: how many observations were actually
compared against the bound.  Without it a column that is entirely missing
emits ``out_of_range_n = 0`` -- byte-identical to a fully observed, entirely
in-range column.  The obligation gate already refuses a receipt that appears
only when the count is nonzero, on the grounds that "no out-of-range rows" and
"we never looked" are different facts; a receipt with no denominator loses that
same distinction one level down, and death and other partly recorded outcomes
are exactly where it bites.
"""

from __future__ import annotations

import textwrap

from ...authority.plausibility import FlagOnlyPlausibilityScope

__all__ = ["render_standard_plausibility_receipt_code"]


def render_standard_plausibility_receipt_code(
    scope: FlagOnlyPlausibilityScope,
    *,
    frame_name: str,
) -> str:
    """Return sandbox source that computes the scope's canonical receipt."""

    if not frame_name.isidentifier():
        raise ValueError("standard executor plausibility frame must be an identifier")
    if not scope.expected_columns:
        return ""

    # The block carries its own imports rather than relying on whichever
    # prologue splices it.  Five executors do, and four of them happened to
    # import `hashlib` for their own reasons; the fifth did not, so the
    # host's own generated script died at `NameError: name 'hashlib' is not
    # defined` on a real run -- inside a step the host had just claimed as
    # deterministic, which then spent a runtime repair and a post-mutation
    # concept repair on a missing import and was blocked.  A shared fragment
    # that compiles only when its caller happens to have the right names is a
    # coupling, not a contract.  Re-importing is idempotent.
    return textwrap.dedent(
        f"""
        import hashlib
        import json
        import os
        from pathlib import Path

        import pandas as pd

        plausibility_expected_columns = {scope.expected_columns!r}
        plausibility_manifest = json.loads(
            Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]).read_text(
                encoding="utf-8"
            )
        )
        raw_input_contracts = plausibility_manifest.get("raw_input_contracts")
        raw_input_contracts_payload = (
            dict(raw_input_contracts)
            if isinstance(raw_input_contracts, dict)
            else {{}}
        )
        declared_contracts_sha256 = raw_input_contracts_payload.pop(
            "contracts_sha256",
            None,
        )
        computed_contracts_sha256 = hashlib.sha256(
            json.dumps(
                raw_input_contracts_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if (
            not isinstance(raw_input_contracts, dict)
            or declared_contracts_sha256
            != {scope.source_contracts_sha256!r}
            or computed_contracts_sha256 != declared_contracts_sha256
        ):
            raise RuntimeError(
                "Resolved plausibility contracts do not match the step authority"
            )
        contracts = raw_input_contracts.get("contracts")
        if not isinstance(contracts, dict):
            raise RuntimeError("Resolved plausibility contracts are missing")

        plausibility_audit = {{}}
        for column, contract in plausibility_manifest[
            "raw_input_contracts"
        ]["contracts"].items():
            if column not in plausibility_expected_columns:
                continue
            if not isinstance(contract, dict) or column not in {frame_name}.columns:
                raise RuntimeError(
                    "Flag-only plausibility column is absent from its bound frame"
                )
            plausibility_range = contract.get("analysis_plausibility_range")
            plausibility_policy = contract.get("plausibility_policy")
            if (
                not isinstance(plausibility_range, dict)
                or not isinstance(plausibility_policy, dict)
                or plausibility_policy.get("range_policy") != "flag_only"
                or plausibility_policy.get("out_of_range_action")
                != "retain_and_flag"
            ):
                raise RuntimeError(
                    "Flag-only plausibility policy is absent or changed"
                )
            minimum = plausibility_range.get("minimum")
            maximum = plausibility_range.get("maximum")
            if minimum is None and maximum is None:
                raise RuntimeError("Flag-only plausibility range has no bound")
            numeric = pd.to_numeric({frame_name}[column], errors="coerce")
            below_minimum_n = (
                int((numeric < float(minimum)).sum())
                if minimum is not None
                else 0
            )
            above_maximum_n = (
                int((numeric > float(maximum)).sum())
                if maximum is not None
                else 0
            )
            plausibility_audit[column] = {{
                "policy": "retain_and_flag",
                "below_minimum_n": below_minimum_n,
                "above_maximum_n": above_maximum_n,
                "out_of_range_n": below_minimum_n + above_maximum_n,
                "compared_n": int(numeric.notna().sum()),
                "observed_n": int({frame_name}[column].notna().sum()),
            }}
        if set(plausibility_audit) != set(plausibility_expected_columns):
            raise RuntimeError(
                "Flag-only plausibility scope is absent from the sealed contracts"
            )
        """
    ).strip()
