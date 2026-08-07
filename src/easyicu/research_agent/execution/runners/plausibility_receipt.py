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

__all__ = [
    "host_plausibility_receipt_injected",
    "render_standard_plausibility_receipt_code",
]


def host_plausibility_receipt_injected(
    code: str,
    *,
    scope: "FlagOnlyPlausibilityScope | None",
    already_satisfied: bool,
) -> str:
    """Return ``code`` with the host's own receipt appended, when it is owed.

    MEASURED over every recorded run, ``flag_only_plausibility_obligation`` is
    the single largest pre-execution blocker: 37 findings across 32 distinct
    steps in 8 of the 9 tasks, 53 % of all mechanical-preflight findings. The
    obligation is mechanical -- read each declared column's bounds from the
    sealed manifest, count what falls outside, file the counts under one exact
    key -- and the host renders it correctly for its own executors. Only
    agent-authored steps must hand-write it, and they get it wrong: h2's
    causal step spent BOTH of its LLM repairs on this one message, with five
    provider calls still unspent, and died anyway.

    The alternative considered and rejected was a host helper the agent calls.
    It fails on the decisive point: it still depends on the agent REMEMBERING
    to call it, which is the exact thing that fails 37 times. This module's own
    docstring gives the second reason -- the comparisons are rendered into the
    source so the static gate can verify the code that will actually run, which
    an imported helper defeats.

    Injection happens before the concept audit, so the approved digest and the
    executed digest both cover the assembled script and their identity is
    preserved by construction.
    """

    body = str(code or "")
    if scope is None or not scope.expected_columns or already_satisfied:
        return body
    if not body.strip():
        return body

    receipt = render_standard_plausibility_receipt_code(
        scope,
        frame_name="plausibility_frame",
    )
    # The receipt runs AFTER the body, and the names are PLAIN.
    #
    # Both were forced by measurement. Prefixing the names BREAKS the gate: it
    # follows the receipt value by NAME into the summary write, and a rename --
    # even one rebound immediately before the write -- loses the flow and the
    # step stays refused. And a prologue would have been worse than useless:
    # agent bodies routinely bind ``plausibility_audit`` themselves (`= {}`,
    # `= build_plausibility_audit(...)`), so a body running after a prologue
    # silently overwrites the host's value. Running last makes the host's value
    # win by construction and leaves the delivery flowing straight out of the
    # host's own comparisons, which is exactly what the gate reads.
    tail = "\n\n".join(
        (
            textwrap.dedent(
                """
                import json
                import os
                from pathlib import Path

                import pandas as pd

                plausibility_frame = pd.read_parquet(
                    os.environ.get("EASYICU_UNIVERSE_PARQUET")
                    or os.environ["COHORT_PARQUET"]
                )
                """
            ).strip(),
            receipt,
            # The body may not write a summary at all -- a step that failed
            # earlier, or one whose product is a figure. Creating one here would
            # manufacture a record the step never produced, so the patch is
            # guarded on the file the body actually wrote. (Measured: the guard
            # costs the gate nothing; guarded and unguarded both clear it.)
            textwrap.dedent(
                """
                _easyicu_host_summary_path = (
                    Path(os.environ["STEP_OUT_DIR"]) / "step_summary.json"
                )
                if _easyicu_host_summary_path.exists():
                    _easyicu_host_summary = json.loads(
                        _easyicu_host_summary_path.read_text(encoding="utf-8")
                    )
                    _easyicu_host_summary["plausibility_audit"] = plausibility_audit
                    _easyicu_host_summary_path.write_text(
                        json.dumps(
                            _easyicu_host_summary, indent=2, sort_keys=True
                        ),
                        encoding="utf-8",
                    )
                """
            ).strip(),
        )
    )
    return body.rstrip() + "\n\n" + tail + "\n"


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
