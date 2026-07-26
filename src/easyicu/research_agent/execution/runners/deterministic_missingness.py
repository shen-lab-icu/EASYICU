"""Deterministic missingness / measurement-process audit runner.

This auxiliary runner emits a per-concept measurement-missingness table that
distinguishes **structural no-source** (the concept is not sourced for the stay
at all) from **measurement missingness** (the concept is sourced but was not
measured in the window). It returns a self-contained runner script that computes
the audit WITHOUT an LLM coder call.

Motivation (E3 / M1, 2026-07-08): the missingness/measurement audit is a PURE
COUNT — per concept, how many stays have a measured value vs none, and the
percentages. Yet on real runs the LLM coder for this step reliably exhausted its
retry budget (~27.6 min, IDENTICAL across two runs) and failed with no code,
blocking the whole run on ``execution_complete``. A deterministic runner removes
both the flakiness and the dominant coder round-trip (the runner itself is
<1 % of wall-clock; the coder call is ~60 %).

The generated script:

* reads ``COHORT_PARQUET`` + ``STEP_OUT_DIR`` and the run's
  ``research_context.json``;
* discovers the concepts to audit case-neutrally — every base concept ``X`` that
  carries a paired ``X_measured`` indicator, plus any concepts named in
  ``user_preferences`` — excluding ids / demographics / the outcome;
* for each concept computes ``n_total``, ``measured_one_n`` (measured >= once),
  ``value_missing_n`` (never measured) and their percentages, and the
  structural-vs-measurement split (``value_present_but_measured_zero_n`` and a
  ``missingness_kind`` label). ``_measured`` is the authoritative availability
  signal except for a narrowly detected binary event-status encoding where the
  complete 0/1 value, positive flag, and event-count signal agree exactly;
* writes ``missingness_measurement_audit.csv`` (the schema the deterministic
  missingness figure renderer consumes), ``analytic_denominators.csv`` from the
  current step's declared inputs, ``cohort_flow.csv``, and a
  ``step_summary.json`` with ``analysis_family='data_quality'`` and
  ``adjusted_effect=None`` (a descriptive audit, never an effect estimate).

It intentionally emits NO figure: the family figure renderer builds the
manuscript figure from ``missingness_measurement_audit.csv`` in the figure step.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Sequence

from ...icu_rules import companion_count_column_for_measured
from ...schema import AnalysisStep

__all__ = [
    "is_compact_missingness_measurement_contract",
    "is_missingness_complete_case_contract",
    "is_missingness_measurement_availability_contract",
    "missingness_audit_cohort_input_key",
    "missingness_audit_executor_owns_step",
    "missingness_audit_input_scope_supported",
    "missingness_measurement_audit_code",
    "source_availability_audit_executor_owns_step",
]


_MISSINGNESS_AVAILABILITY_METHOD_TOKENS = frozenset(
    {
        "and",
        "audit",
        "availability",
        "frequency",
        "informative",
        "measurement",
        "missingness",
        "source",
    }
)
_MEASUREMENT_AVAILABILITY_PRODUCT_TOKENS = frozenset(
    {
        "audit",
        "availability",
        "measurement",
        "source",
    }
)
_MISSINGNESS_COMPLETE_CASE_METHOD_TOKENS = frozenset(
    {"and", "audit", "case", "complete", "missingness"}
)
_COMPACT_MISSINGNESS_MEASUREMENT_TOKENS = frozenset(
    {"audit", "measurement", "missingness"}
)


def _contract_tokens(value: object) -> frozenset[str]:
    """Return normalised structured-name tokens, not prose keywords."""

    return frozenset(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def is_missingness_measurement_availability_contract(
    method: object,
    expected_outputs: Sequence[object],
) -> bool:
    """Classify the closed, descriptive missingness/availability analysis kind.

    Planner method labels may use harmless compositional synonyms, but method
    prose alone never grants executor ownership.  The contract must declare
    exactly two typed tables: the missingness audit and one measurement/source
    availability audit.  Unknown method or product tokens fail closed, which
    prevents this count-only executor from swallowing a model, test, figure, or
    richer scientific reconciliation step.
    """

    method_tokens = _contract_tokens(method)
    method_is_closed_audit = bool(
        method_tokens
        and method_tokens <= _MISSINGNESS_AVAILABILITY_METHOD_TOKENS
        and {"missingness", "audit"} <= method_tokens
        and (
            "measurement" in method_tokens
            or {"source", "availability"} <= method_tokens
        )
    )
    if not method_is_closed_audit:
        return False

    outputs = [str(value or "").strip().casefold() for value in expected_outputs]
    if len(outputs) != 2 or len(set(outputs)) != 2:
        return False
    if any(not value.startswith("table:") for value in outputs):
        return False

    product_tokens = [_contract_tokens(value.split(":", 1)[1]) for value in outputs]
    missingness_products = [
        tokens for tokens in product_tokens if tokens == {"missingness", "audit"}
    ]
    availability_products = [
        tokens
        for tokens in product_tokens
        if (
            tokens <= _MEASUREMENT_AVAILABILITY_PRODUCT_TOKENS
            and "measurement" in tokens
            and bool(tokens & {"source", "availability"})
        )
    ]
    return len(missingness_products) == 1 and len(availability_products) == 1


def is_missingness_complete_case_contract(
    method: object,
    expected_outputs: Sequence[object],
) -> bool:
    """Classify one closed missingness-profile/complete-case count contract."""

    method_tokens = _contract_tokens(method)
    if method_tokens != _MISSINGNESS_COMPLETE_CASE_METHOD_TOKENS:
        return False
    outputs = {str(value or "").strip().casefold() for value in expected_outputs}
    return outputs == {
        "table:missingness_profile",
        "table:complete_case_attrition",
    }


def is_compact_missingness_measurement_contract(
    method: object,
    expected_outputs: Sequence[object],
) -> bool:
    """Classify one closed per-concept missingness/measurement audit."""

    method_tokens = _contract_tokens(method)
    outputs = [str(value or "").strip().casefold() for value in expected_outputs]
    return bool(
        method_tokens == _COMPACT_MISSINGNESS_MEASUREMENT_TOKENS
        and outputs == ["table:missingness_measurement_audit"]
    )


def _cohort_input_scope(step: AnalysisStep) -> tuple[bool, str | None]:
    """Resolve an optional single typed row-membership authority."""

    typed_inputs = {
        str(value or "").strip()
        for value in step.inputs
        if ":" in str(value or "").strip()
    }
    if not typed_inputs:
        return True, None
    if len(typed_inputs) != 1:
        return False, None
    input_key = next(iter(typed_inputs))
    kind, separator, product = input_key.partition(":")
    if separator and product and (
        kind == "cohort" or input_key == "artifact:analysis_cohort"
    ):
        return True, input_key
    return False, None


def missingness_audit_input_scope_supported(step: AnalysisStep) -> bool:
    """Return whether the runner can consume every declared typed input."""

    supported, _ = _cohort_input_scope(step)
    return supported


def missingness_audit_cohort_input_key(step: AnalysisStep) -> str | None:
    """Return the exact typed cohort key, after scope validation."""

    supported, input_key = _cohort_input_scope(step)
    return input_key if supported else None


def missingness_audit_executor_owns_step(step: AnalysisStep) -> bool:
    """Own a closed, auxiliary count-only missingness contract."""

    contract_is_supported = is_missingness_measurement_availability_contract(
        step.method,
        step.expected_outputs,
    ) or is_missingness_complete_case_contract(
        step.method,
        step.expected_outputs,
    ) or is_compact_missingness_measurement_contract(
        step.method,
        step.expected_outputs,
    )
    return bool(
        contract_is_supported
        # AnalysisStep bare columns are evaluated against the orchestrator's
        # already-locked COHORT_PARQUET by construction. One explicit cohort
        # product is loaded and digest-verified; every other typed source
        # rejects ownership.
        and missingness_audit_input_scope_supported(step)
    )


def source_availability_audit_executor_owns_step(step: AnalysisStep) -> bool:
    """Own one closed, non-scientific missingness/availability contract."""

    return bool(
        is_missingness_measurement_availability_contract(
            step.method,
            step.expected_outputs,
        )
        and missingness_audit_executor_owns_step(step)
    )


def _measurement_provenance_code(step: AnalysisStep | None) -> str:
    """Render provenance checks within the exact declared pair scope."""

    if step is None:
        loop_header = textwrap.dedent(
            """
            for measured_column in requested_inputs:
                count_column = companion_count_column_for_measured(measured_column)
                if count_column is None:
                    continue
            """
        ).rstrip()
    else:
        declared_inputs = {
            str(value).strip()
            for value in step.inputs
            if str(value).strip() and ":" not in str(value)
        }
        declared_pairs = sorted(
            (measured_column, count_column)
            for measured_column in declared_inputs
            if (count_column := companion_count_column_for_measured(measured_column))
            is not None
            and count_column in declared_inputs
        )
        if not declared_pairs:
            return "measurement_checks = []"
        loop_header = f"for measured_column, count_column in {declared_pairs!r}:"
    loop_body = textwrap.dedent(
        """
        if measured_column not in df.columns:
            # The declared-input denominator check below will block this
            # step. Do not fabricate a receipt for a missing flag.
            continue
        resolved_count_column = (
            count_column
            if count_column in df.columns
            else low.get(count_column.lower())
        )
        if resolved_count_column is None:
            measurement_checks.append(
                {
                    "measured_column": measured_column,
                    "count_column": count_column,
                    "status": "unavailable",
                    "comparison_n": None,
                    "invalid_pair_n": None,
                    "discordant_n": None,
                    "role": "audit_only",
                    "reason": "Declared structural count companion is absent.",
                }
            )
            continue
        measurement_checks.append(
            measurement_provenance_receipt(
                df,
                measured_column=measured_column,
                count_column=resolved_count_column,
            )
        )
        """
    ).strip()
    return "measurement_checks = []\n" + loop_header + "\n" + textwrap.indent(
        loop_body,
        "    ",
    )


def missingness_measurement_audit_code(
    step: AnalysisStep | None = None,
) -> str:
    """Return a runner script that computes the per-concept missingness audit."""

    if step is not None and not missingness_audit_input_scope_supported(step):
        raise ValueError("missingness runner cannot consume the declared typed inputs")
    typed_cohort_input = (
        missingness_audit_cohort_input_key(step) if step is not None else None
    )
    template = textwrap.dedent(r"""
        import hashlib
        import json
        import os
        from pathlib import Path

        import numpy as np
        import pandas as pd

        from easyicu.research_agent.icu_rules import (
            companion_count_column_for_measured,
        )
        from easyicu.research_agent.methods.descriptive_inputs import (
            measurement_provenance_receipt,
        )

        out_dir = Path(os.environ["STEP_OUT_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(os.environ.get("EASYICU_RUN_DIR") or out_dir.parents[2])
        current_step_id = os.environ.get("EASYICU_STEP_ID") or out_dir.parent.name

        def sha256_file(path):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        def load_typed_cohort(input_key):
            resolved_run_dir = run_dir.resolve()
            manifest_path = Path(
                os.environ["EASYICU_RESOLVED_INPUTS_JSON"]
            ).resolve()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            inputs = manifest.get("inputs")
            if not isinstance(inputs, dict) or input_key not in inputs:
                raise RuntimeError(
                    "Missing exact typed cohort binding: %s" % input_key
                )
            binding = inputs[input_key]
            relative_path = binding.get("relative_path")
            expected_sha256 = binding.get("sha256")
            contract = binding.get("product_contract")
            if (
                not isinstance(relative_path, str)
                or not relative_path
                or not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or not isinstance(contract, dict)
            ):
                raise RuntimeError("Typed cohort binding is incomplete")
            cohort_path = (resolved_run_dir / relative_path).resolve()
            try:
                cohort_path.relative_to(resolved_run_dir)
            except ValueError as exc:
                raise RuntimeError(
                    "Typed cohort binding escapes EASYICU_RUN_DIR"
                ) from exc
            if not cohort_path.is_file():
                raise RuntimeError("Typed cohort binding does not name a file")
            if sha256_file(cohort_path) != expected_sha256:
                raise RuntimeError("Typed cohort digest verification failed")
            columns = contract.get("columns")
            row_count = contract.get("row_count")
            if (
                not isinstance(columns, list)
                or not columns
                or not all(isinstance(value, str) and value for value in columns)
                or len(set(columns)) != len(columns)
                or not isinstance(row_count, int)
                or isinstance(row_count, bool)
                or row_count < 0
            ):
                raise RuntimeError(
                    "Typed cohort product_contract is incomplete"
                )
            suffix = cohort_path.suffix.lower()
            if suffix in {".parquet", ".pq"}:
                frame = pd.read_parquet(cohort_path)
            elif suffix == ".csv":
                frame = pd.read_csv(cohort_path)
            elif suffix == ".tsv":
                frame = pd.read_csv(cohort_path, sep="\t")
            else:
                raise RuntimeError("Typed cohort table format is unsupported")
            if list(frame.columns) != columns:
                raise RuntimeError(
                    "Typed cohort columns do not match product_contract"
                )
            if len(frame) != row_count:
                raise RuntimeError(
                    "Typed cohort row count does not match product_contract"
                )
            return frame, cohort_path

        typed_cohort_input = __EASYICU_TYPED_COHORT_INPUT__
        if typed_cohort_input is None:
            cohort_path = Path(os.environ["COHORT_PARQUET"])
            df = pd.read_parquet(cohort_path).copy()
        else:
            df, cohort_path = load_typed_cohort(typed_cohort_input)
            df = df.copy()
        n_total = int(len(df))

        # --- research context: optional explicit concept list ------------------
        # Both files below are host-owned. An ABSENT file is a legitimate
        # legacy state (broad discovery mode); a present-but-unparseable file
        # must fail this step loudly — swallowing it would silently change the
        # audit scope from the declared inputs to every paired concept.
        req_concepts = []
        requested_inputs = []
        requested_outputs = []
        context_path = run_dir / "research_context.json"
        if context_path.is_file():
            ctx = json.loads(context_path.read_text("utf-8"))
            prefs = ctx.get("user_preferences") or {}
            if isinstance(prefs, dict):
                for key in ("audit_concepts", "feature_concepts", "concepts", "features"):
                    vals = prefs.get(key)
                    if isinstance(vals, (list, tuple)):
                        req_concepts.extend(str(v).strip() for v in vals if str(v).strip())

        # The plan owns the complete-case contract.  Reading the current
        # step's declared inputs lets the deterministic audit emit the requested
        # analytic denominator instead of returning only a compact concept
        # count and falsely satisfying a richer step.
        plan_path = run_dir / "analysis_plan.json"
        manifest_path = run_dir / "manifest_partial.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text("utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("manifest_partial.json must contain an object")
            declared_plan = manifest.get("plan_path")
            if declared_plan is not None:
                declared_plan_path = Path(str(declared_plan).strip())
                if (
                    declared_plan_path.is_absolute()
                    or declared_plan_path.suffix != ".json"
                    or not declared_plan_path.parts
                    or any(part in {"", ".", ".."} for part in declared_plan_path.parts)
                ):
                    raise ValueError(
                        "manifest_partial.json carries an unsafe plan_path"
                    )
                resolved_run_dir = run_dir.resolve()
                resolved_plan_path = (run_dir / declared_plan_path).resolve()
                if not resolved_plan_path.is_relative_to(resolved_run_dir):
                    raise ValueError(
                        "manifest_partial.json plan_path escapes the run directory"
                    )
                plan_path = resolved_plan_path
        if plan_path.is_file():
            plan = json.loads(plan_path.read_text("utf-8"))
            for planned_step in plan.get("steps") or []:
                if str(planned_step.get("step_id") or "") == current_step_id:
                    requested_inputs = [
                        str(value).strip()
                        for value in (planned_step.get("inputs") or [])
                        if str(value).strip() and ":" not in str(value)
                    ]
                    requested_outputs = [
                        str(value).strip()
                        for value in (planned_step.get("expected_outputs") or [])
                        if str(value).strip()
                    ]
                    break

        cols = list(df.columns)
        low = {c.lower(): c for c in cols}

        # Replay every Planner-declared measurement flag against its structural
        # count companion on the exact locked cohort.  The host helper raises
        # before any result is sealed when a present pair is invalid or
        # discordant.  A genuinely unavailable count is recorded explicitly;
        # the summary never invents a count column or silently omits a planned
        # measurement flag.
        __EASYICU_MEASUREMENT_PROVENANCE_SCOPE__

        # IDs are never audit variables. Demographics and outcomes are excluded
        # only from broad discovery; if the current step explicitly declares
        # them, their direct value availability belongs in that step's audit.
        _IDENTIFIER_COLUMNS = {
            "stay_id", "hadm_id", "subject_id", "icustay_id", "patient_id", "id",
        }
        _NON_CONCEPT = {
            *_IDENTIFIER_COLUMNS,
            "age", "sex", "gender", "adm", "admission_type", "ethnicity", "race",
            "death", "died", "mortality", "hospital_mortality", "hospital_expire_flag",
            "los_icu", "los_hosp", "icu_los", "hospital_los", "length_of_stay",
            "followup_time_hours", "event_observed",
        }
        _SUFFIX_SKIP = ("_measured", "_first_time", "_last_time", "_n", "_time")

        def _is_flag(colname):
            return colname.lower().endswith("_measured")

        def _representative_value_column(base):
            '''Resolve one value aggregate paired with ``<base>_measured``.

            Wide ICU exports commonly pair ``crea_measured`` with
            ``crea_first`` or ``aki_stage_measured`` with ``aki_stage_max``;
            requiring an exact bare ``base`` silently loses the value/flag
            discordance audit.  The closed aggregate suffix list is structural
            and case-neutral.
            '''
            suffixes = ("", "_first", "_max", "_last", "_mean", "_min")
            base_lower = base.lower()
            for requested in requested_inputs:
                resolved = (
                    requested
                    if requested in df.columns
                    else low.get(requested.lower())
                )
                if resolved is None:
                    continue
                if resolved.lower() in {
                    base_lower + suffix for suffix in suffixes
                }:
                    return resolved

            candidates = [
                base,
                base + "_first",
                base + "_max",
                base + "_last",
                base + "_mean",
                base + "_min",
            ]
            for candidate in candidates:
                if candidate in df.columns:
                    return candidate
                matched = low.get(candidate.lower())
                if matched is not None:
                    return matched
            return None

        # --- discover the concepts to audit (case-neutral) ---------------------
        # Primary source of truth: every base concept X that carries an
        # ``X_measured`` indicator. Add any explicitly requested concept present
        # in the cohort. Never audit ids / demographics / the outcome.
        concepts = []
        seen = set()

        def _add(base, *, declared=False):
            b = str(base)
            if (
                not b
                or b.lower() in _IDENTIFIER_COLUMNS
                or (not declared and b.lower() in _NON_CONCEPT)
                or b in seen
            ):
                return
            if (
                b not in df.columns
                and (b + "_measured") not in df.columns
                and _representative_value_column(b) is None
            ):
                return
            seen.add(b)
            concepts.append(b)

        _FAMILY_SUFFIXES = (
            "_measured", "_first_time", "_last_time", "_first", "_last",
            "_max", "_mean", "_min", "_n",
        )

        def _family_base(column):
            text = str(column)
            lowered = text.lower()
            for suffix in _FAMILY_SUFFIXES:
                if lowered.endswith(suffix) and len(text) > len(suffix):
                    return text[: -len(suffix)]
            return text

        if requested_inputs:
            # The current plan is the audit scope. Collapse aggregate-family
            # members (X_max/X_measured/...) to one concept row, but retain
            # explicitly declared direct variables such as age or outcome.
            for requested in requested_inputs:
                resolved = (
                    requested
                    if requested in df.columns
                    else low.get(requested.lower(), requested)
                )
                _add(_family_base(resolved), declared=True)
        else:
            # Backward-compatible discovery for legacy plans with no input
            # contract: audit every paired measurement concept.
            for c in cols:
                if _is_flag(c):
                    _add(c[: -len("_measured")])
            for name in req_concepts:
                _add(low.get(name.lower(), name), declared=True)

        # Fallback: if no _measured flags exist at all, audit every non-id,
        # non-aggregate value column so the step still produces a real audit.
        if not concepts:
            for c in cols:
                cl = c.lower()
                if cl in _NON_CONCEPT or any(cl.endswith(s) for s in _SUFFIX_SKIP):
                    continue
                _add(c)

        def _fail(reason):
            summary = {
                "step": current_step_id,
                "status": "blocked",
                "analysis_family": "data_quality",
                "blocking_reason": reason,
                "adjusted_effect": None,
                "primary_estimand": "Blocked: " + reason,
                "n_total": n_total,
                "outputs": [],
            }
            (out_dir / "step_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False)
            )
            print(json.dumps(summary))

        if n_total == 0:
            _fail("Analysis cohort is empty; nothing to audit.")
            raise SystemExit(0)
        if not concepts:
            _fail(
                "No auditable concept columns found (no '<concept>_measured' "
                "indicators and no non-id value columns)."
            )
            raise SystemExit(0)

        # --- per-concept measurement audit -------------------------------------
        rows = []
        for base in concepts:
            flag_col = base + "_measured"
            value_col = _representative_value_column(base)
            has_flag = flag_col in df.columns
            raw_measured_flag = pd.Series(np.nan, index=df.index, dtype=float)
            measured_flag = pd.Series(np.nan, index=df.index, dtype=float)

            if has_flag:
                raw_measured_flag = pd.to_numeric(df[flag_col], errors="coerce")
                measured_flag = raw_measured_flag.fillna(0)
                measured_mask = measured_flag >= 1
            elif value_col is not None:
                # no explicit indicator -> a non-null value counts as measured.
                measured_mask = df[value_col].notna()
            else:
                continue

            # Structural no-source: the concept is not sourced for ANY stay in
            # this cohort/database (indicator all-zero, or value column entirely
            # absent/NaN). Distinct from measurement missingness (sourced, but
            # not measured for a given stay).
            if value_col is not None:
                value_present = df[value_col].notna()
            else:
                value_present = pd.Series(False, index=df.index)
            value_present_n = int(value_present.sum())

            # Some wide exports use ``X_measured`` for an event-presence flag,
            # not measurement availability: a fully observed binary X is 0/1
            # and the paired flag is exactly 1 where X == 1. Treating negative
            # states as unmeasured would turn a complete binary status into
            # near-total missingness (for example, a rare procedure). Keep this
            # inference narrow and case-neutral: both binary states must occur,
            # every value and flag must be present, and an independently emitted
            # ``X_n > 0`` event-count signal must exactly agree with the value and
            # flag. Partial continuous labs/vitals and an unknown bad binary flag
            # cannot enter this branch.
            indicator_semantics = "measurement_availability"
            raw_indicator_one_n = int(measured_mask.sum())
            count_candidate = base + "_n"
            count_col = (
                count_candidate
                if count_candidate in df.columns
                else low.get(count_candidate.lower())
            )
            if (
                has_flag
                and value_col is not None
                and count_col is not None
                and value_present_n == n_total
            ):
                numeric_value = pd.to_numeric(df[value_col], errors="coerce")
                event_count = pd.to_numeric(df[count_col], errors="coerce")
                value_levels = set(numeric_value.dropna().unique().tolist())
                flag_levels = set(measured_flag.dropna().unique().tolist())
                is_complete_binary_status = (
                    numeric_value.notna().all()
                    and raw_measured_flag.notna().all()
                    and event_count.notna().all()
                    and bool(event_count.ge(0).all())
                    and value_levels == {0, 1}
                    and flag_levels.issubset({0, 1})
                    and bool((measured_mask == numeric_value.eq(1)).all())
                    and bool((measured_mask == event_count.gt(0)).all())
                )
                if is_complete_binary_status:
                    indicator_semantics = "binary_event_presence"
                    measured_mask = value_present

            measured_one_n = int(measured_mask.sum())
            value_missing_n = int(n_total - measured_one_n)
            structural_no_source = bool(
                measured_one_n == 0 and value_present_n == 0
            )
            # Rows with a value present but the measurement indicator says zero
            # (a genuine present-but-unmeasured, e.g. a derived source flag).
            if has_flag:
                present_but_zero = int((value_present & ~measured_mask).sum())
                measured_but_missing = int((measured_mask & ~value_present).sum())
            else:
                present_but_zero = 0
                measured_but_missing = 0

            if structural_no_source:
                kind = "structural_no_source"
            elif indicator_semantics == "binary_event_presence":
                kind = "binary_event_status_complete"
            elif present_but_zero or measured_but_missing:
                kind = "measurement_flag_conflict"
            else:
                kind = "measurement_missing"
            rows.append(
                {
                    "concept": base,
                    "variable": base,
                    "label": base.replace("_", " "),
                    "value_column": value_col or "",
                    "n_total": n_total,
                    "measured_one_n": measured_one_n,
                    "measured_one_pct": round(100.0 * measured_one_n / n_total, 6),
                    "value_missing_n": value_missing_n,
                    "value_missing_pct": round(100.0 * value_missing_n / n_total, 6),
                    # aliases so every downstream resolver (figure renderer /
                    # validator) finds a column it recognises.
                    "measured_n": measured_one_n,
                    "n_nonmissing": measured_one_n,
                    "missing_n": value_missing_n,
                    "missing_pct": round(100.0 * value_missing_n / n_total, 6),
                    "measured_pct": round(100.0 * measured_one_n / n_total, 6),
                    "value_present_but_measured_zero_n": present_but_zero,
                    "measured_but_value_missing_n": measured_but_missing,
                    "raw_value_missing_n": int(n_total - value_present_n),
                    "raw_indicator_one_n": raw_indicator_one_n,
                    "indicator_semantics": indicator_semantics,
                    "event_count_column": (
                        count_col
                        if indicator_semantics == "binary_event_presence"
                        else ""
                    ),
                    "missingness_kind": kind,
                    "has_measured_indicator": bool(has_flag),
                }
            )

        audit = pd.DataFrame(rows)
        audit = audit.sort_values("value_missing_pct", ascending=False).reset_index(drop=True)
        audit.to_csv(out_dir / "missingness_measurement_audit.csv", index=False)

        missingness_audit = audit[
            ["concept", "variable", "value_column", "n_total", "raw_value_missing_n"]
        ].copy()
        missingness_audit["n_nonmissing"] = (
            missingness_audit["n_total"] - missingness_audit["raw_value_missing_n"]
        )
        missingness_audit["missing_n"] = missingness_audit["raw_value_missing_n"]
        missingness_audit["missing_pct"] = (
            100.0
            * missingness_audit["missing_n"]
            / missingness_audit["n_total"]
        )
        missingness_audit.drop(columns=["raw_value_missing_n"]).to_csv(
            out_dir / "missingness_audit.csv", index=False
        )

        source_audit = audit[
            [
                "concept",
                "variable",
                "value_column",
                "n_total",
                "measured_one_n",
                "value_missing_n",
                "value_present_but_measured_zero_n",
                "measured_but_value_missing_n",
                "indicator_semantics",
                "missingness_kind",
                "has_measured_indicator",
            ]
        ].copy()
        source_audit.to_csv(out_dir / "measurement_source_audit.csv", index=False)
        source_audit.to_csv(out_dir / "measurement_availability.csv", index=False)
        source_audit.to_csv(
            out_dir / "measurement_availability_audit.csv", index=False
        )

        # --- declared analytic denominators ----------------------------------
        resolved_inputs = []
        missing_declared_inputs = []
        for requested in requested_inputs:
            resolved = requested if requested in df.columns else low.get(requested.lower())
            if resolved is None:
                # A bare concept name (e.g. ``crea`` present only as
                # ``crea_first``/``crea_measured``) is audited above via
                # ``_representative_value_column``; resolve the analytic
                # denominator the SAME way so a legitimately-audited concept is
                # not spuriously flagged as a missing declared input (which would
                # block an otherwise-complete missingness audit).
                resolved = _representative_value_column(
                    _family_base(low.get(requested.lower(), requested))
                )
            if resolved is None:
                missing_declared_inputs.append(requested)
                continue
            if resolved not in resolved_inputs:
                resolved_inputs.append(resolved)

        denominator_rows = []
        for column in resolved_inputs:
            observed_n = int(df[column].notna().sum())
            denominator_rows.append(
                {
                    "analysis_set": "observed:" + column,
                    "required_variables": column,
                    "n_total": n_total,
                    "n_complete": observed_n,
                    "n_excluded_missing": int(n_total - observed_n),
                    "complete_pct": round(100.0 * observed_n / n_total, 6),
                }
            )
        expects_analytic_denominator = any(
            "analytic_denominator" in value.lower()
            for value in requested_outputs
        )
        denominator_error = None
        if missing_declared_inputs:
            denominator_error = (
                "Declared analytic inputs are absent from the cohort: "
                + ", ".join(missing_declared_inputs)
            )
        elif expects_analytic_denominator and not requested_inputs:
            denominator_error = (
                "The analytic-denominator contract declares no input variables."
            )

        if resolved_inputs and denominator_error is None:
            complete_mask = df[resolved_inputs].notna().all(axis=1)
            complete_n = int(complete_mask.sum())
        else:
            complete_n = None
        denominator_rows.insert(
            0,
            {
                "analysis_set": "all_requested_inputs",
                "required_variables": "|".join(requested_inputs),
                "n_total": n_total,
                "n_complete": complete_n,
                "n_excluded_missing": (
                    int(n_total - complete_n) if complete_n is not None else None
                ),
                "complete_pct": (
                    round(100.0 * complete_n / n_total, 6)
                    if complete_n is not None
                    else None
                ),
            },
        )
        pd.DataFrame(denominator_rows).to_csv(
            out_dir / "analytic_denominators.csv", index=False
        )

        pd.DataFrame(
            [
                {"stage": "universe_or_cohort", "n": n_total},
                {"stage": "concepts_audited", "n": int(len(audit))},
                {
                    "stage": "structural_no_source_concepts",
                    "n": int((audit["missingness_kind"] == "structural_no_source").sum()),
                },
            ]
        ).to_csv(out_dir / "cohort_flow.csv", index=False)

        worst = audit.head(5)[["concept", "value_missing_pct"]].to_dict("records")
        n_structural = int((audit["missingness_kind"] == "structural_no_source").sum())
        n_binary_event_status = int(
            (audit["indicator_semantics"] == "binary_event_presence").sum()
        )
        product_files = {
            "missingness_audit": "missingness_audit.csv",
            "missingness_profile": "missingness_audit.csv",
            "missingness_measurement_audit": "missingness_measurement_audit.csv",
            "measurement_audit": "missingness_measurement_audit.csv",
            "measurement_process_audit": "missingness_measurement_audit.csv",
            "measurement_source_audit": "measurement_source_audit.csv",
            "measurement_availability": "measurement_availability.csv",
            "measurement_availability_audit": "measurement_availability_audit.csv",
            "data_quality_audit": "missingness_measurement_audit.csv",
            "source_coverage": "measurement_source_audit.csv",
            "analytic_denominator": "analytic_denominators.csv",
            "analytic_denominators": "analytic_denominators.csv",
            "complete_case_attrition": "analytic_denominators.csv",
            "cohort_flow": "cohort_flow.csv",
        }
        declared_output_files = {}
        for output in requested_outputs:
            product = output.split(":", 1)[-1].strip()
            if product in product_files:
                declared_output_files[output] = product_files[product]
        summary = {
            "step": current_step_id,
            "status": "blocked" if denominator_error else "ok",
            "analysis_family": "data_quality",
            "interpretation_class": "missingness_measurement_audit",
            "primary_estimand": (
                "Deterministic per-concept measurement-missingness audit "
                "(measured vs missing fraction; structural-no-source vs "
                "measurement-missing distinguished via the '_measured' indicator)."
            ),
            "adjusted_effect": None,
            "cohort_input_key": typed_cohort_input or "COHORT_PARQUET",
            "n_total": n_total,
            "n_concepts_audited": int(len(audit)),
            "n_structural_no_source": n_structural,
            "n_binary_event_status": n_binary_event_status,
            "all_requested_inputs_complete_n": complete_n,
            "requested_input_count": len(requested_inputs),
            "resolved_input_count": len(resolved_inputs),
            "missing_declared_inputs": missing_declared_inputs,
            "blocking_reason": denominator_error,
            "worst_measured_concepts": worst,
            "measurement_provenance_audit": {
                "source": "COHORT_PARQUET",
                "checks": measurement_checks,
            },
            "notes": [
                "Deterministic missingness audit (no LLM coder).",
                "measured_one_n uses the '<concept>_measured' availability indicator "
                "when present, except for an exact complete-binary event-status "
                "encoding; otherwise a non-null value counts as measured.",
                "Labs/vitals are NEVER imputed to 0; missing means unmeasured.",
                "structural_no_source = concept sourced for no stay in this cohort; "
                "measurement_missing = sourced but unmeasured for a given stay.",
            ],
            "output_files": declared_output_files or {
                "missingness_measurement_audit": "missingness_measurement_audit.csv",
                "analytic_denominators": "analytic_denominators.csv",
                "cohort_flow": "cohort_flow.csv",
            },
        }
        (out_dir / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str)
        )
        print(
            json.dumps(
                {
                    "n_total": n_total,
                    "n_concepts_audited": int(len(audit)),
                    "n_structural_no_source": n_structural,
                }
            )
        )
        """).strip()
    template = template.replace(
        "__EASYICU_TYPED_COHORT_INPUT__",
        repr(typed_cohort_input),
    )
    return template.replace(
        "__EASYICU_MEASUREMENT_PROVENANCE_SCOPE__",
        _measurement_provenance_code(step),
    )
