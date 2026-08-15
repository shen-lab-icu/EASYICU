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
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...icu_rules import companion_count_column_for_measured
from ...schema import AnalysisStep, spec_backs_every_declared_product
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_input_binding import sole_typed_cohort_input

__all__ = [
    "MEASUREMENT_AUDIT_KIND_FILES",
    "MISSINGNESS_AUDIT_PRODUCT_FILES",
    "declared_audit_products_are_emittable",
    "declared_audit_spec_is_emittable",
    "is_compact_missingness_measurement_contract",
    "is_missingness_complete_case_contract",
    "is_missingness_measurement_availability_contract",
    "missingness_audit_cohort_input_key",
    "missingness_audit_executor_owns_step",
    "missingness_audit_input_scope_supported",
    "is_measurement_bias_audit_contract",
    "missingness_measurement_audit_code",
    "measurement_audit_product_filename",
    "source_availability_audit_executor_owns_step",
]


#: This executor's own label for the whole-cohort stratum of the exposure
#: completeness table. It reads as a total to a person, and to nothing else.
_ALL_STRATA_LABEL = "__all__"

#: The host's declared row-role vocabulary, so the total row is legible to the
#: aggregate-row validator rather than only to this producer.
#:
#: DUPLICATED, and stated here rather than hidden: the same two strings are
#: defined in ``audits/aggregate_row.py`` (``OVERALL_ROW_ROLE`` /
#: ``LEVEL_ROW_ROLE``) and in ``exposure_outcome_distribution_executor.py``
#: (``_OVERALL_ROLE`` / ``_LEVEL_ROLE``). They are re-declared instead of
#: imported because this module's body is rendered inline into the container
#: script, which imports only ``icu_rules`` and ``methods.*``; pulling the
#: validator package in would drag the schema layer across that boundary for
#: two string constants. ``test_the_total_row_is_legible_to_its_validator``
#: asserts all three spellings agree, so the copy cannot drift silently.
_OVERALL_ROW_ROLE = "overall"
_LEVEL_ROW_ROLE = "exposure_level"


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
# The enriched three-product shape a replanner produces when it asks for the
# observation process and component completeness as well as plain missingness.
# Enriching a step's science must not cost it its deterministic owner.
_MEASUREMENT_BIAS_METHOD_TOKENS = frozenset(
    {
        "and",
        "audit",
        "bias",
        "completeness",
        "component",
        "event",
        "exposure",
        "measurement",
        "missingness",
        "process",
        "timing",
    }
)
# The exact product ids this runner can emit.  Recognition must not be looser
# than production: an earlier revision matched each product by its *token set*,
# which accepts any permutation of the same words -- ``audit_process_measurement``
# passed the contract while the generator's ``product_files`` map, which is keyed
# on the exact id, could not produce it.  The step was then claimed and failed
# afterwards for a missing declared product, which is strictly worse than never
# claiming it.  ``test_measurement_bias_audit_contract`` locks these ids against
# the generated template so the two cannot drift apart.
_MEASUREMENT_BIAS_PRODUCT_IDS = frozenset(
    {
        "missingness_measurement_audit",
        "measurement_process_audit",
        "exposure_component_completeness_audit",
    }
)

# The one declaration of what this runner can emit: audit kind -> output file.
# It is rendered into the generated script rather than duplicated there, because
# the previous arrangement -- a literal inside the template plus several
# hand-enumerated contracts out here -- is what let the two drift apart.
#
# Ownership is a *capability* question ("can I emit exactly these audits?"), and
# this map answers it.  Scientific sufficiency ("must this study declare the
# component-completeness audit at all?") is a different question that belongs to
# the study protocol and the evaluator, not to an executor: being able to
# compute two tables must never be read as a licence to accept two tables where
# the science requires three.
#
# Keyed on the audit, not on a product name.  ``measurement_source`` is written
# to three filenames, so a rule counting distinct *files* would let one step
# declare all three and be satisfied by the same table three times; keyed this
# way that declaration is one audit claimed three times, which fails closed.
MEASUREMENT_AUDIT_KIND_FILES: Mapping[str, str] = MappingProxyType(
    {
        "measurement_missingness": "missingness_measurement_audit.csv",
        "missingness_profile": "missingness_audit.csv",
        "measurement_source": "measurement_source_audit.csv",
        # Its own table, not an alias of the missingness one: "was it ever
        # measured" and "how often, and when" are different questions, and
        # two declared products resolving to one file satisfies a contract
        # without satisfying a reader.
        "measurement_process": "measurement_process_audit.csv",
        "event_timing": "event_timing_audit.csv",
        "component_completeness": "exposure_component_completeness_audit.csv",
        "analytic_denominators": "analytic_denominators.csv",
        "cohort_flow": "cohort_flow.csv",
    }
)

# Product ids seen in recorded plans, mapped to the audit each one names.  This
# is a compatibility shim for plans written before ``measurement_audit_spec``
# existed, NOT a second capability declaration -- every value is looked up in
# MEASUREMENT_AUDIT_KIND_FILES above, so the two cannot drift.  It is also why
# the shim can only ever shrink: a plan that declares the spec does not consult
# it, and a plan that does not is limited to names someone already wrote down.
_LEGACY_PRODUCT_AUDITS: Mapping[str, str] = MappingProxyType(
    {
        "missingness_audit": "missingness_profile",
        "missingness_profile": "missingness_profile",
        "missingness_measurement_audit": "measurement_missingness",
        "measurement_audit": "measurement_missingness",
        "measurement_process_audit": "measurement_process",
        "exposure_component_completeness_audit": "component_completeness",
        "measurement_source_audit": "measurement_source",
        "measurement_availability": "measurement_source",
        "measurement_availability_audit": "measurement_source",
        "data_quality_audit": "measurement_missingness",
        "source_coverage": "measurement_source",
        "analytic_denominator": "analytic_denominators",
        "analytic_denominators": "analytic_denominators",
        "complete_case_attrition": "analytic_denominators",
        "cohort_flow": "cohort_flow",
    }
)

MISSINGNESS_AUDIT_PRODUCT_FILES: Mapping[str, str] = MappingProxyType(
    {
        product: MEASUREMENT_AUDIT_KIND_FILES[audit]
        for product, audit in _LEGACY_PRODUCT_AUDITS.items()
    }
)


def measurement_audit_product_filename(product: str) -> str | None:
    """Return the deterministic producer filename for one typed audit product.

    ``measurement_audit_spec`` products use audit-kind names, while older
    plans use the compatibility product ids above.  Figure renderers need the
    physical producer filename in their ``source_table`` lineage column; the
    typed product label alone is not necessarily that filename.
    """

    normalized = str(product or "").strip()
    if normalized in MEASUREMENT_AUDIT_KIND_FILES:
        return MEASUREMENT_AUDIT_KIND_FILES[normalized]
    return MISSINGNESS_AUDIT_PRODUCT_FILES.get(normalized)

# Every method token that can appear in a declared count-only audit label.  The
# product side is the load-bearing guard -- a model or test step cannot declare
# audit products -- so this vocabulary only has to keep prose out.
_AUDIT_METHOD_VOCABULARY = frozenset(
    _MISSINGNESS_AVAILABILITY_METHOD_TOKENS
    | _MISSINGNESS_COMPLETE_CASE_METHOD_TOKENS
    | _COMPACT_MISSINGNESS_MEASUREMENT_TOKENS
    | _MEASUREMENT_BIAS_METHOD_TOKENS
    | {"data", "quality", "availability", "profile", "denominator", "denominators"}
)


def _render_product_files(step: AnalysisStep | None = None) -> str:
    """Render this step's product -> file map as source for the script.

    With a typed audit spec the map is the step's own declaration resolved
    through the capability map, so the script collects exactly the products the
    Planner asked for under exactly the names it used.  Without one it is the
    legacy shim, unchanged.
    """

    spec = None if step is None else step.measurement_audit_spec
    if spec is None:
        resolved = dict(MISSINGNESS_AUDIT_PRODUCT_FILES)
    else:
        resolved = {
            item.product_id: MEASUREMENT_AUDIT_KIND_FILES[item.audit]
            for item in spec.products
        }
    lines = "".join(
        f"    {product!r}: {filename!r},\n" for product, filename in resolved.items()
    )
    return "{\n" + lines + "}"


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


def is_measurement_bias_audit_contract(
    method: object,
    expected_outputs: Sequence[object],
) -> bool:
    """Classify the enriched three-product measurement-bias audit contract.

    A replanner that adds the observation-process and component-completeness
    products is asking for more science, not different science; the counts are
    still pure per-concept accounting.  Ownership is exact in both directions:
    the three declared products must be precisely the three product ids this
    runner emits, and the method must be drawn only from the closed audit
    vocabulary.  A method label may vary compositionally, but a product id may
    not: it is the key the generator looks the output file up by, so accepting a
    spelling the generator cannot produce would claim the step and then fail it.
    Anything else fails closed rather than letting this count-only runner
    swallow a model or a test.
    """

    method_tokens = _contract_tokens(method)
    if not (
        method_tokens
        and method_tokens <= _MEASUREMENT_BIAS_METHOD_TOKENS
        and "audit" in method_tokens
        and ("bias" in method_tokens or "process" in method_tokens)
    ):
        return False
    outputs = [str(value or "").strip().casefold() for value in expected_outputs]
    if len(outputs) != len(_MEASUREMENT_BIAS_PRODUCT_IDS):
        return False
    if len(set(outputs)) != len(outputs):
        return False
    if any(not value.startswith("table:") for value in outputs):
        return False
    return {
        value.split(":", 1)[1] for value in outputs
    } == _MEASUREMENT_BIAS_PRODUCT_IDS


def declared_audit_products_are_emittable(
    method: object,
    expected_outputs: Sequence[object],
) -> bool:
    """Classify a legacy audit product set this runner can actually emit.

    This is the path for a plan written before ``measurement_audit_spec``: the
    step says nothing about what its products ARE, so the only evidence
    available is the names themselves, and they are matched against the legacy
    shim.  It is why the shim exists and also why it is not enough -- see
    :func:`declared_audit_spec_is_emittable`, which is what a current plan uses.

    * the declared set is non-empty and free of duplicates;
    * every declared product is a ``table:`` product with a known audit;
    * the declared products resolve to as many *distinct audits* as products.

    That last rule is the one doing real work.  Several product ids name the
    same audit, so without it a step could declare two products, be claimed, and
    be satisfied by a single table -- a contract met without a reader being
    given the second thing they were promised.  Ambiguity fails closed.
    """

    method_tokens = _contract_tokens(method)
    if not (
        method_tokens
        and method_tokens <= _AUDIT_METHOD_VOCABULARY
        and bool(method_tokens & {"audit", "missingness", "availability"})
    ):
        return False

    outputs = [str(value or "").strip().casefold() for value in expected_outputs]
    if not outputs or len(set(outputs)) != len(outputs):
        return False
    if any(not value.startswith("table:") for value in outputs):
        return False

    products = [value.split(":", 1)[1] for value in outputs]
    if any(product not in _LEGACY_PRODUCT_AUDITS for product in products):
        return False
    audits = {_LEGACY_PRODUCT_AUDITS[product] for product in products}
    return len(audits) == len(products)


def declared_audit_spec_is_emittable(step: AnalysisStep) -> bool:
    """Whether the step's typed audit declaration is one this runner can emit.

    Recognition here reads what the Planner said each product IS, so the product
    name decides nothing.  That is the whole point: measured over the recorded
    corpus, 162 audit steps -- every declared output a table, no duplicates, an
    input scope this runner supports -- were demoted to the LLM coder solely
    because their names were spellings the legacy shim had not seen.  Ten more
    declared perfectly recognised products and were demoted on the *method*
    string instead.  Both halves were string matching, and a step carrying this
    spec is subject to neither: the spec is the declaration that it is an audit.

    ``AnalysisStep`` has enforced what is malformed on its own terms (one
    product per audit, and no entry naming a product the step never declares).
    Coverage is asked here rather than there, because "this declaration does
    not account for every declared product" has a safe answer -- nobody claims
    the step -- while a schema validator's only answer is to make the plan
    unreadable.
    """

    spec = step.measurement_audit_spec
    if spec is None:
        return False
    if not all(item.audit in MEASUREMENT_AUDIT_KIND_FILES for item in spec.products):
        return False
    return spec_backs_every_declared_product(
        step.expected_outputs,
        spec=spec,
        lookup="audit_for",
        allowed_kinds=frozenset({"table"}),
    )


def _cohort_input_scope(step: AnalysisStep) -> tuple[bool, str | None]:
    """Resolve an optional single typed row-membership authority.

    This audit may run without one, so it keeps its own arity policy (absent is
    supported); *which* keys name the closed cohort product is the published
    vocabulary and is read from its one owner.
    """

    input_key = sole_typed_cohort_input(step)
    if input_key is None:
        return True, None
    if not input_key:
        return False, None
    return True, input_key


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

    # A typed declaration answers the question outright, so it is consulted
    # first and alone: when the Planner has said what each product is, nothing
    # below -- all of which reads names -- may overrule it.
    if step.measurement_audit_spec is not None:
        return bool(
            declared_audit_spec_is_emittable(step)
            and missingness_audit_input_scope_supported(step)
        )
    # The named contracts remain, because they are what gives a recognised shape
    # its specific ``analysis_kind``.  The capability rule is what stops an
    # unnamed-but-computable shape from falling through to the coder.
    contract_is_supported = (
        is_missingness_measurement_availability_contract(
            step.method,
            step.expected_outputs,
        )
        or is_missingness_complete_case_contract(
            step.method,
            step.expected_outputs,
        )
        or is_compact_missingness_measurement_contract(
            step.method,
            step.expected_outputs,
        )
        or is_measurement_bias_audit_contract(
            step.method,
            step.expected_outputs,
        )
        or declared_audit_products_are_emittable(
            step.method,
            step.expected_outputs,
        )
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
    return (
        "measurement_checks = []\n"
        + loop_header
        + "\n"
        + textwrap.indent(
            loop_body,
            "    ",
        )
    )


def missingness_measurement_audit_code(
    step: AnalysisStep | None = None,
    *,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> str:
    """Return a runner script that computes the per-concept missingness audit."""

    if step is not None and not missingness_audit_input_scope_supported(step):
        raise ValueError("missingness runner cannot consume the declared typed inputs")
    if plausibility_scope is not None:
        if step is None:
            raise ValueError(
                "a missingness plausibility scope requires an exact analysis step"
            )
        plausibility_scope.require_step(step.step_id)
    typed_cohort_input = (
        missingness_audit_cohort_input_key(step) if step is not None else None
    )
    plausibility_code = (
        render_standard_plausibility_receipt_code(
            plausibility_scope,
            frame_name="df",
        )
        if plausibility_scope is not None
        else ""
    )
    plausibility_summary_entry = (
        '"plausibility_audit": plausibility_audit,'
        if plausibility_scope is not None and plausibility_scope.expected_columns
        else ""
    )
    template = textwrap.dedent(
        r"""
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
        from easyicu.research_agent.methods.source_status import (
            reconcile_binary_event_presence,
            reconcile_conditional_event_time,
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

        __EASYICU_STANDARD_PLAUSIBILITY_RECEIPT__
        n_total = int(len(df))

        # --- research context: optional explicit concept list ------------------
        # Both files below are host-owned. An ABSENT file is a legitimate
        # legacy state (broad discovery mode); a present-but-unparseable file
        # must fail this step loudly — swallowing it would silently change the
        # audit scope from the declared inputs to every paired concept.
        req_concepts = []
        requested_inputs = []
        requested_outputs = []
        observation_semantics_by_column = {}
        declared_primary_exposure = ""
        context_path = run_dir / "research_context.json"
        if context_path.is_file():
            ctx = json.loads(context_path.read_text("utf-8"))
            # The host declares the exposure; this runner never guesses one from
            # a column name.  It is used only to stratify completeness counts,
            # never to select rows, define a group, or estimate anything.
            declared_primary_exposure = str(ctx.get("primary_exposure") or "").strip()
            for variable in ctx.get("variables") or []:
                if not isinstance(variable, dict):
                    continue
                variable_name = str(variable.get("name") or "").strip()
                semantics = variable.get("observation_semantics")
                if variable_name and isinstance(semantics, dict):
                    observation_semantics_by_column[variable_name] = dict(semantics)
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
                __EASYICU_PLAUSIBILITY_SUMMARY_ENTRY__
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
        semantic_complete_masks = {}
        observation_semantics_audit = {}
        temporal_semantics_findings = []
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

            indicator_semantics = "measurement_availability"
            raw_indicator_one_n = int(measured_mask.sum())
            eligible_n = n_total
            not_applicable_n = 0
            event_present_n = 0
            event_absent_n = 0
            before_origin_n = 0
            count_candidate = base + "_n"
            count_col = (
                count_candidate
                if count_candidate in df.columns
                else low.get(count_candidate.lower())
            )
            typed_semantics = observation_semantics_by_column.get(
                value_col or base,
                {},
            )
            typed_kind = str(typed_semantics.get("kind") or "")
            if typed_kind == "positive_only_event":
                declared_count = str(
                    typed_semantics.get("event_count_column") or ""
                )
                declared_measured = str(
                    typed_semantics.get("measured_column") or ""
                )
                declared_representative = str(
                    typed_semantics.get("representative_column") or ""
                )
                if (
                    declared_count != count_col
                    or declared_measured != flag_col
                    or declared_representative != value_col
                ):
                    raise RuntimeError(
                        "Positive-only event semantics do not match the audited columns"
                    )
                event_result = reconcile_binary_event_presence(
                    df,
                    count_column=declared_count,
                    measured_column=declared_measured,
                    representative_column=declared_representative,
                )
                indicator_semantics = "binary_event_presence"
                measured_mask = pd.Series(True, index=df.index)
                event_present_n = int(event_result.audit["event_present_n"])
                event_absent_n = int(event_result.audit["event_absent_n"])
                semantic_complete_masks[value_col] = pd.Series(
                    True,
                    index=df.index,
                )
                observation_semantics_audit[value_col] = dict(event_result.audit)
            elif typed_kind == "conditional_event_time":
                event_status_column = str(
                    typed_semantics.get("event_status_column") or ""
                )
                if value_col is None or not event_status_column:
                    raise RuntimeError(
                        "Conditional event-time semantics are incomplete"
                    )
                event_time_result = reconcile_conditional_event_time(
                    df,
                    event_status_column=event_status_column,
                    event_time_column=value_col,
                )
                event_audit = event_time_result.audit
                indicator_semantics = "conditional_event_time"
                measured_mask = df[value_col].notna()
                eligible_n = int(event_audit["eligible_event_n"])
                not_applicable_n = int(
                    event_audit["not_applicable_event_absent_n"]
                )
                event_present_n = eligible_n
                event_absent_n = not_applicable_n
                before_origin_n = int(event_audit["before_origin_n"])
                semantic_complete_masks[value_col] = (
                    df[value_col].notna()
                    | pd.to_numeric(
                        df[event_status_column],
                        errors="raise",
                    ).eq(0)
                )
                observation_semantics_audit[value_col] = dict(event_audit)
                if before_origin_n:
                    temporal_semantics_findings.append(
                        "event_time_before_declared_origin:"
                        + value_col
                        + ":"
                        + str(before_origin_n)
                    )
            # Legacy complete 0/1 event-status exports remain accepted when no
            # typed context is available. Positive-only shapes require the
            # typed contract above and the shared reconciliation helper.
            elif (
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
                    event_present_n = int(numeric_value.eq(1).sum())
                    event_absent_n = int(numeric_value.eq(0).sum())

            measured_one_n = int(measured_mask.sum())
            value_missing_n = (
                int(
                    observation_semantics_audit[value_col][
                        "missing_event_time_n"
                    ]
                )
                if indicator_semantics == "conditional_event_time"
                else int(n_total - measured_one_n)
            )
            if value_col is not None and value_col not in semantic_complete_masks:
                semantic_complete_masks[value_col] = measured_mask
            structural_no_source = bool(
                measured_one_n == 0 and value_present_n == 0
            )
            # Rows with a value present but the measurement indicator says zero
            # (a genuine present-but-unmeasured, e.g. a derived source flag).
            if indicator_semantics == "binary_event_presence":
                present_but_zero = 0
                measured_but_missing = 0
            elif has_flag:
                present_but_zero = int((value_present & ~measured_mask).sum())
                measured_but_missing = int((measured_mask & ~value_present).sum())
            else:
                present_but_zero = 0
                measured_but_missing = 0

            if structural_no_source:
                kind = "structural_no_source"
            elif indicator_semantics == "conditional_event_time":
                kind = "conditional_event_time"
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
                    "measured_one_pct": 100.0 * measured_one_n / n_total,
                    "value_missing_n": value_missing_n,
                    "value_missing_pct": 100.0 * value_missing_n / n_total,
                    # aliases so every downstream resolver (figure renderer /
                    # validator) finds a column it recognises.
                    "measured_n": measured_one_n,
                    "n_nonmissing": measured_one_n,
                    "missing_n": value_missing_n,
                    "missing_pct": 100.0 * value_missing_n / n_total,
                    "measured_pct": 100.0 * measured_one_n / n_total,
                    "eligible_n": eligible_n,
                    "not_applicable_n": not_applicable_n,
                    "applicable_pct": 100.0 * eligible_n / n_total,
                    "available_within_applicable_pct": (
                        100.0 * measured_one_n / eligible_n
                        if eligible_n
                        else np.nan
                    ),
                    "missing_within_applicable_pct": (
                        100.0 * value_missing_n / eligible_n
                        if eligible_n
                        else np.nan
                    ),
                    "event_present_n": event_present_n,
                    "event_absent_n": event_absent_n,
                    "event_present_pct": 100.0 * event_present_n / n_total,
                    "before_origin_n": before_origin_n,
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
            [
                "concept",
                "variable",
                "value_column",
                "n_total",
                "measured_one_n",
                "value_missing_n",
                "eligible_n",
                "not_applicable_n",
                "raw_value_missing_n",
                "indicator_semantics",
                "missingness_kind",
            ]
        ].copy()
        missingness_audit["n_nonmissing"] = missingness_audit["measured_one_n"]
        missingness_audit["missing_n"] = missingness_audit["value_missing_n"]
        missingness_audit["missing_pct"] = (
            100.0
            * missingness_audit["missing_n"]
            / missingness_audit["eligible_n"].replace(0, np.nan)
        )
        missingness_audit.to_csv(out_dir / "missingness_audit.csv", index=False)

        source_audit = audit[
            [
                "concept",
                "variable",
                "value_column",
                "n_total",
                "measured_one_n",
                "value_missing_n",
                "eligible_n",
                "not_applicable_n",
                "event_present_n",
                "event_absent_n",
                "before_origin_n",
                "value_present_but_measured_zero_n",
                "measured_but_value_missing_n",
                "indicator_semantics",
                "missingness_kind",
                "has_measured_indicator",
            ]
        ].copy()
        source_audit.to_csv(out_dir / "measurement_source_audit.csv", index=False)

        # --- event timing: what the event columns say about WHEN --------------
        # These counts are already computed above for every concept whose
        # indicator carries event semantics; they were only ever reachable
        # folded into the wide table.  Projected out, they are the audit a
        # plan means when it declares an event-timing product.  The frame is
        # written even when empty: "no audited concept is event-timed" is a
        # finding, and a silently absent file is not.
        event_timing_audit = audit[
            audit["indicator_semantics"].isin(
                ["conditional_event_time", "binary_event_presence"]
            )
        ][
            [
                "concept",
                "variable",
                "value_column",
                "n_total",
                "eligible_n",
                "not_applicable_n",
                "event_present_n",
                "event_absent_n",
                "before_origin_n",
                "value_missing_n",
                "indicator_semantics",
                "missingness_kind",
            ]
        ].copy()
        event_timing_audit.to_csv(out_dir / "event_timing_audit.csv", index=False)

        # --- measurement process: how OFTEN, and when ---------------------------
        # Distinct from the missingness table, which answers "was it measured at
        # all".  This one carries the observation-process facts the ICU rules ask
        # for: repeat-measurement counts, conditional applicability, and event
        # times recorded before the origin.
        process_rows = []
        for record in rows:
            base = record["concept"]
            count_candidate = base + "_n"
            count_col = (
                count_candidate
                if count_candidate in df.columns
                else low.get(count_candidate.lower())
            )
            counts = (
                pd.to_numeric(df[count_col], errors="coerce")
                if count_col is not None
                else pd.Series(dtype=float)
            )
            positive = counts[counts > 0] if len(counts) else counts
            process_rows.append(
                {
                    "concept": base,
                    "variable": record["variable"],
                    "value_column": record["value_column"],
                    "measurement_count_column": count_col or "",
                    "n_total": n_total,
                    "measured_one_n": record["measured_one_n"],
                    "measurement_total_n": (
                        int(counts.fillna(0).sum()) if len(counts) else 0
                    ),
                    "measurement_count_median_when_measured": (
                        float(positive.median()) if len(positive) else float("nan")
                    ),
                    "measurement_count_max": (
                        int(counts.max()) if len(counts) and counts.notna().any() else 0
                    ),
                    "repeat_measured_n": (
                        int((counts > 1).sum()) if len(counts) else 0
                    ),
                    "eligible_n": record["eligible_n"],
                    "not_applicable_n": record["not_applicable_n"],
                    "event_present_n": record["event_present_n"],
                    "event_absent_n": record["event_absent_n"],
                    "before_origin_n": record["before_origin_n"],
                    "indicator_semantics": record["indicator_semantics"],
                    "missingness_kind": record["missingness_kind"],
                }
            )
        measurement_process_audit = pd.DataFrame(process_rows)
        measurement_process_audit.to_csv(
            out_dir / "measurement_process_audit.csv", index=False
        )

        # --- component completeness, stratified by the declared exposure --------
        # Differential completeness between exposure strata is the mechanism by
        # which a derived exposure can look associated with an outcome purely
        # because sicker patients are measured more.  Counting it is
        # deterministic; interpreting it is not, and this runner does not.
        exposure_column = None
        exposure_levels = []
        exposure_note = "no primary exposure declared in the research context"
        if declared_primary_exposure:
            exposure_column = (
                declared_primary_exposure
                if declared_primary_exposure in df.columns
                else low.get(declared_primary_exposure.lower())
            )
            if exposure_column is None:
                exposure_note = (
                    "declared primary exposure %r is not a cohort column"
                    % declared_primary_exposure
                )
            else:
                observed = df[exposure_column].dropna().unique().tolist()
                if len(observed) > 10:
                    exposure_column = None
                    exposure_note = (
                        "declared primary exposure has %d levels; completeness is "
                        "reported unstratified" % len(observed)
                    )
                else:
                    exposure_levels = sorted(observed, key=lambda value: str(value))
                    exposure_note = "stratified by the declared primary exposure"

        completeness_rows = []
        for record in rows:
            base = record["concept"]
            value_col = record["value_column"] or None
            mask = (
                semantic_complete_masks.get(value_col)
                if value_col is not None
                else None
            )
            if mask is None:
                flag_col = base + "_measured"
                resolved_flag = (
                    flag_col if flag_col in df.columns else low.get(flag_col.lower())
                )
                mask = (
                    pd.to_numeric(df[resolved_flag], errors="coerce").eq(1)
                    if resolved_flag is not None
                    else pd.Series(False, index=df.index)
                )
            # Semantic completeness and the raw indicator are BOTH reported.
            # For an event concept the declared ICU rule makes an absent row a
            # complete negative observation, so semantic completeness is 100 %
            # in every stratum — which is exactly where a differential
            # observation process would hide.  The raw indicator rate is what
            # lets a reader judge whether that rule is safe here.
            raw_flag_col = base + "_measured"
            resolved_raw_flag = (
                raw_flag_col
                if raw_flag_col in df.columns
                else low.get(raw_flag_col.lower())
            )
            raw_mask = (
                pd.to_numeric(df[resolved_raw_flag], errors="coerce").eq(1)
                if resolved_raw_flag is not None
                else pd.Series(False, index=df.index)
            )
            strata = [("__all__", pd.Series(True, index=df.index))]
            if exposure_column is not None:
                strata.extend(
                    (str(level), df[exposure_column].eq(level))
                    for level in exposure_levels
                )
            for label, selector in strata:
                stratum_n = int(selector.sum())
                measured_n = int((mask & selector).sum())
                raw_indicator_n = int((raw_mask & selector).sum())
                completeness_rows.append(
                    {
                        "concept": base,
                        "variable": record["variable"],
                        "value_column": record["value_column"],
                        "exposure_variable": exposure_column or "",
                        "exposure_category": label,
                        # THE FIRST ROW IS THE WHOLE COHORT, AND THE TABLE HAS
                        # TO SAY SO IN A SPELLING ITS READERS ALREADY KNOW.
                        # ``exposure_category='__all__'`` said it, but only to
                        # someone who knows this producer; the host's own
                        # aggregate-row validator looks for a declared ROLE
                        # column, so it read the total as a third exposure
                        # group.
                        #
                        # The three strings are LITERALS because this block is
                        # a template rendered into the container script, which
                        # defines no host names -- the module constants above
                        # exist so a test can assert these literals still match
                        # the validator's vocabulary.
                        "row_role": (
                            "overall" if label == "__all__" else "exposure_level"
                        ),
                        "n_stratum": stratum_n,
                        "measured_n": measured_n,
                        "measured_pct": (
                            100.0 * measured_n / stratum_n
                            if stratum_n
                            else float("nan")
                        ),
                        "value_missing_n": stratum_n - measured_n,
                        "value_missing_pct": (
                            100.0 * (stratum_n - measured_n) / stratum_n
                            if stratum_n
                            else float("nan")
                        ),
                        # Empty, not zero, when the concept carries no
                        # ``_measured`` indicator at all: a 0 there reads as
                        # "never measured" when it means "not applicable".
                        "raw_indicator_one_n": (
                            raw_indicator_n
                            if resolved_raw_flag is not None
                            else float("nan")
                        ),
                        "raw_indicator_one_pct": (
                            100.0 * raw_indicator_n / stratum_n
                            if resolved_raw_flag is not None and stratum_n
                            else float("nan")
                        ),
                        "has_measured_indicator": bool(resolved_raw_flag is not None),
                        "indicator_semantics": record["indicator_semantics"],
                        "missingness_kind": record["missingness_kind"],
                    }
                )
        exposure_component_completeness_audit = pd.DataFrame(completeness_rows)
        exposure_component_completeness_audit.to_csv(
            out_dir / "exposure_component_completeness_audit.csv", index=False
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
            complete_mask = semantic_complete_masks.get(
                column,
                df[column].notna(),
            )
            observed_n = int(complete_mask.sum())
            denominator_rows.append(
                {
                    "analysis_set": "observed:" + column,
                    "required_variables": column,
                    "n_total": n_total,
                    "n_complete": observed_n,
                    "n_excluded_missing": int(n_total - observed_n),
                    "complete_pct": 100.0 * observed_n / n_total,
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
            joint_complete_mask = pd.Series(True, index=df.index)
            for column in resolved_inputs:
                joint_complete_mask &= semantic_complete_masks.get(
                    column,
                    df[column].notna(),
                )
            complete_n = int(joint_complete_mask.sum())
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
                    100.0 * complete_n / n_total
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

        # The count travels with the percentage. A manuscript states the
        # numerator ("missing for 696 of 94,458 stays") and cites this step,
        # but publishing only the rate leaves this step owning no claim for
        # the number in that sentence -- the binder then finds the same value
        # registered by other steps, none of them the cited one, and has to
        # refuse. ``value_missing_n`` is the numerator of the percentage
        # already published here and sits in this step's own CSV rows.
        worst = audit.head(5)[
            ["concept", "value_missing_n", "value_missing_pct"]
        ].to_dict("records")
        n_structural = int((audit["missingness_kind"] == "structural_no_source").sum())
        n_binary_event_status = int(
            (audit["indicator_semantics"] == "binary_event_presence").sum()
        )
        product_files = __EASYICU_PRODUCT_FILES__
        declared_output_files = {}
        for output in requested_outputs:
            product = output.split(":", 1)[-1].strip()
            if product in product_files:
                declared_output_files[output] = product_files[product]
        summary = {
            __EASYICU_PLAUSIBILITY_SUMMARY_ENTRY__
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
            "exposure_component_completeness": {
                "exposure_variable": exposure_column or "",
                "exposure_categories": [str(level) for level in exposure_levels],
                "stratified": bool(exposure_column is not None),
                "note": exposure_note,
                "n_rows": int(len(exposure_component_completeness_audit)),
            },
            "n_structural_no_source": n_structural,
            "n_binary_event_status": n_binary_event_status,
            # Zero here is a real answer, not a missing one: it says no audited
            # concept carries event semantics, which the event-timing table
            # alone cannot distinguish from "the table was never written".
            "n_event_timed_concepts": int(len(event_timing_audit)),
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
            "observation_semantics_audit": observation_semantics_audit,
            "temporal_validity_audit": {
                "status": (
                    "flagged_requires_downstream_protocol"
                    if temporal_semantics_findings
                    else "ok"
                ),
                "reason_codes": temporal_semantics_findings,
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
        """
    ).strip()
    template = template.replace(
        "__EASYICU_TYPED_COHORT_INPUT__",
        repr(typed_cohort_input),
    )
    template = template.replace(
        "__EASYICU_PRODUCT_FILES__",
        _render_product_files(step),
    )
    template = template.replace(
        "__EASYICU_MEASUREMENT_PROVENANCE_SCOPE__",
        _measurement_provenance_code(step),
    )
    template = template.replace(
        "__EASYICU_STANDARD_PLAUSIBILITY_RECEIPT__",
        plausibility_code,
    )
    return template.replace(
        "__EASYICU_PLAUSIBILITY_SUMMARY_ENTRY__",
        plausibility_summary_entry,
    )
