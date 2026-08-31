"""Model covariate extraction and exposure/leakage audit rules."""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path
from typing import List, Optional, Sequence

from .step_families import effect_output_authorized
from ..icu_rules import (
    detect_outcome_as_predictor,
    detect_overadjustment,
    outcome_leakage_caution,
    overadjustment_caution,
    treatment_mediator_caution,
)
from ..schema import AnalysisStep, ResearchContext, ValidationFinding

_COEF_TABLE_VALUE_COLUMNS = frozenset(
    {"coef", "beta", "estimate", "log_or", "odds_ratio", "or", "hazard_ratio", "hr"}
)
# A coefficient table's identifier column is named differently across ecosystems:
# statsmodels summary frames use ``variable``; R's broom::tidy and many hand-rolled
# tables use ``term``; others use ``predictor`` / ``covariate`` / ``parameter`` /
# ``feature``. Recognise any of these, but only paired with a coefficient-value
# column (above) — that pairing is what distinguishes a model coefficient table
# from a missingness / table-one CSV, so broadening the id column stays safe.
_COEF_TABLE_ID_COLUMNS = frozenset(
    {"variable", "term", "predictor", "covariate", "parameter", "feature"}
)
_NON_COVARIATE_TERMS = frozenset({"const", "intercept", "(intercept)"})


def read_model_covariate_names(
    directory: Path,
    *,
    files: Optional[Sequence[Path]] = None,
) -> List[str]:
    """Variable names from every model coefficient table under ``directory``.

    De-duplicated, intercept rows dropped, first-seen order preserved. Returns
    ``[]`` when the directory is absent or holds no coefficient table — the
    overadjustment check then stays silent rather than guessing. Filename-agnostic:
    a CSV counts as a coefficient table only when its header has a ``variable``
    column and a coefficient-like column, so non-model tables are ignored.
    """
    names: List[str] = []
    base = Path(directory)
    if files is None and not base.exists():
        return names
    candidates = (
        sorted(base.rglob("*.csv"))
        if files is None
        else sorted(
            Path(path)
            for path in files
            if Path(path).is_file() and Path(path).suffix.lower() == ".csv"
        )
    )
    for path in candidates:
        try:
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                raw_fields = reader.fieldnames or []
                header = {(h or "").strip().lower() for h in raw_fields}
                if header.isdisjoint(_COEF_TABLE_ID_COLUMNS) or header.isdisjoint(
                    _COEF_TABLE_VALUE_COLUMNS
                ):
                    continue
                # The identifier column actually present (first in file order);
                # read variable names from it rather than assuming ``variable``.
                id_field = next(
                    (
                        h
                        for h in raw_fields
                        if (h or "").strip().lower() in _COEF_TABLE_ID_COLUMNS
                    ),
                    None,
                )
                if id_field is None:
                    continue
                for row in reader:
                    value = (row.get(id_field) or "").strip()
                    if (
                        value
                        and value.lower() not in _NON_COVARIATE_TERMS
                        and value not in names
                        and row.get("term_role") not in ("exposure", "outcome")
                    ):
                        names.append(value)
        except (OSError, ValueError):
            continue
    return names


# A coefficient table is the ground truth of what entered a model, but a run
# that reports only a model-level OR summary (rows = models, cols = OR/CI) never
# writes one — the per-covariate adjustment set then lives only in the analysis
# code. These recover it from the code as a fallback, generally: the patterns
# below are how any statsmodels/patsy analysis declares its adjustment set, and
# every extracted token is routed through the dictionary-driven detectors, so no
# case (exposure / covariate / score) is hard-coded here.
#
# A variable whose name *intends* the adjustment set (covariates, confounders,
# adjustment_cols, predictors, ...) assigned a list/tuple of string column names.
# Names that denote the predictor / adjustment side of a model. Deliberately
# NOT "all model variables" names (``model_vars`` / ``vars`` / ``cols``): those
# bundle the outcome in with the predictors, which would let a study endpoint
# leak into the recovered adjustment set and trip a spurious outcome-leakage
# error. X / design / regressors / rhs exclude the outcome by convention.
_COVARIATE_INTENT_SUBSTRINGS = ("covariate", "covar", "confound", "adjust", "predictor")
_COVARIATE_INTENT_EXACT = frozenset(
    {"x_cols", "design_cols", "regressors", "rhs", "rhs_cols"}
)
# Exclusion/negation markers. A list named for what is deliberately kept OUT of
# the model (``renal_source_not_adjusted``, ``excluded_covariates``,
# ``dropped_for_overadjustment``, the columns of the ``unadjusted`` model) is the
# inverse of the adjustment set. Reading it as the adjustment set inverts its
# meaning and manufactures a phantom overadjustment/leakage finding. These
# markers are unambiguous:
# each *means* "not in the model", so suppressing them cannot hide a genuine
# adjustment set (which is never named this way) — no false-negative risk. Only
# clear negations are listed; transformation words ("drop"/"remove"/"omit") are
# excluded because ``covariates_after_dropping_missing`` can name the final set.
_COVARIATE_EXCLUSION_MARKERS = (
    "not_adjust",
    "notadjust",
    "non_adjust",
    "nonadjust",
    "unadjust",
    "overadjust",
    "exclud",
    "not_covariat",
    "not_confound",
)


def _name_intends_covariates(name: str) -> bool:
    low = name.lower()
    if any(marker in low for marker in _COVARIATE_EXCLUSION_MARKERS):
        return False
    if low in _COVARIATE_INTENT_EXACT:
        return True
    return any(sub in low for sub in _COVARIATE_INTENT_SUBSTRINGS)


def _string_list_elements(node: ast.AST) -> List[str]:
    """String constants in a list/tuple literal, or ``[]`` if not one."""
    if not isinstance(node, (ast.List, ast.Tuple)):
        return []
    out: List[str] = []
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            tok = elt.value.strip()
            if tok:
                out.append(tok)
    return out


def _formula_rhs_terms(formula: str) -> List[str]:
    """Right-hand-side column tokens of a patsy/statsmodels formula string.

    ``"death ~ sepsis3 + age + C(sex) + map_max"`` -> ``[sepsis3, age, sex,
    map_max]``. Interaction (``:`` / ``*``) is split to its main terms; the
    ``C(...)`` categorical wrapper is unwrapped; intercept tokens are dropped.
    The exposure may appear on the RHS — that is fine, the detectors exclude the
    exposure itself.

    Conservative: a term is kept only if it is a clean Python identifier, so
    prose strings that merely contain ``~`` (e.g. a note "adjusted OR ~1.11") do
    not masquerade as a formula — their "terms" are not identifiers and the
    string yields nothing.
    """
    if "~" not in formula:
        return []
    # Require an identifier-ish left-hand side so "OR ~1.11" still parses to a
    # RHS, but the identifier check below is what actually rejects the prose.
    rhs = formula.split("~", 1)[1]
    terms: List[str] = []
    for raw in re.split(r"[+*:]", rhs):
        tok = raw.strip()
        # unwrap C(col), C(col, Treatment(...)) -> col
        m = re.match(r"^[A-Za-z_]\w*\(\s*([A-Za-z_]\w*)", tok)
        if m:
            tok = m.group(1)
        if re.fullmatch(r"[A-Za-z_]\w*", tok) and tok not in ("C", "I"):
            terms.append(tok)
    return terms


def _covariate_names_from_code(
    directory: Path,
    *,
    files: Optional[Sequence[Path]] = None,
) -> List[str]:
    """Adjustment-set column names recovered from a run's analysis code.

    General + conservative: parses the analysis ``*.py`` near ``directory`` and
    collects column names from (1) a list/tuple literal assigned to a variable
    whose name intends the adjustment set (``covariates`` / ``confounders`` /
    ``adjustment_cols`` / ``x_cols`` ...) and (2) statsmodels/patsy formula
    strings (the RHS of ``y ~ ...``). Anything it cannot read with confidence is
    skipped (unparseable file, ambiguous slice) so it never invents covariates.
    First-seen order, de-duplicated. Returns ``[]`` when nothing recognisable.
    """
    base = Path(directory)
    seen: List[str] = []

    def _add(tok: str) -> None:
        value = tok.strip()
        if value and value.lower() not in _NON_COVARIATE_TERMS and value not in seen:
            seen.append(value)

    # Search the directory, its parent (a step's outputs/ sits beside analysis.py),
    # and any steps/*/analysis.py beneath it (the post-hoc run-root case). Bounded.
    candidates: List[Path]
    if files is None:
        candidates = []
        for src in (base, base.parent):
            if src.exists():
                candidates.extend(sorted(src.glob("*.py")))
        if base.exists():
            candidates.extend(sorted(base.rglob("analysis.py")))
    else:
        candidates = [
            Path(path)
            for path in files
            if Path(path).is_file() and Path(path).suffix.lower() == ".py"
        ]

    visited: set = set()
    for path in candidates:
        rp = path.resolve()
        if rp in visited or not path.is_file():
            continue
        visited.add(rp)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, SyntaxError):
            continue  # a file we cannot read with confidence is skipped
        for node in ast.walk(tree):
            # (1) covariate-intent list/tuple assignment
            if isinstance(node, ast.Assign):
                targets = [t for t in node.targets if isinstance(t, ast.Name)]
                if any(_name_intends_covariates(t.id) for t in targets):
                    for tok in _string_list_elements(node.value):
                        _add(tok)
            # (2) formula strings anywhere (y ~ rhs)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "~" in node.value and len(node.value) <= 4000:
                    for tok in _formula_rhs_terms(node.value):
                        _add(tok)
    return seen


def read_adjustment_covariates(
    directory: Path,
    *,
    files: Optional[Sequence[Path]] = None,
) -> List[str]:
    """The model's adjustment set, preferring the coefficient table.

    A per-covariate coefficient table is the ground truth of what entered the
    model, so it wins when present. ``files`` restricts discovery to an explicit
    authority list (for example, current manifest evidence); when omitted the
    historical directory scan is preserved. When a run reports only a
    model-level OR summary (no coefficient table), the adjustment set is
    recovered from the analysis code instead, so the overadjustment / leakage
    auditors are not blind to summary-only outputs. Returns ``[]`` when neither
    source yields anything.
    """
    coef_names = read_model_covariate_names(directory, files=files)
    if coef_names:
        return coef_names
    return _covariate_names_from_code(directory, files=files)


def _primary_exposure_overadjustment_findings(
    *,
    step: AnalysisStep,
    context: ResearchContext,
    out_dir: Path,
) -> List[ValidationFinding]:
    """Hard-block overadjustment: adjusting for a constituent of the exposure.

    When the question names a primary exposure that is a known composite/derived
    score and this step's fitted model conditioned on one of that exposure's
    definitional constituents, emit an error finding routed through the same
    in-run contract-repair loop the exposure-contract auditor uses (re-fit in
    run, no full restart). This is an objective design error — conditioning on a
    component of the exposure nulls the very signal under study — never an
    analytical-preference call: it dictates only the removal of the offending
    constituent from the adjustment set, not the model form, covariates beyond
    the offenders, or estimator. The exposure must be declared
    (``context.primary_exposure``); it is never inferred, so the check stays
    silent rather than guessing.
    """
    if not effect_output_authorized(step):
        return []
    exposure = (getattr(context, "primary_exposure", None) or "").strip()
    if not exposure:
        return []
    covariates = read_adjustment_covariates(out_dir)
    covariates = step.without_required_primary_exposure_terms(covariates)
    offenders = detect_overadjustment(exposure, covariates)
    if not offenders:
        # No resolvable constituent matched. If the exposure is nonetheless a
        # derived/composite concept whose inputs could not be resolved (a
        # callback score with an empty dependency closure, e.g. mews/news/sirs),
        # the deterministic check is blind — surface a caution so the risk is
        # not silently passed. A caution is a warning, never a gating error: it
        # prompts the analyst to verify, it does not re-fit or block.
        caution = overadjustment_caution(exposure, covariates)
        if not caution:
            return []
        return [
            ValidationFinding(
                validator="overadjustment_auditor",
                severity="warning",
                message="Overadjustment could not be auto-checked: " + caution,
                detail={
                    "kind": "overadjustment_caution",
                    "step_id": step.step_id,
                    "exposure": exposure,
                    "adjustment_covariates": list(covariates),
                },
            )
        ]
    joined = ", ".join(f"`{o}`" for o in offenders)
    return [
        ValidationFinding(
            validator="overadjustment_auditor",
            severity="error",
            message=(
                f"The primary exposure `{exposure}` is a composite/derived score, "
                f"and this model adjusted for {joined}, which constitute(s) or "
                f"derive(s) it. Conditioning on a component of the exposure removes "
                f"the signal under study (overadjustment). Re-fit dropping {joined} "
                f"from the adjustment set; keep only confounders that are neither "
                f"constituents nor downstream mediators of the exposure."
            ),
            detail={
                "kind": "overadjustment",
                "step_id": step.step_id,
                "exposure": exposure,
                "offending_covariates": list(offenders),
            },
        )
    ]


def _primary_model_leakage_findings(
    *,
    step: AnalysisStep,
    context: ResearchContext,
    out_dir: Path,
) -> List[ValidationFinding]:
    """Outcome-leakage (error) + endpoint/treatment-as-mediator (caution).

    Complements the overadjustment hard-block with two more model-methodology
    checks on this step's fitted covariates, keeping the same impartiality split:

    - ERROR: the declared primary outcome appears among the model's predictors.
      Conditioning a model on its own dependent variable is target leakage by
      construction — an objective design error routed through the same in-run
      re-fit loop (no full restart), like overadjustment.
    - CAUTION (warning, never gates): a *different* endpoint concept used as a
      predictor (timing-dependent leakage), or a treatment/intervention covariate
      that may be a mediator on the exposure→outcome path. Both are defensible
      depending on timing/DAG the auditor cannot see, so they prompt the analyst
      to verify rather than re-fitting or blocking.

    The outcome / exposure must be declared (``context.target_outcome`` /
    ``context.primary_exposure``); they are never inferred, so each check stays
    silent rather than guessing.
    """
    if not effect_output_authorized(step):
        return []
    covariates = read_adjustment_covariates(out_dir)
    if not covariates:
        return []
    outcome = (getattr(context, "target_outcome", None) or "").strip()
    exposure = (getattr(context, "primary_exposure", None) or "").strip()
    findings: List[ValidationFinding] = []

    leak = detect_outcome_as_predictor(covariates, study_outcome=outcome)
    if leak:
        joined = ", ".join(f"`{o}`" for o in leak)
        findings.append(
            ValidationFinding(
                validator="outcome_leakage_auditor",
                severity="error",
                message=(
                    f"The declared primary outcome `{outcome}` appears among this "
                    f"model's predictors ({joined}). Conditioning a model on its own "
                    f"dependent variable is target leakage. Re-fit dropping {joined} "
                    f"from the predictors; the outcome must appear only as the "
                    f"dependent variable."
                ),
                detail={
                    "kind": "outcome_leakage",
                    "step_id": step.step_id,
                    "outcome": outcome,
                    "offending_predictors": list(leak),
                },
            )
        )

    endpoint_caution = outcome_leakage_caution(covariates, study_outcome=outcome)
    if endpoint_caution:
        findings.append(
            ValidationFinding(
                validator="outcome_leakage_auditor",
                severity="warning",
                message="Possible outcome leakage: " + endpoint_caution,
                detail={
                    "kind": "outcome_leakage_caution",
                    "step_id": step.step_id,
                    "outcome": outcome,
                    "adjustment_covariates": list(covariates),
                },
            )
        )

    if exposure:
        mediator_caution = treatment_mediator_caution(exposure, covariates)
        if mediator_caution:
            findings.append(
                ValidationFinding(
                    validator="overadjustment_auditor",
                    severity="warning",
                    message="Possible mediator adjustment: " + mediator_caution,
                    detail={
                        "kind": "treatment_mediator_caution",
                        "step_id": step.step_id,
                        "exposure": exposure,
                        "adjustment_covariates": list(covariates),
                    },
                )
            )
    return findings

