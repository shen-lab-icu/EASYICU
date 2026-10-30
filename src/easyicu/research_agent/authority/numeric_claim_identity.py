"""Typed identity inference for manuscript-bindable numeric claims."""

from __future__ import annotations

import enum
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class NumericEffectScale(str, enum.Enum):
    """Effect-measure identity carried by a manuscript-bindable number."""

    ODDS_RATIO = "odds_ratio"
    HAZARD_RATIO = "hazard_ratio"
    RISK_RATIO = "risk_ratio"


class NumericEstimand(str, enum.Enum):
    """A number's role within an effect estimate and its interval."""

    POINT_ESTIMATE = "point_estimate"
    CONFIDENCE_INTERVAL_LOWER = "confidence_interval_lower"
    CONFIDENCE_INTERVAL_UPPER = "confidence_interval_upper"


def coerce_numeric_effect_scale(value: Any) -> Optional[NumericEffectScale]:
    if value is None or value == "":
        return None
    if isinstance(value, NumericEffectScale):
        return value
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    aliases = {
        "or": NumericEffectScale.ODDS_RATIO,
        "odds_ratio": NumericEffectScale.ODDS_RATIO,
        "adjusted_odds_ratio": NumericEffectScale.ODDS_RATIO,
        "hr": NumericEffectScale.HAZARD_RATIO,
        "hazard_ratio": NumericEffectScale.HAZARD_RATIO,
        "adjusted_hazard_ratio": NumericEffectScale.HAZARD_RATIO,
        "rr": NumericEffectScale.RISK_RATIO,
        "risk_ratio": NumericEffectScale.RISK_RATIO,
        "relative_risk": NumericEffectScale.RISK_RATIO,
        "adjusted_risk_ratio": NumericEffectScale.RISK_RATIO,
    }
    return aliases.get(normalized)


def infer_numeric_claim_identity(
    source_field: str,
    *,
    declared_effect_scale: Any = None,
) -> Tuple[Optional[NumericEffectScale], Optional[NumericEstimand]]:
    """Infer only identities that are explicit in a source coordinate."""

    source = str(source_field or "").strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", source).strip("_")
    tokens = set(normalized.split("_"))
    declared_scale = coerce_numeric_effect_scale(declared_effect_scale)

    scale: Optional[NumericEffectScale] = None
    if "odds_ratio" in normalized or "or" in tokens:
        scale = NumericEffectScale.ODDS_RATIO
    elif "hazard_ratio" in normalized or "hr" in tokens:
        scale = NumericEffectScale.HAZARD_RATIO
    elif (
        "risk_ratio" in normalized or "relative_risk" in normalized or "rr" in tokens
    ):
        scale = NumericEffectScale.RISK_RATIO

    excluded_tokens = {"se", "stderr", "std", "error", "variance", "p", "value"}
    if tokens & excluded_tokens and not ({"ci", "interval"} & tokens):
        return None, None
    if scale is not None and declared_scale is not None and scale is not declared_scale:
        raise ValueError(
            "declared effect scale conflicts with source-field identity: "
            f"{declared_scale.value!r} != {scale.value!r} for {source_field!r}"
        )
    # Regression coefficients and explicitly logged ratios are on a transformed
    # coordinate. They must be exponentiated by an owning producer before they
    # can be published as OR/HR/RR claims.
    transformed_tokens = {"log", "ln", "logit", "coefficient", "coef", "beta"}
    if scale is not None and tokens & transformed_tokens:
        return None, None

    lower = bool(
        re.search(r"(?:ci|interval)_(?:95_)?(?:low|lower|lcl)(?:_|$)", normalized)
        or re.search(
            r"(?:^|_)(?:ci|interval)(?:_[a-z0-9]+)*_0$",
            normalized,
        )
    )
    upper = bool(
        re.search(r"(?:ci|interval)_(?:95_)?(?:high|upper|ucl)(?:_|$)", normalized)
        or re.search(
            r"(?:^|_)(?:ci|interval)(?:_[a-z0-9]+)*_1$",
            normalized,
        )
    )
    if lower:
        estimand = NumericEstimand.CONFIDENCE_INTERVAL_LOWER
    elif upper:
        estimand = NumericEstimand.CONFIDENCE_INTERVAL_UPPER
    else:
        leaf = re.split(r"[.\[]", source)[-1]
        point_names = {
            "adjusted_effect",
            "effect_estimate",
            "estimate",
            "odds_ratio",
            "hazard_ratio",
            "risk_ratio",
            "relative_risk",
        }
        estimand = (
            NumericEstimand.POINT_ESTIMATE
            if scale is not None or leaf in point_names
            else None
        )

    if estimand is not None and scale is None:
        scale = declared_scale
    return scale, estimand


@dataclass
class NumericClaim:
    """One manuscript-bindable numeric leaf and its source identity."""

    value: str
    canonical: float
    evidence_id: str
    step_id: str
    source_field: str
    tolerance: float = 1e-3
    effect_scale: Optional[NumericEffectScale] = None
    estimand: Optional[NumericEstimand] = None
    formula: Optional[str] = None
    explanation: Optional[str] = None
    derived_from: List[Tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        declared_scale = coerce_numeric_effect_scale(self.effect_scale)
        inferred_scale, inferred_estimand = infer_numeric_claim_identity(
            self.source_field,
            declared_effect_scale=declared_scale,
        )
        self.estimand = (
            self.estimand
            if isinstance(self.estimand, NumericEstimand)
            else NumericEstimand(self.estimand)
            if self.estimand
            else inferred_estimand
        )
        normalized_source = re.sub(
            r"[^a-z0-9]+", "_", str(self.source_field or "").strip().lower()
        ).strip("_")
        transformed_ratio = bool(
            {"log", "ln", "logit", "coefficient", "coef", "beta"}
            & set(normalized_source.split("_"))
            and (
                "odds_ratio" in normalized_source
                or "hazard_ratio" in normalized_source
                or "risk_ratio" in normalized_source
                or "relative_risk" in normalized_source
                or {"or", "hr", "rr"} & set(normalized_source.split("_"))
            )
        )
        self.effect_scale = (
            None
            if transformed_ratio
            else inferred_scale
            or (declared_scale if self.estimand is not None else None)
        )

    @property
    def is_derived(self) -> bool:
        return self.formula is not None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.effect_scale is None:
            payload.pop("effect_scale", None)
        else:
            payload["effect_scale"] = self.effect_scale.value
        if self.estimand is None:
            payload.pop("estimand", None)
        else:
            payload["estimand"] = self.estimand.value
        if not self.is_derived:
            payload.pop("formula", None)
            payload.pop("explanation", None)
            payload.pop("derived_from", None)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NumericClaim":
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in known}
        if "derived_from" in clean and clean["derived_from"]:
            clean["derived_from"] = [
                tuple(pair) if not isinstance(pair, tuple) else pair
                for pair in clean["derived_from"]
            ]
        return cls(**clean)


__all__ = [
    "NumericClaim",
    "NumericEffectScale",
    "NumericEstimand",
    "coerce_numeric_effect_scale",
    "infer_numeric_claim_identity",
]


# Coder-generated step_summary entries can request that a derived
# numeric value (an OR confidence-interval bound, a between-cohort
# difference, an AUC delta) be computed from other registered claims
# and registered as its own NumericClaim. The formula is evaluated
# here in a restricted-AST sandbox that accepts ONLY:
#
#   * Numeric constants
#   * Names resolving to entries in ``sources`` (or to a math whitelist
#     constant — pi, e)
#   * Arithmetic: + - * / **
#   * Unary +/-
#   * Calls to a small math whitelist: exp, log, log10, sqrt, abs,
#     min, max
#   * Parentheses
#
# Rejected: attribute access, subscripts, comprehensions, comparisons,
# bool ops, lambda, named expressions, anything else. Names starting
# with ``_`` are rejected outright. The output is a finite float; nan
# / inf raise. Errors raise ``DerivedFormulaError`` with the offending
# AST node type for the audit trail.
#
# Inspired by data-to-paper's ``\num{<formula>, "<explanation>"}``;
# our variant evaluates at register-time and persists the result as a
# regular NumericClaim with ``derived_from`` provenance — so the
# existing reverse-binder picks it up with no extra plumbing.


class DerivedFormulaError(ValueError):
    """Raised when a derived-claim formula contains a disallowed
    expression, references an unknown source, or evaluates to a
    non-finite value. The message is safe to surface in audit findings.
    """


# Math whitelist. ``min`` / ``max`` accept any positive number of
# args; the rest are single-arg. Anything outside this dict is rejected
# at call-time.
import math as _math  # noqa: E402  (deliberate post-typing local-only import)

_DERIVED_FORMULA_FUNCS: Dict[str, Any] = {
    "exp": _math.exp,
    "log": _math.log,
    "log10": _math.log10,
    "sqrt": _math.sqrt,
    "abs": abs,
    "min": min,
    "max": max,
}
_DERIVED_FORMULA_CONSTS: Dict[str, float] = {
    "pi": _math.pi,
    "e": _math.e,
}


def _evaluate_derived_formula(
    formula: str,
    *,
    sources: Dict[str, float],
) -> float:
    """Evaluate ``formula`` in the restricted sandbox.

    ``sources`` maps Python identifier → canonical float of a
    registered claim. Names referenced in the formula that aren't in
    ``sources`` and aren't in the math constants whitelist raise
    ``DerivedFormulaError``. Returns a finite float.
    """
    import ast  # local import — only needed when a derived claim arrives

    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise DerivedFormulaError(f"derived-formula syntax error: {exc.msg}") from exc

    _ALLOWED_BINOPS = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
    )
    _ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)

    def _walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(
                node.value, bool
            ):
                return float(node.value)
            raise DerivedFormulaError(
                f"derived-formula rejected constant of type {type(node.value).__name__}"
            )
        if isinstance(node, ast.Name):
            name = node.id
            if name.startswith("_"):
                raise DerivedFormulaError(
                    f"derived-formula rejected dunder/private name {name!r}"
                )
            if name in sources:
                return float(sources[name])
            if name in _DERIVED_FORMULA_CONSTS:
                return _DERIVED_FORMULA_CONSTS[name]
            raise DerivedFormulaError(
                f"derived-formula references unknown source {name!r}; "
                f"available sources: {sorted(sources)}"
            )
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINOPS):
                raise DerivedFormulaError(
                    f"derived-formula rejected binary operator "
                    f"{type(node.op).__name__}"
                )
            lhs = _walk(node.left)
            rhs = _walk(node.right)
            try:
                if isinstance(node.op, ast.Add):
                    return lhs + rhs
                if isinstance(node.op, ast.Sub):
                    return lhs - rhs
                if isinstance(node.op, ast.Mult):
                    return lhs * rhs
                if isinstance(node.op, ast.Div):
                    if rhs == 0.0:
                        raise DerivedFormulaError("derived-formula division by zero")
                    return lhs / rhs
                if isinstance(node.op, ast.Pow):
                    return lhs**rhs
            except OverflowError as exc:
                # Python raises OverflowError on float ** float when the
                # result would exceed float range; we surface this as a
                # non-finite result so callers see one consistent error
                # type regardless of whether overflow happens silently
                # (→ inf, caught below) or eagerly (→ OverflowError).
                raise DerivedFormulaError(
                    f"derived-formula evaluated to non-finite value "
                    f"(arithmetic overflow): {exc}"
                ) from exc
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARYOPS):
                raise DerivedFormulaError(
                    f"derived-formula rejected unary operator "
                    f"{type(node.op).__name__}"
                )
            operand = _walk(node.operand)
            return +operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise DerivedFormulaError(
                    "derived-formula rejected call with non-name callable"
                )
            fname = node.func.id
            if fname.startswith("_") or fname not in _DERIVED_FORMULA_FUNCS:
                raise DerivedFormulaError(
                    f"derived-formula rejected call to {fname!r}; "
                    f"allowed: {sorted(_DERIVED_FORMULA_FUNCS)}"
                )
            if node.keywords:
                raise DerivedFormulaError(
                    f"derived-formula rejected keyword arguments to {fname!r}"
                )
            args = [_walk(a) for a in node.args]
            try:
                return float(_DERIVED_FORMULA_FUNCS[fname](*args))
            except (ArithmeticError, ValueError, TypeError) as exc:
                raise DerivedFormulaError(
                    f"derived-formula call {fname!r} failed: {exc}"
                ) from exc
        raise DerivedFormulaError(
            f"derived-formula rejected AST node {type(node).__name__}"
        )

    result = _walk(tree)
    if not isinstance(result, (int, float)):
        raise DerivedFormulaError(
            f"derived-formula evaluated to non-numeric {type(result).__name__}"
        )
    fresult = float(result)
    if fresult != fresult or fresult in (float("inf"), float("-inf")):
        raise DerivedFormulaError(
            f"derived-formula evaluated to non-finite value {fresult!r}"
        )
    return fresult
