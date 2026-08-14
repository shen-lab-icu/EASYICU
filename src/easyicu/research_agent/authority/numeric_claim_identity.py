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

    lower = bool(
        re.search(r"(?:ci|interval)_(?:95_)?(?:low|lower|lcl)(?:_|$)", normalized)
        or re.search(r"(?:ci|interval)(?:_|\[|$).*\[0\]$", source)
    )
    upper = bool(
        re.search(r"(?:ci|interval)_(?:95_)?(?:high|upper|ucl)(?:_|$)", normalized)
        or re.search(r"(?:ci|interval)(?:_|\[|$).*\[1\]$", source)
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
        scale = coerce_numeric_effect_scale(declared_effect_scale)
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
        self.effect_scale = inferred_scale or (
            declared_scale if self.estimand is not None else None
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
