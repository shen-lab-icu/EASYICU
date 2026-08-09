"""What an effect scale means to a figure: its null, and how its axis behaves.

Two renderers now draw estimates on a declared effect scale, and both need the
same two facts: where the no-effect line goes, and whether the scale is
multiplicative (so equal ratios in either direction are equally far from the
null on the axis). Keeping one copy of that in each renderer is how the two
drift -- the second copy grows a scale the first never learns, and a reader
comparing the two figures sees the same estimate drawn against different rules.

Abstention is deliberate. An unrecognised scale gets ``null_value=None``, and
the renderers draw no null line at all rather than assuming 1. Drawing the
wrong null is a claim about the result -- a difference plotted against a null
of 1 reads as significant when it crosses zero -- and a missing line is visible
to a reader while a wrong one is not.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "EffectScale",
    "describe_effect_scale",
]


#: Scales whose no-effect value is 1 and whose axis is multiplicative.
#:
#: The abbreviations are here because the producers really write them, not on
#: the chance that someone might: the deterministic robustness owner uppercases
#: the plan's declared measure, and a sweep of every recorded CSV column named
#: ``effect_scale`` (2026-07-31) found exactly three spellings in use --
#: ``odds_ratio`` (209 rows), ``OR`` (4) and ``odds_ratio_per_1_mmol_per_l``
#: (4). Before this, ``OR`` was unrecognised, so every robustness forest the
#: replay owner fed was drawn with no line at no effect.
_RATIO_SCALES = frozenset(
    {
        "odds_ratio",
        "hazard_ratio",
        "risk_ratio",
        "rate_ratio",
        "incidence_rate_ratio",
        "or",
        "hr",
        "rr",
        "irr",
    }
)

#: Scales whose no-effect value is 0 and whose axis is additive.
_DIFFERENCE_SCALES = frozenset(
    {
        "risk_difference",
        "mean_difference",
        "coefficient",
        "log_odds",
        "rd",
        "md",
    }
)


@dataclass(frozen=True, slots=True)
class EffectScale:
    """The declared scale, and what a figure may assume about it."""

    name: str
    #: The no-effect value, or ``None`` when the scale is not recognised.
    null_value: float | None
    #: Whether equal ratios either side of the null are equally far from it.
    #: Only ever ``True`` for a recognised ratio scale; an unrecognised scale
    #: is not assumed to be multiplicative any more than it is assumed to be
    #: null at 1.
    multiplicative: bool

    @property
    def recognised(self) -> bool:
        return self.null_value is not None


def describe_effect_scale(effect_scale: str) -> EffectScale:
    """Return what may be assumed about ``effect_scale``, assuming nothing else.

    A unit qualifier does not change the scale. Producers write the exposure
    unit into the name compositionally -- ``odds_ratio_per_1_mmol_per_l`` is a
    real recorded value -- and an odds ratio per unit is still an odds ratio,
    null at 1 and multiplicative. So the leading term before ``_per_`` decides
    when the whole token is unrecognised. This is the producers' own naming,
    not a guess at what a name might mean: a token with no recognised head
    stays unrecognised rather than matching on a fragment.
    """

    name = str(effect_scale or "").strip()
    key = name.lower()
    if key not in _RATIO_SCALES and key not in _DIFFERENCE_SCALES:
        head, separator, _ = key.partition("_per_")
        if separator and head:
            key = head
    if key in _RATIO_SCALES:
        return EffectScale(name=name, null_value=1.0, multiplicative=True)
    if key in _DIFFERENCE_SCALES:
        return EffectScale(name=name, null_value=0.0, multiplicative=False)
    return EffectScale(name=name, null_value=None, multiplicative=False)
