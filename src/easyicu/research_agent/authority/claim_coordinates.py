"""The coordinate form a host claim uses to name one contrast of an exposure.

Readers see ``variable=level versus variable=reference``, the form descriptive
contrasts already use.  Both the association claim compiler and the
sensitivity-refit compiler name a categorical contrast through this one owner.
"""

from __future__ import annotations

import json
import re


def _coordinate_level(level: str) -> str:
    """A level as a JSON coordinate value; an integer token stays a number."""

    text = str(level)
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", text):
        return text
    return json.dumps(text, ensure_ascii=False)


def contrast_exposure_coordinate(variable: str, level: str, reference: str) -> str:
    """Name one contrast of a categorical exposure as a claim coordinate."""

    return (
        f"{variable}={_coordinate_level(level)} versus "
        f"{variable}={_coordinate_level(reference)}"
    )


__all__ = ["contrast_exposure_coordinate"]
