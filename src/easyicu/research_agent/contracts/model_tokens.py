"""Canonical spelling for typed statistical-model contract tokens."""

from __future__ import annotations

import re
from typing import Any


def normalise_model_contract_token(value: Any) -> str:
    """Collapse presentation spelling without changing contract semantics."""

    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


__all__ = ["normalise_model_contract_token"]
