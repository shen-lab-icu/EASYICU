"""The surface a generated analysis script imports for its own plumbing.

Deliberately NOT under ``methods/``.  That package is the deterministic
statistical-kernel inventory, and every module in it must be reachable by a
declared route -- either a host importer or an entry in
``CURATED_METHOD_KERNELS``, which is ranked as a *method resource* per
analysis family.  This module applies to every family and is not a method,
so declaring it there would crowd out statsmodels/lifelines in exactly the
way ``temporal_features`` is documented to have done.  Its route is this
file being named in the Coder prompt; nothing in the host imports it.

Measured on 2026-07-30 over the 408 recorded generated ``analysis.py`` files:
236 of them (58%) contain a hand-written ``json_default``/``to_jsonable``, and
242 (59%) a hand-written ``sha256_file``.  They were not being creative -- the
Coder prompt pasted the body of ``to_jsonable`` and told the model to "define
this helper near the top of EVERY script".  Twenty lines below, the same prompt
tells it to *import* ``strict_numeric_input`` from
:mod:`easyicu.research_agent.methods.descriptive_inputs`, and 239 scripts duly
import it.  The container can import this package; the pasted body was never a
capability limit, only an instruction.

Every line the model does not write is a line it cannot get wrong.  Both
recorded module-level ``NameError`` deaths -- ``hashlib`` in fresh22 and
``manifest`` in the 2026-07-30 H1 canary -- were inside exactly this
hand-copied plumbing, not inside any analysis.

``sha256_file`` is deliberately not re-exported here: it already has an owner in
:mod:`easyicu.research_agent.authority.provenance`, and a second name for one
function is how two copies start.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

__all__ = ["to_jsonable", "write_json"]


def to_jsonable(value: Any) -> Any:
    """Coerce a numpy/pandas scalar into something ``json.dump`` accepts.

    Pass it as ``default=to_jsonable``.  Non-finite floats become ``None``
    because JSON has no NaN or Infinity; a value this cannot place becomes its
    string form rather than raising, so a summary is still written when one
    unexpected cell appears.
    """

    import numpy as np
    import pandas as pd

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _sanitized(value: Any) -> Any:
    """Walk a payload and replace every non-finite float with ``None``.

    ``default=`` alone cannot do this.  ``numpy.float64`` subclasses Python
    ``float``, so :func:`json.dump` serialises it directly and never consults
    the hook -- a NaN estimate then lands in the file as the bare token ``NaN``,
    which is not JSON and which a strict reader rejects.  The pasted helper the
    Coder prompt shipped had this hole for as long as it was pasted; it
    promised "only Python primitives" and delivered invalid JSON on the one
    case that matters most, a model that did not converge.
    """

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitized(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` as UTF-8 JSON, with every non-finite float as ``null``."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(
            _sanitized(payload),
            handle,
            indent=2,
            default=to_jsonable,
            ensure_ascii=False,
            allow_nan=False,
        )
    return destination
