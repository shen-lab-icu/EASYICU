"""EasyICU's stable, side-effect-free package facade.

``import easyicu`` intentionally performs no runtime configuration, cache
cleanup, data access, plotting setup, or research-agent initialization.
Top-level attributes are resolved lazily from their owning modules so the broad
EasyICU 1.x compatibility surface does not make every import pay for every
subsystem.  New code should prefer the compact stable surface documented by
``easyicu._public_api.STABLE_EXPORTS`` or import specialist submodules directly.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ._public_api import ALL_EXPORTS, PUBLIC_NAMES

__all__ = list(PUBLIC_NAMES)

# Kept as diagnostic state for callers that inspected the old package facade.
# Failures are recorded only when a lazy attribute is actually requested.
_IMPORT_ERRORS: dict[str, ImportError] = {}


def __getattr__(name: str) -> Any:
    """Resolve a public attribute from its single declared owner."""

    target = ALL_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    try:
        module = import_module(module_name)
        value = getattr(module, attribute_name)
    except ImportError as exc:
        _IMPORT_ERRORS[name] = exc
        raise

    # Cache the resolved object exactly as a normal ``from ... import`` would.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy names to introspection without importing their owners."""

    return sorted({*globals(), *ALL_EXPORTS})
