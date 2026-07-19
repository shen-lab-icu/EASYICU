"""Compatibility alias for the canonical planning capability registry."""

from __future__ import annotations

if __name__ == "__main__":  # Preserve the documented legacy regeneration command.
    from .planning.capability_registry import render_capability_matrix_markdown

    print(render_capability_matrix_markdown())
else:
    import sys as _sys

    from .planning import capability_registry as _canonical

    _sys.modules[__name__] = _canonical
