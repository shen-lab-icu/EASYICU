"""Compatibility alias for the canonical planning method-suite registry."""

from __future__ import annotations

if __name__ == "__main__":  # Preserve the documented legacy regeneration command.
    from .planning.analysis_method_suite import render_method_suite_markdown

    print(render_method_suite_markdown())
else:
    import sys as _sys

    from .planning import analysis_method_suite as _canonical

    _sys.modules[__name__] = _canonical
