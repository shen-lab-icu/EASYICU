"""Deprecated ID-mapping compatibility module.

This module used to hold a ricu-shaped ID-metadata API — ``id_map``,
``id_origin``, ``id_windows`` and their helpers. **None of it was ever
operational.** Every entry point began with

    from .data_env import get_src_env, as_src_env

and ``easyicu.io.data_env`` has no ``as_src_env``, so the first line of the
first function raised ``ImportError``. ``id_orig_helper`` and ``id_win_helper``
additionally imported ``as_id_cfg`` from ``.table.convert``, a path that does
not exist from inside ``easyicu.io``. Nothing in the package called any of it,
which is why the breakage survived: there was no caller to fail.

Keeping several hundred lines that look like a working implementation is worse
than having none. A reader — human or agent — cannot tell by looking that this
code has never run, and a reviewer will spend real effort auditing it. So the
bodies are gone and the names now say what is true.

**What to use instead.** Identifier systems, their origins and their windows
are resolved inside the concept layer, from ``data-sources.json``'s ``id_cfg``.
Reach them through the supported API:

* :func:`easyicu.load_concepts` for clinical data on a chosen ID system;
* :func:`easyicu.io.data_load.load_id` to move a table between ID systems;
* ``DataSourceConfig.id_configs`` for the raw declaration
  (``.id`` / ``.start`` / ``.end`` / ``.table``).

Scheduled for removal in EasyICU 2.0.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "id_map",
    "id_map_helper",
    "id_orig_helper",
    "id_origin",
    "id_win_helper",
    "id_windows",
    "as_src_env",
]

_NEVER_OPERATIONAL = (
    "easyicu.io.id_mapping.{name}() was never operational: it imported "
    "'as_src_env' from easyicu.io.data_env, which does not define it, so every "
    "call raised ImportError on its first line. The implementation has been "
    "removed rather than left to look usable.\n\n"
    "Use easyicu.load_concepts() for data on a given ID system, "
    "easyicu.io.data_load.load_id() to move a table between ID systems, or "
    "read DataSourceConfig.id_configs for the raw id_cfg declaration. This "
    "module will be removed in EasyICU 2.0."
)


def _removed(name: str) -> "NotImplementedError":
    return NotImplementedError(_NEVER_OPERATIONAL.format(name=name))


def id_map(*args: Any, **kwargs: Any) -> Any:
    """Removed. See the module docstring."""

    raise _removed("id_map")


def id_map_helper(*args: Any, **kwargs: Any) -> Any:
    """Removed. See the module docstring."""

    raise _removed("id_map_helper")


def id_origin(*args: Any, **kwargs: Any) -> Any:
    """Removed. See the module docstring."""

    raise _removed("id_origin")


def id_orig_helper(*args: Any, **kwargs: Any) -> Any:
    """Removed. See the module docstring."""

    raise _removed("id_orig_helper")


def id_windows(*args: Any, **kwargs: Any) -> Any:
    """Removed. See the module docstring."""

    raise _removed("id_windows")


def id_win_helper(*args: Any, **kwargs: Any) -> Any:
    """Removed. See the module docstring."""

    raise _removed("id_win_helper")


def as_src_env(*args: Any, **kwargs: Any) -> Any:
    """Removed. See the module docstring."""

    raise _removed("as_src_env")
