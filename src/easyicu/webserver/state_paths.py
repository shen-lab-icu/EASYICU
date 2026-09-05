"""Compatibility imports for the shared EasyICU state-root owner.

Web callers retain their existing import path. Policy and environment resolution
live in easyicu.state_paths so extensions and the runtime use the same roots.
"""

from easyicu.state_paths import exports_root, projects_root, state_root, user_home

__all__ = ["user_home", "state_root", "projects_root", "exports_root"]
