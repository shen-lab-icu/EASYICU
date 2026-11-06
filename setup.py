"""Build hooks only; all package metadata lives in pyproject.toml.

Every wheel and sdist carries ``easyicu/_build_identity.json``: the commit,
dirty flag and diff digest of the checkout it was built from, so an installed
release records which code it is (``easyicu.code_identity``). The stamp is
written into the build output, never into the source tree. A wheel built from
an sdist reuses the stamp the sdist carries. Without a checkout or a carried
stamp nothing is written, and the installed package reports its identity as
``unknown``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist

ROOT = Path(__file__).resolve().parent


def _code_identity_module():
    # Load the owner by path: the build must not import the whole package.
    spec = importlib.util.spec_from_file_location(
        "_easyicu_code_identity", ROOT / "src" / "easyicu" / "code_identity.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stamp_bytes(version: str):
    owner = _code_identity_module()
    payload = owner.build_stamp_payload(ROOT, version=version)
    if payload is None:
        return None, owner.BUILD_IDENTITY_FILE
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return text.encode("utf-8"), owner.BUILD_IDENTITY_FILE


class build_py(_build_py):
    def run(self):
        super().run()
        data, name = _stamp_bytes(self.distribution.get_version())
        target = Path(self.build_lib) / "easyicu" / name
        if data is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return
        carried = ROOT / "src" / "easyicu" / name
        if carried.is_file() and not target.is_file():
            # An sdist build: keep the stamp the sdist was made with.
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(carried.read_bytes())


class sdist(_sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        data, name = _stamp_bytes(self.distribution.get_version())
        if data is not None:
            target = Path(base_dir) / "src" / "easyicu" / name
            if target.is_symlink() or target.exists():
                target.unlink()
            target.write_bytes(data)


setup(cmdclass={"build_py": build_py, "sdist": sdist})
