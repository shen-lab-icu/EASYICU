"""A projection reads its own bytes; equal bytes are parsed once.

11-03: the Skills page's research-workflow catalog projected eleven packages
per request and parsed the same host modules for each, 2.5 s per load; on a
slow runner it re-rendered after the page audit opened a menu.  Synthetic
modules only.
"""

from __future__ import annotations

from easyicu.research_agent.skill_packages import host_sources
from easyicu.research_agent.skill_packages.host_sources import SourceDefinition, _Sources


def _module(root, body: str) -> None:
    (root / "pkg").mkdir(exist_ok=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "owner.py").write_text(body, encoding="utf-8")


def test_equal_bytes_share_one_tree_and_changed_bytes_are_read_afresh(tmp_path):
    _module(tmp_path, "def run():\n    return helper()\n\ndef helper():\n    return 1\n")
    first = _Sources(tmp_path).read("easyicu.pkg.owner")
    again = _Sources(tmp_path).read("easyicu.pkg.owner")
    assert first[2] is again[2]
    _module(tmp_path, "def run():\n    return 2\n")
    changed = _Sources(tmp_path)
    assert changed.read("easyicu.pkg.owner")[1] == b"def run():\n    return 2\n"
    assert changed.read("easyicu.pkg.owner")[2] is not first[2]
    assert [node.name for node in changed.read("easyicu.pkg.owner")[2].body] == ["run"]
    node = changed.definition(SourceDefinition("easyicu.pkg.owner", "run"))
    assert host_sources._names_in(node) == frozenset()
