"""A numpy-serialization repair must not disable a caller's fail-closed check.

``_patch_json_dump_numpy_key_sanitizer`` rebinds ``json.dump``/``json.dumps``
on the **stdlib module**, so it applies to every module in the interpreter --
the generated script, and every EasyICU helper that script imports.

Measured on 2026-08-07 (pre-existing, found by delta-debugging an
order-dependent suite failure down to a single file): the sanitizer mapped
every non-finite value to ``None`` unconditionally, including for callers that
had passed ``allow_nan=False`` to demand the opposite.
``cohort_row_identity_sha256`` -- the published cohort-identity recipe that
generated analysis code is told to call -- relies on exactly that flag to fail
closed on a non-canonical identity value. After the repair preamble ran it
returned a digest for ``[1, nan]`` instead of raising.

So a repair for numpy dict keys was silently converting a fail-closed evidence
guard into a fail-open one, for the whole process, in the sandbox where real
analyses run. This is not a test-only artifact; the suite ordering was just how
it became visible.
"""

from __future__ import annotations

import ast
import json

import pytest

from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedMetadataError,
    cohort_row_identity_sha256,
)
from easyicu.research_agent.repairs.helpers import (
    _patch_json_dump_numpy_key_sanitizer,
)


_SCRIPT = """
import json
step_summary = {"score": float("nan")}
with open("step_summary.json", "w", encoding="utf-8") as handle:
    json.dump(step_summary, handle, allow_nan=False)
"""


def test_the_repair_no_longer_rebinds_the_stdlib_module():
    patched = _patch_json_dump_numpy_key_sanitizer(_SCRIPT)
    assert "_easyicu_json_module_v1.dump =" not in patched
    assert "_easyicu_json_module_v1.dumps =" not in patched


def test_the_repair_binds_this_scripts_own_call_sites():
    patched = _patch_json_dump_numpy_key_sanitizer(_SCRIPT)
    assert "_easyicu_json_dump_v1(step_summary, handle" in patched


def test_dumps_is_rewritten_before_dump_so_the_prefix_does_not_win():
    patched = _patch_json_dump_numpy_key_sanitizer(
        "import json\nprint(json.dumps({}))\n"
    )
    assert "_easyicu_json_dumps_v1({})" in patched
    assert "_easyicu_json_dump_v1s(" not in patched


def test_executing_the_repaired_script_leaves_the_module_untouched(
    tmp_path, monkeypatch
):
    """The end-to-end property: repair works, guard survives."""

    patched = _patch_json_dump_numpy_key_sanitizer(_SCRIPT)
    monkeypatch.chdir(tmp_path)
    exec(compile(patched, "<patched>", "exec"), {})

    # The repair did its job for the script that needed it ...
    assert json.loads((tmp_path / "step_summary.json").read_text()) == {"score": None}
    # ... without redefining serialization for anyone else.
    assert json.dumps.__name__ == "dumps"
    with pytest.raises(ValueError):
        json.dumps({"a": float("nan")}, allow_nan=False)
    with pytest.raises(MaterializedMetadataError):
        cohort_row_identity_sha256([1, float("nan")])


def test_an_aliased_import_is_left_alone_rather_than_half_repaired():
    """A missed repair fails the step; a partial one would corrupt output."""

    patched = _patch_json_dump_numpy_key_sanitizer("import json as j\nj.dumps({})\n")
    assert "j.dumps({})" in patched


def test_repair_output_never_assigns_to_an_imported_module_attribute():
    """Static guard against process-global stdlib/third-party monkeypatches."""

    tree = ast.parse(_patch_json_dump_numpy_key_sanitizer(_SCRIPT))
    imported_modules = {
        alias.asname or alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.asname or node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        for node in node.names
    )
    mutations = [
        target
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        for target in (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id in imported_modules
    ]
    assert mutations == []
