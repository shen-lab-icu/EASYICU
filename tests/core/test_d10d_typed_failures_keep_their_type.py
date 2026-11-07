"""A storage refusal keeps its type through every wrapper on the extraction path (Codex delta10d).

The renal AKI bundle wrapped any component failure as
``RuntimeError("renal_aki_bundle_load_failed: ...")``. A storage refusal
(``StorageSourceChanged``, ``TableScanError``) or another typed value failure
therefore reached the caller as a generic error. Other wrappers made a
``ValueError`` or ``ConceptExtractionUnavailable`` of it, which readers treat
as "this source is not available". Every broad catch around a load now
re-raises :data:`easyicu.value_failures.TYPED_VALUE_FAILURES` before it
wraps. Rows are synthetic.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import easyicu
from easyicu.storage_binding import StorageSourceChanged
from easyicu.storage_layout import TableScanError


@pytest.fixture
def hirid(tmp_path):
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "hirid"
    generate_source(root, "hirid", stays=3)
    return root


@pytest.mark.parametrize(
    ("component", "refusal"),
    [
        ("kdigo_urine_input", StorageSourceChanged("storage_source_changed", "observations", "/x")),
        ("crrt_mode_input", TableScanError("table_scan_files_missing", "pharma", "listed output absent")),
    ],
)
def test_a_refused_renal_component_raises_the_refusal_not_a_bundle_failure(
    hirid, monkeypatch, component, refusal
):
    import easyicu.api as api
    from easyicu.api.concepts import clear_global_loader, load_concepts

    real = api.load_concepts

    def component_loader(*args, **kwargs):
        concepts = kwargs.get("concepts", args[0] if args else None)
        if isinstance(concepts, (list, tuple)) and list(concepts) == [component]:
            raise refusal
        return real(*args, **kwargs)

    monkeypatch.setattr(api, "load_concepts", component_loader)
    try:
        with pytest.raises(type(refusal)) as error:
            load_concepts(["aki_stage_reference"], database="hirid", data_path=str(hirid),
                          verbose=False)
    finally:
        clear_global_loader()
    assert error.value is refusal


# --- structural guard -------------------------------------------------------

_VALUE_PATH = ("concept/", "callbacks.py", "load_concepts.py", "scores/", "table/",
               "datasource.py", "api/", "patient_filter.py", "base.py")
_LOADS = {"load_table", "_safe_load_table", "_load_one_concept", "_load_single_concept",
          "change_interval", "load_concepts", "load_concepts_fn", "resolve_target_files",
          "_align_time_to_admission", "load_id_tbl", "_load_recursive_concept",
          "resolve_table_files", "load_renal_aki_bundle", "load_kdigo_aki", "load_circ_failure",
          "load_outcomes", "load_microbiology", "load_comorbidity"}
# The publication refusal is deliberately NativeExportPublicationError (its
# cause kept, with the run's storage_evidence).
_DELIBERATE = {("api/extraction.py", "extract_database")}


def _names(handler):
    if handler.type is None:
        return ["<bare>"]
    elts = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return [ast.unparse(e) for e in elts]


def _raises_another_type(handler):
    return any(
        isinstance(x, ast.Raise) and x.exc is not None
        for stmt in handler.body for x in ast.walk(stmt)
    )


def test_no_broad_catch_around_a_load_rewraps_a_typed_failure():
    root = Path(easyicu.__file__).parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if not rel.startswith(_VALUE_PATH):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            called = {
                x.func.attr if isinstance(x.func, ast.Attribute) else getattr(x.func, "id", "")
                for stmt in node.body for x in ast.walk(stmt) if isinstance(x, ast.Call)
            }
            if not called & _LOADS:
                continue
            owner = node
            while owner in parents and not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                owner = parents[owner]
            if (rel, getattr(owner, "name", "")) in _DELIBERATE:
                continue
            typed = False
            for handler in node.handlers:
                names = _names(handler)
                typed = typed or any(n.endswith("TYPED_VALUE_FAILURES") for n in names)
                broad = any(n in ("Exception", "BaseException", "<bare>") for n in names)
                if broad and not typed and _raises_another_type(handler):
                    offenders.append(f"{rel}:{handler.lineno}")
    assert offenders == []
