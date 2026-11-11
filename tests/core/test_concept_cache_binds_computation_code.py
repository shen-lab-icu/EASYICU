"""A concept cache written by older computation code is never served (P1-2).

11-11 external review: ``CONCEPT_COMPUTATION_VERSION`` stayed at 2026-11-03.1
while computation fixes landed (HiRID millisecond clock, #668/#713/#732/#734/
#741), so an upgraded install served a stale cache: HiRID fluid 580 mL cached
vs 480 mL recomputed. Every concept cache key now binds the computation
version AND a digest of the code that computes concept values. Rows are
synthetic (``easyicu.synthetic_sources``); no patient data.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from easyicu import api
from easyicu.api.concepts import clear_global_loader
from easyicu.concept import ConceptResolver
from easyicu.concept import definition_identity as identity
from easyicu.concept.schema import ConceptDictionary

SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture(scope="module")
def hirid_root(tmp_path_factory):
    from easyicu.synthetic_sources import generate_source

    root = tmp_path_factory.mktemp("p12-synthetic") / "hirid"
    generate_source(root, "hirid", stays=3)
    yield root
    clear_global_loader()


def _load(root: Path, cache_dir: Path) -> pd.DataFrame:
    clear_global_loader()
    return api.load_concept_cached(
        "map", "hirid", root, cache_dir=cache_dir, verbose=False,
        concept_workers=1, parallel_workers=1,
    )


def _entries(cache_dir: Path) -> set[str]:
    return {path.name for path in cache_dir.glob("*.parquet")}


@pytest.mark.parametrize("change", ["code_digest", "computation_version"])
def test_an_upgrade_recomputes_instead_of_serving_the_old_cache(
    monkeypatch, tmp_path, hirid_root, change,
):
    cache_dir = tmp_path / "cache"
    old = _load(hirid_root, cache_dir)
    assert not old.empty and old["map"].notna().any()
    assert len(_entries(cache_dir)) == 1

    # The "new release" computes MAP differently.
    real_load = api.load_concepts
    calls = []

    def new_computation(**kwargs):
        calls.append(kwargs["concepts"])
        result = real_load(**kwargs)
        result = result.copy()
        result["map"] = result["map"] + 7.0
        return result

    monkeypatch.setattr(api, "load_concepts", new_computation)

    # Same identity: the cache is what answers (this is exactly the defect
    # a missed version bump caused before the code digest existed).
    assert _load(hirid_root, cache_dir)["map"].tolist() == old["map"].tolist()
    assert calls == []

    if change == "code_digest":
        monkeypatch.setattr(identity, "concept_code_digest", lambda: "sha256:" + "f" * 64)
    else:
        monkeypatch.setattr(identity, "CONCEPT_COMPUTATION_VERSION", "2099-01-01.1")

    new = _load(hirid_root, cache_dir)
    assert calls == [["map"]], "the stale entry was served instead of recomputing"
    assert new["map"].tolist() == (old["map"] + 7.0).tolist()
    assert len(_entries(cache_dir)) == 2
    # And the new entry is then reused under the new identity.
    assert _load(hirid_root, cache_dir)["map"].tolist() == new["map"].tolist()
    assert calls == [["map"]]


def test_every_cache_identity_binds_the_code_digest(monkeypatch, tmp_path):
    dictionary = ConceptDictionary.from_payload({
        "peep_set": {"sources": {"miiv": [{"table": "chartevents", "ids": [220339]}]},
                     "min": 0, "max": 40, "unit": "cmH2O"},
    })
    dict_file = tmp_path / "dict.json"
    dict_file.write_text("{}", encoding="utf-8")
    before = (
        ConceptResolver(dictionary, cache_dir=tmp_path).dictionary_signature,
        identity.dictionary_files_fingerprint(None),
        identity.dictionary_files_fingerprint(dict_file),
        identity.optional_dictionary_files_fingerprint(dictionary),
    )
    monkeypatch.setattr(identity, "concept_code_digest", lambda: "sha256:" + "0" * 64)
    after = (
        ConceptResolver(dictionary, cache_dir=tmp_path).dictionary_signature,
        identity.dictionary_files_fingerprint(None),
        identity.dictionary_files_fingerprint(dict_file),
        identity.optional_dictionary_files_fingerprint(dictionary),
    )
    assert all(old != new for old, new in zip(before, after)), (before, after)
    assert after[1].startswith("packaged:") and after[2].startswith("files:")


def _copy_covered(destination: Path) -> None:
    root = identity._package_root()
    for name, path in identity._concept_code_files(root):
        if path is not None:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def test_the_code_digest_is_deterministic_and_follows_one_byte(tmp_path):
    current = identity.concept_code_digest()
    assert current.startswith("sha256:")
    assert identity.concept_code_digest() == current
    assert identity._code_digest(identity._package_root()) == current
    assert identity.concept_computation_identity() == (
        f"{identity.CONCEPT_COMPUTATION_VERSION}+{current}"
    )

    # Another process importing the same code gets the same digest.
    probe = subprocess.run(
        [sys.executable, "-c",
         "from easyicu.concept.definition_identity import concept_code_digest as d; print(d())"],
        capture_output=True, text=True, check=True,
        env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"},
    )
    assert probe.stdout.strip() == current

    # Location does not matter, content does.
    copy = tmp_path / "easyicu"
    _copy_covered(copy)
    assert identity._code_digest(copy) == current

    # A file outside the covered set does not change it.
    (copy / "webserver").mkdir()
    (copy / "webserver" / "page.py").write_text("x = 1\n", encoding="utf-8")
    assert identity._code_digest(copy) == current

    # One byte in a covered computation file does.
    for relative in ("utils/time_units.py", "concept/hirid_fluid_counters.py", "datasource.py"):
        target = copy / relative
        original = target.read_bytes()
        target.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        assert identity._code_digest(copy) != current, relative
        target.write_bytes(original)
        assert identity._code_digest(copy) == current

    # A covered file that disappears changes it too.
    (copy / "scores" / "sofa2.py").unlink()
    assert identity._code_digest(copy) != current


def test_without_readable_sources_the_digest_falls_back_to_the_versions(monkeypatch, tmp_path):
    empty = tmp_path / "easyicu"
    (empty / "concept").mkdir(parents=True)
    (empty / "concept" / "definition_identity.pyc").write_bytes(b"\0")
    fallback = identity._code_digest(empty)
    assert fallback.startswith("fallback:")
    assert identity._code_digest(empty) == fallback
    assert identity._code_digest(None) == fallback
    monkeypatch.setattr(identity, "CONCEPT_COMPUTATION_VERSION", "2099-01-01.1")
    assert identity._code_digest(empty) != fallback
    monkeypatch.undo()
    monkeypatch.setattr(identity, "_package_version", lambda: "9.9.9")
    assert identity._code_digest(empty) != fallback


_IMPORT_PROBE = """
import json, sys, tempfile
from pathlib import Path
from easyicu.synthetic_sources import generate_source
from easyicu import load_concepts
root = Path(tempfile.mkdtemp(dir=sys.argv[1]))
for db in ("hirid", "aumc", "sic"):
    generate_source(root / db, db, stays=3)
    load_concepts(["map", "urine", "sofa"], database=db, data_path=root / db,
                  merge=False, verbose=False, concept_workers=1, parallel_workers=1)
print(json.dumps(sorted(
    module.__name__ for module in list(sys.modules.values())
    if getattr(module, "__name__", "").startswith("easyicu") and getattr(module, "__file__", None)
)))
"""

#: Imported while loading concepts but cannot change a value: package
#: re-exports, identity/receipt/state plumbing, pooling and runtime tuning.
_NOT_VALUE_BEARING = {
    "easyicu",
    "easyicu._public_api",
    "easyicu.content_identity",
    "easyicu.data",
    "easyicu.export_publication",
    "easyicu.resolver_errors",
    "easyicu.resolver_pool",
    "easyicu.state_paths",
    "easyicu.storage_seal_publication",
    "easyicu.synthetic_source_tables",
    "easyicu.synthetic_source_values",
    "easyicu.synthetic_sources",
}


def test_every_module_a_concept_load_imports_is_covered_or_declared(tmp_path):
    probe = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, str(tmp_path)],
        capture_output=True, text=True, check=True,
        env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin",
             "TMPDIR": str(tmp_path), "HOME": str(tmp_path)},
    )
    modules = json.loads(probe.stdout.strip().splitlines()[-1])
    covered_dirs = {f"easyicu.{name}" for name in identity.CONCEPT_CODE_DIRECTORIES}
    covered_modules = {
        "easyicu." + name.removesuffix(".py") for name in identity.CONCEPT_CODE_MODULES
    }
    uncovered = [
        name for name in modules
        if name not in covered_modules
        and name not in _NOT_VALUE_BEARING
        and not name.startswith("easyicu.runtime")
        and not any(name == base or name.startswith(base + ".") for base in covered_dirs)
    ]
    assert "easyicu.concept.definition_identity" in modules
    assert uncovered == [], (
        "a concept load imports modules outside the code digest; add them to "
        "CONCEPT_CODE_MODULES/DIRECTORIES or declare them not value-bearing"
    )
