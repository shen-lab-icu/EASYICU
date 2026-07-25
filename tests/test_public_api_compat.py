"""Contracts for the public API surface and its deprecated compatibility layer.

The 2026-07-29 review found a ricu-shaped low-level layer that nothing inside
the package called, that no document mentioned, and whose three call-signature
errors proved it had never run. Cleaning it up is a public-API change on a
released 1.0.0, so the rules are:

* the exported names keep importing;
* the deprecated ones say so, at call time, and say what replaces them;
* one operation has exactly one implementation.

That last rule is the point. The package had grown **three** ``change_id``
functions with three different argument orders and three different meanings for
"upgrade" — so an import resolving to the wrong one changed what the call did,
and a fix applied to the wrong file left the bug in place with the tests green.
"""

from __future__ import annotations

import importlib
import inspect
import warnings

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# The exported names still import
# ---------------------------------------------------------------------------


PUBLIC_NAMES = [
    # advanced low-level loaders (kept, now working)
    "load_src",
    "load_difftime",
    "load_id",
    "load_ts",
    "load_win",
    # canonical table operations
    "change_id",
    "upgrade_id",
    "downgrade_id",
    "cbind_tbl",
    "rbind_tbl",
    # deprecated compatibility surface
    "id_origin",
    "id_windows",
    "id_map",
]


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_every_exported_name_still_imports(name: str) -> None:
    """Removing a public name is a 2.0 change, not a cleanup."""

    import easyicu

    assert hasattr(easyicu, name), f"easyicu.{name} disappeared"


# ---------------------------------------------------------------------------
# One operation, one implementation
# ---------------------------------------------------------------------------


def test_the_canonical_upgrade_id_holds_the_plain_name_in_public_all() -> None:
    """``import *`` must hand over the canonical function, not the inverse one.

    ``easyicu.upgrade_id`` used to resolve to ``easyicu.io.data_utils`` — which
    converts the other way — while ``change_id`` and ``downgrade_id`` beside it
    were canonical, so the three disagreed about direction inside one
    namespace. Checking ``hasattr`` alone would not have caught it: the name
    existed either way. The listing has to name the same object the attribute
    binds, and has to survive ``data_utils`` failing to import.
    """

    import easyicu
    from easyicu.table import id_conversion

    assert "upgrade_id" in easyicu.__all__
    assert easyicu.__all__.count("upgrade_id") == 1
    assert easyicu.upgrade_id is id_conversion.upgrade_id
    assert set(easyicu.__all__) <= set(dir(easyicu))


def test_the_canonical_id_conversion_has_exactly_one_home() -> None:
    """``from easyicu.table import change_id`` must not be a coin flip.

    Diagnosing the 2026-07-29 finding needed ``inspect.getsourcefile`` to
    discover which of three implementations an import resolved to. Whatever the
    package exports and whatever ``easyicu.table`` exports must now be the one
    object defined in ``table.id_conversion``.
    """

    import easyicu
    import easyicu.table as table
    from easyicu.table import id_conversion

    for name in ("change_id", "upgrade_id", "downgrade_id"):
        canonical = getattr(id_conversion, name)
        assert getattr(table, name) is canonical, name
        assert getattr(easyicu, name) is canonical, name
        assert inspect.getsourcefile(canonical).endswith("table/id_conversion.py")


def test_the_diverging_implementations_are_marked_not_silently_forwarded() -> None:
    """The other two ``change_id``s mean different things and must keep saying so.

    Both deprecated ``upgrade_id``s convert fine-to-coarse (icustay to hadm);
    the canonical one converts coarse-to-fine. Forwarding either would invert
    what an existing caller computes, so they stay put and warn. The directions
    are asserted against real rows in ``test_id_conversion_directions`` below —
    this test only checks that no silent redirect was installed.
    """

    from easyicu.io import data_utils
    from easyicu.table import id_conversion
    from easyicu.table import utils as table_utils

    assert table_utils.change_id is not id_conversion.change_id
    assert data_utils.change_id is not id_conversion.change_id
    for module in (table_utils, data_utils):
        assert "deprecated" in (module.__doc__ or "").lower()


# ---------------------------------------------------------------------------
# What each implementation actually computes
# ---------------------------------------------------------------------------


# One hospital admission holding two ICU stays, one holding a single stay. This
# is the smallest shape that tells the directions apart: coarse-to-fine has to
# duplicate hadm 1's row, fine-to-coarse has to combine stays 10 and 11.
ID_MAP = pd.DataFrame({"hadm_id": [1, 1, 2], "stay_id": [10, 11, 20]})
BY_STAY = pd.DataFrame({"stay_id": [10, 11, 20], "hr": [80.0, 86.0, 90.0]})
BY_HADM = pd.DataFrame({"hadm_id": [1, 2], "hr": [80.0, 90.0]})


def test_canonical_upgrade_id_goes_coarse_to_fine_and_expands() -> None:
    from easyicu.table.id_conversion import upgrade_id

    result = upgrade_id(BY_HADM, ID_MAP, "hadm_id", "stay_id")

    assert list(result["stay_id"]) == [10, 11, 20]
    assert list(result["hr"]) == [80.0, 80.0, 90.0]  # hadm 1's row duplicated
    assert "hadm_id" not in result.columns


def test_canonical_downgrade_id_goes_fine_to_coarse_and_aggregates() -> None:
    from easyicu.table.id_conversion import downgrade_id

    result = downgrade_id(BY_STAY, ID_MAP, "stay_id", "hadm_id")

    assert list(result["hadm_id"]) == [1, 2]
    assert list(result["hr"]) == [83.0, 90.0]  # mean of stays 10 and 11
    assert "stay_id" not in result.columns


def test_legacy_table_utils_upgrade_id_keeps_its_opposite_direction() -> None:
    """It targets the *coarse* id — the reverse of the canonical ``upgrade_id``.

    It also annotates rather than aggregates: three stay rows stay three rows,
    now carrying a hadm_id, with both id columns present. Redirecting this to
    the canonical ``upgrade_id`` would turn a 3-row annotation into a different
    table entirely, which is why it is deprecated in place instead.
    """

    from easyicu.table.utils import upgrade_id

    with pytest.warns(DeprecationWarning):
        result = upgrade_id(BY_STAY, "hadm_id", ID_MAP, "stay_id")

    assert list(result["hadm_id"]) == [1, 1, 2]
    assert list(result["stay_id"]) == [10, 11, 20]
    assert list(result["hr"]) == [80.0, 86.0, 90.0]


def test_legacy_table_utils_downgrade_id_keeps_its_opposite_direction() -> None:
    """It expands coarse-to-fine — the reverse of the canonical ``downgrade_id``."""

    from easyicu.table.utils import downgrade_id

    with pytest.warns(DeprecationWarning):
        result = downgrade_id(BY_HADM, "stay_id", ID_MAP, "hadm_id")

    assert list(result["stay_id"]) == [10, 11, 20]
    assert list(result["hr"]) == [80.0, 80.0, 90.0]
    assert len(result) == 3  # expanded, not aggregated


def test_legacy_data_utils_upgrade_id_keeps_its_opposite_direction() -> None:
    """Fine-to-coarse, like ``table.utils``' and unlike the canonical one.

    It drops the stay id without combining rows, so hadm 1 appears twice — a
    third distinct result from a third function of the same name.
    """

    from easyicu.io.data_utils import upgrade_id

    with pytest.warns(DeprecationWarning):
        result = upgrade_id(BY_STAY, "stay_id", "hadm_id", ID_MAP)

    assert list(result["hadm_id"]) == [1, 1, 2]
    assert list(result["hr"]) == [80.0, 86.0, 90.0]
    assert "stay_id" not in result.columns


def test_the_three_upgrade_ids_still_disagree_and_that_is_the_point() -> None:
    """A single assertion of the whole hazard, so it cannot quietly go stale."""

    import easyicu
    from easyicu.io.data_utils import upgrade_id as data_utils_upgrade
    from easyicu.table.utils import upgrade_id as table_utils_upgrade

    canonical = easyicu.upgrade_id(BY_HADM, ID_MAP, "hadm_id", "stay_id")
    with pytest.warns(DeprecationWarning):
        legacy_table = table_utils_upgrade(BY_STAY, "hadm_id", ID_MAP, "stay_id")
    with pytest.warns(DeprecationWarning):
        legacy_data = data_utils_upgrade(BY_STAY, "stay_id", "hadm_id", ID_MAP)

    # canonical produces stay ids; both legacy ones produce hadm ids
    assert "stay_id" in canonical.columns and "hadm_id" not in canonical.columns
    assert "hadm_id" in legacy_table.columns and "stay_id" in legacy_table.columns
    assert "hadm_id" in legacy_data.columns and "stay_id" not in legacy_data.columns


# ---------------------------------------------------------------------------
# An id map that does not cover the data
# ---------------------------------------------------------------------------


PARTIAL_MAP = pd.DataFrame({"hadm_id": [1], "stay_id": [10]})
EMPTY_MAP = pd.DataFrame(
    {"hadm_id": pd.Series([], dtype="int64"), "stay_id": pd.Series([], dtype="int64")}
)


def test_an_empty_id_map_is_refused_rather_than_read_as_one_to_one() -> None:
    """No pairs means no fan-out on either side, which reads as one-to-one.

    The relation test then took the one-to-one branch, ``dict(zip(...))`` built
    an empty lookup, every id mapped to ``NaN``, and the original id column was
    dropped — a frame of measurements with no patient attached, returned
    without an error. A map that failed to load is the common way to get here.
    """

    from easyicu.table.id_conversion import (
        IdMapRelationError,
        change_id,
        classify_id_relation,
        downgrade_id,
        upgrade_id,
    )

    assert classify_id_relation(EMPTY_MAP, "stay_id", "hadm_id") == "empty"

    with pytest.raises(IdMapRelationError, match="no usable pairs"):
        change_id(BY_STAY, EMPTY_MAP, "stay_id", "hadm_id")
    with pytest.raises(IdMapRelationError, match="no usable pairs"):
        downgrade_id(BY_STAY, EMPTY_MAP, "stay_id", "hadm_id")
    with pytest.raises(IdMapRelationError, match="no usable pairs"):
        upgrade_id(BY_HADM, EMPTY_MAP, "hadm_id", "stay_id")


def test_ids_the_map_does_not_cover_stop_the_conversion_by_default() -> None:
    from easyicu.table.id_conversion import UnmappedIdError, change_id

    with pytest.raises(UnmappedIdError) as excinfo:
        change_id(BY_STAY, PARTIAL_MAP, "stay_id", "hadm_id")

    message = str(excinfo.value)
    assert "2 of 3 row(s)" in message
    assert "11" in message and "20" in message  # names the ids it could not map
    assert "on_unmapped" in message


def test_unmapped_rows_can_be_dropped_or_kept_when_the_caller_says_so() -> None:
    from easyicu.table.id_conversion import change_id

    dropped = change_id(BY_STAY, PARTIAL_MAP, "stay_id", "hadm_id", on_unmapped="drop")
    assert list(dropped["hr"]) == [80.0]

    kept = change_id(BY_STAY, PARTIAL_MAP, "stay_id", "hadm_id", on_unmapped="keep")
    assert len(kept) == 3
    assert kept["hadm_id"].isna().sum() == 2


def test_downgrade_id_does_not_let_groupby_delete_the_unmapped_rows() -> None:
    """``groupby`` drops null keys, so 'keep' has to mean keep.

    Before ``on_unmapped``, an incomplete map made ``downgrade_id`` return
    fewer rows than it was given with nothing to say so — the only trace was
    the row count, and a mean computed over the survivors.
    """

    from easyicu.table.id_conversion import downgrade_id

    partial = pd.DataFrame({"stay_id": [10, 11], "hadm_id": [1, 1]})

    kept = downgrade_id(BY_STAY, partial, "stay_id", "hadm_id", on_unmapped="keep")
    assert len(kept) == 2
    assert 90.0 in list(kept["hr"])  # stay 20 survives under a null hadm_id

    dropped = downgrade_id(BY_STAY, partial, "stay_id", "hadm_id", on_unmapped="drop")
    assert list(dropped["hadm_id"]) == [1]
    assert list(dropped["hr"]) == [83.0]


def test_an_unknown_unmapped_policy_is_rejected() -> None:
    from easyicu.table.id_conversion import change_id

    with pytest.raises(ValueError, match="on_unmapped must be one of"):
        change_id(BY_STAY, PARTIAL_MAP, "stay_id", "hadm_id", on_unmapped="ignore")


# ---------------------------------------------------------------------------
# The deprecated surface warns rather than misleading
# ---------------------------------------------------------------------------


def test_the_diverging_id_functions_warn_at_call_time() -> None:
    from easyicu.table.utils import downgrade_id

    with pytest.warns(DeprecationWarning, match="NOT the same operation"):
        downgrade_id(
            pd.DataFrame({"stay_id": [1], "value": [1.0]}),
            "hadm_id",
            pd.DataFrame({"stay_id": [1], "hadm_id": [10]}),
            "stay_id",
        )


def test_a_deprecated_helper_names_its_replacement() -> None:
    from easyicu.io.data_utils import id_origin

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        id_origin(
            pd.DataFrame({"stay_id": [1, 1], "t": pd.to_timedelta([1, 2], unit="h")}),
            "stay_id",
        )

    assert caught, "no warning emitted"
    message = str(caught[0].message)
    assert "deprecated" in message
    assert "2.0" in message


def test_a_function_that_never_worked_says_so_instead_of_pretending() -> None:
    """``id_mapping`` imported a name its own dependency does not define.

    Every entry point raised ImportError on its first line. Several hundred
    lines that look like a working implementation are worse than none: nobody
    reading them can tell, and a reviewer audits them for real.
    """

    from easyicu.io import id_mapping

    with pytest.raises(NotImplementedError) as excinfo:
        id_mapping.id_origin("miiv", "icustay")

    message = str(excinfo.value)
    assert "never operational" in message
    assert "load_concepts" in message


# ---------------------------------------------------------------------------
# The extraction path is not downstream of any of this
# ---------------------------------------------------------------------------


EXTRACTION_MODULES = ("api.py", "concept/*.py", "scores/*.py", "io/data_converter.py")
OFF_PATH_MODULES = (
    "io.data_load",
    "io.data_utils",
    "io.id_mapping",
    "table.utils",
)


def test_the_extraction_path_does_not_import_the_deprecated_layer() -> None:
    """Why prepared data does not have to be re-extracted after this change.

    ``load_concepts`` reaches raw tables through ``ConceptResolver``, never
    through the ricu-shaped ``load_*`` chain or the deprecated helpers, so
    changing them cannot change what an extraction produces. That is an
    argument about reachability, and it is only worth anything if it stays
    true — hence a test rather than a paragraph in a task log.
    """

    import ast
    import pathlib

    import easyicu

    root = pathlib.Path(easyicu.__file__).parent
    offenders = []
    scanned = 0
    for pattern in EXTRACTION_MODULES:
        for path in sorted(root.glob(pattern)):
            scanned += 1
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                else:
                    continue
                for name in names:
                    for off_path in OFF_PATH_MODULES:
                        if off_path in name:
                            offenders.append(f"{path.name} imports {name}")

    assert scanned > 10, f"glob matched too little to be meaningful ({scanned})"
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# Import hygiene
# ---------------------------------------------------------------------------


def test_importing_the_deprecated_modules_is_itself_quiet() -> None:
    """The warning belongs at the call, not at the import.

    ``easyicu/__init__`` imports these modules to re-export their names, so a
    module-level warning would fire on ``import easyicu`` for every user
    whether or not they touch the deprecated API.
    """

    for name in (
        "easyicu.io.data_utils",
        "easyicu.table.utils",
        "easyicu.io.id_mapping",
    ):
        module = importlib.import_module(name)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.reload(module)
        assert not [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ], name


def test_the_package_still_imports_without_deprecation_noise() -> None:
    import easyicu

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(easyicu)

    noisy = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not noisy, [str(w.message) for w in noisy]
