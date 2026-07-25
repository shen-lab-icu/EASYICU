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

    ``easyicu.table.utils.upgrade_id`` converts coarse-to-fine and
    ``easyicu.io.data_utils.upgrade_id`` converts fine-to-coarse — the opposite
    of each other and of the canonical one. Forwarding either would invert what
    an existing caller computes, so they stay put and warn.
    """

    from easyicu.io import data_utils
    from easyicu.table import id_conversion
    from easyicu.table import utils as table_utils

    assert table_utils.change_id is not id_conversion.change_id
    assert data_utils.change_id is not id_conversion.change_id
    for module in (table_utils, data_utils):
        assert "deprecated" in (module.__doc__ or "").lower()


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
