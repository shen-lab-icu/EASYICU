from types import SimpleNamespace

from easyicu.concept import _requested_dependency_overlap


def _definition(*, depends_on=(), sub_concepts=()):
    return SimpleNamespace(
        depends_on=list(depends_on),
        sub_concepts=list(sub_concepts),
    )


def test_requested_dependency_overlap_finds_transitive_requested_descendants():
    dictionary = {
        "sofa": _definition(
            depends_on=("sofa_resp", "sofa_renal"),
            sub_concepts=("sofa_resp", "sofa_renal"),
        ),
        "sofa_resp": _definition(depends_on=("pafi",)),
        "sofa_renal": _definition(depends_on=("crea",)),
        "pafi": _definition(),
        "crea": _definition(),
    }

    assert _requested_dependency_overlap(
        dictionary,
        ["sofa", "sofa_resp", "crea"],
    ) == ["crea", "sofa_resp"]


def test_requested_dependency_overlap_keeps_independent_requests_parallel():
    dictionary = {
        "age": _definition(),
        "sex": _definition(),
        "death": _definition(),
    }

    assert _requested_dependency_overlap(dictionary, ["age", "sex", "death"]) == []
