"""A prior-art label weighs broad recall and how concrete the construct is.

A pairing with no exact-phrase hit but hundreds of broad hits is crowded, not
sparse; few broad hits remain a gap; a direct same-topic hit is already done;
and a vague construct is never called a gap.
"""

from __future__ import annotations


def test_label_prior_art_high_broad_count_blocks_false_sparse() -> None:
    # The bug: a heavily-studied pairing returns 0 on the over-specific exact
    # phrase but hundreds on broad recall; it must NOT be called sparse/gap.
    from easyicu.research_agent.discovery.idea_mining_priorart import _label_prior_art

    label = _label_prior_art(
        broad_count=300,
        exact_count=0,
        direct_same_topic_count=0,
        has_specific_differentiator=True,
    )
    assert label == "crowded_but_differentiable"
    assert label not in ("sparse", "apparently_gap")


def test_label_prior_art_genuinely_sparse_still_sparse() -> None:
    from easyicu.research_agent.discovery.idea_mining_priorart import _label_prior_art

    # Few broad hits and no exact hits remains a genuine gap/sparse signal.
    gap = _label_prior_art(
        broad_count=3,
        exact_count=0,
        direct_same_topic_count=0,
        has_specific_differentiator=True,
    )
    assert gap == "apparently_gap"
    sparse = _label_prior_art(
        broad_count=3,
        exact_count=0,
        direct_same_topic_count=0,
        has_specific_differentiator=False,
    )
    assert sparse == "sparse"


def test_label_prior_art_direct_hit_still_already_done() -> None:
    from easyicu.research_agent.discovery.idea_mining_priorart import _label_prior_art

    assert (
        _label_prior_art(broad_count=300, exact_count=10, direct_same_topic_count=2)
        == "already_done"
    )


def test_construct_is_vague_detection() -> None:
    from easyicu.research_agent.discovery.idea_mining_priorart import (
        _construct_is_vague,
    )

    # vague: decorator/method shells with no substantive clinical noun
    assert _construct_is_vague("robust multiparametric clinical scores") is True
    assert _construct_is_vague("marker") is True
    assert _construct_is_vague("novel biomarkers") is True
    assert _construct_is_vague("") is True
    # concrete: a real measurable construct survives
    assert _construct_is_vague("serum lactate") is False
    assert _construct_is_vague("urea-to-creatinine ratio") is False
    assert _construct_is_vague("physiologic marker") is False  # "physiologic" survives


def test_label_prior_art_vague_construct_blocks_false_sparse() -> None:
    from easyicu.research_agent.discovery.idea_mining_priorart import _label_prior_art

    # Even with 0 broad/exact hits, a vague construct cannot be a sparse gap.
    assert (
        _label_prior_art(
            broad_count=0,
            exact_count=0,
            direct_same_topic_count=0,
            has_specific_differentiator=True,
            construct_is_concrete=False,
        )
        == "crowded_but_differentiable"
    )
    # A concrete construct with genuinely low counts is still a real gap.
    assert (
        _label_prior_art(
            broad_count=2,
            exact_count=0,
            direct_same_topic_count=0,
            has_specific_differentiator=True,
            construct_is_concrete=True,
        )
        == "apparently_gap"
    )
