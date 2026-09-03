"""A patch that could not be posed must not cost a scientific repair.

Live E1 blocker, 2026-07-29, ``run_20260729T062855_175303``, step
``07_primary_adjusted_association`` -- the study's primary step::

    attempt 1 transport: {"mode": "full_rewrite", "state": "completed",
                          "after_code_size_bytes": 30105}
    attempt 2 transport: {"error_type": "CoderPromptBudgetError",
                          "provider_calls": 0, "state": "failed"}
    step_llm_repair_attempts   : 2 of 2   (exhausted)
    step_provider_call_attempts: 3 of 9   (6 unspent)

Attempt 1 produced a 30,105-byte script, so the *patch* prompt built around it
crossed its 30,000-byte envelope. Nothing was sent and nothing was answered,
yet the attempt was spent and the step died reported as ``repair_failed`` --
naming the science rather than the prompt that was never built.

The coordinator already answers this correctly twice: ``_must_skip_patch()``
when the patch is unaffordable, and ``CodePatchError`` when the answer is
unusable. Both fall through to the full-rewrite transport -- which carries its
own envelope, and which attempt 1 had just used successfully. A patch that
cannot be assembled is the third way to have no patch, and was the only one
that killed the attempt.

(The same run's step 06 died on a ``normalization_error`` shadow mismatch. That
one is NOT covered here: ``test_result_table_profile_overflow_remains_fail_closed``
shows that code also carries genuine registered-table overflows at
severity=error, so blocking on it is right in kind. The real inner cause is
undiagnosable from the manifest, which is why the blocking finding now carries
``mismatch_details``.)
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.repairs.coordination import (
    PatchTransportUnavailable,
    RepairCoordinator,
)


# --------------------------------------------------------------------------
# Step 07: the patch that could not be posed
# --------------------------------------------------------------------------


def _coordinator() -> RepairCoordinator:
    return RepairCoordinator(
        provider_budget=None,
        provider_category="concept_repair",
        normalize_script=lambda value: value,
        is_executable_script=lambda value: value.startswith("import "),
        finalize_script=None,
    )


def test_an_unposable_patch_falls_through_to_the_full_rewrite() -> None:
    """The live step-07 failure: the patch prompt exceeded its envelope.

    The full-rewrite transport carries its own envelope and is exactly what the
    *preceding* repair on that step had already used successfully, so there was
    a working path the whole time.
    """

    calls: list[str] = []

    def _patch_call() -> str:
        calls.append("patch")
        return "--- unreachable"

    def _full_rewrite(reason: str) -> str:
        calls.append(f"full_rewrite:{reason}")
        return "import pandas\n"

    result = _coordinator().repair(
        code="import os\n",
        patch_preflight=lambda: (_ for _ in ()).throw(
            PatchTransportUnavailable("prompt 30105 > 30000 bytes")
        ),
        patch_call=_patch_call,
        full_rewrite_call=_full_rewrite,
    )

    assert result.mode == "full_rewrite"
    assert result.code.strip() == "import pandas"
    assert "patch" not in calls, "the unposable patch must not be sent"
    assert any(call.startswith("full_rewrite:") for call in calls)


def test_the_reason_reaches_the_full_rewrite_instead_of_being_swallowed() -> None:
    """Why the patch was skipped is the one clue to the real owner.

    Reported as bare ``repair_failed`` it names the science; carrying the
    reason names the prompt that was never built.
    """

    seen: list[str] = []

    result = _coordinator().repair(
        code="import os\n",
        patch_preflight=lambda: (_ for _ in ()).throw(
            PatchTransportUnavailable("prompt 30105 > 30000 bytes")
        ),
        patch_call=lambda: "--- unreachable",
        full_rewrite_call=lambda reason: (seen.append(reason), "import pandas\n")[1],
    )

    assert result.mode == "full_rewrite"
    assert seen and "30105" in seen[0]


def test_a_preflight_that_fails_for_any_other_reason_still_raises() -> None:
    """Only "could not be posed" earns the fallback.

    Swallowing every preflight exception would turn a genuine host bug into a
    silent extra provider call.
    """

    with pytest.raises(ValueError, match="not a transport concern"):
        _coordinator().repair(
            code="import os\n",
            patch_preflight=lambda: (_ for _ in ()).throw(
                ValueError("not a transport concern")
            ),
            patch_call=lambda: "--- unreachable",
            full_rewrite_call=lambda reason: "import pandas\n",
        )


def test_a_passing_preflight_still_poses_the_patch_first() -> None:
    """The cheap path must stay the default.

    Only the *posing* is asserted here. What happens to an unusable answer is
    the pre-existing ``CodePatchError`` fallback, which this change does not
    touch -- and which is why the patch below is followed by a full rewrite
    rather than being accepted.
    """

    calls: list[str] = []

    _coordinator().repair(
        code="import os\n",
        patch_preflight=lambda: None,
        patch_call=lambda: (calls.append("patch"), "not a patch")[1],
        full_rewrite_call=lambda reason: (
            calls.append("full_rewrite"),
            "import numpy\n",
        )[1],
    )

    assert calls[0] == "patch", "a posable patch must still be tried first"
