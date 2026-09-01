"""Truthfulness guards for native WebApp real-data workflows."""

from pathlib import Path


STATIC_JS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "easyicu"
    / "webserver"
    / "static"
    / "js"
)


def _js(name: str) -> str:
    return (STATIC_JS / name).read_text(encoding="utf-8")


def test_real_extraction_never_falls_through_to_demo_completion() -> None:
    source = _js("screens-extraction.js")

    assert "} else if (!real) {" in source
    assert "Demo mode intentionally uses a seeded, in-browser completion." in source
    assert "Real extraction could not start." in source
    assert "Demo / offline fallback" not in source


def test_real_conversion_and_scan_fail_closed_without_local_runtime() -> None:
    source = _js("screens-extraction.js")

    assert "Real conversion could not start because the local job API" in source
    assert "convResult = { converted: CONV_STEPS.length" not in source
    assert "setInterval(() => {\n        convDone++" not in source
    assert "exScanError = 'scan_api_unavailable'" in source
    assert "this screen will not guess a real data layout" in source
    assert "setTimeout(() => { exReal = 'scanresult'" not in source


# --------------------------------------------------------------------------
# Progress queues: a row may only claim work the app actually observed.
# --------------------------------------------------------------------------
def test_seeded_task_animation_is_refused_in_real_mode() -> None:
    """`streamTasks` invents row states and per-row durations on a timer.

    Real mode must not run it: the caller's real work happens inside `done`,
    so ticking rows first asserts work that has not started. Deleting the
    guard below must turn this test red.
    """
    source = _js("screens-guided.js")

    assert "if (realMode()) { markTasksIndeterminate(sel); done(); return; }" in source
    # The indeterminate state must not carry a fabricated duration.
    assert "function markTasksIndeterminate" in source
    marker = source.split("function markTasksIndeterminate", 1)[1].split("function streamTasks", 1)[0]
    assert "d.textContent = ''" in marker


def test_scan_progress_does_not_pin_steps_it_cannot_observe() -> None:
    """/api/data/scan reports no per-step progress, so no step may show done."""
    source = _js("screens-extraction.js")

    scanning = source.split("function scanningState()", 1)[1].split("function ", 1)[0]
    assert "i === 0 ? 'done'" not in scanning
    assert 'data-progress-source="live-indeterminate"' in scanning


def test_queue_rows_declare_whether_they_are_live_or_scripted() -> None:
    source = _js("screens-guided.js")

    assert source.count('data-progress-source="${realMode()') >= 4


def test_no_fabricated_conversion_failure_state() -> None:
    """A failure card that names a cause the app never observed is not a UI."""
    source = _js("screens-extraction.js")

    assert "convFailState" not in source
    assert "labevents.valueuom" not in source
    assert "convfail" not in source


# --------------------------------------------------------------------------
# The question of record is the user's own sentence.
# --------------------------------------------------------------------------
def test_guided_submits_the_users_question_not_a_template() -> None:
    """`frameFor()` is a proposal. Only `submittedQuestion()` may be bound.

    Regression: the guided run used to submit
    `frameFor(branch) || BRANCH[branch].chip` — a template string built from
    module defaults — as the run's question, which the backend then wrote into
    the run manifest and bound evidence to.
    """
    source = _js("screens-guided.js")

    assert "function submittedQuestion()" in source
    assert "const capturedQuestion = submittedQuestion();" in source
    # No submission path may take the template directly.
    assert "frameFor(capturedBranch)" not in source
    assert "question: branch && BRANCH[branch] ? (frameFor(branch)" not in source


def test_template_wording_requires_explicit_acceptance() -> None:
    source = _js("screens-guided.js")

    assert "acceptedFrame" in source
    assert "tok === '@useFrame'" in source
    # Accepting is the ONLY way a template replaces the user's words.
    body = source.split("function submittedQuestion()", 1)[1].split("function stripQuotes", 1)[0]
    assert "if (userQuestion && !acceptedFrame) return userQuestion;" in body


def test_use_my_own_wording_actually_restores_the_users_words() -> None:
    """The chip used to map to a token literally named @noop and do nothing."""
    source = _js("screens-guided.js")

    handler = source.split("tok === '@noop'", 1)[1].split("return;", 1)[0]
    assert "acceptedFrame = false" in handler


def test_unroutable_free_text_updates_the_question_of_record() -> None:
    """"I'm not studying death, my outcome is AKI" must not be echoed away."""
    source = _js("screens-guided.js")

    assert "function replaceUserQuestion" in source
    assert "if (bindable && replaceUserQuestion(v)) {" in source
