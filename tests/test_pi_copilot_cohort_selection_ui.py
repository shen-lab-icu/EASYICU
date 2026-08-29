"""Frontend ownership contracts for host-confirmed primary cohorts."""

from pathlib import Path


STATIC = Path("src/easyicu/webserver/static")


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


def test_cohort_selection_owner_is_wired_before_guided_pi_shell() -> None:
    index = _read("index.html")

    assert "css/guided-pi-cohort-eligibility.css" in index
    assert "js/screens-guided-pi-cohort-eligibility.js" in index
    assert index.index("js/screens-guided-pi-cohort-eligibility.js") < index.index(
        "js/screens-guided-pi.js"
    )


def test_cohort_selection_renders_exact_server_contracts() -> None:
    owner = _read("js/screens-guided-pi-cohort-eligibility.js")
    shell = _read("js/screens-guided-pi.js")
    api = _read("js/api.js")

    assert "option.primary_cohort_contract" in owner
    assert 'data-gpi-cohort-event="${esc(option.selection_event_id)}"' in owner
    assert "actionFromEvent" in owner
    assert "confirmPiCopilotCohortEligibility" in shell
    assert "confirmPiCopilotCohortEligibility" in api
    assert "/cohort-eligibility-selection" in api
    assert "message.includes" not in owner
    assert "message.match" not in owner


def test_cohort_selection_css_and_event_marker_stay_route_owned() -> None:
    css_owner = STATIC / "css" / "guided-pi-cohort-eligibility.css"
    js_owner = STATIC / "js" / "screens-guided-pi-cohort-eligibility.js"

    assert ".gpi-cohort-eligibility" in css_owner.read_text(encoding="utf-8")
    assert "data-gpi-cohort-option" in js_owner.read_text(encoding="utf-8")
    for path in (STATIC / "css").glob("*.css"):
        if path != css_owner:
            assert ".gpi-cohort-eligibility" not in path.read_text(encoding="utf-8")
    for path in (STATIC / "js").glob("*.js"):
        if path != js_owner:
            assert "data-gpi-cohort-option" not in path.read_text(encoding="utf-8")
