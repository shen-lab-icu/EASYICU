"""The signed survival suite says which record timed its exposure.

A suite signed now times the exposure by its first record as present: its
disclosure carries that representation, and its plan, flow table and executed
design say so.  A suite signed before the representation existed timed the
exposure by its first record of any value.  It keeps its digest and its
request digest, and every reader-facing sentence keeps the words it was
reviewed with.

Synthetic study and seeded synthetic rows only (renal replacement therapy and
90-day mortality).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.contracts.executed_method_design import EXECUTED_METHOD_DESIGN_KEY
from easyicu.research_agent.contracts.manuscript_tables import MANUSCRIPT_TABLES_KEY
from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
from easyicu.research_agent.planning.family_spec import survival_template
from easyicu.webserver.research_launch_scientific import _source_concept_for_operational_column
from tests.support.survival_sealed import (
    run_signed_suite,
    sealed_request,
    sealed_survival,
    synthetic_survival_rows,
)


def _signed_before_the_onset(authority):
    """The same suite as signed before the onset representation existed.

    Such a suite bound ``<c>_first_time``; its digests and words depend on the
    missing representation alone, so the synthetic context keeps its column.
    """

    body = authority.model_dump(
        mode="json", exclude={"execution_contract_sha256", "exposure_onset_representation"}
    )
    return build_current_case_scientific_runtime_authority(body)


def _selected_design(context, authorities):
    request = sealed_request(context, authorities)
    spec = SimpleNamespace(labels={}, literature_design_decisions=[])
    selection = survival_template._design_selection(
        request, spec, method_keys=["strobe_2007"], roster=[]
    )
    selected = next(item for item in selection.candidates if item.disposition == "selected")
    return request, selected


def test_a_suite_signed_now_is_timed_by_its_first_present_record(tmp_path):
    context, authority = sealed_survival(tmp_path)
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)

    assert authority.exposure_onset_column == "rrt_onset_time"
    assert authority.exposure_onset_representation == "first_truthy_event_time"
    assert "first_truthy_event_time" in authorities.planning_contract_context()
    request, selected = _selected_design(context, authorities)
    assert request.sealed_suite.exposure_onset_representation == "first_truthy_event_time"
    assert "from the exposure's first records as present;" in selected.observation_window
    assert selected.assumptions[0].startswith(
        "Exposure timing is the first time the exposure source recorded the exposure as present;"
    )
    assert (
        "timed by its first record as present; exposure first recorded as present at or before "
        "time zero is excluded as prevalent."
    ) in selected.reviewable_plan[1]

    summary = run_signed_suite(authority, synthetic_survival_rows(), tmp_path / "out")
    design = summary[EXECUTED_METHOD_DESIGN_KEY]
    assert design["exposure_onset_representation"] == "first_truthy_event_time"
    tables = json.dumps(summary[MANUSCRIPT_TABLES_KEY])
    assert "Exposure status and time of first present record available" in tables
    assert "first recorded as present at or before hour 0" in tables


def test_a_suite_signed_before_the_onset_keeps_its_digests_and_its_words(tmp_path):
    context, signed = sealed_survival(tmp_path)
    legacy = _signed_before_the_onset(signed)
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=legacy)

    assert legacy.exposure_onset_representation is None
    assert "exposure_onset_representation" not in legacy.model_dump(mode="json")
    # A saved authority signed without the field still verifies on load.
    assert load_current_case_scientific_runtime_authority(legacy.model_dump(mode="json")) == legacy
    assert "exposure_onset_representation" not in authorities.planning_contract_context()
    request, selected = _selected_design(context, authorities)
    assert request.sealed_suite.exposure_onset_representation is None
    # The request keeps the digest it had before the field existed.
    assert "exposure_onset_representation" not in request.model_dump(mode="json")["sealed_suite"]
    assert "from the exposure's first recorded times;" in selected.observation_window
    assert selected.assumptions[0].startswith(
        "Exposure timing is the first recorded time of the exposure source;"
    )
    assert (
        "timed by its first record; exposure first recorded at or before time zero is excluded "
        "as prevalent."
    ) in selected.reviewable_plan[1]

    summary = run_signed_suite(legacy, synthetic_survival_rows(), tmp_path / "out")
    assert "exposure_onset_representation" not in summary[EXECUTED_METHOD_DESIGN_KEY]
    tables = json.dumps(summary[MANUSCRIPT_TABLES_KEY])
    assert "Exposure status and first recorded time available" in tables
    assert "first recorded at or before hour 0" in tables
    assert "as present" not in tables


def test_the_launch_reads_the_onset_as_its_exposure_concept():
    assert _source_concept_for_operational_column("rrt_onset_time", by_id={"rrt": {}}) == "rrt"
