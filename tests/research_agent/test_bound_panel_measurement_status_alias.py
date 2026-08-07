from __future__ import annotations

from easyicu.research_agent.repair_registry import (
    RepairClass,
    automatic_repair_allowed,
    repair_metadata_for,
)
from easyicu.research_agent.repairs.source import _deterministic_runner_repair
from easyicu.research_agent.repairs.typed_input import (
    patch_bound_panel_measurement_status_alias,
)


RUN_LOG = (
    "RuntimeError: Required measurement-status columns are unavailable; "
    "refusing partial feature panel: "
    "['alb_first_measured', 'hr_first_measured']"
)


CODE = '''
PANEL_KEY = "dataset:time_aligned_feature_panel"
FEATURE_NAMES = ("alb_first", "hr_first")
registered_status_columns = tuple(
    f"{feature}_measured" for feature in FEATURE_NAMES
)
'''

LOWERCASE_CODE = '''
PANEL_KEY = "dataset:time_aligned_feature_panel"
feature_names = ["alb_first", "hr_first"]
status_columns = [f"{name}_measured" for name in feature_names]
'''


BOUND = {
    "dataset:time_aligned_feature_panel": {
        "product_contract": {
            "columns": ["stay_id", "alb_measured", "hr_measured"]
        }
    }
}


def test_alias_repair_uses_bound_panel_columns() -> None:
    repaired = patch_bound_panel_measurement_status_alias(
        CODE, RUN_LOG, resolved_input_bindings=BOUND
    )

    assert repaired != CODE
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    assert namespace["registered_status_columns"] == (
        "alb_measured",
        "hr_measured",
    )


def test_alias_repair_is_reachable_from_runner_repair_owner() -> None:
    repair = _deterministic_runner_repair(
        code=CODE,
        run_log=RUN_LOG,
        resolved_input_bindings=BOUND,
    )

    assert repair is not None
    assert repair[0] == "bound_panel_measurement_status_alias_v1"


def test_alias_repair_supports_lowercase_list_comprehension_shape() -> None:
    repaired = patch_bound_panel_measurement_status_alias(
        LOWERCASE_CODE, RUN_LOG, resolved_input_bindings=BOUND
    )

    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    assert namespace["status_columns"] == ["alb_measured", "hr_measured"]


def test_alias_repair_declines_without_exact_bound_aliases() -> None:
    assert (
        patch_bound_panel_measurement_status_alias(
            CODE,
            RUN_LOG,
            resolved_input_bindings={
                "dataset:time_aligned_feature_panel": {
                    "product_contract": {
                        "columns": ["stay_id", "alb_first_measured", "hr_measured"]
                    }
                }
            },
        )
        == CODE
    )


def test_alias_repair_declines_ambiguous_panel_key() -> None:
    ambiguous = CODE + '\nPANEL_KEY = "dataset:other_panel"\n'
    assert (
        patch_bound_panel_measurement_status_alias(
            ambiguous, RUN_LOG, resolved_input_bindings=BOUND
        )
        == ambiguous
    )


def test_alias_repair_is_structural_and_automatic() -> None:
    metadata = repair_metadata_for("bound_panel_measurement_status_alias_v1")

    assert metadata.repair_class is RepairClass.STRUCTURAL
    assert metadata.introduces_numbers is False
    assert automatic_repair_allowed(metadata.repair_id)
