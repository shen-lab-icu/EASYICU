from easyicu.research_agent.repairs.patch import (
    looks_like_executable_python,
    repair_code_excerpt,
)
from easyicu.research_agent.repairs.reasons import RepairPromptAuthority


def test_executable_candidate_detection_keeps_runtime_failure_paths():
    assert looks_like_executable_python("raise RuntimeError('repair me')")
    assert looks_like_executable_python("print('no artifact yet')")
    assert looks_like_executable_python("result = 1")
    assert looks_like_executable_python("pass")


def test_executable_candidate_detection_rejects_inert_or_invalid_payloads():
    for payload in (
        "{}",
        "[]",
        "plain_prose",
        "1 + 2",
        "return 1",
        "not python !",
        'f"Analysis complete; import pandas"',
        '"import " + "pandas"',
        "analysis_complete  # import pandas",
        "pd.DataFrame",
    ):
        assert not looks_like_executable_python(payload)


def test_repair_excerpt_keeps_module_audit_initialization_loop_and_guard():
    decoy_body = "\n".join(f"    unused_{index} = {'x' * 120!r}" for index in range(30))
    code = f"""
import os
import pandas as pd

def generic_decoy():
{decoy_body}
    return None

planner_declared_inputs = ["x_measured", "x_n"]
provenance_checks = []
provenance_failures = []
provenance_stems = set()
for declared_input in planner_declared_inputs:
    if declared_input.endswith("_measured"):
        provenance_stems.add(declared_input[:-9])
    if declared_input.endswith("_n"):
        provenance_stems.add(declared_input[:-2])
for provenance_stem in sorted(provenance_stems):
    provenance_checks.append({{"stem": provenance_stem, "status": "checked"}})
    if provenance_stem == "invalid":
        provenance_failures.append(provenance_stem)
if not provenance_checks:
    provenance_failures.append("no checks")
if provenance_failures:
    raise ValueError(provenance_failures)
first_scientific_sink = pd.DataFrame(provenance_checks)

step_summary = {{"padding": {'y' * 3000!r}}}
""".lstrip()
    authority = RepairPromptAuthority.create(
        typed_ticket=[
            {
                "reason": "provenance_audit_not_fail_closed",
                "issues": [
                    {
                        "failure_mode": (
                            "module_provenance_scope_not_proven_fail_closed"
                        ),
                        "helper_name": "<module>",
                    }
                ],
            }
        ]
    )

    excerpt = repair_code_excerpt(
        code, repair_metadata=authority.metadata(), char_limit=2_400
    )

    assert len(code) > 2_400
    assert len(excerpt) <= 2_400
    assert "import pandas as pd" in excerpt
    assert "provenance_checks = []" in excerpt
    assert "provenance_failures = []" in excerpt
    assert "for declared_input in planner_declared_inputs:" in excerpt
    assert "for provenance_stem in sorted(provenance_stems):" in excerpt
    assert "if not provenance_checks:" in excerpt
    assert "if provenance_failures:\n    raise ValueError" in excerpt
    assert "def generic_decoy" not in excerpt


def test_repair_excerpt_anchors_oversized_function_without_blind_truncation():
    prefix = "\n".join(f"    pre_{index} = {'p' * 100!r}" for index in range(70))
    suffix = "\n".join(f"    post_{index} = {'q' * 100!r}" for index in range(70))
    code = f"""
import os
import pandas as pd

def provenance_audit(frame):
    checks = [{{"status": "checked"}}]
    return {{"checks": checks}}, True

def main(frame):
{prefix}
    provenance_summary, provenance_ok = provenance_audit(frame)
    provenance_checks = provenance_summary.get("checks", [])
    if provenance_ok is not True:
        raise ValueError("provenance failed")
{suffix}
    return provenance_checks
""".lstrip()
    call_line = next(
        index
        for index, line in enumerate(code.splitlines(), start=1)
        if "provenance_summary, provenance_ok" in line
    )
    authority = RepairPromptAuthority.create(
        typed_ticket=[
            {
                "validator": "mechanical_code_preflight",
                "reason": "PROVENANCE_NOT_FAIL_CLOSED",
                "structured_reason": "provenance_audit_not_fail_closed",
                "detail": {
                    "issues": [
                        {
                            "failure_mode": (
                                "provenance_helper_result_not_immediately_guarded"
                            ),
                            "helper_name": "provenance_audit",
                            "call_line": call_line,
                            "following_guard_line": call_line + 1,
                        }
                    ]
                },
            }
        ]
    )

    excerpt = repair_code_excerpt(
        code, repair_metadata=authority.metadata(), char_limit=2_600
    )

    assert len(code) > 10_000
    assert len(excerpt) < 2_600
    assert "import pandas as pd" in excerpt
    assert "def provenance_audit(frame):" in excerpt
    assert 'return {"checks": checks}, True' in excerpt
    assert "provenance_summary, provenance_ok = provenance_audit(frame)" in excerpt
    assert 'provenance_checks = provenance_summary.get("checks", [])' in excerpt
    assert "if provenance_ok is not True:\n        raise ValueError" in excerpt
    assert "post_69" not in excerpt
