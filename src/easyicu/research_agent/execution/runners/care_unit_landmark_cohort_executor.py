"""Publish only a host-gated and sealed SIC landmark cohort."""
from __future__ import annotations

import json

from easyicu.research_agent.authority.case_duration_landmark import CaseDurationLandmarkAuthority
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.contracts.primary_cohort import HOST_BOUND_COHORT_METHOD, HOST_BOUND_COHORT_PRODUCT
from .host_bound_cohort_executor import host_bound_cohort_executor_code


def care_unit_landmark_cohort_executor_code(step, *, plan, authority, plausibility_scope=None):
    if not isinstance(authority, CaseDurationLandmarkAuthority):
        raise ValueError("signed case-duration authority is required")
    authority.validate_plan(plan)
    if step.method != "signed_care_unit_landmark_cohort" or step not in plan.steps:
        raise ValueError("landmark cohort step differs from the signed plan")
    copy_step = step.model_copy(update={"method": HOST_BOUND_COHORT_METHOD, "inputs": [],
                                        "planned_analysis_role": "auxiliary",
                                        "expected_outputs": [HOST_BOUND_COHORT_PRODUCT]})
    code = host_bound_cohort_executor_code(copy_step, plausibility_scope=plausibility_scope)
    plan_sha = canonical_sha256(plan.model_dump(mode="json"))
    contract_sha = authority.execution_contract_sha256
    verify = (
        'from easyicu.research_agent.intake.materialized_metadata import load_verified_materialized_cohort_authority\n'
        'binding = load_verified_materialized_cohort_authority(source)\n'
        'if binding is None:\n'
        '    raise ValueError("landmark cohort has no sealed authority")\n'
        'proof = binding.authority.semantic_provenance\n'
        f'if proof.get("plan_sha256") != {json.dumps(plan_sha)} or '
        f'proof.get("landmark_contract_sha256") != {json.dumps(contract_sha)}:\n'
        '    raise ValueError("landmark cohort differs from signed plan")\n'
    )
    marker = 'source = Path(os.environ["COHORT_PARQUET"]).resolve()\n'
    if code.count(marker) != 1:
        raise ValueError("host cohort publication template changed")
    code = code.replace(repr(HOST_BOUND_COHORT_PRODUCT), repr("artifact:analysis_cohort"))
    return code.replace(marker, marker + verify, 1)
