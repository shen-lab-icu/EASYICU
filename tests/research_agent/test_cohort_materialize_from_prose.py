"""Execute-phase 真强制: translate the agent's prose 纳排 into typed predicates
and materialise the filtered analysis cohort.

E1 run12 showed the framework's cohort enforcement never engaged: the replanner
grew a ``01_cohort_definition`` step but left ``plan.cohort`` empty, the
materialiser no-op'd, and the primary regression ran on the full universe while
the step re-applied 纳排 in its own code. ``5c9537b`` made that auditable; this
closes the loop — when the executing plan carries a cohort step with an empty
structured cohort, ``run_execute_phase`` extracts the criteria the agent stated
in prose, materialises ``cohort_analysis.parquet``, and re-points the runner so
downstream steps read the filtered cohort.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _is_replan(user: str) -> bool:
    upper = user.upper()
    return "PROBE SUMMARY:" in upper and "CURRENT PLAN:" in upper


def _is_extraction(user: str) -> bool:
    return (
        "COHORT-DEFINITION STEP PROSE" in user and "AVAILABLE PER-STAY COLUMNS" in user
    )


@pytest.mark.parametrize(
    ("development_sample_size", "cohort_products"),
    [
        (None, ["table:analysis_cohort"]),
        (100, ["table:analysis_cohort"]),
        (None, ["artifact:analysis_cohort", "table:cohort_flow"]),
    ],
)
def test_prose_cohort_is_extracted_materialised_and_enforced(
    ra,
    synthetic_cohort,
    tmp_path: Path,
    development_sample_size: int | None,
    cohort_products: list[str],
):
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    class ProseCohortLLM(ra.MockLLMClient):
        """Replanner grows an empty-cohort ``01_cohort_definition`` step; the
        extraction call returns typed predicates over real universe columns."""

        def complete(self, messages, **kwargs):
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            if _is_extraction(user):
                return json.dumps(
                    {
                        "inclusion": [
                            {"concept_id": "age", "op": ">=", "value": 18},
                            {"concept_id": "los_icu", "op": ">=", "value": 1},
                        ],
                        "exclusion": [],
                    }
                )
            if _is_replan(user):
                import re

                match = re.search(
                    r"CURRENT PLAN:\n(\{.*?\})\n\nPROBE SUMMARY:",
                    user,
                    flags=re.DOTALL,
                )
                current = (
                    AnalysisPlan.model_validate_json(match.group(1))
                    if match
                    else AnalysisPlan(research_question="q", steps=[])
                )
                steps = list(current.steps)
                if not any("cohort_def" in (s.step_id or "") for s in steps):
                    steps.insert(
                        0,
                        AnalysisStep(
                            step_id="01_cohort_definition",
                            intent=(
                                "Define the adult ICU analysis cohort: include "
                                "age >= 18 and ICU LoS >= 1 day; report attrition."
                            ),
                            inputs=[],
                            expected_outputs=cohort_products,
                            method="cohort_definition",
                        ),
                    )
                revised = current.model_copy(
                    update={"steps": steps, "revision": current.revision + 1}
                )
                return revised.model_dump_json(indent=2)
            return super().complete(messages, **kwargs)

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=ProseCohortLLM(),
        runner_kind="subprocess",
        runner_kwargs={"allow_unsafe_host_fallback": True},
        development_sample_size=development_sample_size,
    )
    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="prose_cohort",
        database="synthetic",
        target_outcome="death",
    )

    run_dir = Path(result.workdir)

    # The filtered analysis cohort was materialised and is smaller than the
    # 800-row universe (age >= 18 keeps all, but los_icu >= 1 drops some).
    analysis_cohort = run_dir / "cohort_analysis.parquet"
    assert analysis_cohort.exists(), "cohort_analysis.parquet was not materialised"
    n_cohort = len(pd.read_parquet(analysis_cohort))
    assert 0 < n_cohort < 800, n_cohort
    if development_sample_size is not None:
        sampled = run_dir / "cohort_analysis_development_sample.parquet"
        assert sampled.exists()
        assert len(pd.read_parquet(sampled)) == development_sample_size
        sample_manifest = json.loads(
            (run_dir / "development_execution_sample.json").read_text(encoding="utf-8")
        )
        assert sample_manifest["paper_authority"] is False
        assert sample_manifest["parent"]["rows"] == n_cohort
        assert sample_manifest["sample"]["rows"] == development_sample_size

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    findings = manifest["findings"]

    materialised = [
        f
        for f in findings
        if f.get("validator") == "cohort_materializer"
        and (f.get("detail") or {}).get("stage") == "execute_repair"
    ]
    assert materialised, "no execute_repair cohort_materializer finding"
    assert materialised[0]["detail"]["n_analysis_cohort"] == n_cohort

    # 真强制, not just auditable: the contract error must NOT fire once the
    # cohort is materialised.
    contract_errors = [
        f
        for f in findings
        if f.get("validator") == "cohort_contract" and f.get("severity") == "error"
    ]
    assert not contract_errors, "cohort_contract error fired despite materialisation"

    # The locked cohort on disk now reflects the real definition, not the empty
    # placeholder.
    locked = json.loads((run_dir / "cohort_locked.json").read_text(encoding="utf-8"))
    assert locked["cohort"]["inclusion"], "cohort_locked.json still has empty 纳排"

    statuses = [r.get("status") for r in manifest.get("per_step_records", [])]
    assert statuses and all(s == "ok" for s in statuses), statuses
    if "table:cohort_flow" in cohort_products:
        flow = pd.read_csv(run_dir / "cohort_analysis_flow.csv")
        assert int(flow.iloc[0]["n_before"]) == 800
        assert int(flow.iloc[-1]["n_remaining"]) == n_cohort
        cohort_step = next(
            record
            for record in manifest["per_step_records"]
            if record["step_id"] == "01_cohort_definition"
        )
        assert cohort_step["generation_mode"] == "deterministic_cohort_materializer"
        assert cohort_step["evidence_ids"] == [
            "analysis_cohort_execute_repair",
            "cohort_flow_execute_repair",
        ]
        assert cohort_step["step_provider_call_categories"] == [
            "cohort_definition_translation"
        ]
