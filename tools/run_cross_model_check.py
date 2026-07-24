#!/usr/bin/env python
"""Run mock cross-model plan/panel concordance checks.

This CLI intentionally does not call real LLM APIs. It verifies the reporting
path for later multi-backend pilots; real backend execution should be wired as a
separate, explicitly monitored run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from easyicu.research_agent.cohort.schema import (  # noqa: E402
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
)
from easyicu.research_agent.evaluation.cross_model_panel import (  # noqa: E402
    BackendIdentity,
    compare_panel_primaries,
    compare_plans,
    write_cross_model_report,
)
from easyicu.research_agent.robustness.panel import (  # noqa: E402
    PRIMARY_SPEC_ID,
    RobustnessPanel,
    RobustnessPanelRow,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare mock backend plans and robustness-panel primary rows. "
            "This infrastructure smoke test does not call real LLM APIs."
        )
    )
    parser.add_argument(
        "--backends", required=True, help="JSON file of backend identities."
    )
    parser.add_argument(
        "--mode",
        choices=["plan_only", "plan_and_panel"],
        default="plan_only",
        help="Whether to compare only plans or also mock panel primary rows.",
    )
    parser.add_argument("--question", help="Research question text.")
    parser.add_argument(
        "--question-file", help="Path to a text file containing the question."
    )
    parser.add_argument(
        "--out-dir",
        default="cross_model_check",
        help="Directory where cross_model_report.json will be written.",
    )
    args = parser.parse_args(argv)

    backends = _load_backends(Path(args.backends))
    if len(backends) < 2:
        parser.error("cross-model check requires at least two backends")
    unsupported = [b.name for b in backends if b.llm_provider != "mock"]
    if unsupported:
        parser.error(
            "this skeleton only supports mock backends; unsupported: "
            + ", ".join(unsupported)
        )
    question = _load_question(args.question, args.question_file)
    backend_by_name = {b.name: b for b in backends}
    plans = {b.name: _mock_plan(question, b) for b in backends}
    plan_conc = compare_plans(plans, backends=backend_by_name)
    panel_conc = None
    if args.mode == "plan_and_panel":
        panels = {b.name: _mock_panel(b) for b in backends}
        panel_conc = compare_panel_primaries(panels, backends=backend_by_name)
    report_path = write_cross_model_report(Path(args.out_dir), plan_conc, panel_conc)
    print(report_path)
    return 0


def _load_backends(path: Path) -> List[BackendIdentity]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("backends", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise SystemExit("--backends must contain a list or {'backends': [...]}")
    return [BackendIdentity.from_dict(item) for item in items]


def _load_question(question: Optional[str], question_file: Optional[str]) -> str:
    if question:
        return question
    if question_file:
        return Path(question_file).read_text(encoding="utf-8").strip()
    raise SystemExit("provide --question or --question-file")


def _mock_plan(question: str, backend: BackendIdentity) -> AnalysisPlan:
    end = float(backend.env_overrides.get("EASYICU_MOCK_WINDOW_END_HOURS", "24"))
    cohort = CohortDefinition(
        name="primary",
        inclusion=(
            ConceptPredicate(
                concept_id="age",
                time_window=TimeWindow(
                    anchor="icu_admit",
                    start_offset_hours=0.0,
                    end_offset_hours=end,
                ),
                aggregation="first",
                op=">=",
                value=18,
            ),
        ),
        exclusion=(),
        derived_from_named=None,
    )
    return AnalysisPlan(
        research_question=question,
        cohort=cohort,
        steps=[
            AnalysisStep(
                step_id="01_primary",
                planned_analysis_role="primary",
                intent="Fit the primary mock association model.",
                inputs=["age", "death"],
                expected_outputs=["statistic:primary_association"],
                method="logistic_regression",
            )
        ],
    )


def _mock_panel(backend: BackendIdentity) -> RobustnessPanel:
    shift = float(backend.env_overrides.get("EASYICU_MOCK_PANEL_SHIFT", "0"))
    row = RobustnessPanelRow(
        spec_id=PRIMARY_SPEC_ID,
        axis="primary",
        n=100,
        point_estimate=1.5 + shift,
        ci_low=1.2 + shift,
        ci_high=1.9 + shift,
        se=0.1,
        evidence_id=f"{backend.name}_primary",
        converged=True,
        notes="mock panel primary row",
    )
    return RobustnessPanel.from_rows([row])


if __name__ == "__main__":
    raise SystemExit(main())
