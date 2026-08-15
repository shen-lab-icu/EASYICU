"""ClinicalConstraintValidator."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


from ..schema import (
    AnalysisStep,
    ResearchContext,
    ValidationFinding,
)

# ---------------------------------------------------------------------------
# CohortAuditor
# ---------------------------------------------------------------------------


# Patient-level identifier column names. Their presence means a cohort can
# be reasoned about at the patient level (within-patient non-independence,
# first-stay selection). Stay-level keys (stay_id, icustay_id) and

class ClinicalConstraintValidator:
    """ICU-specific semantic warnings over planned and executed analyses."""

    name = "clinical_constraint_validator"
    _CAUSAL_STEP_METHODS = {
        "causal_inference",
        "causal_emulation",
        "g_computation",
        "ipw",
        "iptw",
        "propensity_score",
        "psm",
        "target_trial",
        "target_trial_emulation",
        "treatment_response",
        "effect_modification",
        "interaction_model",
    }
    _CAUSAL_STEP_FAMILIES = {
        "causal_inference",
        "treatment_response",
        "reinforcement_learning",
    }

    @staticmethod
    def _normalise(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")

    def audit(
        self,
        *,
        context: ResearchContext,
        step: AnalysisStep,
        out_dir: Path,
        step_summary: Dict[str, Any],
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        family = (
            (context.user_preferences.inferred_analysis_family or "").lower()
            if context.user_preferences
            else ""
        )
        question = (context.research_question or "").lower()
        timing = (
            (context.user_preferences.timing_and_design or "").lower()
            if context.user_preferences and context.user_preferences.timing_and_design
            else ""
        )
        combined = " ".join(
            filter(
                None,
                [
                    question,
                    timing,
                    (step.intent or "").lower(),
                    json.dumps(step_summary, ensure_ascii=False).lower(),
                ],
            )
        )
        method_head = self._normalise(
            str(step.method or "").lower().split(" with ", 1)[0]
        )
        step_family = self._normalise(step_summary.get("analysis_family"))
        causal_step_owner = (
            method_head in self._CAUSAL_STEP_METHODS
            or step_family in self._CAUSAL_STEP_FAMILIES
        )

        if causal_step_owner:
            if not any(
                term in combined
                for term in (
                    "time zero",
                    "time-zero",
                    "eligibility",
                    "anchor",
                    "alignment",
                )
            ):
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="warning",
                        message=(
                            "Treatment-effect style analysis without an explicit time-zero or alignment description "
                            "risks immortal time bias. Document eligibility, anchor time, and treatment assignment timing."
                        ),
                        detail={
                            "analysis_family": step_family or family or "unspecified",
                            "method": method_head,
                        },
                    )
                )
            if "post-treatment" in combined:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="warning",
                        message=(
                            "A post-treatment variable appears in the analysis description. "
                            "Confirm this is not conditioning on a mediator or downstream treatment effect."
                        ),
                    )
                )

        if family == "survival" or any(
            term in combined for term in ("survival", "cox", "kaplan", "hazard")
        ):
            if (
                any(term in combined for term in ("length of stay", "los", "discharge"))
                and "competing" not in combined
            ):
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="warning",
                        message=(
                            "Length-of-stay or discharge-oriented survival analyses often require a competing-risks framing. "
                            "Consider discharge/death competition explicitly rather than a single-event survival model."
                        ),
                    )
                )
            if (
                "time-varying" in combined
                and "landmark" not in combined
                and "time updated" not in combined
            ):
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="warning",
                        message=(
                            "Time-varying covariates are mentioned without an explicit handling strategy. "
                            "Specify landmarking, time-updated modeling, or another deterministic design."
                        ),
                    )
                )

        return findings
