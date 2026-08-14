"""CohortAuditor — cohort hygiene validation."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from ..schema import (
    ResearchContext,
    ValidationFinding,
    VariableRole,
)

# ---------------------------------------------------------------------------
# CohortAuditor
# ---------------------------------------------------------------------------


# Patient-level identifier column names. Their presence means a cohort can
# be reasoned about at the patient level (within-patient non-independence,
# first-stay selection). Stay-level keys (stay_id, icustay_id) and

from ._v_support import cohort_hygiene_findings

class CohortAuditor:
    """Confirm the dataframe matches the descriptor it claims to."""

    name = "cohort_auditor"

    def audit(
        self,
        *,
        context: ResearchContext,
        cohort_path: Path,
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        try:
            df = pd.read_parquet(cohort_path)
        except Exception as exc:
            return [
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=f"Could not read cohort parquet: {exc}",
                )
            ]

        # Row count
        if context.cohort.n_stays != int(len(df)):
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Row count mismatch: descriptor says n_stays={context.cohort.n_stays:,} "
                        f"but cohort parquet has {len(df):,} rows."
                    ),
                )
            )

        # Required id columns
        for col in context.cohort.id_columns:
            if col not in df.columns:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=f"Declared id column '{col}' missing from cohort.",
                    )
                )

        # Target outcome present and binary if labelled binary
        outcome = context.target_outcome
        if outcome:
            if outcome not in df.columns:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=f"Target outcome '{outcome}' missing from cohort.",
                    )
                )
            else:
                v = context.variable(outcome)
                if v and v.role == VariableRole.OUTCOME:
                    s = df[outcome].dropna()
                    if not s.empty and set(s.unique()) - {0, 1, True, False, 0.0, 1.0}:
                        findings.append(
                            ValidationFinding(
                                validator=self.name,
                                severity="warning",
                                message=(
                                    f"Target outcome '{outcome}' has non-binary values "
                                    f"({sorted(set(s.unique()))[:5]}…); confirm this is intended."
                                ),
                            )
                        )

        # NaN-only columns
        for col in df.columns:
            if df[col].isna().all():
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="warning",
                        message=f"Column '{col}' is entirely missing in the cohort.",
                    )
                )

        # High-missing flag for any declared variable
        for v in context.variables:
            if v.missingness and v.missingness.fraction_missing > 0.5:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="warning",
                        message=(
                            f"Variable '{v.name}' has {v.missingness.fraction_missing:.0%} "
                            "missingness; downstream associations are at risk of selection bias."
                        ),
                        detail={"fraction_missing": v.missingness.fraction_missing},
                    )
                )

        # Impartial, advisory cohort-hygiene flags (patient-level
        # non-independence, short-stay exposure). Always severity="warning",
        # so they record the question without enforcing a choice.
        findings.extend(cohort_hygiene_findings(df, context))

        return findings
