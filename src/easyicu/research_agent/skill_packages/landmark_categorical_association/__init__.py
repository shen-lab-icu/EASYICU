"""Fixed-landmark categorical association skill (public entry points).

Standard workflow (see ``SKILL.md`` in this directory)::

    from easyicu.research_agent.skill_packages.landmark_categorical_association import (
        LandmarkCategoricalSpec, load_cohort, run_analysis, generate_all_plots, export_all,
    )

    cohort = load_cohort("cohort.parquet", spec)
    result = run_analysis(cohort, work_dir="results")
    figures = generate_all_plots(result, "results")
    export_all(result, "results", figures=figures)

or, in one call, ``run_all(cohort, spec, "results")``.
"""

from __future__ import annotations

from .scripts.example_data import example_spec, example_truth, make_example_cohort
from .scripts.export_all import (
    EXPORT_TOKEN,
    ExportConsistencyError,
    ExportReceipt,
    caveat_sentences,
    export_all,
)
from .scripts.generate_all_plots import PLOTS_TOKEN, FigureArtifact, generate_all_plots
from .scripts.load_cohort import (
    LOAD_TOKEN,
    UNDECLARED_PROVENANCE,
    CohortContractError,
    LoadedCohort,
    load_cohort,
)
from .scripts.run_all import SkillRun, run_all
from .scripts.run_analysis import (
    ANALYSIS_TOKEN,
    AnalysisContractError,
    AnalysisResult,
    run_analysis,
)
from .spec import (
    SKILL_ID,
    SKILL_VERSION,
    CovariateSpec,
    LandmarkCategoricalSpec,
    SecondaryOutcomeSpec,
)

__all__ = [
    "ANALYSIS_TOKEN",
    "EXPORT_TOKEN",
    "LOAD_TOKEN",
    "PLOTS_TOKEN",
    "SKILL_ID",
    "SKILL_VERSION",
    "UNDECLARED_PROVENANCE",
    "AnalysisContractError",
    "AnalysisResult",
    "CohortContractError",
    "CovariateSpec",
    "ExportConsistencyError",
    "ExportReceipt",
    "FigureArtifact",
    "LandmarkCategoricalSpec",
    "LoadedCohort",
    "SecondaryOutcomeSpec",
    "SkillRun",
    "caveat_sentences",
    "example_spec",
    "example_truth",
    "export_all",
    "generate_all_plots",
    "load_cohort",
    "make_example_cohort",
    "run_all",
    "run_analysis",
]
