"""Execute the offline reference workflow on prepared or synthetic rows."""

from pathlib import Path

from ...reference_io import reference_cli
from ..spec import AdjustedAssociationReferenceSpec
from .example_data import write_example
from .export_all import export_all
from .generate_all_plots import generate_all_plots
from .load_cohort import load_cohort
from .run_analysis import run_analysis


def run_all(source, spec: AdjustedAssociationReferenceSpec, out_dir: str | Path, *, provenance=None) -> dict:
    spec = AdjustedAssociationReferenceSpec.model_validate(spec)
    loaded = load_cohort(source, spec, provenance=provenance)
    summary = run_analysis(loaded, spec, out_dir)
    figures = generate_all_plots(out_dir)
    return export_all(loaded, spec, out_dir, summary, figures)


def main(argv=None) -> int:
    return reference_cli(
        argv=argv, description="Adjusted association model reference scripts",
        spec_type=AdjustedAssociationReferenceSpec, run_all=run_all, write_example=write_example,
    )


if __name__ == "__main__":
    raise SystemExit(main())
