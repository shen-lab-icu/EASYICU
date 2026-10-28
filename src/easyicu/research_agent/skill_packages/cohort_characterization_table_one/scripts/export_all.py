"""Record the exact specification and checksums of produced files."""

from pathlib import Path

from ...reference_io import export_manifest
from ..spec import SKILL_ID


def export_all(loaded, spec, out_dir, summary, figures) -> dict:
    out = Path(out_dir)
    names = [*summary["output_files"].values(), "step_summary.json"]
    if summary.get("coefficient_table"):
        names.append(summary["coefficient_table"])
    if (out / "prepared_cohort.parquet").is_file():
        names.append("prepared_cohort.parquet")
    return export_manifest(
        out_dir=out, skill_id=SKILL_ID, loaded=loaded, spec=spec,
        artifacts=[*[out / name for name in names], *figures],
    )
