"""Small, fully synthetic native-v2 packages for aggregate screening tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from easyicu.concept.metadata_sidecar import (
    EXPORT_PHYSICAL_SCOPE,
    ColumnMetadataSidecar,
    write_content_addressed_sidecar,
)
from tests.support.native_outcome_export import file_binding


def screening_export(root: Path, *, planted: bool = False) -> Path:
    """Four invented stays; optionally plant four independent export defects."""
    root.mkdir()
    stays = [910001, 910002, 910003, 910004]
    frames = {
        "demographics": pd.DataFrame({"stay_id": stays, "age": [45.0] * 4}),
        "outcome": pd.DataFrame(
            {"stay_id": stays, "charttime": [0.0] * 4, "los_icu": [2.0] * 4}
        ),
        "vitals": pd.DataFrame(
            {
                "stay_id": stays,
                "charttime": [0.0] * 4,
                "hr": [80.0] * 4,
                "map": [75.0] * 4,
            }
        ),
        "blood_gas": pd.DataFrame(
            {"stay_id": stays, "charttime": [0.0] * 4, "lact": [1.0] * 4}
        ),
        "chemistry": pd.DataFrame(
            {
                "stay_id": stays,
                "charttime": [0.0] * 4,
                "crea": [1.0] * 4,
                "bili": [0.8] * 4,
            }
        ),
        "hematology": pd.DataFrame(
            {"stay_id": stays, "charttime": [0.0] * 4, "plt": [200.0] * 4}
        ),
        "neurological": pd.DataFrame(
            {"stay_id": stays, "charttime": [0.0] * 4, "gcs": [15.0] * 4}
        ),
        "vasopressors": pd.DataFrame(
            {"stay_id": stays, "charttime": [0.0] * 4, "norepi_rate": [0.0] * 4}
        ),
        "other_scores": pd.DataFrame(
            {
                "stay_id": stays,
                "charttime": [0.0] * 4,
                "qsofa": [0.0, 1.0, 2.0, 3.0],
                "sirs": [0.0, 1.0, 2.0, 4.0],
                "news": [0.0, 5.0, 10.0, 20.0],
            }
        ),
    }
    for total, module in (("sofa", "sofa1_score"), ("sofa2", "sofa2_score")):
        frames[module] = pd.DataFrame(
            {
                "stay_id": stays,
                "charttime": [0.0] * 4,
                total: [6.0] * 4,
                **{
                    f"{total}_{organ}": [1.0] * 4
                    for organ in ("resp", "coag", "liver", "cardio", "cns", "renal")
                },
            }
        )
    for number in (1, 2):
        frames[f"sepsis3_sofa{number}"] = pd.DataFrame(
            {
                "stay_id": stays,
                "charttime": [0.0] * 4,
                f"sep3_sofa{number}": [False, True, False, False],
            }
        )
    if planted:
        frames["demographics"] = pd.concat(
            [frames["demographics"], frames["demographics"].iloc[:1]], ignore_index=True
        )
        frames["outcome"].loc[0, "los_icu"] = 500.0
        frames["vitals"].loc[0, "hr"] = 10000.0
        frames["sofa2_score"]["sofa2"] = 5.0

    bindings, members = [], []
    for module, frame in frames.items():
        concepts = [
            column for column in frame if column not in {"stay_id", "charttime"}
        ]
        relative = f"{module}.parquet"
        frame.to_parquet(root / relative, index=False)
        binding = file_binding(
            relative_path=relative, module=module, frame=frame, concepts=concepts
        )
        bindings.append(binding)
        members.append(
            {
                "file": relative,
                "module": module,
                "concept_ids": concepts,
                "concepts": len(concepts),
                "rows": len(frame),
            }
        )
    reference = write_content_addressed_sidecar(
        root,
        ColumnMetadataSidecar(
            source_database="miiv",
            source_database_class_prefixes=(),
            scope=EXPORT_PHYSICAL_SCOPE,
            files=tuple(bindings),
        ),
    )
    (root / "_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu_native_export_v2",
                "time_window_authority": {
                    "required": True, "source": "outcome.los_icu",
                    "bounded_stays": int(frames["vitals"]["stay_id"].nunique()),
                },
                "database": "miiv",
                "format": "parquet",
                "files": members,
                "column_metadata": reference.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    return root


def change_column(root: Path, module: str, column: str, values: list) -> None:
    """Rewrite a tiny fixture column, retaining its original physical type."""
    path = root / f"{module}.parquet"
    frame = pd.read_parquet(path)
    dtype = "boolean" if frame[column].dtype.kind == "b" else frame[column].dtype
    frame[column] = pd.Series(values, dtype=dtype)
    frame.to_parquet(path, index=False)
