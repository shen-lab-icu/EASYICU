from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.concept.metadata_projection import (
    ColumnProjectionSpec,
    ConceptColumnRole,
    project_concept_column_metadata,
)
from easyicu.concept.metadata_sidecar import (
    EXPORT_PHYSICAL_SCOPE,
    ColumnMetadataBinding,
    ColumnMetadataFileBinding,
    ColumnMetadataSidecar,
    DerivationWindow,
    MetadataSidecarError,
    SidecarRef,
    TimeCoordinate,
    canonical_sidecar_bytes,
    parse_column_metadata_sidecar,
    read_content_addressed_sidecar,
    sidecar_sha256,
    write_content_addressed_sidecar,
)
from easyicu.resources import load_dictionary


def _metadata(column: str, role: ConceptColumnRole, aggregation: str | None = None):
    definition = load_dictionary()["lact"]
    return project_concept_column_metadata(
        definition,
        spec=ColumnProjectionSpec(
            column_name=column,
            source_concept="lact",
            role=role,
            aggregation=aggregation,
        ),
        source_database="miiv",
    )


def _sidecar(*, reverse: bool = False) -> ColumnMetadataSidecar:
    pairs = [
        (
            "lact",
            ColumnMetadataBinding(metadata=_metadata("lact", ConceptColumnRole.VALUE)),
        ),
        (
            "lact_mean",
            ColumnMetadataBinding(
                metadata=_metadata(
                    "lact_mean", ConceptColumnRole.NUMERIC_AGGREGATE, "mean"
                ),
                derivation_window=DerivationWindow("icu_admission", 0, 24),
            ),
        ),
        (
            "lact_n",
            ColumnMetadataBinding(
                metadata=_metadata("lact_n", ConceptColumnRole.COUNT),
                derivation_window=DerivationWindow("icu_admission", 0, 24),
            ),
        ),
    ]
    if reverse:
        pairs.reverse()
    return ColumnMetadataSidecar(
        source_database="miiv",
        source_database_class_prefixes=(),
        scope=EXPORT_PHYSICAL_SCOPE,
        files=(
            ColumnMetadataFileBinding(
                relative_path="labs.parquet",
                module="labs",
                identity_column="stay_id",
                time_coordinates=(TimeCoordinate("charttime", "icu_admission", "h"),),
                columns=dict(pairs),
            ),
        ),
    )


def test_sidecar_is_canonical_and_independent_of_mapping_insertion_order():
    first = _sidecar()
    second = _sidecar(reverse=True)

    assert canonical_sidecar_bytes(first) == canonical_sidecar_bytes(second)
    assert sidecar_sha256(first) == sidecar_sha256(second)
    assert parse_column_metadata_sidecar(canonical_sidecar_bytes(first)) == first


def test_semantic_change_changes_content_address():
    first = _sidecar()
    changed = ColumnMetadataSidecar(
        source_database="miiv",
        source_database_class_prefixes=(),
        scope=EXPORT_PHYSICAL_SCOPE,
        files=(
            ColumnMetadataFileBinding(
                relative_path="labs.parquet",
                module="labs",
                identity_column="stay_id",
                time_coordinates=(TimeCoordinate("charttime", "icu_admission", "h"),),
                columns={
                    "lact": ColumnMetadataBinding(
                        metadata=_metadata("lact", ConceptColumnRole.VALUE),
                        representation_transform="unit_normalized",
                    )
                },
            ),
        ),
    )

    assert sidecar_sha256(first) != sidecar_sha256(changed)


def test_content_addressed_writer_is_idempotent_and_digest_bound(tmp_path: Path):
    sidecar = _sidecar()
    first = write_content_addressed_sidecar(tmp_path, sidecar)
    second = write_content_addressed_sidecar(tmp_path, sidecar)

    assert first == second
    assert first.file == f"column_metadata.sha256-{first.sha256}.json"
    loaded = read_content_addressed_sidecar(
        tmp_path / first.file,
        expected_sha256=first.sha256,
        expected_size=first.size,
    )
    assert loaded == sidecar


def test_reader_rejects_tampered_bytes_even_when_file_size_is_unchanged(tmp_path: Path):
    ref = write_content_addressed_sidecar(tmp_path, _sidecar())
    path = tmp_path / ref.file
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b'"miiv"', b'"eicu"', 1))

    with pytest.raises(MetadataSidecarError, match="digest mismatch"):
        read_content_addressed_sidecar(
            path, expected_sha256=ref.sha256, expected_size=ref.size
        )


def test_parser_rejects_duplicate_keys_and_nonfinite_json():
    raw = canonical_sidecar_bytes(_sidecar())
    duplicate = raw.replace(b'{"files":', b'{"files":[],"files":', 1)
    with pytest.raises(MetadataSidecarError, match="duplicate JSON key"):
        parse_column_metadata_sidecar(duplicate)

    nonfinite = raw.replace(b'"start_hours":0.0', b'"start_hours":NaN', 1)
    with pytest.raises(MetadataSidecarError, match="non-finite"):
        parse_column_metadata_sidecar(nonfinite)


def test_parser_rejects_unknown_fields_and_noncanonical_pretty_json():
    payload = _sidecar().to_dict()
    payload["unknown"] = True
    compact = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(MetadataSidecarError, match="keys do not match"):
        parse_column_metadata_sidecar(compact)

    pretty = json.dumps(_sidecar().to_dict(), indent=2, sort_keys=True).encode()
    with pytest.raises(MetadataSidecarError, match="not canonical"):
        parse_column_metadata_sidecar(pretty)


def test_sidecar_rejects_source_database_or_resolution_chain_mismatch():
    file_binding = _sidecar().files[0]
    with pytest.raises(MetadataSidecarError, match="source_database"):
        ColumnMetadataSidecar(
            source_database="eicu",
            source_database_class_prefixes=(),
            scope=EXPORT_PHYSICAL_SCOPE,
            files=(file_binding,),
        )


def test_sidecar_rejects_binding_key_or_time_coordinate_duplicates():
    with pytest.raises(MetadataSidecarError, match="key must match"):
        ColumnMetadataFileBinding(
            relative_path="labs.parquet",
            module="labs",
            identity_column="stay_id",
            time_coordinates=(),
            columns={
                "wrong": ColumnMetadataBinding(
                    metadata=_metadata("lact", ConceptColumnRole.VALUE)
                )
            },
        )

    with pytest.raises(MetadataSidecarError, match="must be unique"):
        ColumnMetadataFileBinding(
            relative_path="labs.parquet",
            module="labs",
            identity_column="stay_id",
            time_coordinates=(
                TimeCoordinate("charttime", "icu_admission", "h"),
                TimeCoordinate("charttime", "icu_admission", "h"),
            ),
            columns={
                "lact": ColumnMetadataBinding(
                    metadata=_metadata("lact", ConceptColumnRole.VALUE)
                )
            },
        )


@pytest.mark.parametrize("structural_column", ["stay_id", "charttime"])
def test_file_binding_rejects_value_authority_over_structural_coordinates(
    structural_column: str,
):
    with pytest.raises(
        MetadataSidecarError,
        match="identity/time coordinates must not also be value bindings",
    ):
        ColumnMetadataFileBinding(
            relative_path="labs.parquet",
            module="labs",
            identity_column="stay_id",
            time_coordinates=(TimeCoordinate("charttime", "icu_admission", "h"),),
            columns={
                structural_column: ColumnMetadataBinding(
                    metadata=_metadata(structural_column, ConceptColumnRole.VALUE)
                )
            },
        )


def test_file_binding_rejects_identity_time_overlap():
    with pytest.raises(MetadataSidecarError, match="must be disjoint"):
        ColumnMetadataFileBinding(
            relative_path="labs.parquet",
            module="labs",
            identity_column="stay_id",
            time_coordinates=(TimeCoordinate("stay_id", "icu_admission", "h"),),
            columns={},
        )


def test_sidecar_reference_parser_is_exact_and_canonical(tmp_path: Path):
    reference = write_content_addressed_sidecar(tmp_path, _sidecar())
    assert SidecarRef.from_dict(reference.to_dict()) == reference

    with pytest.raises(MetadataSidecarError, match="keys do not match schema"):
        SidecarRef.from_dict({**reference.to_dict(), "extra": True})
    with pytest.raises(MetadataSidecarError, match="64 lowercase hex"):
        SidecarRef.from_dict(
            {**reference.to_dict(), "sha256": reference.sha256.upper()}
        )
    with pytest.raises(MetadataSidecarError, match="non-negative integer"):
        SidecarRef.from_dict({**reference.to_dict(), "size": True})


def test_reader_rejects_symlink(tmp_path: Path):
    ref = write_content_addressed_sidecar(tmp_path, _sidecar())
    link = tmp_path / "sidecar-link.json"
    link.symlink_to(ref.file)
    with pytest.raises(MetadataSidecarError, match="symlink"):
        read_content_addressed_sidecar(
            link, expected_sha256=ref.sha256, expected_size=ref.size
        )
