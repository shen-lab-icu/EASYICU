"""Synthetic-only closed adapter for the host producer integration test."""
from __future__ import annotations

from pathlib import Path

from easyicu.synthetic_sources import verify_source
from easyicu.webserver.reviewed_snapshot_producer import ClosedWorkerSpec


class SyntheticAdapter:
    def __init__(self, source, python, runtime_roots):
        self.source = Path(source).absolute()
        self.python = python
        self.runtime_roots = tuple(runtime_roots)

    def validate_scope(self, scope):
        if scope.get("source") != str(self.source) or scope.get("database") != "sic":
            raise ValueError("fixture_scope_mismatch")
        if scope.get("purpose") != "synthetic_fixture_only":
            raise ValueError("fixture_scope_invalid")
        return ClosedWorkerSpec(
            source=self.source, source_identity_sha256="a" * 64,
            files=tuple(scope["files"]), worker=Path(__file__).with_name("scoped_synthetic_worker.py"),
            python=self.python, runtime_roots=self.runtime_roots,
        )

    def verify_source(self, scope, spec):
        manifest = verify_source(spec.source)
        if not manifest or manifest["database"] != "sic":
            raise ValueError("fixture_source_not_verified_synthetic")
        expected = sorted([*manifest["files"], "README.md", "synthetic_source.json"])
        if list(spec.files) != expected:
            raise ValueError("fixture_scope_roster_mismatch")

    def arguments(self, snapshot, output):
        return str(snapshot), str(output)

    def verify_output(self, scope, spec, output):
        import hashlib
        import re
        import pandas as pd

        expected = {"README.md", "_manifest.json", "outcome.parquet", "blood_gas.parquet", "sic_case_landmark.parquet"}
        actual = {path.name for path in output.iterdir()}
        metadata = {name for name in actual if re.fullmatch(r"column_metadata\.sha256-[0-9a-f]{64}\.json", name)}
        if actual - metadata != expected or len(metadata) != 1:
            raise ValueError(f"fixture_unexpected_output_roster: {sorted(actual)}")
        sidecar = next(iter(metadata))
        if hashlib.sha256((output / sidecar).read_bytes()).hexdigest() != sidecar.removeprefix("column_metadata.sha256-").removesuffix(".json"):
            raise ValueError("fixture_output_metadata_invalid")
        for name, column in (("blood_gas.parquet", "lact"),
                             ("sic_case_landmark.parquet", "care_unit_to_case_closure_days")):
            table = pd.read_parquet(output / name)
            if table.empty or column not in table.columns:
                raise ValueError("fixture_output_invalid")
