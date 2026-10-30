"""Capture selected native resolver inputs before reads; refuse input drift."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from easyicu.concept.provenance import (
    CodeProvenance, CohortFilterProvenance, ExportDataProvenance, capture_data_provenance,
    metadata_digest, resolver_implementation_digest,
    observing_resolver_inputs,
)
from easyicu.concept.schema import ConceptDictionary
from easyicu.config import DataSourceConfig
from easyicu.resources import package_path


class ExtractionProvenanceCapture:
    """Run-local metadata recorder. Does not receive a frame or patient IDs."""

    def __init__(self) -> None:
        self._inputs = self._input_identity()
        self._snapshots = []
        self._observations = []

    @staticmethod
    def _input_identity() -> str:
        import hashlib

        digests = {}
        for name in ("concept-dict.json", "sofa2-dict.json", "data-sources.json", "clinical-contracts.json"):
            with package_path(name) as path:
                digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return metadata_digest([digests, resolver_implementation_digest()])

    def record(
        self, *, dictionary: ConceptDictionary, source_config: DataSourceConfig,
        concepts: Sequence[str], cohort_contract: Mapping[str, object],
        resolver_parameters: Mapping[str, object] | None = None,
    ) -> None:
        filters = tuple(CohortFilterProvenance(
            field=key, values=tuple(value) if isinstance(value, (list, tuple)) else (value,),
            codes=tuple(CodeProvenance(code=str(code)) for code in value)
            if key in {"icd_include", "icd_exclude"} else (),
        ) for key, value in cohort_contract.items() if key in {
            "preset", "age_min", "age_max", "min_icu_los_hours",
            "observation_window_hours", "exclude_readmissions",
            "first_icu_stay_per_hospitalization", "icd_enabled", "icd_include", "icd_exclude",
        } and value is not None)
        parameters = dict(
            concepts=tuple(concepts), source_database=source_config.name,
            class_prefixes=tuple(source_config.class_prefix), cohort_filters=filters,
            resource_inputs_sha256=self._inputs, resolver_parameters=resolver_parameters,
            source_config_sha256=metadata_digest(source_config.model_dump(mode="json")),
            source_config=source_config,
        )
        snapshot = capture_data_provenance(dictionary, **parameters)
        self._snapshots.append(snapshot)
        self._observations.append((dictionary, source_config, parameters, snapshot.snapshot_sha256))

    def recorder(
        self, *, concepts: Sequence[str], cohort_contract: Mapping[str, object],
        resolver_parameters: Mapping[str, object] | None = None,
    ) -> Callable[[ConceptDictionary, DataSourceConfig], None]:
        def observe(dictionary: ConceptDictionary, config: DataSourceConfig) -> None:
            self.record(
                dictionary=dictionary, source_config=config, concepts=concepts,
                cohort_contract=cohort_contract, resolver_parameters=resolver_parameters,
            )
        return observe

    def observing(self, *, concepts: Sequence[str]):
        return observing_resolver_inputs(self.recorder(concepts=concepts, cohort_contract={}))

    def finish(self) -> dict:
        if self._inputs != self._input_identity():
            raise ValueError("extraction provenance inputs changed during extraction")
        for dictionary, config, parameters, expected in self._observations:
            current = dict(parameters, source_config_sha256=metadata_digest(config.model_dump(mode="json")))
            if capture_data_provenance(dictionary, **current).snapshot_sha256 != expected:
                raise ValueError("extraction provenance resolver inputs changed during extraction")
        return ExportDataProvenance(snapshots=tuple(self._snapshots)).model_dump(mode="json")
