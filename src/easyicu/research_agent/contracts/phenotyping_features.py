"""The planned fit roster is distinct from readable audit/profile inputs."""

from __future__ import annotations

from typing import Any, Iterable, Sequence


PHENOTYPING_PRIMARY_ACTION = "phenotyping.cluster_solution"
_FORBIDDEN_FEATURE_ROLES = frozenset({"id", "time", "index", "meta", "outcome"})


def require_phenotyping_features(
    columns: Sequence[str] | None, *, inputs: Iterable[str],
    descriptors: Sequence[Any] | None = None, outcome_columns: Iterable[str] = (),
) -> tuple[str, ...]:
    if columns is None:
        raise ValueError("phenotyping_feature_roster_missing: explicitly separate fit features from profile and outcome inputs")
    features = tuple(columns)
    if (len(features) < 2 or len(features) != len(set(features))
            or any(not name.strip() or name != name.strip() or ":" in name for name in features)):
        raise ValueError("phenotyping_feature_roster_invalid: require at least two distinct exact column names")
    if not set(features) <= set(inputs):
        raise ValueError("phenotyping_feature_input_mismatch: every fit feature must be a declared raw input")
    if descriptors is not None:
        known = {item.name: item for item in descriptors}
        outcomes = set(outcome_columns)
        for name in features:
            item = known.get(name)
            role = getattr(getattr(item, "role", None), "value", getattr(item, "role", None))
            lineage = {name, getattr(item, "source_concept", None), *getattr(item, "derived_from_concepts", ())}
            if item is None or role in _FORBIDDEN_FEATURE_ROLES or outcomes & lineage:
                raise ValueError(f"phenotyping_feature_role_forbidden: {name} is not a non-outcome fitting feature")
    return features
