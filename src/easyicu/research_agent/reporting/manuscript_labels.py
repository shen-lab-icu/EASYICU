"""Language-aware display of sealed variable definitions, without translation guesses."""

import re
from typing import Mapping


def _recorded_term(variable):
    semantics = getattr(variable, "observation_semantics", None)
    definition = getattr(variable, "clinical_definition", None)
    domain = getattr(variable, "observed_domain", None) or {}
    if (getattr(semantics, "kind", None) != "positive_only_event"
            or definition is None or domain.get("is_binary") is not True
            or set(domain.get("levels") or []) != {0, 1}):
        return None
    term = str(definition.definition or "").strip()
    return term if term and not re.search(r"[\u3400-\u9fff]", term) else None


def source_bound_manuscript_labels(context, labels: Mapping[str, str], *, language="en", include_unlabeled=False):
    """Use existing English source metadata when the UI label is Chinese.

    Only typed positive-only binary event records admit the recorded/not-recorded
    projection. Zero must never become clinical absence for an arbitrary code.
    The plan, cohort, source labels and scientific definition remain unchanged.
    Unknown translations retain their supplied label for explicit review.
    """
    result = dict(labels)
    if context is None or not str(language).lower().startswith("en"):
        return result
    variables = {variable.name: variable for variable in context.variables}
    for name, variable in (variables.items() if include_unlabeled else ()):
        if name in result and result[name] != name:
            continue
        description = str(variable.description or "").strip()
        if description and not re.search(r"[\u3400-\u9fff]", description):
            term = _recorded_term(variable)
            result[name] = f"Recorded {term} status" if term else description
    if include_unlabeled:
        # Audit products identify the source concept rather than a particular
        # aggregation. Only the typed event representative can supply this alias.
        for name, variable in variables.items():
            semantics = getattr(variable, "observation_semantics", None)
            concept = getattr(variable, "source_concept", None)
            term = _recorded_term(variable)
            if concept and term and getattr(semantics, "representative_column", None) == name:
                result.setdefault(concept, f"Recorded {term} status")
    for key, label in labels.items():
        if not re.search(r"[\u3400-\u9fff]", str(label)):
            continue
        name, separator, level = key.partition("=")
        variable = variables.get(name)
        if variable is None:
            continue
        description = str(variable.description or "").strip()
        if not separator:
            if description and not re.search(r"[\u3400-\u9fff]", description):
                term = _recorded_term(variable)
                result[key] = f"Recorded {term} status" if term else description
            continue
        semantics = variable.observation_semantics
        definition = variable.clinical_definition
        domain = variable.observed_domain or {}
        if (semantics is None or semantics.kind != "positive_only_event"
                or definition is None or level not in {"0", "1"}
                or domain.get("is_binary") is not True
                or set(domain.get("levels") or []) != {0, 1}):
            continue
        term = str(definition.definition or "").strip()
        if term and not re.search(r"[\u3400-\u9fff]", term):
            result[key] = ("No recorded " if level == "0" else "Recorded ") + term
    return result


def recorded_definition_section_errors(manuscript: str, context) -> dict[str, tuple[str, ...]]:
    """Positive-only record membership must not become confirmed diagnosis."""
    if context is None or not any(
        getattr(getattr(variable, "observation_semantics", None), "kind", None) == "positive_only_event"
        for variable in context.variables
    ):
        return {}
    errors = {}
    for match in re.finditer(r"(?ms)^## (Introduction|Discussion)\s*\n(.*?)(?=^## |\Z)", manuscript):
        for sentence in re.split(r"(?<=[.!?])\s+", match.group(2)):
            if re.search(r"diagnosis status|anchored to (?:that|the) clinical definition", sentence, re.I) and not re.search(
                r"\brecorded\b|\boperational\b|\bproxy\b|not (?:a |an )?(?:independent|confirmed)", sentence, re.I,
            ):
                errors[match.group(1).lower()] = (
                    "This section treats a positive-only source record as a clinical diagnosis. "
                    "Make a limited correction using the existing source definitions: the executed grouping is recorded status, "
                    "and no positive record does not establish clinical absence. Do not claim independent clinical confirmation. "
                    "Retain the existing citations and other supported prose; do not add new analyses or literature. "
                    "Describe source operationalization separately from the clinical definition used in background literature.",
                )
                break
    return errors
