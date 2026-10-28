"""Provider-neutral compilation of closed Pydantic JSON Schemas.

Strict-schema providers require every object property to be required and every
object to reject unknown keys.  This module owns that mechanical projection so
individual agents do not grow subtly different copies of the same transport
rule.  Scientific narrowing (run-bound variables, actions, citations, and
cross-field branches) remains the responsibility of the contract owner that
calls these helpers.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from typing import Any, Mapping

from pydantic import BaseModel


class StrictJsonSchemaError(ValueError):
    """A model schema could not be represented as one closed JSON contract."""


def strictify_json_schema(node: Any) -> None:
    """Mutate a JSON Schema into the strict provider subset.

    Metadata is removed to keep the transport compact.  Field names such as
    ``description`` remain untouched because the traversal never treats a
    ``properties`` mapping itself as a schema node.
    """

    if not isinstance(node, dict):
        return
    for key in ("default", "description", "examples", "title"):
        node.pop(key, None)
    properties = node.get("properties")
    if isinstance(properties, dict):
        node["required"] = list(properties)
        node["additionalProperties"] = False
        for property_schema in properties.values():
            strictify_json_schema(property_schema)
    definitions = node.get("$defs")
    if isinstance(definitions, dict):
        for definition_schema in definitions.values():
            strictify_json_schema(definition_schema)
    for key in ("items", "additionalProperties", "not", "if", "then", "else"):
        value = node.get(key)
        if isinstance(value, dict):
            strictify_json_schema(value)
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        values = node.get(key)
        if isinstance(values, list):
            for value in values:
                strictify_json_schema(value)


def assert_closed_json_schema(node: Any, *, path: str = "$") -> None:
    """Fail when a schema is open, unconstrained, or structurally invalid."""

    if not isinstance(node, dict):
        return
    if not node:
        raise StrictJsonSchemaError(f"unconstrained JSON schema at {path}")
    properties = node.get("properties")
    if isinstance(properties, dict):
        if set(node.get("required") or ()) != set(properties):
            raise StrictJsonSchemaError(
                f"strict object does not require every property at {path}"
            )
        if node.get("additionalProperties") is not False:
            raise StrictJsonSchemaError(
                f"strict object permits additional properties at {path}"
            )
        for key, value in properties.items():
            assert_closed_json_schema(value, path=f"{path}/properties/{key}")
    definitions = node.get("$defs")
    if isinstance(definitions, dict):
        for key, value in definitions.items():
            assert_closed_json_schema(value, path=f"{path}/$defs/{key}")
    additional = node.get("additionalProperties", False)
    if additional is not False:
        raise StrictJsonSchemaError(f"open mapping is not permitted at {path}")
    for key in ("items", "not", "if", "then", "else"):
        value = node.get(key)
        if isinstance(value, dict):
            assert_closed_json_schema(value, path=f"{path}/{key}")
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        values = node.get(key)
        if isinstance(values, list):
            if not values:
                raise StrictJsonSchemaError(
                    f"strict schema keyword {key} must be non-empty at {path}"
                )
            for index, value in enumerate(values):
                assert_closed_json_schema(value, path=f"{path}/{key}/{index}")
    enum_values = node.get("enum")
    if isinstance(enum_values, list) and not enum_values:
        raise StrictJsonSchemaError(
            f"strict schema keyword enum must be non-empty at {path}"
        )


def closed_pydantic_json_schema(
    model: type[BaseModel],
    *,
    replacements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a detached, strict schema for a Pydantic model.

    ``replacements`` is intentionally shallow and optional.  Contract owners
    normally narrow nested fields directly before calling
    :func:`strictify_json_schema`; this convenience exists for exact top-level
    property replacements only.
    """

    schema = copy.deepcopy(model.model_json_schema(mode="validation"))
    if replacements:
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            raise StrictJsonSchemaError("Pydantic schema root has no properties")
        for key, value in replacements.items():
            if key not in properties:
                raise StrictJsonSchemaError(
                    f"cannot replace unknown root property {key!r}"
                )
            properties[key] = copy.deepcopy(value)
    strictify_json_schema(schema)
    assert_closed_json_schema(schema)
    return schema


__all__ = [
    "StrictJsonSchemaError",
    "assert_closed_json_schema",
    "closed_pydantic_json_schema",
    "strictify_json_schema",
]


def transport_definition_name(name: str) -> str:
    # These names are read by downstream transport projections.
    if name in {"AnalysisStep", "KnowHowDecision", "CohortDefinition"}:
        return name
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:4]


def compact_schema_definition_names(schema: dict) -> dict:
    definitions = schema.get("$defs", {})
    names = {name: transport_definition_name(name) for name in definitions}
    if len(set(names.values())) != len(names):
        raise ValueError("transport schema definition alias collision")

    def rewrite(node):
        if isinstance(node, list):
            return [rewrite(value) for value in node]
        if not isinstance(node, dict):
            return node
        result = {}
        for key, value in node.items():
            if key == "$defs":
                result[key] = {names[name]: rewrite(shape) for name, shape in value.items()}
            elif key == "$ref" and value.startswith("#/$defs/"):
                result[key] = "#/$defs/" + names[value.removeprefix("#/$defs/")]
            else:
                result[key] = rewrite(value)
        return result
    return rewrite(schema)


def intern_schema_shapes(schema: dict) -> dict:
    """Reference repeated schema shapes when doing so reduces wire bytes.

    Each extraction replaces identical schema nodes with a local reference.
    Property names, enum values, required lists and validation constraints are
    preserved. Fresh definitions are inserted after replacement, avoiding a
    self reference. The input schema is never mutated.
    """
    result = copy.deepcopy(schema)
    def encode(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    while True:
        counts = Counter()
        def count(node):
            if isinstance(node, dict):
                if isinstance(node.get("type"), str) or isinstance(node.get("anyOf"), list):
                    counts[encode(node)] += 1
                for value in node.values():
                    count(value)
            elif isinstance(node, list):
                for value in node:
                    count(value)
        count(result)
        definitions = result.setdefault("$defs", {})
        index = 0
        while f"_s{index}" in definitions:
            index += 1
        name = f"_s{index}"
        reference = {"$ref": f"#/$defs/{name}"}
        candidates = [(n * (len(shape) - len(encode(reference))) - len(shape) - len(name) - 4, shape)
                      for shape, n in counts.items() if n > 1]
        if not candidates or max(candidates)[0] <= 0:
            return result
        _, selected = max(candidates)
        def replace(node):
            if isinstance(node, dict):
                if encode(node) == selected:
                    return dict(reference)
                return {key: replace(value) for key, value in node.items()}
            if isinstance(node, list):
                return [replace(value) for value in node]
            return node
        result = replace(result)
        result["$defs"][name] = json.loads(selected)
