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
    """Fail when a schema contains an open object or unconstrained mapping."""

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
            for index, value in enumerate(values):
                assert_closed_json_schema(value, path=f"{path}/{key}/{index}")


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
