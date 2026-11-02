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
from urllib.parse import unquote

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
    if name in {"AnalysisStep", "KnowHowDecision", "CohortDefinition", "LiteratureDesignBinding"}:
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
    single = {"additionalItems", "additionalProperties", "contains", "contentSchema",
              "else", "if", "items", "not", "propertyNames", "then",
              "unevaluatedItems", "unevaluatedProperties"}
    mapping = {"$defs", "definitions", "dependencies", "dependentSchemas",
               "patternProperties", "properties"}
    sequence = {"allOf", "anyOf", "oneOf", "prefixItems"}

    def walk(node, visit):
        if not isinstance(node, dict):
            return node
        replacement = visit(node)
        if replacement is not node:
            return replacement
        out = {}
        for key, value in node.items():
            if key in single and isinstance(value, dict):
                out[key] = walk(value, visit)
            elif key == "items" and isinstance(value, list):
                out[key] = [walk(item, visit) for item in value]
            elif key in mapping and isinstance(value, dict):
                out[key] = {name: walk(child, visit) for name, child in value.items()}
            elif key in sequence and isinstance(value, list):
                out[key] = [walk(child, visit) for child in value]
            else:
                out[key] = value
        return out

    # Rehoming a node can break resource scopes or pointers into its children.
    # Only direct references to a named definition are safe to preserve.
    scoped = False
    reserved = set()
    def inspect_scope(node):
        nonlocal scoped
        if any(key in node for key in ("$id", "$anchor", "$dynamicAnchor",
                                       "$dynamicRef", "$recursiveAnchor", "$recursiveRef")):
            scoped = True
        if "$ref" in node:
            ref = node["$ref"]
            decoded = unquote(ref) if isinstance(ref, str) else ""
            name = decoded.removeprefix("#/$defs/")
            if not decoded.startswith("#/$defs/") or not name or "/" in name:
                scoped = True
            else:
                reserved.add(name)
        return node
    walk(result, inspect_scope)
    if scoped:
        return result
    if "$defs" in result and not isinstance(result["$defs"], dict):
        return result

    def encode(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    while True:
        counts = Counter()
        def count(node):
            if isinstance(node.get("type"), str) or isinstance(node.get("anyOf"), list):
                counts[encode(node)] += 1
            return node
        walk(result, count)
        definitions = result.get("$defs", {})
        index = 0
        while f"_s{index}" in definitions or f"_s{index}" in reserved:
            index += 1
        name = f"_s{index}"
        reference = {"$ref": f"#/$defs/{name}"}
        candidates = [(n * (len(shape) - len(encode(reference))) - len(shape) - len(name) - 4, shape)
                      for shape, n in counts.items() if n > 1]
        if not candidates or max(candidates)[0] <= 0:
            return result
        _, selected = max(candidates)
        def replace(node):
            if encode(node) == selected:
                return dict(reference)
            return node
        result = walk(result, replace)
        result.setdefault("$defs", {})[name] = json.loads(selected)


def compact_planner_wire_schema(schema: dict) -> dict:
    """Compile planner wire v2 without changing output property names or values.

    This only walks JSON-Schema-bearing locations: literal enum/const/default
    objects are data, never references. Unsupported reference scopes fail
    closed instead of being rewritten optimistically.
    """
    import string

    single = {"additionalItems", "additionalProperties", "contains", "contentSchema", "else", "if", "items", "not", "propertyNames", "then", "unevaluatedItems", "unevaluatedProperties"}
    mapping = {"$defs", "definitions", "dependencies", "dependentSchemas", "patternProperties", "properties"}
    sequence = {"allOf", "anyOf", "oneOf", "prefixItems"}

    def walk(node, visit):
        if not isinstance(node, dict):
            return node
        replacement = visit(node)
        if replacement is not node:
            return replacement
        result = {}
        for key, value in node.items():
            if key in single and isinstance(value, dict):
                result[key] = walk(value, visit)
            elif key == "items" and isinstance(value, list):
                result[key] = [walk(v, visit) for v in value]
            elif key in mapping and isinstance(value, dict):
                result[key] = {name: walk(v, visit) for name, v in value.items()}
            elif key in sequence and isinstance(value, list):
                result[key] = [walk(v, visit) for v in value]
            else:
                result[key] = copy.deepcopy(value)
        return result

    def named_ref(node):
        if "$ref" not in node:
            return None
        ref = node["$ref"]
        if set(node) != {"$ref"} or not isinstance(ref, str) or not ref.startswith("#/$defs/"):
            raise StrictJsonSchemaError("planner wire schema has unsupported reference")
        name = ref.removeprefix("#/$defs/")
        if not name or any(char in name for char in "~/%") or unquote(name) != name:
            raise StrictJsonSchemaError("planner wire schema has unsupported JSON pointer")
        return name

    def validate(value):
        definitions = value.get("$defs")
        if not isinstance(definitions, dict):
            raise StrictJsonSchemaError("planner wire schema definitions missing")
        if any(not isinstance(name, str) or not name or any(char in name for char in "~/%") for name in definitions):
            raise StrictJsonSchemaError("planner wire schema has unsupported definition name")
        # The planner emits this closed modern-dialect subset. Unknown
        # keywords and explicit dialects must not be reinterpreted as v2.
        supported = {"$defs", "$ref", "additionalProperties", "anyOf", "const", "enum",
                     "exclusiveMaximum", "exclusiveMinimum", "items", "maxItems", "maxLength",
                     "maximum", "minItems", "minLength", "minimum", "pattern", "properties",
                     "required", "type", "default", "title", "description", "examples", "format", "multipleOf"}
        def check_dialect(node):
            if set(node) - supported or (node is not value and "$defs" in node):
                raise StrictJsonSchemaError("planner wire schema uses an unsupported dialect or keyword")
            return node
        walk(value, check_dialect)
        graph = {}
        for name, definition in definitions.items():
            refs = set()
            def collect(node):
                if any(k in node for k in ("$id", "$anchor", "$dynamicAnchor", "$dynamicRef", "$recursiveAnchor", "$recursiveRef")):
                    raise StrictJsonSchemaError("planner wire schema uses scoped references")
                ref = named_ref(node)
                if ref is not None:
                    refs.add(ref)
                return node
            walk(definition, collect)
            graph[name] = refs
        root_refs = set()
        def collect_root(node):
            if any(k in node for k in ("$id", "$anchor", "$dynamicAnchor", "$dynamicRef", "$recursiveAnchor", "$recursiveRef")):
                raise StrictJsonSchemaError("planner wire schema uses scoped references")
            ref = named_ref(node)
            if ref is not None:
                root_refs.add(ref)
            return node
        walk(value, collect_root)
        if not root_refs.issubset(definitions):
            raise StrictJsonSchemaError("planner wire schema reference is missing")
        done, active = set(), set()
        def visit(name):
            if name in active or name not in graph:
                raise StrictJsonSchemaError("planner wire schema is cyclic or missing")
            if name in done:
                return
            active.add(name)
            for ref in graph[name]:
                visit(ref)
            active.remove(name)
            done.add(name)
        for name in graph:
            visit(name)

    validate(schema)
    result = intern_schema_shapes(copy.deepcopy(schema))
    validate(result)
    # A definition used exactly once is cheaper inline; retain no assumptions
    # about historic definition names in the separate transport copy.
    while True:
        counts = Counter()
        def count(node):
            name = named_ref(node)
            if name is not None:
                counts[name] += 1
            return node
        walk(result, count)
        names = sorted(name for name in result["$defs"] if counts[name] == 1)
        if not names:
            break
        name = names[0]
        target = result["$defs"].pop(name)
        def inline(node):
            return copy.deepcopy(target) if named_ref(node) == name else node
        result = walk(result, inline)
    allowed = {
        "string": {"minLength", "maxLength", "pattern", "format"},
        "number": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"},
        "integer": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"},
        "boolean": set(), "array": {"items", "minItems", "maxItems"},
        "object": {"properties", "required", "additionalProperties"},
    }
    def nullable(node):
        branches = node.get("anyOf")
        if set(node) != {"anyOf"} or not isinstance(branches, list) or len(branches) != 2 or {"type": "null"} not in branches:
            return node
        value = next((v for v in branches if v != {"type": "null"}), None)
        if not isinstance(value, dict) or not isinstance(value.get("type"), str) or value["type"] not in allowed:
            return node
        kind = value["type"]
        if set(value) <= {"type", *allowed[kind]}:
            return {**value, "type": [kind, "null"]}
        if kind == "string" and set(value) == {"type", "enum"} and isinstance(value["enum"], list):
            return {"type": ["string", "null"], "enum": [*value["enum"], None]}
        return node
    result = walk(result, nullable)
    def singleton_enum(node):
        if isinstance(node.get("enum"), list) and len(node["enum"]) == 1 and "const" not in node:
            return {**{k: v for k, v in node.items() if k != "enum"}, "const": node["enum"][0]}
        return node
    result = walk(result, singleton_enum)
    validate(result)
    counts = Counter()
    def count_for_alias(node):
        name = named_ref(node)
        if name is not None:
            counts[name] += 1
        return node
    walk(result, count_for_alias)
    names = sorted(result["$defs"], key=lambda n: (-counts[n], n))
    short = list(string.ascii_letters) + [a + b for a in string.ascii_letters for b in string.ascii_letters]
    if len(names) > len(short):
        raise StrictJsonSchemaError("too many planner wire definitions")
    aliases = dict(zip(names, short))
    def rewrite(node):
        name = named_ref(node)
        return {"$ref": "#/$defs/" + aliases[name]} if name is not None else node
    result = walk(result, rewrite)
    result["$defs"] = {aliases[name]: value for name, value in result["$defs"].items()}
    validate(result)

    def expanded(value):
        definitions = value["$defs"]
        def expand_node(node):
            name = named_ref(node)
            return walk(definitions[name], expand_node) if name is not None else node
        root = {key: value for key, value in value.items() if key != "$defs"}
        return walk(root, expand_node)

    def canonical_nullable(node):
        kind = node.get("type")
        if not isinstance(kind, list) or len(kind) != 2 or kind[1] != "null" or kind[0] not in allowed:
            return node
        if kind[0] == "string" and set(node) == {"type", "enum"} and isinstance(node["enum"], list) and node["enum"][-1:] == [None]:
            return {"anyOf": [{"type": "string", "enum": node["enum"][:-1]}, {"type": "null"}]}
        if set(node) <= {"type", *allowed[kind[0]]}:
            return {"anyOf": [{**node, "type": kind[0]}, {"type": "null"}]}
        return node

    def canonical_enum(node):
        if "const" in node and "enum" not in node:
            return {**{k: v for k, v in node.items() if k != "const"}, "enum": [node["const"]]}
        return node

    def normal(value):
        current = expanded(value)
        while True:
            updated = walk(current, canonical_nullable)
            if updated == current:
                return walk(updated, canonical_enum)
            current = updated

    if normal(schema) != normal(result):
        raise StrictJsonSchemaError("planner wire schema changed the expanded contract")
    return result
