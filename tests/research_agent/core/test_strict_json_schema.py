from __future__ import annotations

import pytest

from easyicu.research_agent.providers.strict_json_schema import (
    StrictJsonSchemaError,
    assert_closed_json_schema,
)


@pytest.mark.parametrize(
    "keyword",
    ("allOf", "anyOf", "oneOf", "prefixItems", "enum"),
)
def test_strict_schema_rejects_empty_structural_arrays(keyword: str) -> None:
    schema = {"type": "string", keyword: []}

    with pytest.raises(
        StrictJsonSchemaError,
        match=rf"keyword {keyword} must be non-empty",
    ):
        assert_closed_json_schema(schema)


def test_intern_preserves_enum_and_const_object_and_array_literals() -> None:
    import copy
    from jsonschema import Draft202012Validator
    from easyicu.research_agent.providers.strict_json_schema import intern_schema_shapes

    literal = {"type": "string", "long_annotation": "a" * 100}
    array = [literal, {"anyOf": [{"type": "integer"}, {"type": "null"}]}]
    schema = {"type": "object", "properties": {
        "enum_object": {"enum": [literal]}, "const_object": {"const": literal},
        "enum_array": {"enum": [array]}, "const_array": {"const": array},
    }, "required": ["enum_object", "const_object", "enum_array", "const_array"],
        "additionalProperties": False}
    before = copy.deepcopy(schema)
    compiled = intern_schema_shapes(schema)
    assert schema == before
    value = {"enum_object": literal, "const_object": literal,
             "enum_array": array, "const_array": array}
    for current in (schema, compiled):
        assert Draft202012Validator(current).is_valid(value)
        assert not Draft202012Validator(current).is_valid({**value, "const_object": {}})
    assert compiled["properties"] == schema["properties"]


def test_intern_respects_existing_definition_name_and_recursive_references() -> None:
    from jsonschema import Draft202012Validator
    from easyicu.research_agent.providers.strict_json_schema import intern_schema_shapes

    text = {"type": "string", "pattern": "[A-Z]+" * 20}
    node = {"type": "object", "properties": {"value": text,
        "child": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]}},
        "required": ["value", "child"], "additionalProperties": False}
    schema = {"$defs": {"_s0": {"type": "number"}, "node": node},
        "type": "object", "properties": {"tree": {"$ref": "#/$defs/node"},
            "label": text}, "required": ["tree", "label"], "additionalProperties": False}
    result = intern_schema_shapes(schema)
    assert result["$defs"]["_s0"] == {"type": "number"}
    assert "_s1" in result["$defs"]
    valid = {"tree": {"value": "A" * 20, "child": {"value": "Z" * 20, "child": None}}, "label": "A" * 20}
    invalid = {**valid, "tree": {"value": "abc", "child": None}}
    for current in (schema, result):
        validator = Draft202012Validator(current)
        assert validator.is_valid(valid)
        assert not validator.is_valid(invalid)


def test_intern_does_not_move_schema_with_nested_identifier_scope() -> None:
    import copy
    from easyicu.research_agent.providers.strict_json_schema import intern_schema_shapes

    nested = {"$id": "https://example.test/inner", "type": "string", "minLength": 100}
    schema = {"type": "object", "properties": {"a": nested, "b": nested},
              "required": ["a", "b"], "additionalProperties": False}
    result = intern_schema_shapes(schema)
    assert result == schema
    assert result is not schema
    assert schema == copy.deepcopy(result)


@pytest.mark.parametrize("reference", ["#/$defs/_s0", "#/$defs/%5Fs0"])
def test_intern_never_turns_dangling_reference_into_a_definition(reference: str) -> None:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import RefResolutionError
    from easyicu.research_agent.providers.strict_json_schema import intern_schema_shapes

    shape = {"type": "string", "pattern": "[A-Z]+" * 20}
    schema = {"type": "object", "properties": {"a": shape, "b": shape,
        "invalid": {"$ref": reference}}, "required": ["a", "b", "invalid"],
        "additionalProperties": False}
    result = intern_schema_shapes(schema)
    assert "_s0" not in result["$defs"]
    assert "_s1" in result["$defs"]
    value = {"a": "A" * 20, "b": "B" * 20, "invalid": 1}
    for current in (schema, result):
        with pytest.raises(RefResolutionError):
            Draft202012Validator(current).is_valid(value)


@pytest.mark.parametrize("reference", ["#/$defs", "#/%24defs"])
@pytest.mark.parametrize("repeated", [False, True])
def test_intern_never_creates_a_missing_root_defs_reference(reference: str, repeated: bool) -> None:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import RefResolutionError
    from easyicu.research_agent.providers.strict_json_schema import intern_schema_shapes

    shape = {"type": "string", "pattern": "[A-Z]+" * 20}
    properties = {"invalid": {"$ref": reference, "type": "string"}}
    value = {"invalid": "test"}
    if repeated:
        properties.update(a=shape, b=shape)
        value.update(a="A" * 20, b="B" * 20)
    schema = {"type": "object", "properties": properties,
              "required": list(properties), "additionalProperties": False}
    result = intern_schema_shapes(schema)
    assert "$defs" not in result
    for current in (schema, result):
        with pytest.raises(RefResolutionError):
            Draft202012Validator(current).is_valid(value)


@pytest.mark.parametrize("reference", [
    "#/properties/a/items", "#/%70roperties/a/items", "#/$defs/old/items", "#/%24defs/old/items",
])
def test_intern_leaves_nested_reference_targets_in_place(reference: str) -> None:
    import copy
    from jsonschema import Draft202012Validator
    from easyicu.research_agent.providers.strict_json_schema import intern_schema_shapes

    shape = {"type": "array", "items": {"type": "string", "minLength": 3}, "minItems": 1}
    schema = {"$defs": {"old": shape}, "type": "object", "properties": {
        "a": shape, "b": shape, "selected": {"$ref": reference}},
        "required": ["a", "b", "selected"], "additionalProperties": False}
    value = {"a": ["abc"], "b": ["xyz"], "selected": "hello"}
    assert Draft202012Validator(schema).is_valid(value)
    result = intern_schema_shapes(schema)
    assert result == schema
    assert result is not schema
    assert schema == copy.deepcopy(result)
    assert Draft202012Validator(result).is_valid(value)
    assert not Draft202012Validator(result).is_valid({**value, "selected": "x"})


@pytest.mark.parametrize("reference", ["#", "#local", "https://example.test/schema", "#/properties/a"])
def test_intern_leaves_other_reference_forms_unchanged(reference: str) -> None:
    from easyicu.research_agent.providers.strict_json_schema import intern_schema_shapes

    shape = {"type": "string", "minLength": 100}
    schema = {"type": "object", "properties": {"a": shape, "b": shape,
        "referenced": {"$ref": reference}}, "required": ["a", "b", "referenced"],
        "additionalProperties": False}
    assert intern_schema_shapes(schema) == schema


def test_intern_does_not_add_unused_defs_without_an_extraction() -> None:
    from easyicu.research_agent.providers.strict_json_schema import intern_schema_shapes

    schema = {"type": "object", "properties": {"x": {"type": "string"}},
              "required": ["x"], "additionalProperties": False}
    assert intern_schema_shapes(schema) == schema
