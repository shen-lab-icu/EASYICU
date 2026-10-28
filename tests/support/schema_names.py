"""Inspect compact provider schemas using their authority model type names."""

import json
import re

from easyicu.research_agent.agents import plan_payload
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanSkeleton, ProgressivePlanOutline, ProgressiveFoundationMaterialization,
    ProgressiveStepMaterialization, ProgressiveSuffixRevision,
)
from easyicu.research_agent.providers.strict_json_schema import transport_definition_name


def authority_schema_json(source):
    wire = json.loads(source)
    if not isinstance(wire, dict) or "$defs" not in wire:
        return wire
    definitions = wire["$defs"]
    names = set(plan_payload._planner_transport_schema()["$defs"])
    for model in (ProgressivePlanSkeleton, ProgressivePlanOutline, ProgressiveFoundationMaterialization,
                  ProgressiveStepMaterialization, ProgressiveSuffixRevision):
        names.update(model.model_json_schema().get("$defs", {}))
    reverse = {transport_definition_name(name): name for name in names}
    def restore(node):
        if isinstance(node, list):
            return [restore(value) for value in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            name = node["$ref"].removeprefix("#/$defs/")
            if re.fullmatch(r"_s\d+", name):
                return restore(definitions[name])
            return {"$ref": "#/$defs/" + reverse.get(name, name)}
        return {key: ({reverse.get(name, name): restore(shape) for name, shape in value.items()
                       if not re.fullmatch(r"_s\d+", name)} if key == "$defs" else restore(value))
                for key, value in node.items()}
    return restore(wire)
