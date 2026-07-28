"""Whether a renderer can read the typed inputs a step declares.

Every figure executor used to answer this itself, and every one answered it as
``tuple(step.inputs) == <its exact constant>``.  That is an enumerated shape,
not a capability, and it fails in two directions:

* **Too strict.**  Equality on a tuple is order-sensitive, so the same two
  tables declared the other way round are refused by a renderer that reads
  both by key.  Nothing about the rendering depends on the order the Planner
  happened to list them in.
* **Too blunt.**  A renderer that can work without a secondary panel has no
  way to say so, so an otherwise complete declaration is refused for a table
  the renderer does not need.

The replacement asks the renderer what it can read: which typed inputs it
*requires*, and which it can do without.  The shape of the declaration then
follows from the capability instead of being remembered alongside it.

What this deliberately does **not** relax:

* An input the renderer does not know is refusal, not something to ignore.  A
  step declaring an extra table is asking for a figure that reads it, and a
  renderer that quietly dropped it would publish a figure answering a
  different question than the plan promised.
* Duplicates are refused.  Two identical keys cannot both be bound, and taking
  the set would hide the contradiction.
* The declared inputs and the consumption contracts must be the same set.  A
  contract for an input that is not declared, or an input with no contract, is
  a plan that has not decided what it reads.

Marking an input optional is a claim about the *renderer*: that its code path
produces a correct figure when that binding is absent.  Declaring an input
optional that the rendering code then indexes turns a clean decline into a
runtime failure inside the sandbox, which is strictly worse.  Every renderer
today requires each of its inputs; the optional set exists so that the claim
is stated where it can be checked, not so it can be assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

__all__ = [
    "TypedInputCapability",
    "declared_typed_inputs",
]


def declared_typed_inputs(step: Any) -> tuple[str, ...]:
    """Return the step's declared inputs as trimmed strings, order preserved."""

    return tuple(str(value or "").strip() for value in step.inputs or ())


@dataclass(frozen=True, slots=True)
class TypedInputCapability:
    """The typed product inputs one renderer can read."""

    required: frozenset[str]
    optional: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.required:
            raise ValueError("a renderer capability must require at least one input")
        overlap = self.required & self.optional
        if overlap:
            raise ValueError(
                "an input cannot be both required and optional: "
                + ", ".join(sorted(overlap))
            )
        for key in self.required | self.optional:
            if not key or ":" not in key:
                raise ValueError(f"capability inputs must be typed keys: {key!r}")

    @property
    def readable(self) -> frozenset[str]:
        return self.required | self.optional

    def admits(self, declared: Sequence[str]) -> bool:
        """Whether this renderer can read exactly what the step declared."""

        values = [str(value or "").strip() for value in declared]
        if not values or any(not value for value in values):
            return False
        if len(set(values)) != len(values):
            return False
        if not self.required <= set(values):
            return False
        return set(values) <= self.readable

    def admits_step(self, step: Any) -> bool:
        """Whether the declared inputs *and* their contracts are readable."""

        declared = declared_typed_inputs(step)
        if not self.admits(declared):
            return False
        return _contracts_match(step.input_consumption_contracts, declared)


def _contracts_match(contracts: Iterable[Any], declared: Sequence[str]) -> bool:
    """Every declared input is consumed whole, once, and nothing else is."""

    consumption = list(contracts)
    keys = [str(contract.input_key or "").strip() for contract in consumption]
    if len(keys) != len(declared) or set(keys) != set(declared):
        return False
    return all(
        contract.mode == "all_rows"
        and contract.role_column is None
        and not contract.expected_roles
        for contract in consumption
    )
