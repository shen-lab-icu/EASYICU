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
* A contract for an input that is not declared is refused.  It is a plan that
  has not decided what it reads.
* Every declared input that *has rows* must carry one.  A contract for a
  row-bearing input is the plan saying it consumes the whole thing rather than
  a subset, which is the decision this check exists to force.

  Which inputs owe one is ``schema.inputs_owing_a_consumption_contract`` --
  the same function ``AnalysisStep`` validates with, deliberately not a second
  copy.  A ``statistic:`` input owes nothing, because the product is one finite
  number in a JSON sidecar and ``mode="all_rows"`` over it decides nothing.

  This module used to answer that question itself and answered it more
  strictly, demanding a contract for *every* declared input.  A step that
  satisfied the schema was then refused by the capability sitting behind it, so
  the renderer declined without a word and the figure went to the Coder.
  Measured 2026-07-30: 7 of 21 visualization steps lost their owner that way,
  and in 100% of them the inputs without a contract were exactly the
  statistics.

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

from ...schema import inputs_owing_a_consumption_contract

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
    supported_consumption_modes: frozenset[str] = frozenset({"all_rows"})

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
        if not self.supported_consumption_modes or not (
            self.supported_consumption_modes
            <= {"all_rows", "single_row", "one_per_role"}
        ):
            raise ValueError("renderer capability has unsupported consumption modes")

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
        return _contracts_match(
            step.input_consumption_contracts,
            declared,
            supported_modes=self.supported_consumption_modes,
        )


def _contracts_match(
    contracts: Iterable[Any],
    declared: Sequence[str],
    *,
    supported_modes: frozenset[str],
) -> bool:
    """Every input that owes a contract has one, whole and once, and only those.

    Which inputs owe one is not decided here.  ``AnalysisStep`` already
    validated it for every visualization step, and this asks the same function
    so the two answers cannot drift -- which they had.
    """

    consumption = list(contracts)
    keys = [str(contract.input_key or "").strip() for contract in consumption]
    if len(keys) != len(set(keys)):
        return False
    if set(keys) != inputs_owing_a_consumption_contract(declared):
        return False
    for contract in consumption:
        if contract.mode not in supported_modes:
            return False
        if contract.mode == "one_per_role":
            if not str(contract.role_column or "").strip() or not list(
                contract.expected_roles or ()
            ):
                return False
        elif contract.role_column is not None or contract.expected_roles:
            return False
    return True
