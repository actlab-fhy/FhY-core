"""Python-defined parts of Rust values, with registered wire forms.

Each class is a part the Rust core holds behind an adapter: a constraint, a
domain, a type, a frame, and a member value. Their V2 form is a foreign
part, ``{"type_id": .., "data": <canonical JSON text>}``, under the family's
foreign variant. The golden serialization corpus and the interface suites
use them.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from fhy_core.identifier import Identifier
from fhy_core.serialization import Serializable, SerializedDict, register_serializable
from fhy_core.symbol_table import SymbolTableFrame
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintBindings,
    ConstraintOutcome,
)
from fhy_core.symbolic.expression import Expression, LiteralExpression
from fhy_core.symbolic.param import ParamDomain
from fhy_core.symbolic.symbol_type import SymbolType
from fhy_core.types import Type
from fhy_core.utils.override import override

__all__ = [
    "GoldenDomain",
    "GoldenEven",
    "GoldenFrame",
    "GoldenToken",
    "GoldenType",
]


@register_serializable(type_id="golden.token")
class GoldenToken(Serializable):
    """A member value, equal, hashed and ordered by its value."""

    def __init__(self, value: int) -> None:
        self.value = value

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, GoldenToken) and self.value == other.value

    @override
    def __hash__(self) -> int:
        return hash(self.value)

    def __lt__(self, other: "GoldenToken") -> bool:
        return self.value < other.value

    @override
    def __repr__(self) -> str:
        return f"GoldenToken({self.value})"

    @override
    def serialize_to_dict(self) -> SerializedDict:
        return {"value": self.value}

    @classmethod
    @override
    def deserialize_from_dict(cls, data: SerializedDict) -> "GoldenToken":
        value = data["value"]
        if not isinstance(value, int):
            raise TypeError("a token's value is an int")
        return cls(value)


@register_serializable(type_id="golden.even")
@dataclass(frozen=True, eq=False)
class GoldenEven(Constraint):
    """A constraint: its variable's value is even."""

    variable: Identifier

    @override
    def get_free_identifiers(self) -> frozenset[Identifier]:
        return frozenset({self.variable})

    @override
    def evaluate_with_bindings(self, bindings: ConstraintBindings) -> ConstraintOutcome:
        value = bindings.get(self.variable)
        if not isinstance(value, int):
            return ConstraintOutcome.UNDECIDED
        return (
            ConstraintOutcome.SATISFIED
            if value % 2 == 0
            else ConstraintOutcome.VIOLATED
        )

    @override
    def convert_to_expression(self) -> Expression:
        return LiteralExpression(True)

    @override
    def build_ordering_key(self) -> str:
        return f"GoldenEven|{self.variable.id}"

    @override
    def __repr__(self) -> str:
        return f"GoldenEven({self.variable!r})"

    @override
    def __str__(self) -> str:
        return f"even({self.variable!r})"


@register_serializable(type_id="golden.domain")
class GoldenDomain(ParamDomain):
    """A domain of the even integers."""

    @property
    @override
    def symbol_type(self) -> SymbolType | None:
        return SymbolType.INT

    @override
    def is_value_admissible(self, value: Any) -> bool:
        return isinstance(value, int) and value % 2 == 0

    @override
    def normalize_value(self, value: Any) -> Any:
        return value

    @override
    def validate_constraint(self, constraint: Constraint, variable: Identifier) -> None:
        return None

    @override
    def get_implied_constraints(self, variable: Identifier) -> tuple[Constraint, ...]:
        return ()

    @override
    def is_value_set_subset(self, other: ParamDomain) -> bool:
        return False

    @override
    def compute_feasibility_subset(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
    ) -> ConstraintOutcome:
        return ConstraintOutcome.UNDECIDED

    @override
    def has_feasible_value(
        self, constraints: Sequence[Constraint], variable: Identifier
    ) -> ConstraintOutcome:
        return ConstraintOutcome.SATISFIED

    @override
    def compute_intersection(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
        variable: Identifier,
    ) -> tuple[ParamDomain, tuple[Constraint, ...]]:
        return self, ()

    @override
    def is_structurally_equivalent(self, other: object) -> bool:
        return isinstance(other, GoldenDomain)

    @override
    def render_set_string(self) -> str:
        return "2Z"

    @override
    def render_set_repr(self) -> str:
        return ""

    @override
    def serialize_data_to_dict(self) -> SerializedDict:
        return {"modulus": 2}

    @classmethod
    @override
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "GoldenDomain":
        if data != {"modulus": 2}:
            raise ValueError("a golden domain has modulus 2")
        return cls()


@register_serializable(type_id="golden.type")
class GoldenType(Type):
    """A type named by its tag."""

    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag

    @override
    def serialize_data_to_dict(self) -> SerializedDict:
        return {"tag": self.tag}

    @classmethod
    @override
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "GoldenType":
        tag = data["tag"]
        if not isinstance(tag, str):
            raise TypeError("a golden type's tag is a str")
        return cls(tag)


@register_serializable(type_id="golden.frame")
@dataclass(frozen=True)
class GoldenFrame(SymbolTableFrame):
    """A frame with a note."""

    note: str
