"""Benchmarks of the constraint package.

They measure ``fhy_core.symbolic.constraint`` before and after it moves to
the Rust core (S13 of ``docs/design/python-switch.md``), through the public
API only. The set rows use integer members unless a row names another
kind; ``_Token`` is a ``Serializable`` member only Python can compare, the
kind the binding keeps behind an opaque adapter (D-S13-3). The rows that
evaluate an equation reach the default solver's simplifier; the questions
reach its SMT backend.
"""

import pickle
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.serialization import Serializable, register_serializable
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintSystem,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
    create_constraint_system,
)
from fhy_core.symbolic.expression import (
    IdentifierExpression,
    LiteralExpression,
    logical_and,
)
from fhy_core.symbolic.symbol_type import SymbolType
from fhy_core.term import AlphaRenaming
from fhy_core.utils.override import override

from .conftest import Benchmark

pytestmark = pytest.mark.benchmark(group="constraint")

# How many members the large set constraints hold, and how many
# constraints the systems join.
_LARGE_SET_SIZE = 100
_SYSTEM_SIZE = 20
# The bound the equation rows compare against.
_BOUND = 10


@register_serializable(type_id="benchmarks.constraint.token")
class _Token(Serializable):
    """A ``Serializable`` member, equal and hashed by its value."""

    def __init__(self, value: int) -> None:
        self._value = value

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Token) and self._value == other._value

    @override
    def __hash__(self) -> int:
        return hash(self._value)

    @override
    def __repr__(self) -> str:
        return f"_Token({self._value})"

    @override
    def serialize_to_dict(self) -> dict[str, Any]:
        return {"value": self._value}

    @classmethod
    @override
    def deserialize_from_dict(cls, data: dict[str, Any]) -> "_Token":
        return cls(int(data["value"]))


@pytest.fixture()
def x() -> Identifier:
    """Return the constrained variable."""
    return Identifier("x")


def _build_large_in_set(x: Identifier) -> InSetConstraint:
    return InSetConstraint(x, range(_LARGE_SET_SIZE))


def _build_bound_system(
    identifiers: list[Identifier],
) -> ConstraintSystem:
    """Return ``_SYSTEM_SIZE`` interval bounds over ``identifiers``."""
    constraints: list[Constraint] = []
    for index in range(_SYSTEM_SIZE):
        reference = IdentifierExpression(identifiers[index % len(identifiers)])
        constraints.append(
            EquationConstraint(reference > -index)
            if index % 2 == 0
            else EquationConstraint(reference < index + 100)
        )
    return create_constraint_system(*constraints)


def _build_set_system(identifiers: list[Identifier]) -> ConstraintSystem:
    """Return ``_SYSTEM_SIZE`` set constraints over ``identifiers``."""
    constraints: list[Constraint] = []
    for index in range(_SYSTEM_SIZE):
        variable = identifiers[index % len(identifiers)]
        kind = InSetConstraint if index % 2 == 0 else NotInSetConstraint
        constraints.append(kind(variable, range(index, index + 10)))
    return create_constraint_system(*constraints)


@pytest.fixture()
def identifiers() -> list[Identifier]:
    """Return five variables."""
    return [Identifier(f"v{index}") for index in range(5)]


def test_equation_constraint_construction(benchmark: Benchmark, x: Identifier) -> None:
    """Benchmark building an equation constraint."""
    expression = IdentifierExpression(x) >= 0
    benchmark(EquationConstraint, expression)


@pytest.mark.parametrize("kind", ["4", "100", "tuples", "serializable"])
def test_set_constraint_construction(
    benchmark: Benchmark, x: Identifier, kind: str
) -> None:
    """Benchmark building a set constraint, normalizing its members."""
    members: list[Any]
    if kind == "4":
        members = [1, 2, 3, 4]
    elif kind == "100":
        members = list(range(_LARGE_SET_SIZE))
    elif kind == "tuples":
        members = [(index, "a", index * 0.5) for index in range(10)]
    else:
        members = [_Token(index) for index in range(10)]
    benchmark(InSetConstraint, x, members)


def test_set_constraint_members(benchmark: Benchmark, x: Identifier) -> None:
    """Benchmark reading the canonical members of a fresh constraint."""

    def read() -> Any:
        return _build_large_in_set(x).members

    benchmark(read)


def test_set_constraint_values(benchmark: Benchmark, x: Identifier) -> None:
    """Benchmark reading the stored values."""
    constraint = _build_large_in_set(x)
    benchmark(lambda: constraint.values)


@pytest.mark.parametrize(
    "kind", ["member", "non_member", "unbound", "literal_expression", "serializable"]
)
def test_set_constraint_evaluate_with_bindings(
    benchmark: Benchmark, x: Identifier, kind: str
) -> None:
    """Benchmark deciding membership under bindings."""
    if kind == "serializable":
        constraint: InSetConstraint = InSetConstraint(
            x, [_Token(index) for index in range(10)]
        )
        bindings: dict[Identifier, Any] = {x: _Token(7)}
    else:
        constraint = _build_large_in_set(x)
        choices: dict[str, dict[Identifier, Any]] = {
            "member": {x: 42},
            "non_member": {x: 1000},
            "unbound": {},
            "literal_expression": {x: LiteralExpression(42)},
        }
        bindings = choices[kind]
    benchmark(constraint.evaluate_with_bindings, bindings)


@pytest.mark.parametrize("kind", ["ground", "partial"])
def test_equation_constraint_evaluate_with_bindings(
    benchmark: Benchmark, identifiers: list[Identifier], kind: str
) -> None:
    """Benchmark an equation decided through the simplifier."""
    a, b = identifiers[0], identifiers[1]
    constraint = EquationConstraint(
        IdentifierExpression(a) + IdentifierExpression(b) <= _BOUND
    )
    bindings = {a: 3, b: 4} if kind == "ground" else {a: 3}
    benchmark(constraint.evaluate_with_bindings, bindings)


def test_set_constraint_convert_to_expression(
    benchmark: Benchmark, x: Identifier
) -> None:
    """Benchmark lowering a 50-member set constraint to an expression."""
    constraint = InSetConstraint(x, range(50))
    benchmark(constraint.convert_to_expression)


@pytest.mark.parametrize("kind", ["equation", "set_100"])
def test_constraint_build_ordering_key(
    benchmark: Benchmark, identifiers: list[Identifier], kind: str
) -> None:
    """Benchmark the canonical ordering key."""
    constraint: Constraint
    if kind == "equation":
        reference = IdentifierExpression(identifiers[0])
        constraint = EquationConstraint(
            logical_and(reference > 0, reference < _BOUND, reference.not_equals(5))
        )
    else:
        constraint = _build_large_in_set(identifiers[0])
    benchmark(constraint.build_ordering_key)


def test_constraint_structural_equivalence_of_large_sets(
    benchmark: Benchmark, x: Identifier
) -> None:
    """Benchmark two equal 100-member set constraints built apart."""
    left = _build_large_in_set(x)
    right = InSetConstraint(x, list(reversed(range(_LARGE_SET_SIZE))))
    assert benchmark(left.is_structurally_equivalent, right) is True


def test_constraint_alpha_equivalence(
    benchmark: Benchmark, identifiers: list[Identifier]
) -> None:
    """Benchmark two equations compared under a free renaming."""
    a, b = identifiers[0], identifiers[1]
    left = EquationConstraint(IdentifierExpression(a) > 0)
    right = EquationConstraint(IdentifierExpression(b) > 0)
    renaming = AlphaRenaming.with_free_renaming({a: b})
    assert benchmark(left.is_alpha_equivalent_under, right, renaming) is True


def test_constraint_serialize_to_dict(benchmark: Benchmark, x: Identifier) -> None:
    """Benchmark serializing a 100-member set constraint."""
    constraint = _build_large_in_set(x)
    benchmark(constraint.serialize_to_dict)


def test_constraint_deserialize_from_dict(benchmark: Benchmark, x: Identifier) -> None:
    """Benchmark deserializing a 100-member set constraint."""
    payload = _build_large_in_set(x).serialize_to_dict()
    benchmark(Constraint.deserialize_from_dict, payload)


def test_constraint_pickle_round_trip(benchmark: Benchmark, x: Identifier) -> None:
    """Benchmark pickling and unpickling an equation constraint."""
    constraint = EquationConstraint(IdentifierExpression(x) >= 0)
    benchmark(lambda: pickle.loads(pickle.dumps(constraint)))


def test_constraint_repr(benchmark: Benchmark, x: Identifier) -> None:
    """Benchmark rendering a 100-member set constraint."""
    constraint = _build_large_in_set(x)
    benchmark(repr, constraint)


def test_constraint_system_construction(
    benchmark: Benchmark, identifiers: list[Identifier]
) -> None:
    """Benchmark building a system of 20 bounds, sorting them canonically."""
    system = _build_bound_system(identifiers)
    members = list(reversed(system.constraints))
    benchmark(create_constraint_system, *members)


@pytest.mark.parametrize("kind", ["sets_20", "mixed"])
def test_constraint_system_evaluate_with_bindings(
    benchmark: Benchmark, identifiers: list[Identifier], kind: str
) -> None:
    """Benchmark folding a system's member outcomes."""
    bindings: dict[Identifier, Any] = dict.fromkeys(identifiers, 9)
    if kind == "sets_20":
        system = _build_set_system(identifiers)
    else:
        a = identifiers[0]
        system = create_constraint_system(
            InSetConstraint(a, range(20)),
            NotInSetConstraint(a, [3, 5]),
            EquationConstraint(IdentifierExpression(a) >= 0),
        )
    benchmark(system.evaluate_with_bindings, bindings)


def test_constraint_system_check_satisfiability_of_bounds(
    benchmark: Benchmark, identifiers: list[Identifier]
) -> None:
    """Benchmark a system's satisfiability question over 20 bounds."""
    system = _build_bound_system(identifiers)
    symbol_types = dict.fromkeys(identifiers, SymbolType.INT)
    benchmark(system.check_satisfiability, symbol_types)


def test_constraint_system_check_satisfiability_with_bindings(
    benchmark: Benchmark, identifiers: list[Identifier]
) -> None:
    """Benchmark a satisfiability question with decided set leaves."""
    a, b = identifiers[0], identifiers[1]
    system = create_constraint_system(
        InSetConstraint(a, range(10)),
        EquationConstraint(IdentifierExpression(a) + IdentifierExpression(b) < _BOUND),
    )
    benchmark(system.check_satisfiability_with_bindings, {a: 3}, {b: SymbolType.INT})


def test_constraint_system_serialize_to_dict(
    benchmark: Benchmark, identifiers: list[Identifier]
) -> None:
    """Benchmark serializing a system of 20 set constraints."""
    system = _build_set_system(identifiers)
    benchmark(system.serialize_to_dict)


def test_constraint_system_structural_equivalence(
    benchmark: Benchmark, identifiers: list[Identifier]
) -> None:
    """Benchmark two equal systems of 20 set constraints built apart."""
    left = _build_set_system(identifiers)
    right = _build_set_system(identifiers)
    assert benchmark(left.is_structurally_equivalent, right) is True
