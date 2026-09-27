"""Benchmarks of the param package.

They measure ``fhy_core.symbolic.param`` before and after it moves to the
Rust core (S16 of ``docs/design/python-switch.md``), through the public API
only. ``_Level`` is a ``Serializable`` value with ``==``, ``hash`` and
``<``, the kind the binding keeps behind an opaque adapter (D-S16-3). The
rows over a bounded natural or real param reach the default solver: its
simplifier for value checks, its SMT backend for the questions.
"""

import pickle
from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.serialization import Serializable, register_serializable
from fhy_core.symbolic.constraint import (
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.symbolic.expression import IdentifierExpression
from fhy_core.symbolic.param import (
    CategoricalDomain,
    OrdinalDomain,
    Param,
    ParamAssignment,
    ParamError,
    create_categorical_param,
    create_integer_param,
    create_integer_param_between,
    create_intersection_param,
    create_interval_integer_param_between,
    create_natural_param,
    create_natural_param_between,
    create_ordinal_param,
    create_permutation_param,
    create_real_param_between,
    create_union_param,
)
from fhy_core.utils.override import override

from .conftest import Benchmark

pytestmark = pytest.mark.benchmark(group="param")

# How many values the ordinal params hold, and the large domains.
_ORDINAL_SIZE = 20
_LARGE_DOMAIN_SIZE = 100
# The bound the in-set param's equation keeps its candidates above.
_IN_SET_FLOOR = 5


@register_serializable(type_id="benchmarks.param.level")
class _Level(Serializable):
    """A ``Serializable`` value, equal, hashed and ordered by its value."""

    def __init__(self, value: int) -> None:
        self._value = value

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Level) and self._value == other._value

    @override
    def __hash__(self) -> int:
        return hash(self._value)

    def __lt__(self, other: "_Level") -> bool:
        return self._value < other._value

    @override
    def __repr__(self) -> str:
        return f"_Level({self._value})"

    @override
    def serialize_to_dict(self) -> dict[str, Any]:
        return {"value": self._value}

    @classmethod
    @override
    def deserialize_from_dict(cls, data: dict[str, Any]) -> "_Level":
        return cls(int(data["value"]))


def _build_in_set_param() -> Param[Any]:
    """Return an integer param whose in-set constraint makes it finite."""
    variable = Identifier("x")
    return create_integer_param(
        name=variable,
        constraints=[
            InSetConstraint(variable, range(10)),
            NotInSetConstraint(variable, {3, 4}),
            EquationConstraint(IdentifierExpression(variable) > _IN_SET_FLOOR),
        ],
    )


_PARAM_BUILDERS: dict[str, Callable[[], Param[Any]]] = {
    "integer": create_integer_param,
    "natural_between": lambda: create_natural_param_between(1, 10),
    "interval_between": lambda: create_interval_integer_param_between(0, 10),
    "real_between": lambda: create_real_param_between(0.5, "2.5"),
    "ordinal_20": lambda: create_ordinal_param(list(range(_ORDINAL_SIZE))),
    "categorical_4": lambda: create_categorical_param(["a", "b", "c", "d"]),
    "permutation_4": lambda: create_permutation_param([1, 2, 3, 4]),
    "serializable": lambda: create_ordinal_param(
        [_Level(index) for index in range(_ORDINAL_SIZE)]
    ),
    "in_set": _build_in_set_param,
}


def _build_param(kind: str) -> Param[Any]:
    """Return a fresh param of `kind`."""
    return _PARAM_BUILDERS[kind]()


@pytest.mark.parametrize(
    "kind",
    [
        "integer",
        "natural_between",
        "interval_between",
        "ordinal_20",
        "categorical_4",
        "permutation_4",
    ],
)
def test_param_construction(benchmark: Benchmark, kind: str) -> None:
    """Benchmark building a param with a factory: its domain and constraints."""
    benchmark(_build_param, kind)


@pytest.mark.parametrize(
    "kind", ["ordinal_100", "categorical_100", "ordinal_serializable"]
)
def test_domain_construction(benchmark: Benchmark, kind: str) -> None:
    """Benchmark building a finite domain: its checks and its order."""
    if kind == "ordinal_100":
        values: list[Any] = list(reversed(range(_LARGE_DOMAIN_SIZE)))
        benchmark(OrdinalDomain, tuple(values))
    elif kind == "categorical_100":
        values = [f"c{index}" for index in reversed(range(_LARGE_DOMAIN_SIZE))]
        benchmark(CategoricalDomain, tuple(values))
    else:
        values = [_Level(index) for index in reversed(range(_ORDINAL_SIZE))]
        benchmark(OrdinalDomain, tuple(values))


def test_param_attribute_read(benchmark: Benchmark) -> None:
    """Benchmark reading a param's constraints and domain."""
    param = _build_param("natural_between")

    def read() -> object:
        return (param.constraints, param.domain, param.variable)

    benchmark(read)


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("natural_between", 3),
        ("ordinal_20", 7),
        ("categorical_4", "c"),
        ("permutation_4", (4, 3, 2, 1)),
        ("serializable", _Level(7)),
    ],
)
def test_param_is_value_valid(benchmark: Benchmark, kind: str, value: Any) -> None:
    """Benchmark a value check: admissibility, then the constraints."""
    param = _build_param(kind)
    assert benchmark(param.is_value_valid, value) is True


def test_param_validate_value_violation(benchmark: Benchmark) -> None:
    """Benchmark a value check that fails, naming the violated constraint."""
    param = _build_param("natural_between")

    def validate() -> None:
        try:
            param.validate_value(11)
        except ParamError:
            return
        raise AssertionError("11 is valid")

    benchmark(validate)


def test_param_assign(benchmark: Benchmark) -> None:
    """Benchmark assigning a value to a bounded natural param."""
    param = _build_param("natural_between")
    benchmark(param.assign, 3)


@pytest.mark.parametrize(
    "kind",
    ["natural_between", "real_between", "in_set", "ordinal_20", "permutation_4"],
)
def test_param_check_feasibility(benchmark: Benchmark, kind: str) -> None:
    """Benchmark deciding whether a param admits a value."""
    param = _build_param(kind)
    benchmark(param.check_feasibility)


@pytest.mark.parametrize("kind", ["integer", "in_set", "ordinal_20"])
def test_param_check_subset(benchmark: Benchmark, kind: str) -> None:
    """Benchmark deciding whether a param's feasible set is within another's."""
    if kind == "integer":
        left = create_integer_param_between(2, 8)
        right = create_integer_param_between(0, 10)
    elif kind == "in_set":
        left = _build_param("in_set")
        right = create_integer_param_between(0, 10)
    else:
        left = create_ordinal_param(list(range(_ORDINAL_SIZE // 2)))
        right = _build_param("ordinal_20")
    benchmark(left.check_subset, right)


@pytest.mark.parametrize("kind", ["ordinal_20", "categorical_4"])
def test_param_union(benchmark: Benchmark, kind: str) -> None:
    """Benchmark the union of two finite params."""
    left: Param[Any]
    right: Param[Any]
    if kind == "ordinal_20":
        left = create_ordinal_param(list(range(_ORDINAL_SIZE)))
        right = create_ordinal_param(list(range(10, 10 + _ORDINAL_SIZE)))
    else:
        left = create_categorical_param(["a", "b", "c", "d"])
        right = create_categorical_param(["c", "d", "e", "f"])
    benchmark(create_union_param, left, right)


@pytest.mark.parametrize("kind", ["integer", "ordinal_20", "permutation_4"])
def test_param_intersection(benchmark: Benchmark, kind: str) -> None:
    """Benchmark the intersection of two params, with its emptiness check."""
    left: Param[Any]
    right: Param[Any]
    if kind == "integer":
        left = create_integer_param_between(0, 10)
        right = create_integer_param_between(5, 20)
    elif kind == "ordinal_20":
        left = create_ordinal_param(list(range(_ORDINAL_SIZE)))
        right = create_ordinal_param(list(range(10, 10 + _ORDINAL_SIZE)))
    else:
        left = create_permutation_param([1, 2, 3, 4])
        right = create_permutation_param([4, 3, 2, 1])
    benchmark(create_intersection_param, left, right)


@pytest.mark.parametrize("operation", ["add", "sub", "mul", "neg"])
def test_param_arithmetic(benchmark: Benchmark, operation: str) -> None:
    """Benchmark interval arithmetic on two bounded interval params."""
    left = create_interval_integer_param_between(0, 10)
    right = create_interval_integer_param_between(-3, 4)
    if operation == "add":
        benchmark(lambda: left + right)
    elif operation == "sub":
        benchmark(lambda: left - right)
    elif operation == "mul":
        benchmark(lambda: left * right)
    else:
        benchmark(lambda: -left)


def test_param_add_lower_bound(benchmark: Benchmark) -> None:
    """Benchmark adding a gated lower bound to a natural param."""
    param = create_natural_param()
    benchmark(param.add_lower_bound_constraint, 3)


def test_param_structural_equivalence(benchmark: Benchmark) -> None:
    """Benchmark two equal ordinal params built apart, structurally."""
    variable = Identifier("x")
    left = create_ordinal_param(list(range(_ORDINAL_SIZE)), name=variable)
    right = create_ordinal_param(list(range(_ORDINAL_SIZE)), name=variable)
    assert benchmark(left.is_structurally_equivalent, right)


def test_param_alpha_equivalence(benchmark: Benchmark) -> None:
    """Benchmark two bounded natural params over distinct variables."""
    left = _build_param("natural_between")
    right = _build_param("natural_between")
    assert benchmark(left.is_alpha_equivalent, right)


def test_param_serialize_to_dict(benchmark: Benchmark) -> None:
    """Benchmark the payload of a bounded natural param."""
    param = _build_param("natural_between")
    benchmark(param.serialize_to_dict)


def test_param_deserialize_from_dict(benchmark: Benchmark) -> None:
    """Benchmark rebuilding a bounded natural param from its payload."""
    payload = _build_param("natural_between").serialize_to_dict()
    benchmark(Param.deserialize_from_dict, payload)


def test_param_pickle_round_trip(benchmark: Benchmark) -> None:
    """Benchmark pickling and unpickling an ordinal param."""
    param = _build_param("ordinal_20")
    benchmark(lambda: pickle.loads(pickle.dumps(param)))


def test_param_assignment_deserialize_from_dict(benchmark: Benchmark) -> None:
    """Benchmark rebuilding an assignment, which re-checks its value."""
    payload = create_natural_param().assign(3).serialize_to_dict()
    benchmark(ParamAssignment.deserialize_from_dict, payload)


def test_param_repr(benchmark: Benchmark) -> None:
    """Benchmark the ``repr`` of a bounded natural param."""
    param = _build_param("natural_between")
    benchmark(repr, param)


def test_param_str(benchmark: Benchmark) -> None:
    """Benchmark the ``str`` of an ordinal param."""
    param = _build_param("ordinal_20")
    benchmark(str, param)
