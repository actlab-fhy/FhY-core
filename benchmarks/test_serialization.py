"""Benchmarks of the wire formats.

They measure ``fhy_core.serialization`` and the payloads of the classes in
both wire formats: the core's serde format (V2), the default, and the
envelope format (V1), selected through ``wire_version``. Each row is
parametrized by version; the ``v1`` rows run inside
``wire_version(WireVersion.V1)``. The cases use the public API only, with
their own Python-defined classes: a derived dataclass (``_Kernel``), a
Python-defined constraint (``_EvenConstraint``) and a ``Serializable``
member value (``_Level``).
"""

import contextlib
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.provenance import FileProvenance, FusedProvenance, Span
from fhy_core.serialization import (
    Serializable,
    WireVersion,
    deserialize_registry_wrapped_value,
    deserialize_value,
    register_serializable,
    serialize_registry_wrapped_value,
    serialize_value,
    wire_version,
)
from fhy_core.symbol_table import SymbolTable, VariableSymbolTableFrame
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintBindings,
    ConstraintOutcome,
    ConstraintSystem,
    EquationConstraint,
    InSetConstraint,
)
from fhy_core.symbolic.expression import (
    Expression,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.symbolic.param import create_ordinal_param
from fhy_core.types import (
    CoreDataType,
    NumericalType,
    PrimitiveDataType,
    TypeQualifier,
)
from fhy_core.utils.override import override

from .conftest import Benchmark

pytestmark = [
    pytest.mark.benchmark(group="serialization"),
    # Reading V1 warns; the rows measure the reader.
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

_VERSIONS = ("v1", "v2")
# The depth of the deep expression, as `test_expression.py`'s deep tree.
_DEEP_TREE_DEPTH = 100
# How many terms the wide, balanced sum adds.
_WIDE_TERMS = 1_000
# How many literals of each kind the literal case holds.
_LITERALS_PER_KIND = 34
# How many members, constraints, values and symbols the other cases hold.
_SET_SIZE = 100
_SYSTEM_SIZE = 20
_ORDINAL_SIZE = 20
_SYMBOL_COUNT = 20
_FUSED_FILES = 10
_VALUE_COUNT = 100


@register_serializable(type_id="benchmarks.serialization.level")
class _Level(Serializable):
    """A ``Serializable`` member value, equal and hashed by its value."""

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


@register_serializable(type_id="benchmarks.serialization.kernel")
@dataclass(frozen=True)
class _Kernel(Serializable):
    """A derived dataclass nesting a Rust-backed value."""

    name: str
    rank: int
    extent: Expression


@register_serializable(type_id="benchmarks.serialization.even")
@dataclass(frozen=True, eq=False)
class _EvenConstraint(Constraint):
    """A Python-defined constraint: its variable's value is even."""

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
        return f"_EvenConstraint|{self.variable.id}"

    @override
    def __repr__(self) -> str:
        return f"_EvenConstraint({self.variable!r})"

    @override
    def __str__(self) -> str:
        return f"even({self.variable!r})"


def _build_deep_expression() -> Expression:
    identifiers = [Identifier(f"d{index}") for index in range(4)]
    tree: Expression = IdentifierExpression(identifiers[0])
    for level in range(1, _DEEP_TREE_DEPTH + 1):
        if level % 10 == 0:
            tree = -tree
        elif level % 2:
            tree = tree + identifiers[(level // 2) % len(identifiers)]
        else:
            tree = tree * LiteralExpression(level)
    return tree


def _build_wide_expression() -> Expression:
    """Return a balanced sum of `_WIDE_TERMS` terms, about 10 levels deep.

    Balanced, since V1's writer recurses once per level and a chain of
    1,000 additions would exceed Python's recursion limit.
    """
    identifiers = [Identifier(f"w{index}") for index in range(10)]
    terms: list[Expression] = [
        IdentifierExpression(identifiers[index % len(identifiers)])
        for index in range(_WIDE_TERMS)
    ]
    while len(terms) > 1:
        pairs = [
            left + right for left, right in zip(terms[::2], terms[1::2], strict=False)
        ]
        terms = pairs + terms[len(pairs) * 2 :]
    return terms[0]


def _build_literals() -> Expression:
    tree: Expression = LiteralExpression(0)
    for index in range(_LITERALS_PER_KIND):
        tree = tree + LiteralExpression(index + 0.25)
        tree = tree + LiteralExpression(2**80 + index)
        tree = tree + LiteralExpression(Decimal(f"{index}.125"))
    return tree


def _build_provenance() -> FusedProvenance:
    return FusedProvenance(
        sources=tuple(
            FileProvenance(Path(f"src/f{index}.fhy"), Span(index, index + 3))
            for index in range(_FUSED_FILES)
        ),
        metadata="fuse",
    )


def _build_type() -> NumericalType:
    return NumericalType(
        PrimitiveDataType(CoreDataType.INT32),
        [LiteralExpression(4), IdentifierExpression(Identifier("N"))],
    )


def _build_set_constraint() -> InSetConstraint:
    return InSetConstraint(Identifier("s"), range(_SET_SIZE))


def _build_constraint_system() -> ConstraintSystem:
    variables = [Identifier(f"c{index}") for index in range(_SYSTEM_SIZE)]
    constraints: list[Constraint] = []
    for index, variable in enumerate(variables):
        if index % 2:
            constraints.append(InSetConstraint(variable, range(5)))
        else:
            constraints.append(
                EquationConstraint(IdentifierExpression(variable) > index)
            )
    return ConstraintSystem(tuple(constraints))


def _build_symbol_table() -> SymbolTable:
    int32 = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    namespace = Identifier("ns")
    table = SymbolTable()
    table.add_namespace(namespace)
    for index in range(_SYMBOL_COUNT):
        name = Identifier(f"v{index}")
        table.add_symbol(
            namespace, name, VariableSymbolTableFrame(name, int32, TypeQualifier.STATE)
        )
    return table


def _build_kernel() -> _Kernel:
    return _Kernel("gemm", 2, IdentifierExpression(Identifier("K")) * 4)


def _build_foreign() -> ConstraintSystem:
    variables = [Identifier(f"f{index}") for index in range(10)]
    constraints: list[Constraint] = [
        EquationConstraint(IdentifierExpression(variable) > 0)
        for variable in variables[1:]
    ]
    constraints.append(_EvenConstraint(variables[0]))
    constraints.append(
        InSetConstraint(variables[1], [_Level(index) for index in range(20)])
    )
    return ConstraintSystem(tuple(constraints))


_CASES: dict[str, Callable[[], Serializable]] = {
    "deep_expression": _build_deep_expression,
    "wide_expression": _build_wide_expression,
    "literals": _build_literals,
    "provenance": _build_provenance,
    "type": _build_type,
    "set_constraint_100": _build_set_constraint,
    "constraint_system_20": _build_constraint_system,
    "param_ordinal_20": lambda: create_ordinal_param(list(range(_ORDINAL_SIZE))),
    "symbol_table_20": _build_symbol_table,
    "kernel": _build_kernel,
    "foreign": _build_foreign,
}


@contextlib.contextmanager
def _version(version: str) -> Iterator[None]:
    """Write in `version`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with wire_version(WireVersion[version.upper()]):
            yield


def _decoder(instance: Serializable) -> Callable[[Any], Any]:
    """Return the `deserialize_from_dict` that reads `instance`'s payload."""
    if isinstance(instance, Expression):
        return Expression.deserialize_from_dict
    if isinstance(instance, Constraint) and not isinstance(instance, ConstraintSystem):
        return Constraint.deserialize_from_dict
    return type(instance).deserialize_from_dict


def _json_decoder(instance: Serializable) -> Callable[[Any], Any]:
    """Return the `from_json` that reads `instance`'s text."""
    if isinstance(instance, Expression):
        return Expression.from_json
    return type(instance).from_json


_CASE_IDS = [f"{version}-{case}" for version in _VERSIONS for case in _CASES]
_CASE_PARAMS = [(version, case) for version in _VERSIONS for case in _CASES]


@pytest.mark.parametrize(("version", "case"), _CASE_PARAMS, ids=_CASE_IDS)
def test_serialize_to_dict(benchmark: Benchmark, version: str, case: str) -> None:
    """Benchmark writing a value's dict payload."""
    instance = _CASES[case]()
    with _version(version):
        benchmark(instance.serialize_to_dict)


@pytest.mark.parametrize(("version", "case"), _CASE_PARAMS, ids=_CASE_IDS)
def test_deserialize_from_dict(benchmark: Benchmark, version: str, case: str) -> None:
    """Benchmark reading a value back from its dict payload."""
    instance = _CASES[case]()
    with _version(version):
        payload = instance.serialize_to_dict()
    benchmark(_decoder(instance), payload)


@pytest.mark.parametrize(("version", "case"), _CASE_PARAMS, ids=_CASE_IDS)
def test_json_round_trip(benchmark: Benchmark, version: str, case: str) -> None:
    """Benchmark writing a value's JSON text and reading it back."""
    instance = _CASES[case]()
    decode = _json_decoder(instance)
    with _version(version):
        benchmark(lambda: decode(instance.to_json()))


@pytest.mark.parametrize("version", _VERSIONS)
@pytest.mark.parametrize("case", ["deep_expression", "param_ordinal_20"])
def test_bytes_round_trip(benchmark: Benchmark, version: str, case: str) -> None:
    """Benchmark writing a value's binary blob and reading it back."""
    instance = _CASES[case]()
    with _version(version):
        benchmark(lambda: Serializable.from_bytes(instance.to_bytes()))


def _mixed_values() -> list[Any]:
    values: list[Any] = []
    for index in range(_VALUE_COUNT // 4):
        values += [index, f"s{index}", index + 0.5, (index, index + 1)]
    return values


@pytest.mark.parametrize("version", _VERSIONS)
def test_value_round_trip(benchmark: Benchmark, version: str) -> None:
    """Benchmark 100 mixed member values through the value functions."""
    values = _mixed_values()
    encode: Callable[[Any], Any] = serialize_registry_wrapped_value
    decode: Callable[[Any], Any] = deserialize_registry_wrapped_value
    if version == "v2":
        encode, decode = serialize_value, deserialize_value

    def round_trip() -> list[Any]:
        return [decode(encode(value)) for value in values]

    assert benchmark(round_trip) == values
