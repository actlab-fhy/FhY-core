"""Benchmarks of expression type checking and the registered-body checks.

They measure ``fhy_core.types.checking``, which the Rust core backs,
through the public API only. The identifiers are looked up through a dict's
``__getitem__``, as a symbol table would answer; the deep tree is the
expression benchmarks' tree, 100 operations over four identifiers.
"""

from collections.abc import Callable, Iterator

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    Expression,
    IdentifierExpression,
    LiteralExpression,
    call,
    logical_and,
)
from fhy_core.symbolic.expression.registry import (
    RegisteredEntry,
    get_registered_entries,
    get_registered_entry,
    register_function,
)
from fhy_core.symbolic.expression.sort import FunctionSort
from fhy_core.types import (
    CoreDataType,
    FhYCoreTypeError,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    Type,
    TypeQualifier,
)
from fhy_core.types.checking import (
    ExpressionTypeChecker,
    check_expression_type,
    check_registered_function_body,
    synthesize_expression_type,
)

from .conftest import Benchmark
from .test_expression import _DEEP_TREE_DEPTH, _build_deep_tree, _build_identifiers
from .test_registry import _restore

pytestmark = pytest.mark.benchmark(group="type_checking")

# How many calls the call-heavy conjunction holds.
_CALL_COUNT = 20

_Lookup = Callable[[Identifier], tuple[Type, TypeQualifier]]


def _scalar(core_data_type: CoreDataType) -> NumericalType:
    return NumericalType(PrimitiveDataType(core_data_type))


def _lookup_of(bindings: dict[Identifier, tuple[Type, TypeQualifier]]) -> _Lookup:
    """Return the dict lookup of `bindings`."""
    return bindings.__getitem__


@pytest.fixture()
def snapshot() -> Iterator[dict[str, RegisteredEntry]]:
    """Return the registry's entries, and restore them after the benchmark."""
    entries = dict(get_registered_entries())
    try:
        yield entries
    finally:
        _restore(entries)


@pytest.fixture()
def x_lookup() -> tuple[Identifier, _Lookup]:
    """Return `x`, an `int32` state variable, and its lookup."""
    x = Identifier("x")
    return x, _lookup_of({x: (_scalar(CoreDataType.INT32), TypeQualifier.STATE)})


@pytest.mark.parametrize("shape", ["identifier", "x_plus_1"])
def test_synthesize_expression_type(
    benchmark: Benchmark, x_lookup: tuple[Identifier, _Lookup], shape: str
) -> None:
    """Synthesize the type of `x` or of `x + 1`: the per-call floor."""
    x, lookup = x_lookup
    expression = (
        IdentifierExpression(x)
        if shape == "identifier"
        else IdentifierExpression(x) + 1
    )
    benchmark(synthesize_expression_type, expression, lookup)


def test_synthesize_expression_type_of_the_deep_tree(benchmark: Benchmark) -> None:
    """Synthesize the type of the 100-operation deep tree: the walk."""
    identifiers = _build_identifiers(4, "v")
    lookup = _lookup_of(
        {
            identifier: (_scalar(CoreDataType.INT64), TypeQualifier.PARAM)
            for identifier in identifiers
        }
    )
    tree = _build_deep_tree(identifiers, _DEEP_TREE_DEPTH)
    benchmark(synthesize_expression_type, tree, lookup)


@pytest.mark.parametrize("case", ["x_plus_1", "literal_into_int8"])
def test_check_expression_type(
    benchmark: Benchmark, x_lookup: tuple[Identifier, _Lookup], case: str
) -> None:
    """Check `x + 1` against `int32`, and the literal `100` against `int8`."""
    x, lookup = x_lookup
    if case == "x_plus_1":
        expression: Expression = IdentifierExpression(x) + 1
        expected = _scalar(CoreDataType.INT32)
    else:
        expression = LiteralExpression(100)
        expected = _scalar(CoreDataType.INT8)
    benchmark(check_expression_type, expression, expected, lookup)


def test_synthesize_of_calls(
    benchmark: Benchmark,
    snapshot: dict[str, RegisteredEntry],
    x_lookup: tuple[Identifier, _Lookup],
) -> None:
    """Synthesize a conjunction of 20 comparisons of built-in and user calls."""
    x, lookup = x_lookup
    parameter = Identifier("p")
    register_function(
        "bench_scale",
        [parameter],
        [FunctionSort.REAL],
        FunctionSort.REAL,
        IdentifierExpression(parameter) * 2,
    )
    comparisons = [
        (call("max", x, index) if index % 2 else call("bench_scale", x)) > index
        for index in range(_CALL_COUNT)
    ]
    benchmark(synthesize_expression_type, logical_and(*comparisons), lookup)


def test_synthesize_of_index_arithmetic(benchmark: Benchmark) -> None:
    """Synthesize `2 * i + 1` over an index type: shift and scale."""
    i, n = Identifier("i"), Identifier("N")
    index = IndexType(LiteralExpression(0), IdentifierExpression(n))
    lookup = _lookup_of({i: (index, TypeQualifier.PARAM)})
    benchmark(synthesize_expression_type, 2 * IdentifierExpression(i) + 1, lookup)


def test_synthesize_with_a_python_resolver(
    benchmark: Benchmark, x_lookup: tuple[Identifier, _Lookup]
) -> None:
    """Synthesize `max(x, 1) + min(x, 2)` through a Python call resolver."""
    x, lookup = x_lookup

    def resolve(name: str) -> RegisteredEntry:
        return get_registered_entry(name)

    checker = ExpressionTypeChecker(lookup, resolve_call_target=resolve)
    benchmark(checker.synthesize, call("max", x, 1) + call("min", x, 2))


def test_type_error_of_a_failing_check(
    benchmark: Benchmark, x_lookup: tuple[Identifier, _Lookup]
) -> None:
    """Fail to check `x + 1.5` against `int32`: the error path and its frame."""
    x, lookup = x_lookup

    def fail() -> None:
        try:
            check_expression_type(
                IdentifierExpression(x) + 1.5, _scalar(CoreDataType.INT32), lookup
            )
        except FhYCoreTypeError:
            return
        raise AssertionError("the check passed")

    benchmark(fail)


def test_expression_type_checker_pass_call(
    benchmark: Benchmark, x_lookup: tuple[Identifier, _Lookup]
) -> None:
    """Run the checker as a pass over `x + 1`: the pass framework's floor."""
    x, lookup = x_lookup
    checker = ExpressionTypeChecker(lookup, resolve_call_target=get_registered_entry)
    benchmark(checker, IdentifierExpression(x) + 1)


def test_check_registered_function_body(benchmark: Benchmark) -> None:
    """Check the body `x * 2 + 1` against the sort `INT`."""
    x = Identifier("x")
    body = IdentifierExpression(x) * 2 + 1
    benchmark(
        check_registered_function_body,
        "f",
        (x,),
        (FunctionSort.INT,),
        FunctionSort.INT,
        body,
        get_registered_entry,
    )
