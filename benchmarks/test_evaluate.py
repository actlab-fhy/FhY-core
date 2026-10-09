"""Benchmarks of the expression evaluators and the pretty formatter.

They measure ``evaluate_expression``, ``evaluate_expression_with_numpy``
and ``ExpressionPrettyFormatter``, which the Rust core backs. The array
rows use seeded ``float64`` data unless a row names another dtype. The deep
tree is the expression benchmarks' tree, 100 operations over four
identifiers, with float bindings, since its integer products overflow
``int64``.
"""

import math
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryOperation,
    CallExpression,
    Expression,
    FunctionSort,
    IdentifierExpression,
    LiteralExpression,
    RegisteredEntry,
    call,
    evaluate_expression,
    evaluate_expression_with_numpy,
    get_native_constant_identifier,
    get_registered_entries,
    logical_and,
    logical_not,
    logical_or,
    make_binary_expression,
    piecewise,
    register_native_function,
)
from fhy_core.symbolic.expression.pprint import ExpressionPrettyFormatter
from fhy_core.testing_patches import set_function_registry_state

from .conftest import Benchmark
from .test_expression import (
    _DEEP_TREE_DEPTH,
    _DEEP_TREE_IDENTIFIER_COUNT,
    _build_deep_tree,
    _build_identifiers,
)

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.benchmark(group="evaluate")

# How many native calls the nested row nests.
_NESTED_CALL_DEPTH = 10
# How many constant references the constant row sums.
_CONSTANT_REFERENCE_COUNT = 100
# How many bindings the unused-bindings row passes, two of them referenced.
_UNUSED_BINDING_COUNT = 100
# The array sizes.
_SMALL = 1_000
_MEDIUM = 10_000
_LARGE = 1_000_000


@pytest.fixture()
def snapshot() -> Iterator[Mapping[str, RegisteredEntry]]:
    """Return the registry's entries, and restore them after the benchmark."""
    entries = dict(get_registered_entries())
    try:
        yield entries
    finally:
        set_function_registry_state(entries)


def _rng() -> Any:
    return np.random.default_rng(0)


def _polynomial(x: Identifier) -> Expression:
    """Return ``x * x + x * 2.0 + 1.0``."""
    reference = IdentifierExpression(x)
    return reference * reference + reference * 2.0 + 1.0


def _deep_tree() -> tuple[Expression, tuple[Identifier, ...]]:
    identifiers = _build_identifiers(_DEEP_TREE_IDENTIFIER_COUNT, "v")
    return _build_deep_tree(identifiers, _DEEP_TREE_DEPTH), identifiers


# ---------------------------------------------------------------------------
# The fold: evaluate_expression
# ---------------------------------------------------------------------------


def test_evaluate_expression_of_a_builtin_native_call(benchmark: Benchmark) -> None:
    """Time folding ``exp(1.0)``: the fold's floor, one pass run."""
    tree = CallExpression("exp", (LiteralExpression(1.0),))
    result = benchmark(evaluate_expression, tree)
    assert isinstance(result, LiteralExpression)


def test_evaluate_expression_of_a_user_native_call(
    benchmark: Benchmark, snapshot: Mapping[str, RegisteredEntry]
) -> None:
    """Time folding a user native backed by ``math.atan2``: the callback."""
    register_native_function(
        "bench_atan2",
        parameter_sorts=[FunctionSort.REAL, FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        implementation=math.atan2,
    )
    tree = CallExpression(
        "bench_atan2", (LiteralExpression(1.0), LiteralExpression(2.0))
    )
    result = benchmark(evaluate_expression, tree)
    assert isinstance(result, LiteralExpression)


def test_evaluate_expression_of_the_deep_tree(benchmark: Benchmark) -> None:
    """Time the fold over a tree with nothing to fold: the walk."""
    tree, _ = _deep_tree()
    benchmark(evaluate_expression, tree)


def test_evaluate_expression_of_nested_native_calls(benchmark: Benchmark) -> None:
    """Time folding ten nested built-in natives over a literal."""
    tree: Expression = LiteralExpression(0.5)
    for level in range(_NESTED_CALL_DEPTH):
        tree = call("sin" if level % 2 else "cos", tree)
    result = benchmark(evaluate_expression, tree)
    assert isinstance(result, LiteralExpression)


def test_evaluate_expression_of_constant_references(benchmark: Benchmark) -> None:
    """Time resolving a sum of 100 references to ``pi`` and ``e``."""
    pi = IdentifierExpression(get_native_constant_identifier("pi"))
    e = IdentifierExpression(get_native_constant_identifier("e"))
    tree: Expression = e
    for index in range(1, _CONSTANT_REFERENCE_COUNT):
        tree = tree + (pi if index % 2 else e)
    benchmark(evaluate_expression, tree)


# ---------------------------------------------------------------------------
# The NumPy evaluator: scalar environments
# ---------------------------------------------------------------------------


def _scalar_case(kind: str) -> tuple[Expression, dict[Identifier, float]]:
    if kind == "poly":
        x = Identifier("x")
        return _polynomial(x), {x: 3.0}
    if kind == "sigmoid":
        x = Identifier("x")
        return call("sigmoid", x), {x: 0.5}
    tree, identifiers = _deep_tree()
    return tree, dict.fromkeys(identifiers, 0.5)


@pytest.mark.parametrize("kind", ["poly", "sigmoid", "deep_tree"])
def test_evaluate_with_numpy_of_scalars(benchmark: Benchmark, kind: str) -> None:
    """Time a scalar environment: the per-call floor."""
    tree, environment = _scalar_case(kind)
    result = benchmark(evaluate_expression_with_numpy, tree, environment)
    assert np.ndim(result) == 0


# ---------------------------------------------------------------------------
# The NumPy evaluator: arrays
# ---------------------------------------------------------------------------


def _array_tree(kind: str, x: Identifier, y: Identifier) -> Expression:
    """Return the tree of an array row over ``x`` and, for ``logical``, ``y``."""
    reference = IdentifierExpression(x)
    trees: dict[str, Expression] = {
        "poly": _polynomial(x),
        "exp": call("exp", x),
        "tanh": call("tanh", x),
        "sigmoid": call("sigmoid", x),
        "piecewise": piecewise((reference > 0, reference), otherwise=-reference),
        "guarded": piecewise(
            (reference >= 0, call("floor", call("sqrt", x))),
            otherwise=LiteralExpression(0),
        ),
        "integer": reference // 7 + reference % 5,
        "logical": logical_or(
            logical_and(reference > 0, IdentifierExpression(y) < 1),
            logical_not(
                make_binary_expression(
                    BinaryOperation.EQUAL, reference, IdentifierExpression(y)
                )
            ),
        ),
    }
    return trees[kind]


def _array_case(case: str) -> tuple[Expression, dict[Identifier, Any]]:
    kind, _, size_text = case.rpartition("-")
    size = int(float(size_text))
    rng = _rng()
    if kind == "deep_tree":
        tree, identifiers = _deep_tree()
        return tree, {identifier: rng.random(size) for identifier in identifiers}
    x = Identifier("x")
    y = Identifier("y")
    if kind == "integer":
        values = rng.integers(-1_000_000, 1_000_000, size, dtype=np.int64)
    else:
        values = rng.standard_normal(size)
    environment = {x: values, y: rng.standard_normal(size)}
    return _array_tree(kind, x, y), environment


@pytest.mark.parametrize(
    "case",
    [
        "poly-1e3",
        "poly-1e6",
        "exp-1e6",
        "tanh-1e6",
        "sigmoid-1e6",
        "piecewise-1e6",
        "guarded-1e6",
        "integer-1e6",
        "logical-1e6",
        "deep_tree-1e4",
    ],
)
def test_evaluate_with_numpy_of_arrays(benchmark: Benchmark, case: str) -> None:
    """Time evaluating over arrays: throughput."""
    tree, environment = _array_case(case)
    result = benchmark(evaluate_expression_with_numpy, tree, environment)
    assert isinstance(result, np.ndarray)


def test_evaluate_with_numpy_of_float32_arrays(benchmark: Benchmark) -> None:
    """Time the polynomial over 10^6 ``float32`` values."""
    x = Identifier("x")
    values = _rng().standard_normal(_LARGE).astype(np.float32)
    result = benchmark(evaluate_expression_with_numpy, _polynomial(x), {x: values})
    assert isinstance(result, np.ndarray)


def test_evaluate_with_numpy_with_unused_bindings(benchmark: Benchmark) -> None:
    """Time an environment of 100 bindings, two of them referenced."""
    identifiers = _build_identifiers(_UNUSED_BINDING_COUNT, "u")
    tree = IdentifierExpression(identifiers[0]) + IdentifierExpression(identifiers[1])
    values = _rng().standard_normal(_SMALL)
    environment = dict.fromkeys(identifiers, values)
    result = benchmark(evaluate_expression_with_numpy, tree, environment)
    assert isinstance(result, np.ndarray)


def test_pretty_formatter_of_the_deep_tree(benchmark: Benchmark) -> None:
    """Time ``ExpressionPrettyFormatter()`` over the deep tree."""
    tree, _ = _deep_tree()
    formatter = ExpressionPrettyFormatter()
    result = benchmark(formatter, tree)
    assert isinstance(result, str)
