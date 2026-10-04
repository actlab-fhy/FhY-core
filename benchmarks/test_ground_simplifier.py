"""Benchmarks of ground simplification: the ground simplifier, the chain, SymPy.

They ask ``Solver.simplify_expression`` of a solver holding each simplifier
to simplify, after substituting the environment, the shapes a constraint
check takes once its identifiers are bound: an integer comparison, a bound
check, and a small arithmetic tree. The ground simplifier folds them in
Rust with no SymPy; the chain tries it first and asks SymPy for what it
declines, which these ground shapes never reach; SymPy lowers, simplifies
and lifts them.

``test_free_expression`` asks each of them an expression with a free
identifier, which the ground fold declines, so the chain pays the declined
fold on top of SymPy's work.
"""

from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    logical_and,
)
from fhy_core.symbolic.solver import GroundSimplifier, Solver

from .conftest import Benchmark

pytestmark = pytest.mark.benchmark(group="ground-simplifier")

# The extent a bound check keeps its identifier below.
_TILE_EXTENT = 64

sympy_bridge = pytest.importorskip("fhy_core.symbolic.expression.passes.sympy")

_SIMPLIFIERS: dict[str, Callable[[], Any]] = {
    "ground": GroundSimplifier,
    "chain": lambda: GroundSimplifier(sympy_bridge.SympySimplifier()),
    "sympy": sympy_bridge.SympySimplifier,
}


def _binary(
    operation: BinaryOperation, left: Expression, right: Expression
) -> Expression:
    return BinaryExpression(operation, left, right)


@pytest.fixture(params=sorted(_SIMPLIFIERS))
def solver(request: pytest.FixtureRequest) -> Solver:
    """Return a solver holding each simplifier in turn."""
    simplifier = _SIMPLIFIERS[request.param]()
    solver = Solver(simplifier=simplifier)
    # SymPy imports on its first question: pay that before timing.
    solver.simplify_expression(LiteralExpression(1))
    return solver


def test_integer_comparison(benchmark: Benchmark, solver: Solver) -> None:
    """Benchmark ``x >= 0`` with ``x`` bound to ``3``."""
    x = Identifier("x")
    comparison = IdentifierExpression(x) >= 0
    environment = {x: LiteralExpression(3)}

    result = benchmark(solver.simplify_expression, comparison, environment)

    assert result == LiteralExpression(True)


def test_bound_check(benchmark: Benchmark, solver: Solver) -> None:
    """Benchmark ``0 <= x < 64 and x % 8 == 0`` with ``x`` bound to ``24``."""
    x = Identifier("x")
    reference = IdentifierExpression(x)
    check = logical_and(
        reference >= 0,
        reference < _TILE_EXTENT,
        _binary(
            BinaryOperation.EQUAL,
            _binary(BinaryOperation.MODULO, reference, LiteralExpression(8)),
            LiteralExpression(0),
        ),
    )
    environment = {x: LiteralExpression(24)}

    result = benchmark(solver.simplify_expression, check, environment)

    assert result == LiteralExpression(True)


def test_small_arithmetic_tree(benchmark: Benchmark, solver: Solver) -> None:
    """Benchmark ``(a * b + c) // 4 - a % 3`` with ``a``, ``b``, ``c`` bound."""
    a, b, c = Identifier("a"), Identifier("b"), Identifier("c")
    tree = _binary(
        BinaryOperation.SUBTRACT,
        _binary(
            BinaryOperation.FLOOR_DIVIDE,
            IdentifierExpression(a) * IdentifierExpression(b) + IdentifierExpression(c),
            LiteralExpression(4),
        ),
        _binary(BinaryOperation.MODULO, IdentifierExpression(a), LiteralExpression(3)),
    )
    environment = {
        a: LiteralExpression(7),
        b: LiteralExpression(9),
        c: LiteralExpression(5),
    }

    result = benchmark(solver.simplify_expression, tree, environment)

    assert result == LiteralExpression(15)


def test_free_expression(benchmark: Benchmark, solver: Solver) -> None:
    """Benchmark ``x + y - y`` with nothing bound: the ground fold declines."""
    x, y = Identifier("x"), Identifier("y")
    expression = (
        IdentifierExpression(x) + IdentifierExpression(y)
    ) - IdentifierExpression(y)

    benchmark(solver.simplify_expression, expression)
