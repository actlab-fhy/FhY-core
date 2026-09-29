"""Benchmarks of the SymPy bridge and simplifier.

They measure the bridge, whose lowering, simplification and lifting run in
the Rust core, through the public functions of
``fhy_core.symbolic.expression.passes.sympy``, ``SympySimplifier`` and
``simplify_expression``.

The trees reuse the expression benchmarks' deep tree, 100 operations over
four identifiers. ``test_first_simplification_in_a_fresh_interpreter``
times a fresh interpreter importing the package and simplifying once, five
rounds, so it covers the lazy load of SymPy and of the backend.
"""

import subprocess
import sys
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    Expression,
    IdentifierExpression,
    LiteralExpression,
    logical_and,
    piecewise,
)
from fhy_core.symbolic.solver import simplify_expression

from .conftest import Benchmark
from .test_expression import (
    _DEEP_TREE_DEPTH,
    _DEEP_TREE_IDENTIFIER_COUNT,
    _build_deep_tree,
    _build_identifiers,
)

sympy_bridge = pytest.importorskip("fhy_core.symbolic.expression.passes.sympy")

pytestmark = [pytest.mark.benchmark(group="sympy"), pytest.mark.sympy]

# How many fresh interpreters the first-simplification row starts.
_FRESH_ROUNDS = 5
_FRESH_SIMPLIFICATION = (
    "from fhy_core.identifier import Identifier\n"
    "from fhy_core.symbolic.expression import IdentifierExpression, "
    "LiteralExpression\n"
    "from fhy_core.symbolic.solver import simplify_expression\n"
    "x = Identifier('x')\n"
    "result = simplify_expression(IdentifierExpression(x) >= 0, "
    "{x: LiteralExpression(3)})\n"
    "assert result == LiteralExpression(True)\n"
)


@pytest.fixture()
def deep_identifiers() -> tuple[Identifier, ...]:
    """Return the identifiers the deep tree is built over."""
    return _build_identifiers(_DEEP_TREE_IDENTIFIER_COUNT, "v")


@pytest.fixture()
def deep_tree(deep_identifiers: tuple[Identifier, ...]) -> Expression:
    """Return a tree `_DEEP_TREE_DEPTH` operations deep."""
    return _build_deep_tree(deep_identifiers, _DEEP_TREE_DEPTH)


@pytest.fixture()
def lowered_deep_tree(deep_tree: Expression) -> Any:
    """Return the SymPy form of the deep tree."""
    return sympy_bridge.convert_expression_to_sympy_expression(deep_tree)


def test_lower_to_sympy_of_a_deep_tree(
    benchmark: Benchmark, deep_tree: Expression
) -> None:
    """Benchmark lowering the deep tree to SymPy."""
    benchmark(sympy_bridge.convert_expression_to_sympy_expression, deep_tree)


def test_lift_from_sympy_of_a_deep_tree(
    benchmark: Benchmark, lowered_deep_tree: Any
) -> None:
    """Benchmark lifting the SymPy form of the deep tree."""
    benchmark(sympy_bridge.convert_sympy_expression_to_expression, lowered_deep_tree)


def test_substitute_sympy_variables_of_a_deep_tree(
    benchmark: Benchmark,
    lowered_deep_tree: Any,
    deep_identifiers: tuple[Identifier, ...],
) -> None:
    """Benchmark substituting a literal for each identifier of the deep tree."""
    environment: dict[Identifier, Expression] = {
        identifier: LiteralExpression(index + 2)
        for index, identifier in enumerate(deep_identifiers)
    }
    benchmark(
        sympy_bridge.substitute_sympy_expression_variables,
        lowered_deep_tree,
        environment,
    )


def test_sympy_simplifier_of_a_ground_comparison(benchmark: Benchmark) -> None:
    """Benchmark the simplifier alone, without the solver around it."""
    simplifier = sympy_bridge.SympySimplifier()
    expression = LiteralExpression(3) < LiteralExpression(10)
    assert benchmark(simplifier.simplify, expression) == LiteralExpression(True)


def test_simplify_expression_of_a_bound_piecewise(benchmark: Benchmark) -> None:
    """Benchmark a bound piecewise with Boolean case conditions."""
    x, b = Identifier("x"), Identifier("b")
    reference = IdentifierExpression(x)
    expression = (
        piecewise(
            (IdentifierExpression(b), reference + 1),
            (reference > 0, reference * 2),
            otherwise=reference - 1,
        )
        > 3  # noqa: PLR2004
    )
    environment: dict[Identifier, Expression] = {
        x: LiteralExpression(2),
        b: LiteralExpression(False),
    }
    result = benchmark(simplify_expression, expression, environment)
    assert result == LiteralExpression(True)


def test_simplify_expression_of_a_boolean_comparison(benchmark: Benchmark) -> None:
    """Benchmark `(x < 1) == b` conjoined with `x > -5`, with `b` bound."""
    x, b = Identifier("x"), Identifier("b")
    reference = IdentifierExpression(x)
    expression = logical_and(
        (reference < 1).equals(IdentifierExpression(b)),
        reference > -5,  # noqa: PLR2004
    )
    environment: dict[Identifier, Expression] = {b: LiteralExpression(True)}
    benchmark(simplify_expression, expression, environment)


def _simplify_in_a_fresh_interpreter() -> None:
    subprocess.run([sys.executable, "-c", _FRESH_SIMPLIFICATION], check=True)


def test_first_simplification_in_a_fresh_interpreter(benchmark: Benchmark) -> None:
    """Benchmark importing the package and simplifying once, in a new process."""
    benchmark.pedantic(_simplify_in_a_fresh_interpreter, rounds=_FRESH_ROUNDS)
