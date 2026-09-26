"""Benchmarks of the solver, its bridges, and the layers that ask it.

They measure the solver API before and after it moves to the Rust core
(S8 of ``docs/design/python-switch.md``). Every call whose spelling S8
changes sits in a helper marked with its decision:

- :func:`_ask_an_instant_backend` asks a satisfiability question of a
  backend that answers at once, so the row measures the screens and the
  checks around the backend. Before S8 it replaced the z3 bridge's
  implication; after S8 it asks a ``Solver`` holding a Python
  ``SmtSolver`` that answers ``sat`` (D-S8-11), so the row also lowers the
  question and calls the backend once.

The trees reuse the expression benchmarks' deep tree, 100 operations over
four identifiers. ``test_lower_to_smtlib2_of_a_deep_tree`` exists only
after S8. ``test_import_fhy_core`` times a fresh interpreter importing the
package, five rounds (D-S8-16).
"""

import subprocess
import sys
from collections.abc import Mapping

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic import solver
from fhy_core.symbolic.constraint import EquationConstraint, create_constraint_system
from fhy_core.symbolic.expression import (
    Expression,
    IdentifierExpression,
    LiteralExpression,
    convert_expression_to_z3_expression,
    logical_and,
)
from fhy_core.symbolic.param import (
    create_integer_param_between,
    create_intersection_param,
    create_natural_param,
)
from fhy_core.symbolic.symbol_type import SymbolType
from fhy_core.utils.override import override

from .conftest import Benchmark
from .test_expression import (
    _DEEP_TREE_DEPTH,
    _DEEP_TREE_IDENTIFIER_COUNT,
    _build_deep_tree,
    _build_identifiers,
)

pytestmark = pytest.mark.benchmark(group="solver")

# How many interval bounds the benchmarked conjunction joins, and how many
# identifiers they cycle through.
_BOUND_COUNT = 50
_BOUND_IDENTIFIER_COUNT = 5
# How many fresh interpreters the import row starts.
_IMPORT_ROUNDS = 5


class _InstantSmtSolver(solver.SmtSolver):
    """An SMT backend that finds every script satisfiable at once."""

    @override
    def check(
        self, script: solver.SmtScript, *, timeout_milliseconds: int | None
    ) -> solver.SatResult:
        return solver.SatResult.SAT


_INSTANT_SOLVER = solver.Solver(smt_solver=_InstantSmtSolver())


def _ask_an_instant_backend(
    expression: Expression, symbol_types: Mapping[Identifier, SymbolType]
) -> bool | None:
    """Ask whether `expression` is satisfiable of a backend that answers at once.

    D-S8-11: a ``Solver`` holding a Python ``SmtSolver`` that answers
    ``sat``, so the call runs the capability, timeout, symbol-type,
    ill-typedness and hazard checks, the lowering, and one Python call.
    """
    return _INSTANT_SOLVER.check_expression_satisfiability(expression, symbol_types)


@pytest.fixture()
def deep_identifiers() -> tuple[Identifier, ...]:
    """Return the identifiers the deep tree is built over."""
    return _build_identifiers(_DEEP_TREE_IDENTIFIER_COUNT, "v")


@pytest.fixture()
def deep_tree(deep_identifiers: tuple[Identifier, ...]) -> Expression:
    """Return a tree `_DEEP_TREE_DEPTH` operations deep."""
    return _build_deep_tree(deep_identifiers, _DEEP_TREE_DEPTH)


@pytest.fixture()
def deep_symbol_types(
    deep_identifiers: tuple[Identifier, ...],
) -> dict[Identifier, SymbolType]:
    """Return an integer sort for each identifier of the deep tree."""
    return dict.fromkeys(deep_identifiers, SymbolType.INT)


@pytest.fixture()
def x() -> Identifier:
    """Return an integer variable."""
    return Identifier("x")


def test_screen_of_a_deep_predicate(
    benchmark: Benchmark,
    deep_tree: Expression,
    deep_symbol_types: dict[Identifier, SymbolType],
) -> None:
    """Benchmark the screens over a 100-operation comparison."""
    predicate = deep_tree > 0
    assert benchmark(_ask_an_instant_backend, predicate, deep_symbol_types) is True


def test_lower_to_z3_of_a_deep_tree(
    benchmark: Benchmark,
    deep_tree: Expression,
    deep_symbol_types: dict[Identifier, SymbolType],
) -> None:
    """Benchmark lowering the deep tree to a z3 term."""
    benchmark(convert_expression_to_z3_expression, deep_tree, deep_symbol_types)


def test_lower_to_smtlib2_of_a_deep_tree(
    benchmark: Benchmark,
    deep_tree: Expression,
    deep_symbol_types: dict[Identifier, SymbolType],
) -> None:
    """Benchmark lowering the deep tree to SMT-LIB2 text (after S8 only)."""
    benchmark(solver.convert_expression_to_smtlib2, deep_tree, deep_symbol_types)


def test_check_satisfiability_of_bounds(benchmark: Benchmark, x: Identifier) -> None:
    """Benchmark the smallest satisfiability question: `0 < x && x < 10`."""
    reference = IdentifierExpression(x)
    expression = logical_and(reference > 0, reference < 10)  # noqa: PLR2004
    result = benchmark(
        solver.check_expression_satisfiability, expression, {x: SymbolType.INT}
    )
    assert result is True


def test_check_satisfiability_of_a_conjunction_of_50_bounds(
    benchmark: Benchmark,
) -> None:
    """Benchmark 50 interval bounds over five identifiers."""
    identifiers = _build_identifiers(_BOUND_IDENTIFIER_COUNT, "b")
    bounds: list[Expression] = []
    for index in range(_BOUND_COUNT):
        reference = IdentifierExpression(identifiers[index % len(identifiers)])
        bounds.append(reference > -index if index % 2 else reference < index + 100)
    expression = logical_and(*bounds)
    symbol_types = dict.fromkeys(identifiers, SymbolType.INT)
    result = benchmark(solver.check_expression_satisfiability, expression, symbol_types)
    assert result is True


def test_does_expression_imply_of_bounds(benchmark: Benchmark, x: Identifier) -> None:
    """Benchmark `x >= 1` implies `x >= 0`."""
    reference = IdentifierExpression(x)
    result = benchmark(
        solver.does_expression_imply,
        reference >= 1,
        reference >= 0,
        {x: SymbolType.INT},
    )
    assert result is True


def test_holds_for_all_free_assignments_with_a_witness(
    benchmark: Benchmark, x: Identifier
) -> None:
    """Benchmark the quantified question: for every `x` there is a `y > x`."""
    y = Identifier("y")
    expression = IdentifierExpression(y) > IdentifierExpression(x)
    result = benchmark(
        solver.holds_for_all_free_assignments,
        frozenset({y}),
        expression,
        {x: SymbolType.INT, y: SymbolType.INT},
    )
    assert result is True


def test_check_satisfiability_refused_by_the_screen(
    benchmark: Benchmark, x: Identifier
) -> None:
    """Benchmark a question the screen refuses: a division by a variable."""
    y = Identifier("y")
    expression = IdentifierExpression(x) / IdentifierExpression(y) > 0
    result = benchmark(
        solver.check_expression_satisfiability,
        expression,
        {x: SymbolType.REAL, y: SymbolType.REAL},
    )
    assert result is None


def test_simplify_expression_of_a_ground_comparison(
    benchmark: Benchmark, x: Identifier
) -> None:
    """Benchmark simplifying a fully bound comparison, as a value check does."""
    expression = IdentifierExpression(x) >= 0
    environment: dict[Identifier, Expression] = {x: LiteralExpression(3)}
    result = benchmark(solver.simplify_expression, expression, environment)
    assert result == LiteralExpression(True)


def test_simplify_expression_symbolic(benchmark: Benchmark, x: Identifier) -> None:
    """Benchmark simplifying `x + x - x`."""
    reference = IdentifierExpression(x)
    benchmark(solver.simplify_expression, reference + reference - reference)


def test_equation_constraint_evaluate_with_bindings(
    benchmark: Benchmark, x: Identifier
) -> None:
    """Benchmark the constraint layer over the simplifier."""
    constraint = EquationConstraint(IdentifierExpression(x) >= 0)
    benchmark(constraint.evaluate_with_bindings, {x: 3})


def test_constraint_system_check_implication(
    benchmark: Benchmark, x: Identifier
) -> None:
    """Benchmark the constraint layer over the implication."""
    reference = IdentifierExpression(x)
    antecedent = create_constraint_system(EquationConstraint(reference >= 1))
    consequent = create_constraint_system(EquationConstraint(reference >= 0))
    benchmark(antecedent.check_implication, consequent, {x: SymbolType.INT})


def test_nat_param_is_value_valid(benchmark: Benchmark) -> None:
    """Benchmark a nat param's value check."""
    param = create_natural_param()
    assert benchmark(param.is_value_valid, 3) is True


def test_int_param_intersection_feasibility(benchmark: Benchmark) -> None:
    """Benchmark a param intersection whose feasibility asks the solver."""
    left = create_integer_param_between(0, 10)
    right = create_integer_param_between(5, 20)
    benchmark(create_intersection_param, left, right)


def _import_fhy_core() -> None:
    subprocess.run([sys.executable, "-c", "import fhy_core"], check=True)


def test_import_fhy_core(benchmark: Benchmark) -> None:
    """Benchmark a fresh interpreter importing the package (D-S8-16)."""
    benchmark.pedantic(_import_fhy_core, rounds=_IMPORT_ROUNDS)
