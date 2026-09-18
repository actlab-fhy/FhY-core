"""Hypothesis property tests for the solver seam.

Covers ``simplify_expression`` (SymPy backend: evaluation-preserving on
integer and Boolean trees, evaluates fully under a total literal
environment, and simplifying twice still preserves evaluation) and the
Z3-backed query trio -- satisfiability, implication, and universal
validity -- cross-checked against brute-force enumeration over a small
bounded integer domain and both values of every Boolean identifier.

Every property draws from two identifier pools: integer identifiers, and
Boolean identifiers that stand only where a Boolean belongs (a piecewise
condition, a Boolean piecewise branch, a connective operand, or an
operand of ``==``/``!=`` between Booleans). Environments bind the first
to ints and the second to bools, and the Z3 queries declare them
``INT`` and ``BOOL``.

The integer-tree and boolean-tree simplify properties, which the NumPy
oracle judges, draw division-free trees. A ``FLOOR_DIVIDE`` lowers to
``floor(a / b)``, and a simplified quotient can hold rational
coefficients, as in ``floor(-v0**2 / 5 - v0 / 5)``. The oracle computes
those by float true division, and ``floor`` can turn the rounding error
into an off-by-one. The full-environment and twice-simplified properties
do draw division, since both are judged exactly: the first binds every
identifier before SymPy simplifies and checks against
``evaluate_with_python``; the second simplifies twice with nothing bound,
then binds every identifier of the result to a literal, which SymPy
evaluates in exact rational arithmetic, and checks that value against
``evaluate_with_python`` on the original tree. SymPy lifts a
``Rational`` to decimal text only when a binary float equals it, and to
an exact integer ``DIVIDE`` otherwise;
``test_simplify_expression_preserves_a_rational_coefficient_comparison``
pins one such ``DIVIDE``. ``simplify_expression`` is not structurally
idempotent; ``tests/symbolic/test_solver.py`` pins a counterexample as a
strict xfail. The Z3-backed queries draw division-free trees too: a
symbolic ``FLOOR_DIVIDE``/``MODULO`` is one of the solver module's
documented divergences. The expression strategies thread every option
through every subtree they draw (including a piecewise condition or
value), so ``include_division=False`` at the root is enough to keep the
whole tree division-free.

The SymPy-backed properties additionally enable calls, but only to
:data:`SYMPY_STABLE_CALL_FUNCTIONS` (``floor``, ``ceil``, ``round``),
the natives that survive the SymPy round trip: ``sign``, an
expression-bodied ``RegisteredFunction`` that ``simplify_expression``
could not lower without first running ``inline_functions``, is not one
of the gate grammar's native functions at all.

The Z3-backed queries keep calls off entirely: the Z3 bridge refuses
every native function call outright
(``TypeError: Z3 does not support native function calls; {name!r} cannot
be lowered``, from ``ExpressionToZ3Converter.visit_call_expression``),
so a call anywhere in one of their trees would raise before the query
could run. Piecewise stays enabled for them: ``ExpressionToZ3Converter``
lowers it to a right-folded ``z3.If`` chain with no such restriction.
"""

import itertools
from collections.abc import Iterator, Sequence
from typing import Final

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryOperation,
    Expression,
    LiteralExpression,
    evaluate_expression_with_numpy,
    logical_and,
    logical_not,
    logical_or,
    make_binary_expression,
)
from fhy_core.symbolic.solver import (
    check_expression_satisfiability,
    does_expression_imply,
    holds_for_all_free_assignments,
    simplify_expression,
)
from fhy_core.symbolic.symbol_type import SymbolType

from ..strategies.expressions import (
    SYMPY_STABLE_CALL_FUNCTIONS,
    build_boolean_expression_strategy,
    draw_boolean_tree_with_environment,
    draw_numeric_tree_with_environment,
    evaluate_with_python,
)
from ..strategies.identifiers import (
    build_boolean_identifier_pool,
    build_identifier_pool,
)
from ..strategies.settings import cap_max_examples

pytestmark = pytest.mark.property

_POOL: Final[tuple[Identifier, ...]] = build_identifier_pool(3)
_BOOLEAN_POOL: Final[tuple[Identifier, ...]] = build_boolean_identifier_pool(2)
_SYMBOL_TYPES: Final[dict[Identifier, SymbolType]] = {
    **dict.fromkeys(_POOL, SymbolType.INT),
    **dict.fromkeys(_BOOLEAN_POOL, SymbolType.BOOL),
}

# Pinned by test_simplify_expression_preserves_a_rational_coefficient_comparison:
# SymPy solves the equation "21 == 15 * v0" to "v0 == 7/5". Seven fifths
# has finite decimal text (1.4), but no binary float equals it, so the
# SymPy lifter writes it as the exact quotient DIVIDE(7, 5), which
# evaluate_expression_with_numpy computes by true division, rather than as
# a decimal string literal that coerce_literal_value would refuse.
_RATIONAL_COEFFICIENT_COMPARISON: Final[Expression] = make_binary_expression(
    BinaryOperation.EQUAL,
    LiteralExpression(21),
    make_binary_expression(BinaryOperation.MULTIPLY, LiteralExpression(15), _POOL[0]),
)
_RATIONAL_COEFFICIENT_ENVIRONMENT: Final[dict[Identifier, int]] = dict.fromkeys(
    _POOL, 0
)

# The SymPy-backed properties below enable calls restricted to
# SYMPY_STABLE_CALL_FUNCTIONS and piecewise. The integer-tree and
# boolean-tree properties keep division off; see the module docstring. See
# test_check_expression_satisfiability_agrees_with_brute_force et al.
# below for why the Z3-backed properties keep calls off.


# =============================================================================
# simplify_expression preserves evaluation
# =============================================================================


@given(
    tree_and_environment=draw_numeric_tree_with_environment(
        _POOL,
        6,
        boolean_identifiers=_BOOLEAN_POOL,
        include_division=False,
        include_calls=True,
        native_functions=SYMPY_STABLE_CALL_FUNCTIONS,
        include_piecewise=True,
    )
)
def test_simplify_expression_preserves_evaluation_on_integer_trees(
    tree_and_environment: tuple[Expression, dict[Identifier, int | bool]],
) -> None:
    """Test simplification never changes an integer tree's value.

    Oracle: ``evaluate_expression_with_numpy``, an independent lowering
    from the SymPy bridge simplification runs through.
    """
    expression, environment = tree_and_environment

    simplified = simplify_expression(expression)

    assert int(evaluate_expression_with_numpy(simplified, environment)) == int(
        evaluate_expression_with_numpy(expression, environment)
    )


@given(
    tree_and_environment=draw_boolean_tree_with_environment(
        _POOL,
        6,
        boolean_identifiers=_BOOLEAN_POOL,
        include_calls=True,
        include_division=False,
        native_functions=SYMPY_STABLE_CALL_FUNCTIONS,
        include_piecewise=True,
    )
)
def test_simplify_expression_preserves_evaluation_on_boolean_trees(
    tree_and_environment: tuple[Expression, dict[Identifier, int | bool]],
) -> None:
    """Test simplification never changes a boolean tree's truth value.

    Oracle: ``evaluate_expression_with_numpy``, an independent lowering
    from the SymPy bridge simplification runs through.
    """
    expression, environment = tree_and_environment

    simplified = simplify_expression(expression)

    assert bool(evaluate_expression_with_numpy(simplified, environment)) == bool(
        evaluate_expression_with_numpy(expression, environment)
    )


def test_simplify_expression_preserves_a_rational_coefficient_comparison() -> None:
    """Test simplifying ``21 == 15 * v0`` keeps its truth value at ``v0 = 0``.

    Integer-only input simplifies to a comparison against seven fifths, a
    rational the NumPy oracle reads only as a quotient of integers, never
    as decimal text.
    """
    simplified = simplify_expression(_RATIONAL_COEFFICIENT_COMPARISON)

    assert bool(
        evaluate_expression_with_numpy(simplified, _RATIONAL_COEFFICIENT_ENVIRONMENT)
    ) == bool(
        evaluate_expression_with_numpy(
            _RATIONAL_COEFFICIENT_COMPARISON, _RATIONAL_COEFFICIENT_ENVIRONMENT
        )
    )


# =============================================================================
# A total literal environment makes simplification evaluate fully
# =============================================================================


# Division is on: both properties below are judged exactly (see the module
# docstring).
_DIVISION_TREES_WITH_ENVIRONMENTS: Final = draw_numeric_tree_with_environment(
    _POOL,
    6,
    boolean_identifiers=_BOOLEAN_POOL,
    include_division=True,
    include_calls=True,
    native_functions=SYMPY_STABLE_CALL_FUNCTIONS,
    include_piecewise=True,
)


def _convert_to_literal_environment(
    environment: dict[Identifier, int | bool],
) -> dict[Identifier, Expression]:
    """Return ``environment`` with every bound value wrapped as a literal."""
    return {
        identifier: LiteralExpression(value)
        for identifier, value in environment.items()
    }


@given(tree_and_environment=_DIVISION_TREES_WITH_ENVIRONMENTS)
def test_simplify_expression_with_full_environment_evaluates(
    tree_and_environment: tuple[Expression, dict[Identifier, int | bool]],
) -> None:
    """Test a full literal environment makes ``simplify_expression`` evaluate.

    Oracle: ``evaluate_with_python``, the gate-grammar reference
    evaluator. The docstring promises a ``LiteralExpression`` result
    whenever every free identifier is bound to one.
    """
    expression, environment = tree_and_environment

    result = simplify_expression(
        expression, _convert_to_literal_environment(environment)
    )

    assert isinstance(result, LiteralExpression)
    assert result.value == evaluate_with_python(expression, environment)


# =============================================================================
# Simplifying twice preserves evaluation
# =============================================================================


@given(tree_and_environment=_DIVISION_TREES_WITH_ENVIRONMENTS)
def test_simplify_expression_twice_preserves_evaluation(
    tree_and_environment: tuple[Expression, dict[Identifier, int | bool]],
) -> None:
    """Test simplifying a simplified tree still denotes the original tree's value.

    Oracle: ``evaluate_with_python`` on the original tree. The twice
    simplified tree can hold ``DIVIDE`` and ``POWER`` nodes that oracle
    does not evaluate, so it is evaluated by binding every identifier to
    a literal and simplifying once more, which SymPy computes in exact
    rational arithmetic rather than in floats. Structural idempotence is
    not claimed: ``tests/symbolic/test_solver.py`` pins a tree whose
    second simplification re-associates the first.
    """
    expression, environment = tree_and_environment
    once = simplify_expression(expression)

    twice = simplify_expression(once)

    result = simplify_expression(twice, _convert_to_literal_environment(environment))
    assert isinstance(result, LiteralExpression)
    assert result.value == evaluate_with_python(expression, environment)


# =============================================================================
# The Z3-backed query trio agrees with brute-force enumeration
# =============================================================================


@st.composite
def _draw_domain(draw: st.DrawFn) -> tuple[int, int]:
    """Draw a bounded integer domain ``[lo, hi]`` with ``1 <= hi - lo <= 7``."""
    lo = draw(st.integers(min_value=-5, max_value=5))
    width = draw(st.integers(min_value=1, max_value=7))
    return lo, lo + width


def _build_domain_conjunction(
    identifiers: Sequence[Identifier], lo: int, hi: int
) -> Expression:
    """Return ``lo <= v`` and ``v <= hi`` for every ``v``, conjoined together."""
    conjuncts: list[Expression] = []
    for identifier in identifiers:
        conjuncts.append(
            make_binary_expression(BinaryOperation.LESS_EQUAL, lo, identifier)
        )
        conjuncts.append(
            make_binary_expression(BinaryOperation.LESS_EQUAL, identifier, hi)
        )
    return logical_and(*conjuncts)


def _enumerate_domain_assignments(
    lo: int, hi: int
) -> Iterator[dict[Identifier, int | bool]]:
    """Yield every assignment of the pools: integers in ``[lo, hi]``, both bools."""
    integer_values = range(lo, hi + 1)
    for integer_combination in itertools.product(integer_values, repeat=len(_POOL)):
        for boolean_combination in itertools.product(
            (False, True), repeat=len(_BOOLEAN_POOL)
        ):
            yield {
                **dict(zip(_POOL, integer_combination, strict=True)),
                **dict(zip(_BOOLEAN_POOL, boolean_combination, strict=True)),
            }


# Z3 refuses every native function call outright (see the module
# docstring for the exact TypeError), so include_calls stays off for all
# three Z3-backed properties below; include_division stays off for the
# reason given at the top of this module. Piecewise lowers to a
# right-folded z3.If chain with no such restriction, so it stays on, and
# a Boolean identifier lowers to a z3.Bool under its BOOL declaration.
_Z3_BOOLEAN_TREES: Final = build_boolean_expression_strategy(
    _POOL,
    6,
    boolean_identifiers=_BOOLEAN_POOL,
    include_calls=False,
    include_division=False,
    include_piecewise=True,
)


# Z3-backed: this query routes through the solver.
@pytest.mark.z3
@cap_max_examples(50)
@given(expression=_Z3_BOOLEAN_TREES, domain=_draw_domain())
def test_check_expression_satisfiability_agrees_with_brute_force(
    expression: Expression, domain: tuple[int, int]
) -> None:
    """Test Z3 satisfiability over a bounded domain agrees with brute force.

    Oracle: enumerating every assignment of the integer identifiers in
    ``[lo, hi]`` and of the Boolean identifiers, and evaluating with
    ``evaluate_with_python``. The domain is encoded as a conjunct so
    the query is over the same bounded space brute force enumerates; the
    result must never be ``None`` for this shape.
    """
    lo, hi = domain
    domain_conjunction = _build_domain_conjunction(_POOL, lo, hi)

    result = check_expression_satisfiability(
        logical_and(domain_conjunction, expression), _SYMBOL_TYPES
    )

    assert result is not None, (
        "check_expression_satisfiability returned None (solver unknown or "
        "hazard-screened) for a bounded-domain boolean gate tree"
    )
    brute_force = any(
        evaluate_with_python(expression, assignment)
        for assignment in _enumerate_domain_assignments(lo, hi)
    )
    assert result == brute_force


# Z3-backed: this query routes through the solver.
@pytest.mark.z3
@cap_max_examples(50)
@given(
    antecedent=_Z3_BOOLEAN_TREES, consequent=_Z3_BOOLEAN_TREES, domain=_draw_domain()
)
def test_does_expression_imply_agrees_with_brute_force(
    antecedent: Expression, consequent: Expression, domain: tuple[int, int]
) -> None:
    """Test Z3 implication over a bounded domain agrees with brute force.

    Oracle: every domain assignment that satisfies the antecedent also
    satisfies the consequent.
    """
    lo, hi = domain
    domain_conjunction = _build_domain_conjunction(_POOL, lo, hi)

    result = does_expression_imply(
        logical_and(domain_conjunction, antecedent), consequent, _SYMBOL_TYPES
    )

    brute_force = all(
        evaluate_with_python(consequent, assignment)
        for assignment in _enumerate_domain_assignments(lo, hi)
        if evaluate_with_python(antecedent, assignment)
    )
    assert result == brute_force


# Z3-backed: this query routes through the solver.
@pytest.mark.z3
@cap_max_examples(50)
@given(expression=_Z3_BOOLEAN_TREES, domain=_draw_domain())
def test_holds_for_all_free_assignments_agrees_with_brute_force(
    expression: Expression, domain: tuple[int, int]
) -> None:
    """Test Z3 universal validity over a bounded domain agrees with brute force.

    Oracle: every domain assignment satisfies the expression. The domain
    is folded into the checked formula itself (``not B or e``) rather
    than passed as considered identifiers, so every identifier of both
    pools stays universally (freely) quantified.
    """
    lo, hi = domain
    domain_conjunction = _build_domain_conjunction(_POOL, lo, hi)

    screened_expression = logical_or(logical_not(domain_conjunction), expression)
    result = holds_for_all_free_assignments(
        frozenset(), screened_expression, _SYMBOL_TYPES
    )

    brute_force = all(
        evaluate_with_python(expression, assignment)
        for assignment in _enumerate_domain_assignments(lo, hi)
    )
    assert result == brute_force
