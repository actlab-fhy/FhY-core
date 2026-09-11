"""Hypothesis property tests for the solver seam (P2, P3, P7).

Covers ``simplify_expression`` (SymPy backend: evaluation-preserving,
evaluates fully under a total literal environment, structural
idempotence) and the Z3-backed query trio -- satisfiability, implication,
and universal validity -- cross-checked against brute-force enumeration
over a small bounded integer domain.

Every strategy here draws no native-function call, no division, and no
piecewise node, all for reasons specific to this seam rather than to the
gate grammar in general:

- ``simplify_expression`` cannot lower a call to an expression-bodied
  ``RegisteredFunction`` (for example ``sign``, one of
  ``INTEGER_RESULT_NATIVE_FUNCTIONS``) without first running
  ``inline_functions`` (it raises ``TypeError: Cannot lower an
  expression-bodied function call to SymPy``), and the Z3 bridge refuses
  every call outright (``TypeError: Z3 does not support native function
  calls``).
- A symbolic ``FLOOR_DIVIDE``/``MODULO`` is one of the solver module's
  documented divergences: SymPy lifts a resulting ``Rational`` such as
  ``1/5`` to the exact-decimal string literal ``"0.2"``, which the NumPy
  oracle then refuses with ``StringLiteralPrecisionError`` (no binary
  float equals ``0.2`` exactly); the Z3 bridge separately hazard-screens
  a non-positive divisor.
- The shared ``build_boolean_expression_strategy`` (reached from a
  piecewise condition, direct or nested) draws each comparison's numeric
  operands with every option at its default -- calls, division, and
  piecewise all enabled -- with no parameter to thread this module's
  restrictions through. Excluding piecewise here is what keeps a call or
  a division from re-entering through that path.

This module therefore builds its own call-free, division-free,
piecewise-free numeric strategy from the shared building blocks, and its
own boolean-tree strategy (comparisons, ``LOGICAL_NOT``,
``LOGICAL_AND``/``LOGICAL_OR``) over that numeric strategy, rather than
filtering draws after the fact.
"""

import itertools
from collections.abc import Iterator, Sequence
from typing import Final

import pytest

pytest.importorskip("hypothesis")

from hypothesis import example, given, settings
from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryOperation,
    Expression,
    LiteralExpression,
    UnaryOperation,
    evaluate_expression_with_numpy,
    logical_and,
    logical_not,
    logical_or,
    make_binary_expression,
    make_unary_expression,
)
from fhy_core.symbolic.solver import (
    check_expression_satisfiability,
    does_expression_imply,
    holds_for_all_free_assignments,
    simplify_expression,
)
from fhy_core.symbolic.symbol_type import SymbolType

from ..strategies.expressions import (
    COMPARISON_OPERATIONS,
    LOGICAL_BINARY_OPERATIONS,
    build_integer_environment_strategy,
    build_numeric_expression_strategy,
    evaluate_with_python,
)
from ..strategies.identifiers import build_identifier_pool
from ..strategies.literals import build_boolean_literal_strategy

pytestmark = pytest.mark.property

_POOL: Final[tuple[Identifier, ...]] = build_identifier_pool(3)
_SYMBOL_TYPES: Final[dict[Identifier, SymbolType]] = dict.fromkeys(
    _POOL, SymbolType.INT
)

# Pinned counterexample for
# test_simplify_expression_preserves_evaluation_on_boolean_trees: SymPy
# normalizes the equation "21 == 15 * v0" to "v0 == 21/15", and 21/15
# reduces to 7/5, a terminating decimal (1.4) with no exact binary float
# representation. The lifted expression holds the exact-decimal string
# literal "1.4", and evaluate_expression_with_numpy's coerce_literal_value
# refuses to collapse it to a lossy binary float.
_RATIONAL_COEFFICIENT_COMPARISON: Final[Expression] = make_binary_expression(
    BinaryOperation.EQUAL,
    LiteralExpression(21),
    make_binary_expression(BinaryOperation.MULTIPLY, LiteralExpression(15), _POOL[0]),
)
_RATIONAL_COEFFICIENT_ENVIRONMENT: Final[dict[Identifier, int]] = dict.fromkeys(
    _POOL, 0
)

# Pinned counterexample for test_simplify_expression_is_structurally_idempotent:
# simplifying "-((0 - v0 * -v0) + v0)" once yields a Mul/Add association
# that a second simplification re-associates differently (SymPy's own
# term ordering is not a fixed point of repeated sympy.simplify calls on
# this shape).
_NON_IDEMPOTENT_SIMPLIFICATION: Final[Expression] = make_unary_expression(
    UnaryOperation.NEGATE,
    make_binary_expression(
        BinaryOperation.ADD,
        make_binary_expression(
            BinaryOperation.SUBTRACT,
            LiteralExpression(0),
            make_binary_expression(
                BinaryOperation.MULTIPLY,
                _POOL[0],
                make_unary_expression(UnaryOperation.NEGATE, _POOL[0]),
            ),
        ),
        _POOL[0],
    ),
)


def _build_call_free_numeric_strategy(
    identifiers: Sequence[Identifier], max_leaves: int = 6
) -> st.SearchStrategy[Expression]:
    """Return a numeric gate-tree strategy with no call, division, or piecewise.

    See the module docstring for why all three are excluded rather than
    only the native call.
    """
    return build_numeric_expression_strategy(
        identifiers,
        max_leaves,
        include_division=False,
        include_calls=False,
        include_piecewise=False,
    )


def _build_call_free_boolean_strategy(
    identifiers: Sequence[Identifier], max_leaves: int = 4
) -> st.SearchStrategy[Expression]:
    """Return a boolean gate-tree strategy whose comparisons never call.

    Built with ``st.recursive`` rather than the shared leaf-budget
    machinery: the base case is a boolean literal or a comparison of two
    call-free numeric subtrees, and ``extend`` adds ``LOGICAL_NOT`` and
    ``LOGICAL_AND``/``LOGICAL_OR`` nodes over already-drawn boolean
    subtrees. ``max_leaves`` bounds the recursive expansion, not a
    counted AST leaf total.
    """
    numeric = _build_call_free_numeric_strategy(identifiers, max_leaves=3)
    comparisons = st.builds(
        make_binary_expression,
        st.sampled_from(COMPARISON_OPERATIONS),
        numeric,
        numeric,
    )
    base = st.one_of(build_boolean_literal_strategy(), comparisons)

    def _extend(
        children: st.SearchStrategy[Expression],
    ) -> st.SearchStrategy[Expression]:
        negations = children.map(
            lambda operand: make_unary_expression(UnaryOperation.LOGICAL_NOT, operand)
        )
        binary = st.builds(
            make_binary_expression,
            st.sampled_from(LOGICAL_BINARY_OPERATIONS),
            children,
            children,
        )
        return st.one_of(negations, binary)

    return st.recursive(base, _extend, max_leaves=max_leaves)


@st.composite
def _draw_call_free_numeric_tree_with_environment(
    draw: st.DrawFn, identifiers: Sequence[Identifier]
) -> tuple[Expression, dict[Identifier, int]]:
    """Draw a call-free numeric gate tree with an int binding for each identifier."""
    expression = draw(_build_call_free_numeric_strategy(identifiers))
    environment = draw(build_integer_environment_strategy(identifiers))
    return expression, environment


@st.composite
def _draw_call_free_boolean_tree_with_environment(
    draw: st.DrawFn, identifiers: Sequence[Identifier]
) -> tuple[Expression, dict[Identifier, int]]:
    """Draw a call-free boolean gate tree with an int binding for each identifier."""
    expression = draw(_build_call_free_boolean_strategy(identifiers))
    environment = draw(build_integer_environment_strategy(identifiers))
    return expression, environment


# =============================================================================
# P2a, P2c: simplify_expression preserves evaluation
# =============================================================================


@given(tree_and_environment=_draw_call_free_numeric_tree_with_environment(_POOL))
def test_simplify_expression_preserves_evaluation_on_integer_trees(
    tree_and_environment: tuple[Expression, dict[Identifier, int]],
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "simplify_expression normalizes an equation like '21 == 15 * v0' to "
        "'v0 == 21/15', and 21/15 reduces to the terminating decimal 1.4, "
        "which SymPy lifting turns into the exact-decimal string literal "
        "'1.4'; evaluate_expression_with_numpy's coerce_literal_value then "
        "raises StringLiteralPrecisionError because no binary float equals "
        "1.4 exactly. Integer-only input, non-integer-exact output: a real "
        "tension between simplify_expression's exact-rational semantics and "
        "the NumPy oracle's refusal to round a decimal literal."
    ),
)
@example(
    tree_and_environment=(
        _RATIONAL_COEFFICIENT_COMPARISON,
        _RATIONAL_COEFFICIENT_ENVIRONMENT,
    )
)
@given(tree_and_environment=_draw_call_free_boolean_tree_with_environment(_POOL))
def test_simplify_expression_preserves_evaluation_on_boolean_trees(
    tree_and_environment: tuple[Expression, dict[Identifier, int]],
) -> None:
    """Test simplification never changes a boolean tree's truth value.

    Oracle: ``evaluate_expression_with_numpy``.
    """
    expression, environment = tree_and_environment

    simplified = simplify_expression(expression)

    assert bool(evaluate_expression_with_numpy(simplified, environment)) == bool(
        evaluate_expression_with_numpy(expression, environment)
    )


# =============================================================================
# P2b: a total literal environment makes simplification evaluation
# =============================================================================


@given(tree_and_environment=_draw_call_free_numeric_tree_with_environment(_POOL))
def test_simplify_expression_with_full_environment_evaluates(
    tree_and_environment: tuple[Expression, dict[Identifier, int]],
) -> None:
    """Test a full literal environment makes ``simplify_expression`` evaluate.

    Oracle: ``evaluate_with_python``, the gate-grammar reference
    evaluator. The docstring promises a ``LiteralExpression`` result
    whenever every free identifier is bound to one.
    """
    expression, integer_environment = tree_and_environment
    literal_environment = {
        identifier: LiteralExpression(value)
        for identifier, value in integer_environment.items()
    }

    result = simplify_expression(expression, literal_environment)

    assert isinstance(result, LiteralExpression)
    assert result.value == evaluate_with_python(expression, integer_environment)


# =============================================================================
# P3: structural idempotence of simplify_expression
# =============================================================================


@pytest.mark.xfail(
    strict=True,
    reason=(
        "simplify_expression is not structurally idempotent for every tree: "
        "simplifying '-((0 - v0 * -v0) + v0)' once yields one Mul/Add "
        "association, and simplifying that result again re-associates it "
        "differently, so the two are not is_structurally_equivalent. Per "
        "docs/design/property-based-testing.md Decision 6, this documents "
        "the current behavior and flips loudly if SymPy simplification ever "
        "becomes a structural fixed point."
    ),
)
@example(expression=_NON_IDEMPOTENT_SIMPLIFICATION)
@given(expression=_build_call_free_numeric_strategy(_POOL))
def test_simplify_expression_is_structurally_idempotent(expression: Expression) -> None:
    """Test simplifying twice is structurally the same as simplifying once.

    Oracle: self (idempotence is an algebraic law, not a comparison
    against an independent implementation).
    """
    once = simplify_expression(expression)

    twice = simplify_expression(once)

    assert twice.is_structurally_equivalent(once)


# =============================================================================
# P7: the Z3-backed query trio agrees with brute-force enumeration
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
    identifiers: Sequence[Identifier], lo: int, hi: int
) -> Iterator[dict[Identifier, int]]:
    """Yield every assignment of ``identifiers`` to a value in ``[lo, hi]``."""
    values = range(lo, hi + 1)
    for combination in itertools.product(values, repeat=len(identifiers)):
        yield dict(zip(identifiers, combination, strict=True))


@pytest.mark.z3
@settings(max_examples=50)
@given(
    expression=_build_call_free_boolean_strategy(_POOL, max_leaves=6),
    domain=_draw_domain(),
)
def test_check_expression_satisfiability_agrees_with_brute_force(
    expression: Expression, domain: tuple[int, int]
) -> None:
    """Test Z3 satisfiability over a bounded domain agrees with brute force.

    Oracle: enumerating every assignment in ``[lo, hi]`` and evaluating
    with ``evaluate_with_python``. The domain is encoded as a conjunct so
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
        for assignment in _enumerate_domain_assignments(_POOL, lo, hi)
    )
    assert result == brute_force


@pytest.mark.z3
@settings(max_examples=50)
@given(
    antecedent=_build_call_free_boolean_strategy(_POOL, max_leaves=6),
    consequent=_build_call_free_boolean_strategy(_POOL, max_leaves=6),
    domain=_draw_domain(),
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
        for assignment in _enumerate_domain_assignments(_POOL, lo, hi)
        if evaluate_with_python(antecedent, assignment)
    )
    assert result == brute_force


@pytest.mark.z3
@settings(max_examples=50)
@given(
    expression=_build_call_free_boolean_strategy(_POOL, max_leaves=6),
    domain=_draw_domain(),
)
def test_holds_for_all_free_assignments_agrees_with_brute_force(
    expression: Expression, domain: tuple[int, int]
) -> None:
    """Test Z3 universal validity over a bounded domain agrees with brute force.

    Oracle: every domain assignment satisfies the expression. The domain
    is folded into the checked formula itself (``not B or e``) rather
    than passed as considered identifiers, so every pool identifier stays
    universally (freely) quantified.
    """
    lo, hi = domain
    domain_conjunction = _build_domain_conjunction(_POOL, lo, hi)

    screened_expression = logical_or(logical_not(domain_conjunction), expression)
    result = holds_for_all_free_assignments(
        frozenset(), screened_expression, _SYMBOL_TYPES
    )

    brute_force = all(
        evaluate_with_python(expression, assignment)
        for assignment in _enumerate_domain_assignments(_POOL, lo, hi)
    )
    assert result == brute_force
