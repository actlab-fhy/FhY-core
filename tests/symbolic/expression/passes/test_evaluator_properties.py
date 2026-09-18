"""Hypothesis property tests for ``evaluate_expression``.

Covers three invariants of the bottom-up native-call/native-constant
folding pass: folding never changes what a tree evaluates to (oracle:
the NumPy evaluator), folding is idempotent, and -- with one documented
exception -- no literal-argument native call survives a fold. Trees draw
Boolean identifiers into their Boolean positions (piecewise conditions
and everything beneath them), and environments bind those identifiers
to bools.
"""

from typing import Final

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    CallExpression,
    Expression,
    LiteralExpression,
    evaluate_expression,
    evaluate_expression_with_numpy,
)

from ....strategies.expressions import (
    build_numeric_expression_strategy,
    draw_numeric_tree_with_environment,
)
from ....strategies.identifiers import (
    build_boolean_identifier_pool,
    build_identifier_pool,
)

pytestmark = pytest.mark.property

pytest.importorskip("numpy")

_POOL: Final[tuple[Identifier, ...]] = build_identifier_pool(3)
_BOOLEAN_POOL: Final[tuple[Identifier, ...]] = build_boolean_identifier_pool(2)


def _find_foldable_call(expression: Expression) -> CallExpression | None:
    """Return a ``CallExpression`` with all-``LiteralExpression`` arguments, if any.

    Walks the whole tree via ``get_visit_children`` rather than only the
    root, so a foldable call nested under a surviving node (a piecewise
    branch, a binary operand) is still found.
    """
    if isinstance(expression, CallExpression) and all(
        isinstance(argument, LiteralExpression) for argument in expression.arguments
    ):
        return expression
    for child in expression.get_visit_children():
        found = _find_foldable_call(child)
        if found is not None:
            return found
    return None


# =============================================================================
# evaluate_expression never changes what a tree evaluates to
# =============================================================================


@given(
    tree_and_environment=draw_numeric_tree_with_environment(
        _POOL, boolean_identifiers=_BOOLEAN_POOL
    )
)
def test_evaluate_expression_preserves_evaluation(
    tree_and_environment: tuple[Expression, dict[Identifier, int | bool]],
) -> None:
    """Test folding never changes the value under any sampled assignment.

    Oracle: ``evaluate_expression_with_numpy``, an independent evaluator
    that does not go through the pass under test.
    """
    expression, environment = tree_and_environment

    folded = evaluate_expression(expression)

    assert int(evaluate_expression_with_numpy(folded, environment)) == int(
        evaluate_expression_with_numpy(expression, environment)
    )


# =============================================================================
# folding is idempotent
# =============================================================================


@given(
    expression=build_numeric_expression_strategy(
        _POOL, boolean_identifiers=_BOOLEAN_POOL
    )
)
def test_evaluate_expression_is_idempotent(expression: Expression) -> None:
    """Test folding twice is structurally the same as folding once."""
    once = evaluate_expression(expression)

    twice = evaluate_expression(once)

    assert twice.is_structurally_equivalent(once)


# =============================================================================
# No literal-argument native call survives a fold -- except a call to an
# expression-bodied function, which evaluate_expression documents it
# leaves alone (see the module docstring on
# fhy_core.symbolic.expression.passes.evaluate: "run inline_functions
# before evaluate_expression" is the prescribed remedy).
# =============================================================================


# Walks the folded tree with get_visit_children. Every call the gate
# grammar draws targets a NativeFunction, which is exactly the kind
# evaluate_expression promises to fold.
@given(
    expression=build_numeric_expression_strategy(
        _POOL, boolean_identifiers=_BOOLEAN_POOL
    )
)
def test_evaluate_expression_folds_every_literal_argument_call(
    expression: Expression,
) -> None:
    """Test no ``CallExpression`` with all-``LiteralExpression`` arguments survives."""
    folded = evaluate_expression(expression)

    survivor = _find_foldable_call(folded)

    assert survivor is None, f"literal-argument call survived folding: {survivor!r}"
