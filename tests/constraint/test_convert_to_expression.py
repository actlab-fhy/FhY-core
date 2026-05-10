"""Tests for `convert_to_expression` cardinality branches.

The empty / singleton / multi-value branches are the largest cluster of
mutants. Singleton tests are particularly load-bearing: when the
``elif len(...) == 1`` branch is mutated to a predicate that reduces to
``False`` (e.g. ``< 1``, ``== 0``), the singleton case falls through to
the ``else`` branch that calls ``Expression.logical_or`` /
``logical_and`` with a single argument, which raises ``ValueError``.
"""

from typing import Callable

import pytest

from fhy_core.constraint import (
    Constraint,
    ConstraintError,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.identifier import Identifier

from .conftest import SerializableEqualHashable, mock_identifier

SetConstraintFactory = Callable[[Identifier, object], Constraint]

# (factory, empty_literal, leaf_op, combinator_op)
_KIND_SHAPES = [
    pytest.param(
        InSetConstraint,
        True,  # not used; placeholder
        BinaryOperation.EQUAL,
        BinaryOperation.LOGICAL_OR,
        id="in_set",
    ),
    pytest.param(
        NotInSetConstraint,
        False,  # not used; placeholder
        BinaryOperation.NOT_EQUAL,
        BinaryOperation.LOGICAL_AND,
        id="not_in_set",
    ),
]

_EMPTY_LITERALS = [
    pytest.param(InSetConstraint, False, id="in_set"),
    pytest.param(NotInSetConstraint, True, id="not_in_set"),
]


@pytest.mark.parametrize("factory, expected_literal", _EMPTY_LITERALS)
def test_empty_set_returns_literal(
    factory: SetConstraintFactory, expected_literal: bool
) -> None:
    """Test empty in-set returns ``False`` and empty not-in-set returns ``True``."""
    constraint = factory(mock_identifier("x", 0), [])
    expression = constraint.convert_to_expression()
    assert isinstance(expression, LiteralExpression)
    assert expression.value is expected_literal


@pytest.mark.parametrize("factory, _empty, leaf_op, _combinator", _KIND_SHAPES)
def test_singleton_set_returns_single_leaf(
    factory: SetConstraintFactory,
    _empty: bool,
    leaf_op: BinaryOperation,
    _combinator: BinaryOperation,
) -> None:
    """Test a singleton constraint returns a bare leaf, not a logical combinator."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {42})
    expression = constraint.convert_to_expression()
    expected = BinaryExpression(leaf_op, IdentifierExpression(x), LiteralExpression(42))
    assert isinstance(expression, BinaryExpression)
    assert expression.operation == leaf_op
    assert expected.is_structurally_equivalent(expression)


@pytest.mark.parametrize("factory, _empty, leaf_op, combinator", _KIND_SHAPES)
def test_multi_value_set_returns_combinator_of_leaves(
    factory: SetConstraintFactory,
    _empty: bool,
    leaf_op: BinaryOperation,
    combinator: BinaryOperation,
) -> None:
    """Test a multi-value constraint returns a combinator over leaf comparisons."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2})
    expression = constraint.convert_to_expression()
    expected = BinaryExpression(
        combinator,
        BinaryExpression(leaf_op, IdentifierExpression(x), LiteralExpression(1)),
        BinaryExpression(leaf_op, IdentifierExpression(x), LiteralExpression(2)),
    )
    assert expected.is_structurally_equivalent(expression)


@pytest.mark.parametrize(
    "factory", [InSetConstraint, NotInSetConstraint], ids=["in_set", "not_in_set"]
)
def test_non_literal_member_rejected_by_conversion(
    factory: SetConstraintFactory,
) -> None:
    """Test non-`LiteralType` members are rejected by ``convert_to_expression``."""
    constraint = factory(mock_identifier("x", 0), {SerializableEqualHashable(1)})
    with pytest.raises(ConstraintError):
        constraint.convert_to_expression()
