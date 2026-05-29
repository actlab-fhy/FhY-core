"""Tests for `convert_to_expression` cardinality branches."""

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
    LiteralExpression,
    make_binary_expression,
)
from fhy_core.identifier import Identifier

from .conftest import SerializableEqualHashable, mock_identifier

SetConstraintFactory = Callable[[Identifier, object], Constraint]

# (factory, leaf_op, combinator_op)
_KIND_SHAPES = [
    pytest.param(
        InSetConstraint,
        BinaryOperation.EQUAL,
        BinaryOperation.LOGICAL_OR,
        id="in_set",
    ),
    pytest.param(
        NotInSetConstraint,
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


@pytest.mark.parametrize("factory, leaf_op, _combinator", _KIND_SHAPES)
def test_singleton_set_returns_single_leaf(
    factory: SetConstraintFactory,
    leaf_op: BinaryOperation,
    _combinator: BinaryOperation,
) -> None:
    """Test a singleton constraint returns a bare leaf, not a logical combinator."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {42})

    expression = constraint.convert_to_expression()

    expected = make_binary_expression(leaf_op, x, 42)
    assert isinstance(expression, BinaryExpression)
    assert expression.operation == leaf_op
    assert expected.is_structurally_equivalent(expression)


@pytest.mark.parametrize("factory, leaf_op, combinator", _KIND_SHAPES)
def test_multi_value_set_returns_combinator_of_leaves(
    factory: SetConstraintFactory,
    leaf_op: BinaryOperation,
    combinator: BinaryOperation,
) -> None:
    """Test a multi-value constraint returns a combinator over leaf comparisons."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2})

    expression = constraint.convert_to_expression()

    expected = make_binary_expression(
        combinator,
        make_binary_expression(leaf_op, x, 1),
        make_binary_expression(leaf_op, x, 2),
    )
    assert expected.is_structurally_equivalent(expression)


@pytest.mark.parametrize(
    "factory", [InSetConstraint, NotInSetConstraint], ids=["in_set", "not_in_set"]
)
def test_non_literal_member_rejected_by_conversion(
    factory: SetConstraintFactory,
) -> None:
    """Test non-`LiteralType` members raise ``ConstraintError`` from conversion."""
    constraint = factory(mock_identifier("x", 0), {SerializableEqualHashable(1)})

    with pytest.raises(ConstraintError):
        constraint.convert_to_expression()


@pytest.mark.parametrize(
    "factory", [InSetConstraint, NotInSetConstraint], ids=["in_set", "not_in_set"]
)
def test_non_grammar_string_member_rejected_by_conversion_as_constraint_error(
    factory: SetConstraintFactory,
) -> None:
    """Test non-grammar string members raise ``ConstraintError`` not ``ValueError``."""
    constraint = factory(mock_identifier("x", 0), ["hello"])

    with pytest.raises(ConstraintError):
        constraint.convert_to_expression()


@pytest.mark.parametrize(
    "factory", [InSetConstraint, NotInSetConstraint], ids=["in_set", "not_in_set"]
)
def test_multi_value_convert_to_expression_is_deterministic_across_construction_orders(
    factory: SetConstraintFactory,
) -> None:
    """Test independent constraint instances produce structurally identical trees."""
    x = mock_identifier("x", 0)
    left = factory(x, [3, 1, 2])
    right = factory(x, [2, 3, 1])

    left_expression = left.convert_to_expression()
    right_expression = right.convert_to_expression()

    assert left_expression.is_structurally_equivalent(right_expression)


@pytest.mark.parametrize(
    "factory", [InSetConstraint, NotInSetConstraint], ids=["in_set", "not_in_set"]
)
def test_multi_value_convert_to_expression_orders_leaves_by_repr(
    factory: SetConstraintFactory,
) -> None:
    """Test the produced combinator's leaves match `repr`-sorted member order.

    The chosen integer members give a lexicographic-by-repr order
    (``[12, 3, 7, 9]``) that differs from both numeric order and
    typical ``frozenset`` iteration order, so the test will fail if
    the implementation forgets to sort by ``repr`` and falls through
    to either of those orderings.
    """
    x = mock_identifier("x", 0)
    members = [3, 7, 9, 12]
    constraint = factory(x, members)

    expression = constraint.convert_to_expression()

    leaf_literals: list[object] = []

    def _walk(node: object) -> None:
        if isinstance(node, BinaryExpression) and node.operation in (
            BinaryOperation.LOGICAL_OR,
            BinaryOperation.LOGICAL_AND,
        ):
            _walk(node.left)
            _walk(node.right)
        elif isinstance(node, BinaryExpression):
            assert isinstance(node.right, LiteralExpression)
            leaf_literals.append(node.right.value)

    _walk(expression)
    assert leaf_literals == sorted(members, key=repr)
