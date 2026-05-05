"""Tests for the `NatParam` constraint-detection helper predicates.

The three helpers — ``is_the_basic_nat_param_constraint``,
``is_zero_included_constraint_exists``, and ``is_zero_not_included_constraint_exists``
— each match a constraint expression of a fixed shape: ``variable >= 0`` or
``variable > 0``. The tests in this file build constraints whose expression
breaks exactly one conjunct of the predicate at a time, so each ``and→or``
weakening, each ``==``-vs-``is``/``is not`` swap, and each ``right.value``
threshold mutation is killed by a single discriminating input.
"""

from collections.abc import Callable
from typing import Any

import pytest
from _pytest.mark.structures import ParameterSet

from fhy_core.constraint import Constraint, EquationConstraint
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.identifier import Identifier
from fhy_core.param.fundamental import (
    is_the_basic_nat_param_constraint,
    is_zero_included_constraint_exists,
    is_zero_not_included_constraint_exists,
)

# =============================================================================
# Constraint builders
# =============================================================================


def _build_basic_constraint(
    variable: Identifier, operation: BinaryOperation, literal_value: Any = 0
) -> EquationConstraint:
    """Build ``variable <operation> literal_value`` as an `EquationConstraint`."""
    return EquationConstraint(
        variable,
        BinaryExpression(
            operation, IdentifierExpression(variable), LiteralExpression(literal_value)
        ),
    )


def _wrap_with_expression(variable: Identifier, expression: Expression) -> Constraint:
    """Wrap an expression in an `EquationConstraint` for the given variable."""
    return EquationConstraint(variable, expression)


# =============================================================================
# `is_the_basic_nat_param_constraint`
# =============================================================================


class TestIsTheBasicNatParamConstraint:
    """Tests for `is_the_basic_nat_param_constraint`."""

    def test_returns_true_for_zero_included_constraint(self) -> None:
        """Test the predicate is ``True`` for ``var >= 0`` when zero is included."""
        var = Identifier("x")
        constraint = _build_basic_constraint(var, BinaryOperation.GREATER_EQUAL)
        assert is_the_basic_nat_param_constraint(constraint, var, True)

    def test_returns_true_for_zero_excluded_constraint(self) -> None:
        """Test the predicate returns ``True`` for ``var > 0`` when zero is excluded."""
        var = Identifier("x")
        constraint = _build_basic_constraint(var, BinaryOperation.GREATER)
        assert is_the_basic_nat_param_constraint(constraint, var, False)

    def test_returns_true_when_left_identifier_is_same_object_as_variable(self) -> None:
        """Test the predicate compares identifiers by ``==``, not identity.

        The same `Identifier` object is used in both the constraint expression
        and the predicate call, so an identity-based check (``is``) would
        return the same result. To pin down ``==`` against ``is not``, the
        predicate must return ``True`` when the two are the *same* object.
        """
        var = Identifier("x")
        constraint = _build_basic_constraint(var, BinaryOperation.GREATER_EQUAL)
        assert is_the_basic_nat_param_constraint(constraint, var, True)

    def test_returns_false_when_expression_is_not_a_binary_expression(self) -> None:
        """Test the predicate returns ``False`` for a non-binary expression."""
        var = Identifier("x")
        constraint = _wrap_with_expression(var, LiteralExpression(0))
        assert not is_the_basic_nat_param_constraint(constraint, var, True)

    def test_returns_false_for_zero_included_with_wrong_operation(self) -> None:
        """Test the predicate is ``False`` when the op is not ``GREATER_EQUAL``."""
        var = Identifier("x")
        constraint = _build_basic_constraint(var, BinaryOperation.LESS_EQUAL)
        assert not is_the_basic_nat_param_constraint(constraint, var, True)

    def test_returns_false_for_zero_excluded_with_wrong_operation(self) -> None:
        """Test the predicate is ``False`` when the operation is not ``GREATER``."""
        var = Identifier("x")
        constraint = _build_basic_constraint(var, BinaryOperation.LESS)
        assert not is_the_basic_nat_param_constraint(constraint, var, False)

    def test_returns_false_when_left_operand_is_not_an_identifier_expression(
        self,
    ) -> None:
        """Test the predicate is ``False`` when left is not an identifier expr."""
        var = Identifier("x")
        constraint = _wrap_with_expression(
            var,
            BinaryExpression(
                BinaryOperation.GREATER_EQUAL,
                LiteralExpression(0),
                LiteralExpression(0),
            ),
        )
        assert not is_the_basic_nat_param_constraint(constraint, var, True)

    def test_returns_false_when_left_identifier_does_not_match_variable(self) -> None:
        """Test the predicate returns ``False`` when the identifier does not match."""
        var = Identifier("x")
        other_var = Identifier("y")
        constraint = _build_basic_constraint(other_var, BinaryOperation.GREATER_EQUAL)
        assert not is_the_basic_nat_param_constraint(constraint, var, True)

    def test_returns_false_when_right_operand_is_not_a_literal_expression(self) -> None:
        """Test the predicate is ``False`` when the right operand is not a literal."""
        var = Identifier("x")
        other_var = Identifier("y")
        constraint = _wrap_with_expression(
            var,
            BinaryExpression(
                BinaryOperation.GREATER_EQUAL,
                IdentifierExpression(var),
                IdentifierExpression(other_var),
            ),
        )
        assert not is_the_basic_nat_param_constraint(constraint, var, True)

    def test_returns_false_when_right_literal_value_is_not_an_int(self) -> None:
        """Test the predicate is ``False`` when the right literal is not an `int`."""
        var = Identifier("x")
        constraint = _build_basic_constraint(var, BinaryOperation.GREATER_EQUAL, 0.0)
        assert not is_the_basic_nat_param_constraint(constraint, var, True)

    def test_returns_false_when_right_literal_value_is_negative(self) -> None:
        """Test the predicate returns ``False`` when the right literal is non-zero.

        A non-zero literal that satisfies ``<= 0`` (like ``-5``) pins down
        the ``== 0`` check against an ``<= 0`` weakening: the original
        rejects, and the mutated form would accept.
        """
        var = Identifier("x")
        constraint = _build_basic_constraint(var, BinaryOperation.GREATER_EQUAL, -5)
        assert not is_the_basic_nat_param_constraint(constraint, var, True)


# =============================================================================
# `is_zero_included_constraint_exists`
# =============================================================================


class TestIsZeroIncludedConstraintExists:
    """Tests for `is_zero_included_constraint_exists`."""

    def test_returns_true_for_a_matching_constraint_in_the_iterable(self) -> None:
        """Test the predicate is ``True`` when a ``var >= 0`` constraint is present."""
        var = Identifier("x")
        constraints = [_build_basic_constraint(var, BinaryOperation.GREATER_EQUAL)]
        assert is_zero_included_constraint_exists(constraints, var)

    def test_returns_true_when_left_identifier_is_same_object_as_variable(self) -> None:
        """Test the predicate compares identifiers by ``==``, not identity.

        Pins down ``==`` against ``is not`` on ``left.identifier == variable``.
        """
        var = Identifier("x")
        constraints = [_build_basic_constraint(var, BinaryOperation.GREATER_EQUAL)]
        assert is_zero_included_constraint_exists(constraints, var)

    def test_returns_false_for_an_empty_constraint_iterable(self) -> None:
        """Test the predicate returns ``False`` for an empty constraint iterable."""
        var = Identifier("x")
        assert not is_zero_included_constraint_exists([], var)

    def test_returns_false_when_the_only_match_uses_a_different_variable(self) -> None:
        """Test the predicate is ``False`` when constraint targets another variable."""
        var = Identifier("x")
        other_var = Identifier("y")
        constraints = [
            _build_basic_constraint(other_var, BinaryOperation.GREATER_EQUAL)
        ]
        assert not is_zero_included_constraint_exists(constraints, var)

    def test_returns_false_when_only_a_strict_greater_constraint_is_present(
        self,
    ) -> None:
        """Test the predicate returns ``False`` when only ``var > 0`` is present."""
        var = Identifier("x")
        constraints = [_build_basic_constraint(var, BinaryOperation.GREATER)]
        assert not is_zero_included_constraint_exists(constraints, var)

    def test_returns_false_when_right_literal_value_is_not_an_int(self) -> None:
        """Test the predicate is ``False`` when the right literal is not an `int`."""
        var = Identifier("x")
        constraints = [
            _build_basic_constraint(var, BinaryOperation.GREATER_EQUAL, 0.0),
        ]
        assert not is_zero_included_constraint_exists(constraints, var)

    def test_returns_false_when_right_literal_value_is_negative(self) -> None:
        """Test the predicate returns ``False`` when the right literal is ``-5``.

        Pins down ``== 0`` against an ``<= 0`` weakening.
        """
        var = Identifier("x")
        constraints = [_build_basic_constraint(var, BinaryOperation.GREATER_EQUAL, -5)]
        assert not is_zero_included_constraint_exists(constraints, var)


# =============================================================================
# `is_zero_not_included_constraint_exists`
# =============================================================================


_NON_MATCHING_CONSTRAINT_BUILDERS: list[ParameterSet] = [
    pytest.param(
        lambda v, w: _wrap_with_expression(v, LiteralExpression(0)),
        id="non-binary-expression",
    ),
    pytest.param(
        lambda v, w: _build_basic_constraint(v, BinaryOperation.GREATER_EQUAL),
        id="wrong-operation",
    ),
    pytest.param(
        lambda v, w: _build_basic_constraint(w, BinaryOperation.GREATER),
        id="mismatched-variable",
    ),
    pytest.param(
        lambda v, w: _wrap_with_expression(
            v,
            BinaryExpression(
                BinaryOperation.GREATER,
                IdentifierExpression(v),
                IdentifierExpression(w),
            ),
        ),
        id="non-literal-right-operand",
    ),
    pytest.param(
        lambda v, w: _build_basic_constraint(v, BinaryOperation.GREATER, 0.0),
        id="non-int-right-literal",
    ),
    pytest.param(
        lambda v, w: _build_basic_constraint(v, BinaryOperation.GREATER, -5),
        id="negative-right-literal",
    ),
    pytest.param(
        lambda v, w: _build_basic_constraint(v, BinaryOperation.GREATER, 5),
        id="positive-right-literal",
    ),
]


class TestIsZeroNotIncludedConstraintExists:
    """Tests for `is_zero_not_included_constraint_exists`."""

    def test_returns_true_for_a_matching_constraint_in_the_iterable(self) -> None:
        """Test the predicate is ``True`` when a ``var > 0`` constraint is present."""
        var = Identifier("x")
        constraints = [_build_basic_constraint(var, BinaryOperation.GREATER)]
        assert is_zero_not_included_constraint_exists(constraints, var)

    def test_returns_true_when_left_identifier_is_same_object_as_variable(self) -> None:
        """Test the predicate compares identifiers by ``==``, not identity.

        Pins down ``==`` against ``is not`` on ``left.identifier == variable``.
        """
        var = Identifier("x")
        constraints = [_build_basic_constraint(var, BinaryOperation.GREATER)]
        assert is_zero_not_included_constraint_exists(constraints, var)

    def test_returns_false_for_an_empty_constraint_iterable(self) -> None:
        """Test the predicate returns ``False`` for an empty constraint iterable.

        Pins down the trailing ``return False`` against a ``return True`` flip.
        """
        var = Identifier("x")
        assert not is_zero_not_included_constraint_exists([], var)

    @pytest.mark.parametrize("builder", _NON_MATCHING_CONSTRAINT_BUILDERS)
    def test_returns_false_when_each_individual_conjunct_is_broken(
        self, builder: Callable[[Identifier, Identifier], Constraint]
    ) -> None:
        """Test the predicate returns ``False`` when any single conjunct fails.

        Each parametrized case breaks exactly one conjunct of the predicate's
        ``and``-chain. Together they kill every ``and→or`` mutation in the
        function as well as the ``== 0`` / ``is`` / ``is not`` mutations on
        the same line.
        """
        var = Identifier("x")
        other_var = Identifier("y")
        constraints = [builder(var, other_var)]
        assert not is_zero_not_included_constraint_exists(constraints, var)
