"""Constraint utility."""

__all__ = [
    "Constraint",
    "EquationConstraint",
    "InSetConstraint",
    "NotInSetConstraint",
]

from abc import ABC, abstractmethod
from typing import Any

from .expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    LiteralType,
    copy_expression,
    simplify_expression,
)
from .identifier import Identifier


class Constraint(ABC):
    """Abstract base class for constraints."""

    _variable: Identifier

    def __init__(self, constrained_variable: Identifier) -> None:
        self._variable = constrained_variable

    @property
    def variable(self) -> Identifier:
        return self._variable

    def __call__(self, values: dict[Identifier, Any]) -> bool:
        return self.is_satisfied(values)

    @abstractmethod
    def is_satisfied(self, variable_value: Any) -> bool:
        """Check if the value satisfies the constraint.

        Args:
            variable_value: Value to check.

        Returns:
            True if the value satisfies the constraint; False otherwise.

        """

    @abstractmethod
    def copy(self) -> "Constraint":
        """Return a shallow copy of the constraint."""

    @abstractmethod
    def convert_to_expression(self) -> Expression:
        """Return an expression equivalent to the constraint.

        Raises:
            ValueError: If the constraint cannot be converted to an expression.

        """


class EquationConstraint(Constraint):
    """Represents an equation constraint."""

    _expression: Expression

    def __init__(
        self, constrained_variable: Identifier, expression: Expression
    ) -> None:
        super().__init__(constrained_variable)
        self._expression = expression

    def is_satisfied(self, value: Expression) -> bool:
        result = simplify_expression(self._expression, {self.variable: value})
        return (
            isinstance(result, LiteralExpression)
            and isinstance(result.value, bool)
            and result.value
        )

    def copy(self) -> "EquationConstraint":
        new_constraint = EquationConstraint(
            self.variable, copy_expression(self._expression)
        )
        return new_constraint

    def convert_to_expression(self) -> Expression:
        return copy_expression(self._expression)


class InSetConstraint(Constraint):
    """Represents an in-set constraint."""

    _valid_values: set[Any]

    def __init__(
        self, constrained_variable: Identifier, valid_values: set[Any]
    ) -> None:
        super().__init__(constrained_variable)
        self._valid_values = valid_values

    def is_satisfied(self, value: Any) -> bool:
        return value in self._valid_values

    def copy(self) -> "InSetConstraint":
        new_constraint = InSetConstraint(self.variable, self._valid_values.copy())
        return new_constraint

    def convert_to_expression(self) -> Expression:
        if len(self._valid_values) == 0:
            return LiteralExpression(False)
        elif len(self._valid_values) == 1:
            return self._generate_single_value_constraint(
                next(iter(self._valid_values))
            )
        else:
            return Expression.logical_or(
                *map(self._generate_single_value_constraint, self._valid_values)
            )

    def _generate_single_value_constraint(self, value: Any) -> Expression:
        if not isinstance(value, LiteralType):
            raise ValueError(
                f"Conversion of type {type(value)} to an expression is not "
                "supported."
            )
        variable = IdentifierExpression(self.variable)
        return BinaryExpression(
            BinaryOperation.EQUAL,
            variable,
            LiteralExpression(value),
        )


class NotInSetConstraint(Constraint):
    """Represents a not-in-set constraint."""

    _invalid_values: set[Any]

    def __init__(
        self, constrained_variable: Identifier, invalid_values: set[Any]
    ) -> None:
        super().__init__(constrained_variable)
        self._invalid_values = invalid_values

    def is_satisfied(self, value: Any) -> bool:
        return value not in self._invalid_values

    def copy(self) -> "NotInSetConstraint":
        new_constraint = NotInSetConstraint(self.variable, self._invalid_values.copy())
        return new_constraint

    def convert_to_expression(self) -> Expression:
        if len(self._invalid_values) == 0:
            return LiteralExpression(True)
        elif len(self._invalid_values) == 1:
            return self._generate_single_value_constraint(
                next(iter(self._invalid_values))
            )
        else:
            return Expression.logical_and(
                *map(self._generate_single_value_constraint, self._invalid_values)
            )

    def _generate_single_value_constraint(self, value: Any) -> Expression:
        if not isinstance(value, LiteralType):
            raise ValueError(
                f"Conversion of type {type(value)} to an expression is not "
                "supported."
            )
        variable = IdentifierExpression(self.variable)
        return BinaryExpression(
            BinaryOperation.NOT_EQUAL,
            variable,
            LiteralExpression(value),
        )
