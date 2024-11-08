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
    Expression,
    LiteralExpression,
    copy_expression,
    simplify_expression,
)
from .identifier import Identifier


class Constraint(ABC):
    """Abstract base class for constraints."""

    def __call__(self, values: dict[Identifier, Any]) -> bool:
        return self.is_satisfied(values)

    @abstractmethod
    def is_satisfied(self, values: dict[Identifier, Any]) -> bool:
        """Check if the value satisfies the constraint.

        Args:
            values: Variable values.

        Returns:
            True if the value satisfies the constraint; False otherwise.

        """

    @abstractmethod
    def copy(self) -> "Constraint":
        """Return a shallow copy of the constraint."""


class EquationConstraint(Constraint):
    """Represents an equation constraint."""

    _expression: Expression

    def __init__(self, expression: Expression) -> None:
        self._expression = expression

    def is_satisfied(self, values: dict[Identifier, Expression]) -> bool:
        result = simplify_expression(self._expression, values)
        return (
            isinstance(result, LiteralExpression)
            and isinstance(result.value, bool)
            and result.value
        )

    def copy(self) -> "EquationConstraint":
        new_constraint = EquationConstraint(copy_expression(self._expression))
        return new_constraint


class InSetConstraint(Constraint):
    """Represents an in-set constraint."""

    _variables: set[Identifier]
    _valid_values: set[Any]

    def __init__(
        self, constrained_variables: set[Identifier], valid_values: set[Any]
    ) -> None:
        self._variables = constrained_variables
        self._valid_values = valid_values

    def is_satisfied(self, values: dict[Identifier, Any]) -> bool:
        return all(
            values[variable] in self._valid_values for variable in self._variables
        )

    def copy(self) -> "InSetConstraint":
        new_constraint = InSetConstraint(
            self._variables.copy(), self._valid_values.copy()
        )
        return new_constraint


class NotInSetConstraint(Constraint):
    """Represents a not-in-set constraint."""

    _variables: set[Identifier]
    _invalid_values: set[Any]

    def __init__(
        self, constrained_variables: set[Identifier], invalid_values: set[Any]
    ) -> None:
        self._variables = constrained_variables
        self._invalid_values = invalid_values

    def is_satisfied(self, values: dict[Identifier, Any]) -> bool:
        return any(
            values[variable] not in self._invalid_values for variable in self._variables
        )

    def copy(self) -> "NotInSetConstraint":
        new_constraint = NotInSetConstraint(
            self._variables.copy(), self._invalid_values.copy()
        )
        return new_constraint
