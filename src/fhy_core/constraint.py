"""Constraint utility."""

from abc import ABC, abstractmethod
from typing import Any

from .expression import Expression, LiteralExpression, simplify_expression
from .identifier import Identifier


class Constraint(ABC):
    """Abstract base class for constraints."""

    def __call__(self, values: dict[Identifier, Any]) -> bool:
        return self.check(values)

    @abstractmethod
    def check(self, values: dict[Identifier, Any]) -> bool:
        """Check if the value satisfies the constraint.

        Args:
            values: Variable values.

        Returns:
            True if the value satisfies the constraint; False otherwise.

        """


class EquationConstraint(Constraint):
    """Represents an equation constraint."""

    _expression: Expression

    def __init__(self, expression: Expression) -> None:
        self._expression = expression

    def check(self, values: dict[Identifier, Expression]) -> bool:
        result = simplify_expression(self._expression, values)
        return (
            isinstance(result, LiteralExpression)
            and isinstance(result.value, bool)
            and result.value
        )
