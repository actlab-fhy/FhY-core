"""Fundamental parameter classes."""

__all__ = ["NatParam"]

from fhy_core.constraint import EquationConstraint
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.identifier import Identifier

from .core import IntParam


class NatParam(IntParam):
    """Natural number parameter."""

    def __init__(
        self, name: Identifier | None = None, is_zero_included: bool = True
    ) -> None:
        super().__init__(name)
        if is_zero_included:
            self.add_constraint(
                EquationConstraint(
                    self.variable,
                    BinaryExpression(
                        BinaryOperation.GREATER_EQUAL,
                        IdentifierExpression(self.variable),
                        LiteralExpression(0),
                    ),
                )
            )
        else:
            self.add_constraint(
                EquationConstraint(
                    self.variable,
                    BinaryExpression(
                        BinaryOperation.GREATER,
                        IdentifierExpression(self.variable),
                        LiteralExpression(0),
                    ),
                )
            )
