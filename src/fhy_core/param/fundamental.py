"""Fundamental parameter classes."""

__all__ = ["NatParam"]

from typing import Any

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

    _is_zero_included: bool

    def __init__(
        self,
        name: Identifier | None = None,
        is_zero_included: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(name)
        self._is_zero_included = is_zero_included
        if self._is_zero_included:
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

    def add_lower_bound_constraint(
        self, lower_bound: int, is_inclusive: bool = True
    ) -> None:
        if self._is_zero_included:
            if lower_bound < 0:
                raise ValueError("Lower bound must be non-negative.")
            if not is_inclusive and lower_bound < 1:
                raise ValueError(
                    "Lower bound must be at least 1 if zero is included and "
                    "bound is exclusive."
                )
        elif is_inclusive:
            if lower_bound < 1:
                raise ValueError(
                    "Lower bound must be at least 1 when zero is not included."
                )
        elif lower_bound < 0:
            raise ValueError(
                "Lower bound must be non-negative when zero is not included "
                "and bound is exclusive."
            )

        return super().add_lower_bound_constraint(lower_bound, is_inclusive)

    def add_upper_bound_constraint(
        self, upper_bound: int, is_inclusive: bool = True
    ) -> None:
        if self._is_zero_included:
            if is_inclusive:
                if upper_bound < 0:
                    raise ValueError(
                        "Upper bound must be non-negative when zero is included."
                    )
            elif upper_bound < 1:
                raise ValueError(
                    "Upper bound must be at least 1 if zero is included and "
                    "bound is exclusive."
                )
        elif is_inclusive:
            if upper_bound < 1:
                raise ValueError(
                    "Upper bound must be at least 1 when zero is not included."
                )
        elif upper_bound < 2:  # noqa: PLR2004
            raise ValueError(
                "Upper bound must be at least 2 when zero is not included "
                "and bound is exclusive."
            )

        return super().add_upper_bound_constraint(upper_bound, is_inclusive)

    # TODO: Add deserialization method
