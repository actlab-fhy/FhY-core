"""Fundamental parameter classes."""

__all__ = [
    "NatParam",
    "is_the_basic_nat_param_constraint",
    "is_zero_included_constraint_exists",
    "is_zero_not_included_constraint_exists",
]

from collections.abc import Iterable
from typing import Any, cast

from fhy_core.constraint import Constraint, EquationConstraint
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    SerializedDict,
    register_serializable,
)

from .core import (
    IntParam,
    ParamData,
    finalize_param_construction_from_data,
    is_valid_param_data,
)


def is_the_basic_nat_param_constraint(
    constraint: Constraint, variable: Identifier, is_zero_included: bool
) -> bool:
    """Return if the constraint is the basic constraint bounding the `NatParam`."""
    expression = constraint.convert_to_expression()
    if isinstance(expression, BinaryExpression) and (
        is_zero_included
        and expression.operation == BinaryOperation.GREATER_EQUAL
        or not is_zero_included
        and expression.operation == BinaryOperation.GREATER
    ):
        left = expression.left
        right = expression.right
        return (
            isinstance(left, IdentifierExpression)
            and left.identifier == variable
            and isinstance(right, LiteralExpression)
            and isinstance(right.value, int)
            and right.value == 0
        )
    return False


def is_zero_included_constraint_exists(
    constraints: Iterable[Constraint], variable: Identifier
) -> bool:
    """Return if there is a constraint bounding the variable to be GE to zero."""
    for constraint in constraints:
        expression = constraint.convert_to_expression()
        if (
            isinstance(expression, BinaryExpression)
            and expression.operation == BinaryOperation.GREATER_EQUAL
        ):
            left = expression.left
            right = expression.right
            if (
                isinstance(left, IdentifierExpression)
                and left.identifier == variable
                and isinstance(right, LiteralExpression)
                and isinstance(right.value, int)
                and right.value == 0
            ):
                return True
    return False


def is_zero_not_included_constraint_exists(
    constraints: Iterable[Constraint], variable: Identifier
) -> bool:
    """Return if there is a constraint bounding the variable to be GT to zero."""
    for constraint in constraints:
        expression = constraint.convert_to_expression()
        if (
            isinstance(expression, BinaryExpression)
            and expression.operation == BinaryOperation.GREATER
        ):
            left = expression.left
            right = expression.right
            if (
                isinstance(left, IdentifierExpression)
                and left.identifier == variable
                and isinstance(right, LiteralExpression)
                and isinstance(right.value, int)
                and right.value == 0
            ):
                return True
    return False


@register_serializable(type_id="nat_param")
class NatParam(IntParam):
    """Natural number parameter."""

    _is_zero_included: bool

    def __init__(
        self,
        *,
        name: Identifier | None = None,
        is_zero_included: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name)
        object.__setattr__(self, "_is_zero_included", is_zero_included)
        if self._is_zero_included:
            basic_constraint = EquationConstraint(
                self.variable,
                BinaryExpression(
                    BinaryOperation.GREATER_EQUAL,
                    IdentifierExpression(self.variable),
                    LiteralExpression(0),
                ),
            )
        else:
            basic_constraint = EquationConstraint(
                self.variable,
                BinaryExpression(
                    BinaryOperation.GREATER,
                    IdentifierExpression(self.variable),
                    LiteralExpression(0),
                ),
            )
        object.__setattr__(
            self, "_constraints", self._constraints + (basic_constraint,)
        )

    def add_lower_bound_constraint(
        self, lower_bound: int, *, is_inclusive: bool = True
    ) -> "NatParam":
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

        return super().add_lower_bound_constraint(
            lower_bound, is_inclusive=is_inclusive
        )

    def add_upper_bound_constraint(
        self, upper_bound: int, *, is_inclusive: bool = True
    ) -> "NatParam":
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

        return super().add_upper_bound_constraint(
            upper_bound, is_inclusive=is_inclusive
        )

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "NatParam":
        if not is_valid_param_data(data):
            raise DeserializationDictStructureError(
                cls, ParamData.__annotations__, data
            )

        variable = Identifier.deserialize_from_dict(data["variable"])
        constraints = [Constraint.deserialize_from_dict(c) for c in data["constraints"]]

        if is_zero_included_constraint_exists(constraints, variable):
            is_zero_included = True
        elif is_zero_not_included_constraint_exists(constraints, variable):
            is_zero_included = False
        else:
            raise DeserializationValueError(
                cls,
                "constraints",
                (
                    "a constraint bounding the variable to be greater than or "
                    "equal to zero or greater than zero"
                ),
                data["constraints"],
            )

        param = NatParam(name=variable, is_zero_included=is_zero_included)
        return cast(
            NatParam,
            finalize_param_construction_from_data(
                param,
                data,
                constraint_filter_function=lambda c: (
                    not is_the_basic_nat_param_constraint(c, variable, is_zero_included)
                ),
            ),
        )
