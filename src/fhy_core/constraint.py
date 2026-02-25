"""Constraint utility."""

__all__ = [
    "Constraint",
    "EquationConstraint",
    "InSetConstraint",
    "NotInSetConstraint",
]

from abc import ABC, abstractmethod
from typing import Any, TypedDict, TypeGuard

from fhy_core.serialization import (
    DeserializationDictStructureError,
    SerializedDict,
    WrappedFamilySerializable,
    is_serialized_dict,
    register_serializable,
)
from fhy_core.utils import Self, format_comma_separated_list

from .expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    LiteralType,
    copy_expression,
    pformat_expression,
    simplify_expression,
)
from .identifier import Identifier


class Constraint(WrappedFamilySerializable, ABC):
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
    def copy(self) -> Self:
        """Return a shallow copy of the constraint."""

    @abstractmethod
    def convert_to_expression(self) -> Expression:
        """Return an expression equivalent to the constraint.

        Raises:
            ValueError: If the constraint cannot be converted to an expression.

        """

    def __copy__(self) -> Self:
        return self.copy()

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def __str__(self) -> str: ...


class _EquationConstraintData(TypedDict):
    variable: SerializedDict
    expression: SerializedDict


def _is_valid_equation_constraint_data(
    data: SerializedDict,
) -> TypeGuard[_EquationConstraintData]:
    return (
        "variable" in data
        and is_serialized_dict(data["variable"])
        and "expression" in data
        and is_serialized_dict(data["expression"])
    )


@register_serializable(type_id="equation_constraint")
class EquationConstraint(Constraint):
    """Represents an equation constraint."""

    _expression: Expression

    def __init__(
        self, constrained_variable: Identifier, expression: Expression
    ) -> None:
        super().__init__(constrained_variable)
        self._expression = expression

    def is_satisfied(self, value: Expression | LiteralType) -> bool:
        if isinstance(value, (str, float, int, bool)):
            value = LiteralExpression(value)
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

    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            "variable": self.variable.serialize_to_dict(),
            "expression": self._expression.serialize_to_dict(),
        }

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "EquationConstraint":
        if not _is_valid_equation_constraint_data(data):
            raise DeserializationDictStructureError(
                cls, _EquationConstraintData.__annotations__, data
            )
        return cls(
            Identifier.deserialize_from_dict(data["variable"]),
            Expression.deserialize_from_dict(data["expression"]),
        )

    def __repr__(self) -> str:
        return repr(self._expression)

    def __str__(self) -> str:
        return pformat_expression(self._expression)


class _InSetConstraintData(TypedDict):
    variable: SerializedDict
    valid_values: list[Any]


def _is_valid_in_set_constraint_data(
    data: SerializedDict,
) -> TypeGuard[_InSetConstraintData]:
    return (
        "variable" in data
        and is_serialized_dict(data["variable"])
        and "valid_values" in data
        and isinstance(data["valid_values"], list)
    )


@register_serializable(type_id="in_set_constraint")
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
                f"Conversion of type {type(value)} to an expression is not supported."
            )
        variable = IdentifierExpression(self.variable)
        return BinaryExpression(
            BinaryOperation.EQUAL,
            variable,
            LiteralExpression(value),
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            "variable": self.variable.serialize_to_dict(),
            "valid_values": sorted(self._valid_values, key=repr),
        }

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "InSetConstraint":
        if not _is_valid_in_set_constraint_data(data):
            raise DeserializationDictStructureError(
                cls, _InSetConstraintData.__annotations__, data
            )
        return cls(
            Identifier.deserialize_from_dict(data["variable"]),
            set(data["valid_values"]),
        )

    def __repr__(self) -> str:
        return repr(self._valid_values)

    def __str__(self) -> str:
        return (
            f"{self.variable} in {{"
            f"{format_comma_separated_list(self._valid_values, str_func=str)}}}"
        )


class _NotInSetConstraintData(TypedDict):
    variable: SerializedDict
    invalid_values: list[Any]


def _is_valid_not_in_set_constraint_data(
    data: SerializedDict,
) -> TypeGuard[_NotInSetConstraintData]:
    return (
        "variable" in data
        and is_serialized_dict(data["variable"])
        and "invalid_values" in data
        and isinstance(data["invalid_values"], list)
    )


@register_serializable(type_id="not_in_set_constraint")
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
                f"Conversion of type {type(value)} to an expression is not supported."
            )
        variable = IdentifierExpression(self.variable)
        return BinaryExpression(
            BinaryOperation.NOT_EQUAL,
            variable,
            LiteralExpression(value),
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            "variable": self.variable.serialize_to_dict(),
            "invalid_values": sorted(self._invalid_values, key=repr),
        }

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "NotInSetConstraint":
        if not _is_valid_not_in_set_constraint_data(data):
            raise DeserializationDictStructureError(
                cls, _NotInSetConstraintData.__annotations__, data
            )
        return cls(
            Identifier.deserialize_from_dict(data["variable"]),
            set(data["invalid_values"]),
        )

    def __repr__(self) -> str:
        return repr(self._invalid_values)

    def __str__(self) -> str:
        return (
            f"{self.variable} not in {{"
            f"{format_comma_separated_list(self._invalid_values, str_func=str)}}}"
        )
