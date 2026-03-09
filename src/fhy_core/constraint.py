"""Constraint utility."""

__all__ = [
    "Constraint",
    "EquationConstraint",
    "InSetConstraint",
    "NotInSetConstraint",
]

from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable
from functools import singledispatch
from typing import (
    Any,
    Generic,
    TypeAlias,
    TypedDict,
    TypeGuard,
    TypeVar,
    cast,
)

from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    Serializable,
    SerializedDict,
    SerializedValue,
    WrappedFamilySerializable,
    deserialize_registry_wrapped_value,
    is_serialized_dict,
    register_serializable,
    serialize_registry_wrapped_value,
)
from fhy_core.trait import FrozenMixin, StructuralEquivalenceMixin
from fhy_core.utils import format_comma_separated_list

from .expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    LiteralType,
    pformat_expression,
    simplify_expression,
)
from .identifier import Identifier


class Constraint(
    WrappedFamilySerializable, FrozenMixin, StructuralEquivalenceMixin, ABC
):
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
    def convert_to_expression(self) -> Expression:
        """Return an expression equivalent to the constraint.

        Raises:
            ValueError: If the constraint cannot be converted to an expression.

        """

    def is_structurally_equivalent(self, other: object) -> bool:
        return _is_constraint_structurally_equivalent(self, other)

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def __str__(self) -> str: ...


ConstraintPrimitive: TypeAlias = str | int | float | bool

ConstraintMember: TypeAlias = (
    ConstraintPrimitive
    | Serializable
    | tuple["ConstraintMember", ...]
    | frozenset["ConstraintMember"]
)
_ConstraintMemberT = TypeVar("_ConstraintMemberT", bound=ConstraintMember)


def _is_valid_constraint_primitive(value: Any) -> TypeGuard[ConstraintPrimitive]:
    return isinstance(value, (str, int, float, bool))


def _is_serializable_hashable(value: Any) -> bool:
    return isinstance(value, Serializable) and isinstance(value, Hashable)


def _validate_constraint_member(value: Any) -> None:
    if value is None:
        raise ValueError("Constraint members cannot be `None`.")
    if isinstance(value, (tuple, frozenset)):
        for nested_value in value:
            _validate_constraint_member(nested_value)
        if not isinstance(value, Hashable):
            raise ValueError(
                "Constraint member containers must be hashable, but got "
                f"value {value} of type {type(value)}.",
            )
        return
    if _is_valid_constraint_primitive(value) or _is_serializable_hashable(value):
        return
    raise ValueError(
        "Constraint member must be either a primitive literal "
        "(`str`, `int`, `float`, `bool`), both `Serializable` and `Hashable`, "
        "or a tuple/frozenset containing valid constraint members, but got value "
        f"{value} of type {type(value)}.",
    )


def _validate_constraint_members(values: Collection[ConstraintMember]) -> None:
    for value in values:
        _validate_constraint_member(value)


def _normalize_constraint_member_collection(
    values: Collection[ConstraintMember],
) -> frozenset[ConstraintMember]:
    _validate_constraint_members(values)
    try:
        return frozenset(values)
    except TypeError as exc:
        raise ValueError(
            "Constraint members must be hashable after validation."
        ) from exc


def _is_equivalent_constraint_member_collections(
    values_1: Collection[ConstraintMember], values_2: Collection[ConstraintMember]
) -> bool:
    return frozenset(values_1) == frozenset(values_2)


def _serialize_constraint_member(value: ConstraintMember) -> SerializedValue:
    return serialize_registry_wrapped_value(value)


def _deserialize_constraint_member(
    owner_class: type[Any], field_name: str, value: SerializedValue
) -> ConstraintMember:
    if not is_serialized_dict(value):
        raise DeserializationValueError(
            owner_class,
            field_name,
            "a wrapped dictionary value",
            value,
        )

    try:
        member = deserialize_registry_wrapped_value(value)
    except (DeserializationDictStructureError, DeserializationValueError) as exc:
        raise DeserializationValueError(
            f'Invalid serialized member in field "{field_name}": {exc}'
        ) from exc

    _validate_constraint_member(member)
    return member


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
        self.freeze(deep=True)

    def is_satisfied(self, value: Expression | LiteralType) -> bool:
        if isinstance(value, (str, float, int, bool)):
            value = LiteralExpression(value)
        result = simplify_expression(self._expression, {self.variable: value})
        return (
            isinstance(result, LiteralExpression)
            and isinstance(result.value, bool)
            and result.value
        )

    def convert_to_expression(self) -> Expression:
        return self._expression

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
    valid_values: list[SerializedValue]


def _is_valid_in_set_constraint_data(
    data: SerializedDict,
) -> TypeGuard[_InSetConstraintData]:
    return (
        "variable" in data
        and is_serialized_dict(data["variable"])
        and "valid_values" in data
        and isinstance(data["valid_values"], list)
        and all(is_serialized_dict(value) for value in data["valid_values"])
    )


@register_serializable(type_id="in_set_constraint")
class InSetConstraint(Constraint, Generic[_ConstraintMemberT]):
    """Represents an in-set constraint."""

    _valid_values: frozenset[_ConstraintMemberT]

    def __init__(
        self,
        constrained_variable: Identifier,
        valid_values: Collection[_ConstraintMemberT],
    ) -> None:
        super().__init__(constrained_variable)
        self._valid_values = cast(
            frozenset[_ConstraintMemberT],
            _normalize_constraint_member_collection(
                cast(Collection[ConstraintMember], valid_values)
            ),
        )
        self.freeze(deep=True)

    def is_satisfied(self, value: _ConstraintMemberT) -> bool:
        return value in self._valid_values

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
            "valid_values": sorted(
                [_serialize_constraint_member(value) for value in self._valid_values],
                key=repr,
            ),
        }

    @classmethod
    def deserialize_data_from_dict(
        cls: type["InSetConstraint[_ConstraintMemberT]"], data: SerializedDict
    ) -> "InSetConstraint[_ConstraintMemberT]":
        if not _is_valid_in_set_constraint_data(data):
            raise DeserializationDictStructureError(
                cls, _InSetConstraintData.__annotations__, data
            )
        members = {
            _deserialize_constraint_member(cls, "valid_values", value)
            for value in data["valid_values"]
        }
        return cls(
            Identifier.deserialize_from_dict(data["variable"]),
            cast(set[_ConstraintMemberT], members),
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
    invalid_values: list[SerializedValue]


def _is_valid_not_in_set_constraint_data(
    data: SerializedDict,
) -> TypeGuard[_NotInSetConstraintData]:
    return (
        "variable" in data
        and is_serialized_dict(data["variable"])
        and "invalid_values" in data
        and isinstance(data["invalid_values"], list)
        and all(is_serialized_dict(value) for value in data["invalid_values"])
    )


@register_serializable(type_id="not_in_set_constraint")
class NotInSetConstraint(Constraint, Generic[_ConstraintMemberT]):
    """Represents a not-in-set constraint."""

    _invalid_values: frozenset[_ConstraintMemberT]

    def __init__(
        self,
        constrained_variable: Identifier,
        invalid_values: Collection[_ConstraintMemberT],
    ) -> None:
        super().__init__(constrained_variable)
        self._invalid_values = cast(
            frozenset[_ConstraintMemberT],
            _normalize_constraint_member_collection(
                cast(Collection[ConstraintMember], invalid_values)
            ),
        )
        self.freeze(deep=True)

    def is_satisfied(self, value: _ConstraintMemberT) -> bool:
        return value not in self._invalid_values

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
            "invalid_values": sorted(
                [_serialize_constraint_member(value) for value in self._invalid_values],
                key=repr,
            ),
        }

    @classmethod
    def deserialize_data_from_dict(
        cls: type["NotInSetConstraint[_ConstraintMemberT]"], data: SerializedDict
    ) -> "NotInSetConstraint[_ConstraintMemberT]":
        if not _is_valid_not_in_set_constraint_data(data):
            raise DeserializationDictStructureError(
                cls, _NotInSetConstraintData.__annotations__, data
            )
        members = {
            _deserialize_constraint_member(cls, "invalid_values", value)
            for value in data["invalid_values"]
        }
        return cls(
            Identifier.deserialize_from_dict(data["variable"]),
            cast(set[_ConstraintMemberT], members),
        )

    def __repr__(self) -> str:
        return repr(self._invalid_values)

    def __str__(self) -> str:
        return (
            f"{self.variable} not in {{"
            f"{format_comma_separated_list(self._invalid_values, str_func=str)}}}"
        )


@singledispatch
def _is_constraint_structurally_equivalent(
    constraint: Constraint, other: object
) -> bool:
    return False


@_is_constraint_structurally_equivalent.register
def _(constraint: EquationConstraint, other: object) -> bool:
    return (
        isinstance(other, EquationConstraint)
        and constraint.variable == other.variable
        and constraint._expression.is_structurally_equivalent(other._expression)
    )


@_is_constraint_structurally_equivalent.register
def _(
    constraint: InSetConstraint,  # type: ignore[type-arg]
    other: object,
) -> bool:
    return (
        isinstance(other, InSetConstraint)
        and constraint.variable == other.variable
        and _is_equivalent_constraint_member_collections(
            constraint._valid_values, other._valid_values
        )
    )


@_is_constraint_structurally_equivalent.register
def _(
    constraint: NotInSetConstraint,  # type: ignore[type-arg]
    other: object,
) -> bool:
    return (
        isinstance(other, NotInSetConstraint)
        and constraint.variable == other.variable
        and _is_equivalent_constraint_member_collections(
            constraint._invalid_values, other._invalid_values
        )
    )
