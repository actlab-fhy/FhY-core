"""Named-variable constraints over expressions and value sets.

This module provides the three constraint kinds used by the parameter
infrastructure in ``fhy_core.param.*``:

- ``EquationConstraint`` -- a Boolean expression that must hold when
  the variable is bound to a candidate value.
- ``InSetConstraint`` -- the variable must take a value from a
  permitted set.
- ``NotInSetConstraint`` -- the variable must NOT take a value from a
  forbidden set.

Every constraint is keyed by a single ``Identifier``, is frozen on
construction via ``FrozenMixin``'s ``freeze_on_init`` mechanism, is
serializable through ``WrappedFamilySerializable``, and supports
structural-equivalence comparison via a singledispatch table.

Each constraint is callable as a predicate -- ``constraint(value)`` is
the same as ``constraint.is_satisfied(value)`` -- and can be converted
to an equivalent ``Expression`` over its variable through
``Constraint.convert_to_expression``.

Set-constraint members are stored with type-strict equality: ``int``,
``float``, and ``bool`` are not interchangeable, including inside
nested ``tuple`` and ``frozenset`` members. The wrapping that
implements this is private; the public API accepts and returns raw
values.

Set-constraint serialization is deterministic: members are emitted in
``repr``-sorted order, and the corresponding ``convert_to_expression``
leaves are emitted in the same order.
"""

__all__ = [
    "Constraint",
    "ConstraintError",
    "ConstraintMember",
    "EquationConstraint",
    "InSetConstraint",
    "NotInSetConstraint",
]

from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable
from functools import singledispatch
from typing import Any, TypeAlias, TypedDict, TypeGuard

from fhy_core.error import register_error
from fhy_core.logger import get_logger
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    Serializable,
    SerializedDict,
    WrappedFamilySerializable,
    deserialize_registry_wrapped_value,
    is_serialized_dict,
    register_serializable,
    serialize_registry_wrapped_value,
)
from fhy_core.trait import FrozenMixin, StructuralEquivalenceMixin
from fhy_core.utils import format_comma_separated_list

from .expression import (
    BinaryOperation,
    Expression,
    LiteralExpression,
    LiteralType,
    make_binary_expression,
    pformat_expression,
    simplify_expression,
)
from .identifier import Identifier

_LOGGER = get_logger(__name__)


@register_error
class ConstraintError(ValueError):
    """Domain error for constraint construction, validation, and conversion."""


_ConstraintPrimitive: TypeAlias = str | int | float | bool

ConstraintMember: TypeAlias = (
    _ConstraintPrimitive
    | Serializable
    | tuple["ConstraintMember", ...]
    | frozenset["ConstraintMember"]
)
"""Allowed constraint member kinds.

A constraint member is one of: the four primitive Python types
(``str``, ``int``, ``float``, ``bool``); any ``Serializable`` instance
that is also ``Hashable``; or a tuple or frozenset of valid members.
Members are stored with type-strict equality: ``int``, ``float``, and
``bool`` are not interchangeable, even at the leaves of nested
containers.
"""


def _is_valid_constraint_primitive(value: Any) -> TypeGuard[_ConstraintPrimitive]:
    return isinstance(value, (str, int, float, bool))


def _is_serializable_hashable(value: Any) -> bool:
    return isinstance(value, Serializable) and isinstance(value, Hashable)


def _validate_constraint_member(value: Any) -> None:
    if value is None:
        raise ConstraintError("Constraint members cannot be `None`.")
    elif isinstance(value, (tuple, frozenset)):
        if not isinstance(value, Hashable):
            raise ConstraintError(
                "Constraint member containers must be hashable, but got "
                f"value {value} of type {type(value)}.",
            )
        for nested_value in value:
            _validate_constraint_member(nested_value)
        return
    elif _is_valid_constraint_primitive(value) or _is_serializable_hashable(value):
        return
    else:
        raise ConstraintError(
            "Constraint member must be either a primitive literal "
            "(`str`, `int`, `float`, `bool`), both `Serializable` and `Hashable`, "
            "or a tuple/frozenset containing valid constraint members, but got value "
            f"{value} of type {type(value)}.",
        )


class _TypedMember:
    """Internal wrapper providing type-strict equality and hashing.

    Wraps a constraint member so that ``__eq__`` and ``__hash__`` use
    ``(type(value), value)`` instead of the value's own equality. This
    keeps ``True`` and ``1`` distinct, ``1`` and ``1.0`` distinct, and
    ensures nested ``tuple`` and ``frozenset`` containers preserve the
    same type-strict semantics for their elements.

    The stored ``_value`` is typed ``Any`` because the wrap stores both
    raw primitives and nested containers whose elements are themselves
    ``_TypedMember`` instances (so the static type for nested containers
    is ``tuple[_TypedMember, ...]`` / ``frozenset[_TypedMember]``, not
    ``tuple[ConstraintMember, ...]``). Callers should always go through
    ``_unwrap_member`` rather than reading ``value`` directly.
    """

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        self._value = value

    @property
    def value(self) -> Any:
        return self._value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _TypedMember):
            return NotImplemented
        return type(self._value) is type(other._value) and self._value == other._value

    def __hash__(self) -> int:
        return hash((type(self._value), self._value))

    def __repr__(self) -> str:
        return repr(self._value)

    def __str__(self) -> str:
        return str(self._value)


def _wrap_member(value: Any) -> _TypedMember:
    """Recursively wrap a validated constraint member for type-strict storage."""
    if isinstance(value, tuple):
        return _TypedMember(tuple(_wrap_member(v) for v in value))
    elif isinstance(value, frozenset):
        return _TypedMember(frozenset(_wrap_member(v) for v in value))
    else:
        return _TypedMember(value)


def _unwrap_member(wrapped: Any) -> Any:
    """Recursively unwrap a stored member to its raw form."""
    if isinstance(wrapped, _TypedMember):
        return _unwrap_member(wrapped.value)
    elif isinstance(wrapped, tuple):
        return tuple(_unwrap_member(v) for v in wrapped)
    elif isinstance(wrapped, frozenset):
        return frozenset(_unwrap_member(v) for v in wrapped)
    else:
        return wrapped


def _normalize_constraint_member_collection(
    values: Collection[ConstraintMember],
) -> frozenset[_TypedMember]:
    """Validate, wrap, and freeze a collection of constraint members.

    Validation, wrapping, and hashing all happen in a single pass over
    ``values`` so a one-shot iterable would still produce the right
    result even though the parameter type is ``Collection``.

    Args:
        values: Iterable of raw constraint members.

    Returns:
        Frozen set of wrapped members with type-strict equality and hashing.

    Raises:
        ConstraintError: If any member fails validation or is unhashable
            after validation.

    """
    wrapped_values: list[_TypedMember] = []
    for value in values:
        _validate_constraint_member(value)
        wrapped = _wrap_member(value)
        try:
            hash(wrapped)
        except TypeError as exc:
            raise ConstraintError(
                f"Constraint member is unhashable after validation: "
                f"value {value!r} of type {type(value).__name__}."
            ) from exc
        wrapped_values.append(wrapped)
    return frozenset(wrapped_values)


def _lift_member_to_literal_expression(value: ConstraintMember) -> LiteralExpression:
    """Lifts a constraint member to a ``LiteralExpression``.

    Raises:
        ConstraintError: If the member is not a ``LiteralType`` or if
            ``LiteralExpression`` refuses the value.

    """
    if not isinstance(value, LiteralType):
        raise ConstraintError(
            f"Conversion of type {type(value).__name__} to an expression is "
            "not supported."
        )
    try:
        return LiteralExpression(value)
    except ValueError as exc:
        raise ConstraintError(
            f"Member {value!r} cannot be represented as a literal expression: {exc}"
        ) from exc


def _serialize_constraint_member(value: ConstraintMember) -> SerializedDict:
    return serialize_registry_wrapped_value(value)


def _deserialize_constraint_member(
    field_name: str, value: SerializedDict
) -> ConstraintMember:
    try:
        member = deserialize_registry_wrapped_value(value)
    except (DeserializationDictStructureError, DeserializationValueError) as exc:
        raise DeserializationValueError(
            f'Invalid serialized member in field "{field_name}": {exc}'
        ) from exc
    _validate_constraint_member(member)
    return member


class Constraint(
    WrappedFamilySerializable,
    FrozenMixin,
    StructuralEquivalenceMixin,
    ABC,
    freeze_on_init=True,
    freeze_on_init_deep=True,
):
    """A named-variable predicate.

    Subclasses model the three concrete constraint kinds in this module
    (``EquationConstraint``, ``InSetConstraint``, ``NotInSetConstraint``).
    Instances are frozen at the end of construction; subsequent
    attribute mutation raises ``FrozenMutationError``. Instances are
    callable; ``constraint(value)`` is an alias for
    ``constraint.is_satisfied(value)``.

    Subclassing contract:
        - Override ``is_satisfied`` to define the predicate.
        - Override ``convert_to_expression`` to produce an equivalent
          ``Expression`` over the variable.
        - Override ``serialize_data_to_dict`` and
          ``deserialize_data_from_dict`` to enable serialization.
        - Override ``__repr__`` and ``__str__`` so the textual form
          identifies the kind and the variable.
        - Register a handler with
          ``_is_constraint_structurally_equivalent.register`` so the
          singledispatch table knows how to compare instances of the
          new kind. Unregistered subclasses raise
          ``NotImplementedError`` on structural-equivalence comparison.

    """

    _variable: Identifier

    def __init__(self, constrained_variable: Identifier) -> None:
        self._variable = constrained_variable

    @property
    def variable(self) -> Identifier:
        return self._variable

    def __call__(self, value: Any) -> bool:
        return self.is_satisfied(value)

    @abstractmethod
    def is_satisfied(self, value: Any) -> bool:
        """Return whether the value satisfies the constraint.

        Args:
            value: Candidate value to check.

        Returns:
            True if the value satisfies the constraint; False otherwise.
            Subclasses document their predicate semantics and any
            indeterminate cases.

        """

    @abstractmethod
    def convert_to_expression(self) -> Expression:
        """Return an expression equivalent to the constraint.

        Returns:
            An ``Expression`` over ``self.variable`` whose truth value
            matches ``is_satisfied``.

        Raises:
            ConstraintError: If the constraint cannot be expressed
                (e.g. a set member is not a literal type).

        """

    def is_structurally_equivalent(self, other: object) -> bool:
        return _is_constraint_structurally_equivalent(self, other)

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
    """Boolean-expression predicate over the variable.

    The constraint wraps a Boolean ``Expression`` whose only free
    identifier is meant to be ``self.variable``. ``is_satisfied``
    substitutes the candidate value for that identifier, simplifies the
    resulting expression, and returns True only when the simplifier
    reduces it to a ``LiteralExpression`` with a ``bool`` value of
    ``True``.

    Indeterminate case:
        If the simplifier cannot reduce the substituted expression to a
        ``LiteralExpression`` at all -- for example because the
        expression references additional free identifiers, or because
        the simplifier just cannot decide -- ``is_satisfied`` logs a
        ``WARNING`` through the module logger and returns ``False``.
        Callers that need to distinguish "constraint rejects this
        value" from "constraint cannot decide" should configure log
        capture or escalate the warning channel.

        A reduction to a literal whose value is not ``bool`` (for
        example ``LiteralExpression(1)``) also returns ``False`` but
        is treated as a decided "no" rather than as an indeterminate
        case: no warning is emitted.

    """

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
        if isinstance(result, LiteralExpression) and isinstance(result.value, bool):
            return result.value
        if not isinstance(result, LiteralExpression):
            _LOGGER.warning(
                "%s.is_satisfied: substituted expression %r did not reduce to a "
                "literal; returning False",
                type(self).__name__,
                result,
            )
        return False

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
        return f"EquationConstraint({self.variable!r}, expression={self._expression!r})"

    def __str__(self) -> str:
        return pformat_expression(self._expression)


def _render_member_set(members: frozenset[_TypedMember]) -> str:
    rendered_members = format_comma_separated_list(
        sorted((_unwrap_member(member) for member in members), key=repr),
    )
    return f"{{{rendered_members}}}"


def _render_member_set_str(members: frozenset[_TypedMember]) -> str:
    rendered_members = format_comma_separated_list(
        (_unwrap_member(member) for member in members), str_func=str
    )
    return f"{{{rendered_members}}}"


def _serialize_member_set(members: frozenset[_TypedMember]) -> list[SerializedDict]:
    return sorted(
        [_serialize_constraint_member(_unwrap_member(member)) for member in members],
        key=repr,
    )


class _InSetConstraintData(TypedDict):
    variable: SerializedDict
    valid_values: list[SerializedDict]


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
class InSetConstraint(Constraint):
    """Permitted-set membership predicate.

    ``is_satisfied`` returns True iff ``value`` is in the constraint's
    value set, comparing by type-strict equality (so ``True`` and ``1``
    are distinct members, and ``1`` and ``1.0`` are distinct members,
    including inside nested ``tuple`` or ``frozenset`` members).

    Determinism:
        ``convert_to_expression`` emits its leaves in ``repr``-sorted
        order so structurally equivalent constraints produce
        structurally equivalent expressions. Member serialization is
        also ``repr``-sorted to match.

    """

    _valid_values: frozenset[_TypedMember]

    def __init__(
        self,
        constrained_variable: Identifier,
        valid_values: Collection[ConstraintMember],
    ) -> None:
        super().__init__(constrained_variable)
        self._valid_values = _normalize_constraint_member_collection(valid_values)

    def is_satisfied(self, value: Any) -> bool:
        """Return whether ``value`` is in the permitted set.

        Raises:
            TypeError: If ``value`` is not hashable.

        """
        return _wrap_member(value) in self._valid_values

    def convert_to_expression(self) -> Expression:
        if len(self._valid_values) == 0:
            return LiteralExpression(False)
        sorted_values = sorted(self._valid_values, key=repr)
        if len(sorted_values) == 1:
            return self._build_leaf_expression(sorted_values[0])
        return Expression.logical_or(
            *(self._build_leaf_expression(member) for member in sorted_values)
        )

    def _build_leaf_expression(self, wrapped: _TypedMember) -> Expression:
        literal = _lift_member_to_literal_expression(_unwrap_member(wrapped))
        return make_binary_expression(BinaryOperation.EQUAL, self.variable, literal)

    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            "variable": self.variable.serialize_to_dict(),
            "valid_values": _serialize_member_set(self._valid_values),
        }

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "InSetConstraint":
        if not _is_valid_in_set_constraint_data(data):
            raise DeserializationDictStructureError(
                cls, _InSetConstraintData.__annotations__, data
            )
        members = [
            _deserialize_constraint_member("valid_values", value)
            for value in data["valid_values"]
        ]
        return cls(Identifier.deserialize_from_dict(data["variable"]), members)

    def __repr__(self) -> str:
        return (
            f"InSetConstraint({self.variable!r}, "
            f"values={_render_member_set(self._valid_values)})"
        )

    def __str__(self) -> str:
        return f"{self.variable} in {_render_member_set_str(self._valid_values)}"


class _NotInSetConstraintData(TypedDict):
    variable: SerializedDict
    invalid_values: list[SerializedDict]


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
class NotInSetConstraint(Constraint):
    """Forbidden-set membership predicate.

    Symmetric to ``InSetConstraint``: ``is_satisfied`` returns True iff
    ``value`` is NOT in the constraint's value set, comparing by
    type-strict equality.

    Determinism:
        ``convert_to_expression`` emits its leaves in ``repr``-sorted
        order to match the deterministic serialization contract.

    """

    _invalid_values: frozenset[_TypedMember]

    def __init__(
        self,
        constrained_variable: Identifier,
        invalid_values: Collection[ConstraintMember],
    ) -> None:
        super().__init__(constrained_variable)
        self._invalid_values = _normalize_constraint_member_collection(invalid_values)

    def is_satisfied(self, value: Any) -> bool:
        """Return whether ``value`` is NOT in the forbidden set.

        Raises:
            TypeError: If ``value`` is not hashable.

        """
        return _wrap_member(value) not in self._invalid_values

    def convert_to_expression(self) -> Expression:
        if len(self._invalid_values) == 0:
            return LiteralExpression(True)
        sorted_values = sorted(self._invalid_values, key=repr)
        if len(sorted_values) == 1:
            return self._build_leaf_expression(sorted_values[0])
        return Expression.logical_and(
            *(self._build_leaf_expression(member) for member in sorted_values)
        )

    def _build_leaf_expression(self, wrapped: _TypedMember) -> Expression:
        literal = _lift_member_to_literal_expression(_unwrap_member(wrapped))
        return make_binary_expression(BinaryOperation.NOT_EQUAL, self.variable, literal)

    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            "variable": self.variable.serialize_to_dict(),
            "invalid_values": _serialize_member_set(self._invalid_values),
        }

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "NotInSetConstraint":
        if not _is_valid_not_in_set_constraint_data(data):
            raise DeserializationDictStructureError(
                cls, _NotInSetConstraintData.__annotations__, data
            )
        members = [
            _deserialize_constraint_member("invalid_values", value)
            for value in data["invalid_values"]
        ]
        return cls(Identifier.deserialize_from_dict(data["variable"]), members)

    def __repr__(self) -> str:
        return (
            f"NotInSetConstraint({self.variable!r}, "
            f"values={_render_member_set(self._invalid_values)})"
        )

    def __str__(self) -> str:
        return f"{self.variable} not in {_render_member_set_str(self._invalid_values)}"


@singledispatch
def _is_constraint_structurally_equivalent(
    constraint: Constraint, other: object
) -> bool:
    raise NotImplementedError(
        f"is_structurally_equivalent is not registered for {type(constraint).__name__}."
    )


@_is_constraint_structurally_equivalent.register
def _(constraint: EquationConstraint, other: object) -> bool:
    return (
        isinstance(other, EquationConstraint)
        and constraint.variable == other.variable
        and constraint._expression.is_structurally_equivalent(other._expression)
    )


@_is_constraint_structurally_equivalent.register
def _(constraint: InSetConstraint, other: object) -> bool:
    return (
        isinstance(other, InSetConstraint)
        and constraint.variable == other.variable
        and constraint._valid_values == other._valid_values
    )


@_is_constraint_structurally_equivalent.register
def _(constraint: NotInSetConstraint, other: object) -> bool:
    return (
        isinstance(other, NotInSetConstraint)
        and constraint.variable == other.variable
        and constraint._invalid_values == other._invalid_values
    )
