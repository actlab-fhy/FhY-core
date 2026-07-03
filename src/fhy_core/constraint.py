"""Named-variable constraints over expressions and value sets.

This module provides the three constraint kinds used by the parameter
infrastructure in ``fhy_core.param.*``:

- ``EquationConstraint``: a Boolean expression that must hold when the
  variable is bound to a candidate value.
- ``InSetConstraint``: the variable must take a value from a permitted
  set.
- ``NotInSetConstraint``: the variable must NOT take a value from a
  forbidden set.

``Constraint`` is a sum-type family base. Each concrete constraint is a
``@register_serializable @dataclass(frozen=True, eq=False)`` leaf whose
wrapped serialization and structural equivalence are derived from its
fields rather than hand-written. The base holds no shared state; every
constraint keeps its own ``variable`` field. Instances are frozen on
construction via ``FrozenMixin``.

Each constraint is callable as a predicate: ``constraint(value)`` is the
same as ``constraint.is_satisfied(value)``. Each can be converted to an
equivalent ``Expression`` over its variable through
``Constraint.convert_to_expression``.

Set-constraint members are stored with type-strict equality: ``int``,
``float``, and ``bool`` are not interchangeable, including inside nested
``tuple`` and ``frozenset`` members. The wrapping that implements this is
private; the public API accepts and returns raw values.

Set-constraint serialization is deterministic: members are emitted in
``repr``-sorted order, and the corresponding ``convert_to_expression``
leaves are emitted in the same order.
"""

from fhy_core.utils.override import override

__all__ = [
    "Constraint",
    "ConstraintError",
    "ConstraintMember",
    "ConstraintOutcome",
    "EquationConstraint",
    "InSetConstraint",
    "NotInSetConstraint",
]

from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeGuard,
    TypeVar,
    cast,
    runtime_checkable,
)

from fhy_core.error import register_error
from fhy_core.logger import get_logger
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    FieldCodec,
    Serializable,
    SerializedDict,
    SerializedValue,
    WrappedFamilySerializable,
    deserialize_registry_wrapped_value,
    is_serialized_dict,
    is_serialized_value,
    make_field_codec,
    register_serializable,
    serialize_registry_wrapped_value,
)
from fhy_core.traits import DerivedEquivalenceMixin, FrozenMixin
from fhy_core.traits.derived_equivalence import compared_as_reference, compared_as_value
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


class ConstraintOutcome(Enum):
    """Tri-state result of checking a value against a constraint.

    A constraint check distinguishes three outcomes:

    - ``SATISFIED``: the value provably satisfies the constraint.
    - ``VIOLATED``: the value provably violates the constraint.
    - ``UNDECIDED``: the checker cannot decide (for example, the
      expression simplifier could not reduce the substituted expression
      to a literal). This is neither a satisfaction nor a violation; it
      signals that the value's admissibility could not be determined.
    """

    SATISFIED = auto()
    VIOLATED = auto()
    UNDECIDED = auto()


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

_MemberT_co = TypeVar("_MemberT_co", covariant=True)


@runtime_checkable
class MemberCollection(Protocol[_MemberT_co]):
    """Read-only collection input for a set constraint's members.

    A structural, immutable-by-contract collection: any ``set``, ``list``,
    ``tuple``, or ``frozenset`` of members satisfies it. Used as the
    constructor-input type for the set constraints so callers can pass any of
    those literals while the stored field is a normalized
    ``frozenset[_TypedMember]``. The collection is only iterated during
    construction; the constraint never mutates or retains the caller's
    collection.
    """

    def __iter__(self) -> Iterator[_MemberT_co]: ...

    def __len__(self) -> int: ...

    def __contains__(self, item: object) -> bool: ...


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


class _TypedMember(FrozenMixin):
    """Internal wrapper providing type-strict equality and hashing.

    Wraps a constraint member so that ``__eq__`` and ``__hash__`` use
    ``(type(value), value)`` instead of the value's own equality. This
    keeps ``True`` and ``1`` distinct, ``1`` and ``1.0`` distinct, and
    ensures nested ``tuple`` and ``frozenset`` containers preserve the
    same type-strict semantics for their elements.

    Callers should always go through ``_unwrap_member`` rather than
    reading ``value`` directly: a wrapped container stores ``_TypedMember``
    elements, not raw members.
    """

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        self._value = value

    @property
    def value(self) -> Any:
        return self._value

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _TypedMember):
            return NotImplemented
        return type(self._value) is type(other._value) and self._value == other._value

    @override
    def __hash__(self) -> int:
        return hash((type(self._value), self._value))

    @override
    def __repr__(self) -> str:
        return repr(self._value)

    @override
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
    """Lift a constraint member to a ``LiteralExpression``.

    Raises:
        ConstraintError: If the member is not a ``LiteralType``, if it is
            a ``str``, or if ``LiteralExpression`` refuses the value.

    """
    if not isinstance(value, LiteralType):
        raise ConstraintError(
            f"Conversion of type {type(value).__name__} to an expression is "
            "not supported."
        )
    # Strings are rejected: constraint membership is type-strict, but
    # LiteralExpression equivalence canonicalizes "1" to int 1 and "1.5" to a
    # Decimal, so the converted expression would not match the membership
    # semantics.
    if isinstance(value, str):
        raise ConstraintError(
            f"Member {value!r} is a string; constraint membership uses "
            "type-strict equality, but the converted expression's literal "
            "equivalence would canonicalize the string against int and "
            "float members. Convert the member to the canonical numeric "
            "type before constructing the constraint."
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


class _MemberSetData(TypedDict):
    members: list[SerializedDict]


def _encode_member_set(members: frozenset[_TypedMember]) -> SerializedValue:
    """Encode a wrapped-member set as a ``repr``-sorted list of wrapped dicts."""
    return sorted(
        [_serialize_constraint_member(_unwrap_member(member)) for member in members],
        key=repr,
    )


def _decode_member_set(field_name: str, data: Any) -> list[ConstraintMember]:
    """Decode a list of wrapped-member dicts into raw constraint members.

    Raises:
        DeserializationDictStructureError: If ``data`` is not a list of
            wrapped member dicts.
        DeserializationValueError: If a member dict is individually malformed
            or fails validation.

    """
    if not isinstance(data, list) or not all(
        is_serialized_dict(value) for value in data
    ):
        raise DeserializationDictStructureError(
            Constraint,
            _MemberSetData.__annotations__,
            {field_name: data} if is_serialized_value(data) else {},
        )
    return [_deserialize_constraint_member(field_name, value) for value in data]


def _make_member_set_codec(field_name: str) -> FieldCodec:
    """Build a per-field codec for a set constraint's wrapped-member set.

    Encoding emits a ``repr``-sorted list of registry-wrapped dicts; decoding
    validates and unwraps the list to raw members for the constructor.
    """

    def _decode(data: Any) -> list[ConstraintMember]:
        return _decode_member_set(field_name, data)

    return make_field_codec(_encode_member_set, _decode)


_VALID_VALUES_CODEC: FieldCodec = _make_member_set_codec("valid_values")
_INVALID_VALUES_CODEC: FieldCodec = _make_member_set_codec("invalid_values")


class Constraint(
    WrappedFamilySerializable,
    FrozenMixin,
    DerivedEquivalenceMixin,
    ABC,
):
    """A named-variable predicate.

    Subclasses model the three concrete constraint kinds in this module
    (``EquationConstraint``, ``InSetConstraint``, ``NotInSetConstraint``).
    Each concrete constraint is a ``@register_serializable @dataclass(
    frozen=True, eq=False)`` leaf of this family; serialization and
    structural equivalence are derived from its fields. The base holds no
    shared state; each leaf provides its own ``variable`` field.
    Instances are frozen at the end of construction; subsequent attribute
    mutation raises ``FrozenMutationError``. Instances are callable;
    ``constraint(value)`` is an alias for ``constraint.is_satisfied(value)``.

    Subclassing contract:
        - Declare a ``@dataclass(frozen=True, eq=False)`` with a
          ``variable: Identifier`` field tagged
          ``compared_as_reference()`` plus the kind's own fields.
        - Override ``evaluate`` to define the tri-state predicate; the
          concrete ``is_satisfied`` derives from it.
        - Override ``convert_to_expression`` to produce an equivalent
          ``Expression`` over the variable.
        - Override ``__repr__`` and ``__str__`` so the textual form
          identifies the kind and the variable.

    """

    if TYPE_CHECKING:
        # Every concrete constraint declares a ``variable`` dataclass field;
        # this declaration (no stored state on the base) lets callers that
        # iterate over ``Constraint`` values read ``.variable`` with a known
        # type. Guarded so it does not become a class attribute that would
        # shadow the leaves' dataclass fields at runtime.
        variable: Identifier

    def __call__(self, value: Any) -> bool:
        """Return whether the value satisfies the constraint."""
        return self.is_satisfied(value)

    @abstractmethod
    def evaluate(self, value: Any) -> ConstraintOutcome:
        """Return the tri-state outcome of checking the value.

        Args:
            value: Candidate value to check.

        Returns:
            ``ConstraintOutcome.SATISFIED`` if the value provably
            satisfies the constraint, ``ConstraintOutcome.VIOLATED`` if
            it provably violates it, and ``ConstraintOutcome.UNDECIDED``
            if the checker cannot decide. Subclasses document their
            predicate semantics and which outcomes they can produce.

        """

    def is_satisfied(self, value: Any) -> bool:
        """Return whether the value satisfies the constraint.

        A value is treated as satisfying the constraint only when
        ``evaluate`` decides ``SATISFIED``; both ``VIOLATED`` and the
        indeterminate ``UNDECIDED`` outcome map to ``False``, so an
        undecided check conservatively rejects the value.

        Args:
            value: Candidate value to check.

        Returns:
            True if the value satisfies the constraint; False otherwise.

        """
        return self.evaluate(value) is ConstraintOutcome.SATISFIED

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

    @abstractmethod
    @override
    def __repr__(self) -> str: ...

    @abstractmethod
    @override
    def __str__(self) -> str: ...


@register_serializable(type_id="equation_constraint")
@dataclass(frozen=True, eq=False)
class EquationConstraint(Constraint):
    """Boolean-expression predicate over the variable.

    The constraint wraps a Boolean ``Expression`` whose only free
    identifier is meant to be ``self.variable``. ``evaluate``
    substitutes the candidate value for that identifier, simplifies the
    resulting expression, and reports ``SATISFIED`` only when the
    simplifier reduces it to a ``LiteralExpression`` with a ``bool``
    value of ``True``.

    Outcomes:
        - ``SATISFIED``: the substituted expression reduces to the
          ``bool`` literal ``True``.
        - ``VIOLATED``: the substituted expression reduces to the
          ``bool`` literal ``False``, or to a literal whose value is not
          a ``bool`` (for example ``LiteralExpression(1)``). A non-bool
          literal is a decided "no", not an indeterminate case: no
          warning is emitted.
        - ``UNDECIDED``: the simplifier cannot reduce the substituted
          expression to a ``LiteralExpression`` at all (for example
          because the expression references additional free identifiers,
          or because the simplifier just cannot decide). ``evaluate``
          logs a ``WARNING`` through the module logger in this case.

    ``is_satisfied`` derives from ``evaluate`` and treats both
    ``VIOLATED`` and ``UNDECIDED`` as ``False``, so an undecided check
    conservatively rejects the value.

    """

    variable: Identifier = field(metadata=compared_as_reference())
    expression: Expression

    @override
    def evaluate(self, value: Expression | LiteralType) -> ConstraintOutcome:
        if isinstance(value, (str, float, int, bool)):
            value = LiteralExpression(value)
        result = simplify_expression(self.expression, {self.variable: value})
        if isinstance(result, LiteralExpression):
            if isinstance(result.value, bool) and result.value:
                return ConstraintOutcome.SATISFIED
            return ConstraintOutcome.VIOLATED
        _LOGGER.warning(
            "%s.evaluate: substituted expression %r did not reduce to a "
            "literal; reporting UNDECIDED",
            type(self).__name__,
            result,
        )
        return ConstraintOutcome.UNDECIDED

    @override
    def convert_to_expression(self) -> Expression:
        return self.expression

    @override
    def __repr__(self) -> str:
        return f"EquationConstraint({self.variable!r}, expression={self.expression!r})"

    @override
    def __str__(self) -> str:
        return pformat_expression(self.expression)


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


@register_serializable(type_id="in_set_constraint")
@dataclass(frozen=True, eq=False)
class InSetConstraint(Constraint):
    """Permitted-set membership predicate.

    ``evaluate`` reports ``SATISFIED`` iff ``value`` is in the
    constraint's value set and ``VIOLATED`` otherwise, comparing by
    type-strict equality (so ``True`` and ``1`` are distinct members,
    and ``1`` and ``1.0`` are distinct members, including inside nested
    ``tuple`` or ``frozenset`` members). Membership is always decidable,
    so this constraint never reports ``UNDECIDED``.

    Determinism:
        ``convert_to_expression`` emits its leaves in ``repr``-sorted
        order so structurally equivalent constraints produce
        structurally equivalent expressions. Member serialization is
        also ``repr``-sorted to match.

    """

    variable: Identifier = field(metadata=compared_as_reference())
    # Declared as the constructor-input type. ``__post_init__`` normalizes this
    # in place to a ``frozenset[_TypedMember]``; read the members through the
    # ``members`` property (raw values) or ``_members`` (internal wrappers)
    # rather than this field directly.
    valid_values: MemberCollection[ConstraintMember] = field(
        metadata={
            **compared_as_value(),
            "serialize_codec": _VALID_VALUES_CODEC,
        },
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "valid_values",
            _normalize_constraint_member_collection(self.valid_values),
        )

    @property
    def members(self) -> tuple[ConstraintMember, ...]:
        """Return the permitted members as raw values.

        The ``valid_values`` field stores internal type-strict wrappers; this
        accessor returns the unwrapped members in no particular order.
        """
        return tuple(_unwrap_member(member) for member in self._members)

    @property
    def _members(self) -> frozenset[_TypedMember]:
        """Return the normalized, type-strict member set stored after init."""
        return cast(frozenset[_TypedMember], self.valid_values)

    @classmethod
    @override
    def construct_from_fields(cls, fields: dict[str, Any]) -> "InSetConstraint":
        return cls(fields["variable"], fields["valid_values"])

    @override
    def evaluate(self, value: Any) -> ConstraintOutcome:
        """Return whether ``value`` is in the permitted set.

        Raises:
            TypeError: If ``value`` is not hashable.

        """
        if _wrap_member(value) in self._members:
            return ConstraintOutcome.SATISFIED
        return ConstraintOutcome.VIOLATED

    @override
    def convert_to_expression(self) -> Expression:
        members = self._members
        if len(members) == 0:
            return LiteralExpression(False)
        sorted_values = sorted(members, key=repr)
        if len(sorted_values) == 1:
            return self._build_leaf_expression(sorted_values[0])
        return Expression.logical_or(
            *(self._build_leaf_expression(member) for member in sorted_values)
        )

    def _build_leaf_expression(self, wrapped: _TypedMember) -> Expression:
        literal = _lift_member_to_literal_expression(_unwrap_member(wrapped))
        return make_binary_expression(BinaryOperation.EQUAL, self.variable, literal)

    @override
    def __repr__(self) -> str:
        return (
            f"InSetConstraint({self.variable!r}, "
            f"values={_render_member_set(self._members)})"
        )

    @override
    def __str__(self) -> str:
        return f"{self.variable} in {_render_member_set_str(self._members)}"


@register_serializable(type_id="not_in_set_constraint")
@dataclass(frozen=True, eq=False)
class NotInSetConstraint(Constraint):
    """Forbidden-set membership predicate.

    Symmetric to ``InSetConstraint``: ``evaluate`` reports ``SATISFIED``
    iff ``value`` is NOT in the constraint's value set and ``VIOLATED``
    otherwise, comparing by type-strict equality. Membership is always
    decidable, so this constraint never reports ``UNDECIDED``.

    Determinism:
        ``convert_to_expression`` emits its leaves in ``repr``-sorted
        order to match the deterministic serialization contract.

    """

    variable: Identifier = field(metadata=compared_as_reference())
    # Declared as the constructor-input type. ``__post_init__`` normalizes this
    # in place to a ``frozenset[_TypedMember]``; read the members through the
    # ``members`` property (raw values) or ``_members`` (internal wrappers)
    # rather than this field directly.
    invalid_values: MemberCollection[ConstraintMember] = field(
        metadata={
            **compared_as_value(),
            "serialize_codec": _INVALID_VALUES_CODEC,
        },
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "invalid_values",
            _normalize_constraint_member_collection(self.invalid_values),
        )

    @property
    def members(self) -> tuple[ConstraintMember, ...]:
        """Return the forbidden members as raw values.

        The ``invalid_values`` field stores internal type-strict wrappers; this
        accessor returns the unwrapped members in no particular order.
        """
        return tuple(_unwrap_member(member) for member in self._members)

    @property
    def _members(self) -> frozenset[_TypedMember]:
        """Return the normalized, type-strict member set stored after init."""
        return cast(frozenset[_TypedMember], self.invalid_values)

    @classmethod
    @override
    def construct_from_fields(cls, fields: dict[str, Any]) -> "NotInSetConstraint":
        return cls(fields["variable"], fields["invalid_values"])

    @override
    def evaluate(self, value: Any) -> ConstraintOutcome:
        """Return whether ``value`` is NOT in the forbidden set.

        Raises:
            TypeError: If ``value`` is not hashable.

        """
        if _wrap_member(value) not in self._members:
            return ConstraintOutcome.SATISFIED
        return ConstraintOutcome.VIOLATED

    @override
    def convert_to_expression(self) -> Expression:
        members = self._members
        if len(members) == 0:
            return LiteralExpression(True)
        sorted_values = sorted(members, key=repr)
        if len(sorted_values) == 1:
            return self._build_leaf_expression(sorted_values[0])
        return Expression.logical_and(
            *(self._build_leaf_expression(member) for member in sorted_values)
        )

    def _build_leaf_expression(self, wrapped: _TypedMember) -> Expression:
        literal = _lift_member_to_literal_expression(_unwrap_member(wrapped))
        return make_binary_expression(BinaryOperation.NOT_EQUAL, self.variable, literal)

    @override
    def __repr__(self) -> str:
        return (
            f"NotInSetConstraint({self.variable!r}, "
            f"values={_render_member_set(self._members)})"
        )

    @override
    def __str__(self) -> str:
        return f"{self.variable} not in {_render_member_set_str(self._members)}"
