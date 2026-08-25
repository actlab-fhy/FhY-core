"""Constraint-member representation: validation, wrapping, ordering, codec.

A set constraint's ``values`` collection is stored, compared, and
serialized through this module's machinery rather than through plain
Python collection semantics. ``ConstraintMember`` names the four
primitive Python types plus ``Serializable`` leaves and tuple/frozenset
containers of the same; validation rejects everything else.
``_TypedMember`` wraps every stored member so ``int``, ``float``, and
``bool`` never compare equal even when they carry the same value,
including at the leaves of a nested ``tuple``/``frozenset``.
``_order_members_canonically``/``_build_member_ordering_key`` give a
reproducible iteration order independent of the per-process hash seed,
and the member (de)serialization codec (``_VALUES_CODEC``) emits members
in ``repr``-sorted order so two structurally equivalent set constraints
serialize identically.
"""

__all__ = [
    "ConstraintMember",
    "MemberCollection",
]

from collections.abc import Collection, Hashable, Iterator
from typing import (
    Any,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeGuard,
    TypeVar,
    runtime_checkable,
)

from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    FieldCodec,
    Serializable,
    SerializedDict,
    SerializedValue,
    deserialize_registry_wrapped_value,
    is_serialized_dict,
    is_serialized_value,
    make_field_codec,
    serialize_registry_wrapped_value,
)
from fhy_core.symbolic.expression import LiteralExpression, LiteralType
from fhy_core.traits import FrozenMixin
from fhy_core.utils.override import override

from .errors import ConstraintError

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
    those literals while the stored field is normalized to a deduplicated
    tuple of raw values. The collection is only iterated during
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
        ConstraintError: If ``values`` is itself a ``str``/``bytes``/
            ``bytearray`` (which would silently split into its elements), or
            if any member fails validation or is unhashable after validation.

    """
    if isinstance(values, (str, bytes, bytearray)):
        raise ConstraintError(
            f"Constraint members must be given as a collection of members, not "
            f"a bare {type(values).__name__}, which would be split into its "
            "elements. Wrap a single member in a container, e.g. "
            f"{{{values!r}}}."
        )
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


def _wrap_member_collection(
    values: Collection[ConstraintMember],
) -> frozenset[_TypedMember]:
    """Wrap an already-validated raw member collection for type-strict comparison.

    Unlike ``_normalize_constraint_member_collection``, this performs no
    validation or hashability check: it re-derives the type-strict view of
    a collection that has already passed through construction once (the
    post-``__post_init__`` ``values`` field).

    """
    return frozenset(_wrap_member(value) for value in values)


def _order_members_canonically(
    members: Collection[ConstraintMember],
) -> tuple[ConstraintMember, ...]:
    """Order raw members the same way in every process.

    Normalization stores its result in ``frozenset`` iteration order,
    which is hash-table slot order and therefore depends on the
    per-process hash seed for ``str`` members. Ordering by
    ``_build_member_ordering_key`` gives the accessor a sequence a caller
    can iterate for reproducible output, and keeps type-strict members
    apart, since the key is type-tagged and resolves a ``Serializable``
    member through its serialized form rather than its address-based
    ``repr``.

    Args:
        members: Raw members held by a set constraint.

    Returns:
        The same members in canonical order.

    """
    return tuple(sorted(members, key=_build_member_ordering_key))


def _build_member_ordering_key(value: Any) -> str:
    """Return an ordering key constant on type-strict member equality.

    Members compare by ``(type(value), value)``, so the key carries the
    type name: the string ``"1"``, the integer ``1``, the float ``1.0``,
    and ``True`` are four distinct members and get four distinct keys.
    Containers recurse, and a ``frozenset``'s element keys are sorted so
    the container key does not depend on iteration order. A
    ``Serializable`` member keys on its serialized data, which is stable
    across processes where its ``repr`` need not be.

    Args:
        value: Raw (unwrapped) constraint member.

    Returns:
        Type-tagged textual key.

    """
    if isinstance(value, tuple):
        elements = ",".join(_build_member_ordering_key(item) for item in value)
        return f"tuple({elements})"
    elif isinstance(value, frozenset):
        elements = ",".join(sorted(_build_member_ordering_key(item) for item in value))
        return f"frozenset({elements})"
    elif isinstance(value, (bool, int, float, str)):
        return f"{type(value).__name__}:{value!r}"
    qualified_name = f"{type(value).__module__}.{type(value).__qualname__}"
    if isinstance(value, Serializable):
        return f"{qualified_name}:{value.serialize_to_dict()!r}"
    return f"{qualified_name}:{value!r}"


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


def _encode_member_set(members: Collection[ConstraintMember]) -> SerializedValue:
    """Encode a raw member collection as a ``repr``-sorted list of wrapped dicts."""
    return sorted(
        [_serialize_constraint_member(member) for member in members],
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
        # Deferred import: ``Constraint`` lives in the sibling ``core`` module,
        # which imports this module at load time, so importing it back here at
        # module scope would form a cycle. The error message only needs the
        # class object at call time, well after both modules have finished
        # loading.
        from .core import Constraint  # noqa: PLC0415

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


_VALUES_CODEC: FieldCodec = _make_member_set_codec("values")


def _render_member_set(members: frozenset[_TypedMember]) -> str:
    """Render the members in ``repr`` form, sorted, inside set braces.

    ``repr`` is applied to every member including a ``str`` member, so the
    string ``"5"`` renders as ``'5'`` while the integer ``5`` renders as
    ``5``. Membership is type-strict, so the two are different members and
    the textual form has to keep them apart.

    """
    rendered_members = ", ".join(
        sorted(repr(_unwrap_member(member)) for member in members)
    )
    return f"{{{rendered_members}}}"


def _render_member_set_str(members: frozenset[_TypedMember]) -> str:
    """Render the members in human-readable form, sorted, inside set braces.

    A ``str`` member is quoted so it stays distinguishable from the
    numeric member carrying the same digits, matching the type-strict
    membership semantics. Rendering is sorted so the textual form does not
    depend on set iteration order.

    """
    rendered_members = ", ".join(
        sorted(
            repr(value) if isinstance(value, str) else str(value)
            for value in (_unwrap_member(member) for member in members)
        )
    )
    return f"{{{rendered_members}}}"
