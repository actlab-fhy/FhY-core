"""Named-variable constraints over expressions and value sets.

This module provides the three constraint kinds used by the parameter
infrastructure in ``fhy_core.symbolic.param.*``:

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

Every constraint additionally supports a *bindings* evaluation API --
``evaluate_with_bindings`` / ``is_satisfied_with_bindings`` -- that checks
the constraint against a ``Mapping[Identifier, value]`` instead of a single
positional value. This makes multi-variable (dependent) constraints
usable: an ``EquationConstraint`` whose expression mentions identifiers
beyond ``self.variable`` is decided once every identifier it references
is bound. ``ConstraintSystem`` is the companion set-level value object: a
canonically ordered conjunction of constraints, possibly spanning several
variables, with joint-satisfiability checking backed by
``fhy_core.symbolic.solver.check_expression_satisfiability``. It is not a
``Constraint`` subclass -- joint satisfiability is a property of a
collection, not of any single predicate.
"""

from fhy_core.utils.override import override

__all__ = [
    "Constraint",
    "ConstraintBindings",
    "ConstraintError",
    "ConstraintMember",
    "ConstraintOutcome",
    "ConstraintSystem",
    "EquationConstraint",
    "InSetConstraint",
    "MissingSymbolTypeError",
    "NotInSetConstraint",
    "create_constraint_system",
]

from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable, Iterator, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeGuard,
    TypeVar,
    runtime_checkable,
)

from fhy_core.error import register_error
from fhy_core.identifier import Identifier
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
from fhy_core.term import (
    DerivedEquivalenceMixin,
    compared_as_reference,
    compared_as_value,
)
from fhy_core.traits import FrozenMixin
from fhy_core.utils import format_comma_separated_list

from .expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    LiteralType,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
    make_binary_expression,
    pformat_expression,
)
from .solver import (
    check_expression_satisfiability,
    simplify_expression,
    validate_timeout_milliseconds,
)
from .symbol_type import SymbolType

_LOGGER = get_logger(__name__)

ConstraintBindings: TypeAlias = Mapping[Identifier, "Expression | LiteralType"]
"""Assignment of candidate values (literals or expressions) to identifiers."""


def _coerce_bindings_to_environment(
    bindings: ConstraintBindings,
) -> dict[Identifier, Expression]:
    """Coerce every raw ``LiteralType`` binding value to a ``LiteralExpression``.

    ``Expression`` values (including a non-literal, symbolic ``Expression``)
    pass through unchanged.
    """
    return {
        identifier: (
            LiteralExpression(value)
            if isinstance(value, (str, float, int, bool))
            else value
        )
        for identifier, value in bindings.items()
    }


@register_error
class ConstraintError(ValueError):
    """Domain error for constraint construction, validation, and conversion."""


@register_error
class MissingSymbolTypeError(ValueError):
    """Raised when ``symbol_types`` lacks an entry for a free identifier being lowered.

    ``ConstraintSystem.check_satisfiability`` and
    ``check_satisfiability_with_bindings`` require a Z3 sort for every free
    identifier of the expression they lower. A missing entry is a caller
    precondition violation, not a dictionary lookup miss, so this is a
    ``ValueError`` rather than a ``KeyError``: the missing-identifier
    message must render cleanly in a traceback, and a bare ``except
    KeyError`` elsewhere in a caller's code must not silently swallow it.
    """


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
    post-``__post_init__`` ``valid_values``/``invalid_values`` field).

    """
    return frozenset(_wrap_member(value) for value in values)


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

    def get_free_identifiers(self) -> frozenset[Identifier]:
        """Return every identifier this constraint constrains or references.

        The base implementation returns ``frozenset((self.variable,))``.
        ``EquationConstraint`` overrides it to also include the free
        identifiers of its expression.

        Returns:
            Non-empty frozen set of identifiers; always contains
            ``self.variable``.

        """
        return frozenset((self.variable,))

    def evaluate_with_bindings(self, bindings: ConstraintBindings) -> ConstraintOutcome:
        """Return the tri-state outcome of checking the constraint under bindings.

        The base implementation is sound for any single-variable leaf: it
        looks up ``self.variable`` in ``bindings``, unwraps a
        ``LiteralExpression`` binding to its raw value, and delegates to
        ``evaluate``. A missing binding for ``self.variable`` or a
        non-literal ``Expression`` binding yields ``UNDECIDED``.
        Identifiers in ``bindings`` that the constraint does not reference
        are ignored. ``EquationConstraint`` overrides this to substitute
        every bound identifier simultaneously.

        A non-literal ``Expression`` binding is a structural limitation of
        the leaves that use this implementation, not a per-call outcome:
        ``InSetConstraint`` and ``NotInSetConstraint`` decide membership
        against a concrete value, so no symbolic expression is decidable
        against them and supplying a different one will not help. That
        makes ``ConstraintBindings`` wider than these leaves consume --
        only ``EquationConstraint``, which substitutes into an expression,
        uses the full range. Both this case and a missing binding are
        logged at ``DEBUG`` through the module logger, naming the
        identifier responsible, since a set constraint's ``evaluate``
        never reports ``UNDECIDED`` and so gives no other clue which of
        the two occurred.

        Args:
            bindings: Mapping from identifiers to candidate values. Raw
                ``LiteralType`` values and ``Expression`` values are both
                accepted; raw values behave identically to their
                ``LiteralExpression`` wrapping.

        Returns:
            ``SATISFIED``/``VIOLATED`` when decidable under the given
            (possibly partial) bindings; ``UNDECIDED`` otherwise.

        Raises:
            TypeError: Propagated from ``evaluate`` for leaves that reject
                the bound value's type (e.g. an unhashable value against a
                set constraint).

        """
        snapshot = dict(bindings)
        if self.variable not in snapshot:
            _LOGGER.debug(
                "%s.evaluate_with_bindings: no binding for variable %r; the "
                "bindings supplied %s; reporting UNDECIDED",
                type(self).__name__,
                self.variable,
                format_comma_separated_list(tuple(snapshot)) or "no identifiers",
            )
            return ConstraintOutcome.UNDECIDED
        value = snapshot[self.variable]
        if isinstance(value, Expression):
            if not isinstance(value, LiteralExpression):
                _LOGGER.debug(
                    "%s.evaluate_with_bindings: the binding for %r is the "
                    "non-literal expression %r; this leaf decides against a "
                    "concrete value and cannot consume a symbolic one; "
                    "reporting UNDECIDED",
                    type(self).__name__,
                    self.variable,
                    value,
                )
                return ConstraintOutcome.UNDECIDED
            value = value.value
        return self.evaluate(value)

    def is_satisfied_with_bindings(self, bindings: ConstraintBindings) -> bool:
        """Return whether the bindings provably satisfy the constraint.

        Derived from ``evaluate_with_bindings``; both ``VIOLATED`` and
        ``UNDECIDED`` map to ``False`` (conservative rejection), matching
        ``is_satisfied``.

        """
        return self.evaluate_with_bindings(bindings) is ConstraintOutcome.SATISFIED

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
    def get_free_identifiers(self) -> frozenset[Identifier]:
        """Return the expression's free identifiers united with ``variable``."""
        return self.expression.get_free_identifiers() | {self.variable}

    @override
    def evaluate_with_bindings(self, bindings: ConstraintBindings) -> ConstraintOutcome:
        """Substitute every bound identifier, simplify, and classify.

        Coerces each raw ``LiteralType`` binding value to a
        ``LiteralExpression`` (as ``evaluate`` does), substitutes the full
        multi-key environment through ``simplify_expression``, and reports
        ``SATISFIED`` for the ``bool`` literal ``True``, ``VIOLATED`` for
        any other literal, and ``UNDECIDED`` when no literal results. The
        designated ``variable`` has no special role here; it is bound like
        any other free identifier. Logging on ``UNDECIDED``: DEBUG when
        the residual (substituted and simplified) expression still has
        free identifiers (expected partial evaluation, including the
        case where a symbolic binding introduces a new free identifier),
        WARNING when the residual has none -- every free identifier was
        bound yet the simplifier still failed to reduce it to a literal
        (matches ``evaluate``'s anomaly contract).

        """
        environment = _coerce_bindings_to_environment(bindings)
        result = simplify_expression(self.expression, environment)
        if isinstance(result, LiteralExpression):
            if isinstance(result.value, bool) and result.value:
                return ConstraintOutcome.SATISFIED
            return ConstraintOutcome.VIOLATED
        if result.get_free_identifiers():
            _LOGGER.debug(
                "%s.evaluate_with_bindings: substituted expression %r did not "
                "reduce to a literal; free identifiers remain unbound; "
                "reporting UNDECIDED",
                type(self).__name__,
                result,
            )
        else:
            _LOGGER.warning(
                "%s.evaluate_with_bindings: substituted expression %r did not "
                "reduce to a literal though every free identifier was bound; "
                "reporting UNDECIDED",
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


@register_serializable(type_id="in_set_constraint")
@dataclass(frozen=True, eq=False)
class InSetConstraint(Constraint):
    """Permitted-set membership predicate.

    ``evaluate`` reports ``SATISFIED`` iff ``value`` is in the
    constraint's value set and ``VIOLATED`` otherwise, comparing by
    type-strict equality (so ``True`` and ``1`` are distinct members,
    and ``1`` and ``1.0`` are distinct members, including inside nested
    ``tuple`` or ``frozenset`` members). A membership check against a
    concrete value is always decidable, so ``evaluate`` never reports
    ``UNDECIDED``.

    Determinism:
        ``convert_to_expression`` emits its leaves in ``repr``-sorted
        order so structurally equivalent constraints produce
        structurally equivalent expressions. Member serialization is
        also ``repr``-sorted to match.

    """

    variable: Identifier = field(metadata=compared_as_reference())
    # Declared as the constructor-input type. ``__post_init__`` normalizes this
    # in place to a deduplicated tuple of raw, unwrapped values (the same
    # content the ``members`` property exposes) so no public attribute ever
    # yields the internal ``_TypedMember`` wrapper. Comparison uses
    # ``compared_as_value(key=_wrap_member_collection)`` rather than the
    # default ``==`` so structural/alpha equivalence stays type-strict and
    # order-independent despite the field itself being a plain tuple.
    valid_values: MemberCollection[ConstraintMember] = field(
        metadata={
            **compared_as_value(key=_wrap_member_collection),
            "serialize_codec": _VALID_VALUES_CODEC,
        },
    )

    def __post_init__(self) -> None:
        wrapped = _normalize_constraint_member_collection(self.valid_values)
        object.__setattr__(
            self,
            "valid_values",
            tuple(_unwrap_member(member) for member in wrapped),
        )
        # Seed the ``_members`` cache with the set this normalization pass has
        # already built, so no reader has to derive it a second time.
        object.__setattr__(self, "_members", wrapped)

    @property
    def members(self) -> tuple[ConstraintMember, ...]:
        """Return the permitted members as raw values, in no particular order."""
        return tuple(self.valid_values)

    @cached_property
    def _members(self) -> frozenset[_TypedMember]:
        """Return the type-strict member set membership is decided against.

        Held as a stored set rather than re-derived per read: ``evaluate``
        is then a constant-time frozenset lookup, and ``__repr__`` -- which
        feeds the ``ConstraintSystem`` ordering key -- costs no wrapper
        allocations. ``__post_init__`` seeds it with the set built during
        normalization; this body re-derives it from ``valid_values`` for an
        instance that reaches a reader unseeded, so the public field stays
        the single source of truth.
        """
        return _wrap_member_collection(self.valid_values)

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
    otherwise, comparing by type-strict equality. A membership check
    against a concrete value is always decidable, so ``evaluate`` never
    reports ``UNDECIDED``.

    Determinism:
        ``convert_to_expression`` emits its leaves in ``repr``-sorted
        order to match the deterministic serialization contract.

    """

    variable: Identifier = field(metadata=compared_as_reference())
    # Declared as the constructor-input type. ``__post_init__`` normalizes this
    # in place to a deduplicated tuple of raw, unwrapped values (the same
    # content the ``members`` property exposes) so no public attribute ever
    # yields the internal ``_TypedMember`` wrapper. Comparison uses
    # ``compared_as_value(key=_wrap_member_collection)`` rather than the
    # default ``==`` so structural/alpha equivalence stays type-strict and
    # order-independent despite the field itself being a plain tuple.
    invalid_values: MemberCollection[ConstraintMember] = field(
        metadata={
            **compared_as_value(key=_wrap_member_collection),
            "serialize_codec": _INVALID_VALUES_CODEC,
        },
    )

    def __post_init__(self) -> None:
        wrapped = _normalize_constraint_member_collection(self.invalid_values)
        object.__setattr__(
            self,
            "invalid_values",
            tuple(_unwrap_member(member) for member in wrapped),
        )
        # Seed the ``_members`` cache with the set this normalization pass has
        # already built, so no reader has to derive it a second time.
        object.__setattr__(self, "_members", wrapped)

    @property
    def members(self) -> tuple[ConstraintMember, ...]:
        """Return the forbidden members as raw values, in no particular order."""
        return tuple(self.invalid_values)

    @cached_property
    def _members(self) -> frozenset[_TypedMember]:
        """Return the type-strict member set membership is decided against.

        Held as a stored set rather than re-derived per read: ``evaluate``
        is then a constant-time frozenset lookup, and ``__repr__`` -- which
        feeds the ``ConstraintSystem`` ordering key -- costs no wrapper
        allocations. ``__post_init__`` seeds it with the set built during
        normalization; this body re-derives it from ``invalid_values`` for
        an instance that reaches a reader unseeded, so the public field
        stays the single source of truth.
        """
        return _wrap_member_collection(self.invalid_values)

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


def _validate_symbol_types_cover_free_identifiers(
    expression: Expression, symbol_types: Mapping[Identifier, SymbolType]
) -> None:
    """Raise if ``symbol_types`` lacks an entry for a free identifier of ``expression``.

    Raises:
        MissingSymbolTypeError: If one or more free identifiers of
            ``expression`` have no corresponding ``symbol_types`` entry.

    """
    missing = expression.get_free_identifiers() - set(symbol_types)
    if not missing:
        return
    missing_names = ", ".join(sorted(identifier.name_hint for identifier in missing))
    raise MissingSymbolTypeError(
        f"symbol_types is missing an entry for free identifier(s): {missing_names}."
    )


class _LoweredSort(Enum):
    """Z3 sort an expression node lowers to, as far as it can be determined.

    ``UNDETERMINED`` covers the nodes whose lowered sort this module
    cannot read off the tree: an identifier with no ``symbol_types``
    entry, a call (which the Z3 bridge refuses outright), an unrecognized
    ``Expression`` subclass, and a piecewise whose branches disagree.
    """

    BOOLEAN = auto()
    NUMERIC = auto()
    UNDETERMINED = auto()


_NUMERIC_BINARY_OPERATIONS: frozenset[BinaryOperation] = frozenset(
    {
        BinaryOperation.ADD,
        BinaryOperation.SUBTRACT,
        BinaryOperation.MULTIPLY,
        BinaryOperation.DIVIDE,
        BinaryOperation.FLOOR_DIVIDE,
        BinaryOperation.MODULO,
        BinaryOperation.POWER,
    }
)
"""Binary operations the Z3 bridge lowers to arithmetic on a numeric sort."""

_COMPARISON_BINARY_OPERATIONS: frozenset[BinaryOperation] = frozenset(
    {
        BinaryOperation.EQUAL,
        BinaryOperation.NOT_EQUAL,
        BinaryOperation.LESS,
        BinaryOperation.LESS_EQUAL,
        BinaryOperation.GREATER,
        BinaryOperation.GREATER_EQUAL,
    }
)
"""Binary operations the Z3 bridge lowers to a comparison of two operands."""

_LOGICAL_BINARY_OPERATIONS: frozenset[BinaryOperation] = frozenset(
    {BinaryOperation.LOGICAL_AND, BinaryOperation.LOGICAL_OR}
)
"""Binary operations the Z3 bridge lowers to ``z3.And``/``z3.Or``."""


# One early return per node kind reads clearest here; the alternative is a
# lookup table that would have to be threaded through `symbol_types` anyway.
def _classify_lowered_sort(  # noqa: PLR0911
    expression: Expression, symbol_types: Mapping[Identifier, SymbolType]
) -> _LoweredSort:
    """Return the Z3 sort ``expression`` lowers to, per ``passes.z3``.

    Mirrors ``ExpressionToZ3Converter``: a ``bool``-valued literal becomes
    a ``BoolVal`` and every other literal an ``IntVal``/``RealVal``; an
    identifier takes the sort named by its ``symbol_types`` entry; a
    comparison, a logical operation, and a logical negation are Boolean;
    arithmetic is numeric; a piecewise takes the sort its branches agree
    on.

    Args:
        expression: Node whose lowered sort is wanted.
        symbol_types: Z3 sort for each identifier of the enclosing
            expression.

    Returns:
        The node's ``_LoweredSort``.

    """
    if isinstance(expression, LiteralExpression):
        if type(expression.value) is bool:
            return _LoweredSort.BOOLEAN
        return _LoweredSort.NUMERIC
    elif isinstance(expression, IdentifierExpression):
        symbol_type = symbol_types.get(expression.identifier)
        if symbol_type is SymbolType.BOOL:
            return _LoweredSort.BOOLEAN
        elif symbol_type is None:
            return _LoweredSort.UNDETERMINED
        return _LoweredSort.NUMERIC
    elif isinstance(expression, BinaryExpression):
        if expression.operation in _NUMERIC_BINARY_OPERATIONS:
            return _LoweredSort.NUMERIC
        elif expression.operation in (
            _COMPARISON_BINARY_OPERATIONS | _LOGICAL_BINARY_OPERATIONS
        ):
            return _LoweredSort.BOOLEAN
        return _LoweredSort.UNDETERMINED
    elif isinstance(expression, UnaryExpression):
        if expression.operation is UnaryOperation.LOGICAL_NOT:
            return _LoweredSort.BOOLEAN
        return _LoweredSort.NUMERIC
    elif isinstance(expression, PiecewiseExpression):
        return _join_lowered_sorts(
            _classify_lowered_sort(branch, symbol_types)
            for branch in (*expression.values, expression.otherwise)
        )
    elif isinstance(expression, CallExpression):
        return _LoweredSort.UNDETERMINED
    return _LoweredSort.UNDETERMINED


def _join_lowered_sorts(sorts: Iterator[_LoweredSort]) -> _LoweredSort:
    """Return the sort every input agrees on, or ``UNDETERMINED`` if they differ."""
    distinct = set(sorts)
    if len(distinct) == 1:
        return distinct.pop()
    return _LoweredSort.UNDETERMINED


def _does_node_coerce_a_bool_operand(
    expression: Expression, symbol_types: Mapping[Identifier, SymbolType]
) -> bool:
    """Return whether this one node makes Z3 coerce a Boolean operand to an integer.

    The Z3 Python bindings rewrite a Boolean operand of a numeric context
    into ``If(b, 1, 0)``: ``True`` becomes ``1`` and ``False`` becomes
    ``0``. That collapses the type-strict distinction this module's
    ``evaluate``/``evaluate_with_bindings`` preserve, so a decided outcome
    read back through such a lowering can be provably wrong.

    Three contexts coerce:

    - An arithmetic operation with any Boolean operand.
    - A comparison whose two operands mix a Boolean and a numeric sort. A
      comparison of two Booleans lowers faithfully and is not flagged.
    - A piecewise whose branch values mix a Boolean and a numeric sort,
      since ``z3.If`` forces its two arms to a single sort.

    ``z3.And``/``z3.Or``/``z3.Not`` and unary arithmetic negation do not
    coerce: they raise on an operand of the wrong sort rather than
    silently reinterpreting it, so a Boolean there is either correct or
    already an error.

    Args:
        expression: Node to screen. Children are not visited.
        symbol_types: Z3 sort for each identifier of the enclosing
            expression.

    Returns:
        True if this node's own lowering coerces a Boolean operand.

    """
    if isinstance(expression, BinaryExpression):
        operand_sorts = {
            _classify_lowered_sort(operand, symbol_types)
            for operand in expression.get_operands()
        }
        if expression.operation in _NUMERIC_BINARY_OPERATIONS:
            return _LoweredSort.BOOLEAN in operand_sorts
        elif expression.operation in _COMPARISON_BINARY_OPERATIONS:
            return operand_sorts >= {_LoweredSort.BOOLEAN, _LoweredSort.NUMERIC}
        return False
    elif isinstance(expression, PiecewiseExpression):
        branch_sorts = {
            _classify_lowered_sort(branch, symbol_types)
            for branch in (*expression.values, expression.otherwise)
        }
        return branch_sorts >= {_LoweredSort.BOOLEAN, _LoweredSort.NUMERIC}
    return False


def _find_bool_sort_hazard(
    expression: Expression, symbol_types: Mapping[Identifier, SymbolType]
) -> Expression | None:
    """Return the first node of ``expression`` that coerces a Boolean operand.

    Screens the tree that is actually handed to the solver, so the check
    covers every way a ``BoolVal`` can reach a numeric context: a ``bool``
    set member, a ``bool`` literal written into an equation, and a
    ``bool`` binding value substituted in before the check. The screen is
    per-site: a Boolean literal consumed by a logical operation leaves the
    rest of the tree decidable.

    Args:
        expression: Lowered conjunction, or the residual left after
            substituting bindings into it.
        symbol_types: Z3 sort for each free identifier of ``expression``.

    Returns:
        The offending node, or ``None`` when every node lowers faithfully.
        The node is returned rather than a flag so the caller can name the
        site it refused to lower.

    """
    if _does_node_coerce_a_bool_operand(expression, symbol_types):
        return expression
    for child in expression.get_visit_children():
        hazard = _find_bool_sort_hazard(child, symbol_types)
        if hazard is not None:
            return hazard
    return None


def _render_identifier_sorts(
    expression: Expression, symbol_types: Mapping[Identifier, SymbolType]
) -> str:
    """Return each free identifier of ``expression`` paired with its declared sort.

    Args:
        expression: Node whose free identifiers are described.
        symbol_types: Z3 sort for each free identifier.

    Returns:
        A comma-separated ``identifier: SORT`` listing, ordered by
        identifier id; ``"none"`` when the node has no free identifier,
        and ``"unknown"`` in place of a sort ``symbol_types`` does not
        carry.

    """
    identifiers = sorted(
        expression.get_free_identifiers(), key=lambda identifier: identifier.id
    )
    if not identifiers:
        return "none"
    rendered: list[str] = []
    for identifier in identifiers:
        symbol_type = symbol_types.get(identifier)
        sort_name = "unknown" if symbol_type is None else symbol_type.name
        rendered.append(f"{identifier!r}: {sort_name}")
    return format_comma_separated_list(rendered)


def _decide_satisfiability(
    expression: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
    *,
    context: str,
    timeout_milliseconds: int | None = None,
) -> ConstraintOutcome:
    """Classify satisfiability of ``expression`` via the solver seam.

    Validates the caller's symbol types first, then screens the expression
    for the Z3 Boolean-coercion hazard, and only then consults the solver.
    The precondition therefore raises whether or not the expression is
    hazardous, and a hazardous expression reports ``UNDECIDED`` instead of
    a decided outcome the lowering cannot support. A hazard is logged at
    ``WARNING`` through the module logger: it is a gap in what the Z3
    bridge can express, not the routine partial evaluation that
    ``evaluate_with_bindings`` reports at ``DEBUG``, and no solver setting
    the caller can reach will turn it into a decided answer.

    Args:
        expression: Expression to decide.
        symbol_types: Z3 sort for each free identifier of ``expression``.
        context: Public method name the outcome is reported under, used to
            attribute the warning.
        timeout_milliseconds: Optional bound, in milliseconds, on the
            solver invocation.

    Returns:
        ``SATISFIED``/``VIOLATED`` when the solver decides,
        ``UNDECIDED`` on a hazardous lowering or an inconclusive solver.

    Raises:
        MissingSymbolTypeError: If ``symbol_types`` lacks an entry for a
            free identifier of ``expression``.

    """
    _validate_symbol_types_cover_free_identifiers(expression, symbol_types)
    hazard = _find_bool_sort_hazard(expression, symbol_types)
    if hazard is not None:
        _LOGGER.warning(
            "%s: node %r lowers a Boolean operand into a numeric context, where "
            "the Z3 bindings rewrite it to If(b, 1, 0) and collapse this "
            "package's type-strict semantics; identifier sorts at that node: "
            "%s. The expression is not handed to the solver and the check "
            "reports UNDECIDED; bounding timeout_milliseconds cannot change "
            "this outcome.",
            context,
            hazard,
            _render_identifier_sorts(hazard, symbol_types),
        )
        return ConstraintOutcome.UNDECIDED
    satisfiable = check_expression_satisfiability(
        expression,
        dict(symbol_types),
        timeout_milliseconds=timeout_milliseconds,
    )
    if satisfiable is None:
        return ConstraintOutcome.UNDECIDED
    if satisfiable:
        return ConstraintOutcome.SATISFIED
    return ConstraintOutcome.VIOLATED


def _build_literal_ordering_key(value: LiteralType) -> str:
    """Return an ordering key constant on ``LiteralExpression`` equivalence.

    ``LiteralExpression`` compares literals by bucket and canonical form
    rather than by stored Python type, so the key has to collapse the same
    forms: the integer-grammar strings ``"5"`` and ``"05"`` key alike with
    the integer ``5``, the float-grammar strings ``"1.5"`` and ``"1.50"``
    key alike as one exact decimal, and ``-0.0`` keys alike with ``0.0``.
    A ``bool`` keys apart from every integer, and an exact-decimal string
    apart from the binary ``float`` carrying the same digits.

    Args:
        value: Stored value of a ``LiteralExpression``.

    Returns:
        Bucket-prefixed textual key.

    """
    if isinstance(value, bool):
        return f"bool:{value}"
    elif isinstance(value, int):
        return f"int:{value}"
    elif isinstance(value, float):
        # Adding zero maps -0.0 to 0.0; the two are equal and so must key alike.
        return f"float-binary:{value + 0.0!r}"
    # A string-form literal matches the integer grammar or the float grammar,
    # and only the latter carries a decimal point.
    elif "." in value:
        return f"float-decimal:{Decimal(value).normalize()}"
    return f"int:{int(value)}"


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


def _build_expression_ordering_key(expression: Expression) -> str:
    """Return an ordering key constant on expression structural equivalence.

    Renders the tree as ``NodeType[node data](child keys)``. Node data is
    whatever the node compares by beyond its children: a literal's bucket
    and canonical form, an identifier's ``id``, an operation's name, or a
    call's function name. A ``PiecewiseExpression`` needs none, since its
    children already encode the cases and the fallback.

    Args:
        expression: Expression to key.

    Returns:
        Textual key for the whole subtree.

    """
    children = ",".join(
        _build_expression_ordering_key(child)
        for child in expression.get_visit_children()
    )
    node_data = _render_expression_node_ordering_data(expression)
    return f"{type(expression).__name__}[{node_data}]({children})"


def _render_expression_node_ordering_data(expression: Expression) -> str:
    """Return one node's own ordering data, excluding its children."""
    if isinstance(expression, LiteralExpression):
        return _build_literal_ordering_key(expression.value)
    elif isinstance(expression, IdentifierExpression):
        return f"id:{expression.identifier.id}"
    elif isinstance(expression, (BinaryExpression, UnaryExpression)):
        return expression.operation.value
    elif isinstance(expression, CallExpression):
        return f"call:{expression.function_name}"
    return ""


def _build_constraint_ordering_key(constraint: Constraint) -> str:
    """Return the canonical sort key ``ConstraintSystem`` orders its members by.

    The key is constant on structural-equivalence classes: two
    structurally equivalent constraints always key alike, so a system's
    member order does not depend on construction order. It is keyed on
    the same things equivalence compares -- the concrete kind, the
    variable's ``Identifier.id``, and either the expression tree or the
    type-strict member set -- rather than on ``repr``, which neither
    separates every distinct constraint nor agrees on every equivalent
    pair.

    A ``Constraint`` subclass declared outside this module falls back to
    its ``repr``, which the subclassing contract requires to identify the
    kind and the variable.

    Args:
        constraint: Member to key.

    Returns:
        Textual key ordering the member within its system.

    """
    kind = type(constraint).__name__
    if isinstance(constraint, EquationConstraint):
        expression_key = _build_expression_ordering_key(constraint.expression)
        return f"{kind}|{constraint.variable.id}|{expression_key}"
    elif isinstance(constraint, (InSetConstraint, NotInSetConstraint)):
        members = ",".join(
            sorted(_build_member_ordering_key(member) for member in constraint.members)
        )
        return f"{kind}|{constraint.variable.id}|{{{members}}}"
    return f"{kind}|{constraint!r}"


def create_constraint_system(*constraints: Constraint) -> "ConstraintSystem":
    """Create a constraint system from the given constraints.

    Args:
        constraints: Zero or more constraints; identifiers shared between
            constraints denote the same variable.

    Returns:
        A frozen ``ConstraintSystem`` holding the constraints in canonical
        order.

    Raises:
        ConstraintError: If any argument is not a ``Constraint``.

    """
    return ConstraintSystem(constraints)


@register_serializable(type_id="constraint_system")
@dataclass(frozen=True, eq=False)
class ConstraintSystem(WrappedFamilySerializable, FrozenMixin, DerivedEquivalenceMixin):
    """An ordered conjunction of constraints over shared identifiers.

    Semantically the logical AND of its member constraints. The
    ``constraints`` argument is materialized once before it is traversed,
    so a single-pass iterable is retained in full rather than consumed
    into an empty system. Constraints are then normalized into canonical
    order, keyed on the same things structural equivalence compares, so
    structurally equivalent systems built from differently ordered inputs
    are structurally equivalent and serialize identically. Duplicate
    constraints are retained (conjunction is idempotent). Instances are
    frozen; mutation raises ``FrozenMutationError``.

    ``ConstraintSystem`` is declared ``@dataclass(frozen=True, eq=False)``,
    so ``__eq__`` and ``__hash__`` fall back to object identity rather than
    comparing the ``constraints`` tuple. Two structurally equivalent
    systems are therefore **distinct dict keys** and **distinct set
    members**: use ``is_structurally_equivalent`` for value-equality
    semantics, and avoid using ``ConstraintSystem`` instances as dict keys
    when you expect value-based lookups.

    """

    constraints: tuple[Constraint, ...]

    def __post_init__(self) -> None:
        # Materialize before anything else: validation and canonical ordering
        # each traverse the members, and a one-shot iterator would be empty by
        # the second pass.
        constraints = tuple(self.constraints)
        for constraint in constraints:
            if not isinstance(constraint, Constraint):
                raise ConstraintError(
                    "ConstraintSystem members must be Constraint instances, "
                    f"but got value {constraint!r} of type "
                    f"{type(constraint).__name__}."
                )
        object.__setattr__(
            self,
            "constraints",
            tuple(sorted(constraints, key=_build_constraint_ordering_key)),
        )

    def get_free_identifiers(self) -> frozenset[Identifier]:
        """Return the union of every member constraint's free identifiers."""
        free: frozenset[Identifier] = frozenset()
        for constraint in self.constraints:
            free |= constraint.get_free_identifiers()
        return free

    def evaluate_with_bindings(self, bindings: ConstraintBindings) -> ConstraintOutcome:
        """Return the conjunction outcome of all members under the bindings.

        ``VIOLATED`` if any member is ``VIOLATED`` (a definite violation
        dominates indeterminacy; members are checked in canonical order and
        checking stops at the first violation); ``SATISFIED`` if every
        member is ``SATISFIED``; ``UNDECIDED`` otherwise. Each undecided
        member is logged at ``DEBUG`` through the module logger, so a
        system-level ``UNDECIDED`` identifies the members it came from
        rather than leaving the caller to re-check each one by hand.

        """
        resolved_bindings = dict(bindings)
        saw_undecided = False
        for constraint in self.constraints:
            outcome = constraint.evaluate_with_bindings(resolved_bindings)
            if outcome is ConstraintOutcome.VIOLATED:
                return ConstraintOutcome.VIOLATED
            if outcome is ConstraintOutcome.UNDECIDED:
                _LOGGER.debug(
                    "ConstraintSystem.evaluate_with_bindings: member %r is "
                    "undecided under the given bindings; the conjunction "
                    "reports UNDECIDED unless a later member is violated",
                    constraint,
                )
                saw_undecided = True
        return (
            ConstraintOutcome.UNDECIDED
            if saw_undecided
            else ConstraintOutcome.SATISFIED
        )

    def is_satisfied_with_bindings(self, bindings: ConstraintBindings) -> bool:
        """Return whether the bindings provably satisfy every constraint."""
        return self.evaluate_with_bindings(bindings) is ConstraintOutcome.SATISFIED

    def convert_to_expression(self) -> Expression:
        """Return the conjunction of every member's expression form.

        Empty system yields ``LiteralExpression(True)``; a single member
        yields that member's expression unwrapped; otherwise a
        ``logical_and`` over members in canonical order.

        Raises:
            ConstraintError: If any member cannot be expressed.

        """
        if not self.constraints:
            return LiteralExpression(True)
        expressions = [
            constraint.convert_to_expression() for constraint in self.constraints
        ]
        if len(expressions) == 1:
            return expressions[0]
        return Expression.logical_and(*expressions)

    def check_satisfiability(
        self,
        symbol_types: Mapping[Identifier, SymbolType],
        *,
        timeout_milliseconds: int | None = None,
    ) -> ConstraintOutcome:
        """Return whether some joint assignment satisfies every constraint.

        Lowers ``convert_to_expression()`` to
        ``solver.check_expression_satisfiability``: a satisfying
        assignment provably exists -> ``SATISFIED``; provably none
        exists -> ``VIOLATED``; solver ``unknown`` -> ``UNDECIDED``.
        The empty system returns ``SATISFIED`` without invoking the
        solver.

        Limitation: the lowered conjunction is screened for the Z3
        Boolean-coercion hazard before the solver is consulted, and a
        hazardous conjunction returns ``UNDECIDED`` rather than a
        provably-wrong decided outcome. The hazard is a ``BoolVal``
        reaching a numeric context -- an arithmetic operand, one side of a
        comparison whose other side is numeric, or a piecewise branch
        facing a numeric sibling -- where the Z3 Python bindings silently
        rewrite it to ``If(b, 1, 0)`` and collapse this package's
        type-strict semantics. That covers a ``bool`` set member, a
        ``bool`` literal written into an equation, and a
        ``SymbolType.BOOL`` variable compared against a numeric literal.
        The screen is per-site: a ``bool`` literal consumed by
        ``logical_and``/``logical_or``/``logical_not``, or standing alone
        as the whole expression, lowers faithfully and stays decidable.

        Args:
            symbol_types: Z3 sort for each free identifier of the system.
            timeout_milliseconds: Optional bound, in milliseconds, on the
                underlying Z3 solver invocation. ``None`` (the default)
                leaves the solver unbounded.

        Raises:
            MissingSymbolTypeError: If ``symbol_types`` lacks an entry for
                a free identifier of the lowered conjunction. Checked
                ahead of the hazard screen, so the precondition raises
                even for a conjunction that would otherwise be reported
                ``UNDECIDED``. This is a raise, not the
                ``ConstraintOutcome.UNDECIDED`` degradation
                ``evaluate_with_bindings`` uses for a missing *value*
                binding: a missing symbol type is a caller precondition
                violation the Z3 bridge cannot proceed without, while a
                missing value binding is an ordinary partial assignment
                the symbolic evaluator can report as undecided.
            ConstraintError: If a member cannot be converted to an
                expression.
            ValueError: If ``timeout_milliseconds`` is not None and not
                positive. Checked before the empty-system and hazard
                early returns, so an inadmissible bound is rejected even
                when the outcome is decided without the solver.

        """
        validate_timeout_milliseconds(timeout_milliseconds)
        if not self.constraints:
            return ConstraintOutcome.SATISFIED
        return _decide_satisfiability(
            self.convert_to_expression(),
            symbol_types,
            context="ConstraintSystem.check_satisfiability",
            timeout_milliseconds=timeout_milliseconds,
        )

    def check_satisfiability_with_bindings(
        self,
        bindings: ConstraintBindings,
        symbol_types: Mapping[Identifier, SymbolType],
        *,
        timeout_milliseconds: int | None = None,
    ) -> ConstraintOutcome:
        """Return whether the system is satisfiable given a partial assignment.

        Substitutes the bindings into the conjunction, then decides
        satisfiability of the residual over the remaining free identifiers
        via the z3 bridge. ``symbol_types`` needs entries only for the
        identifiers left free after substitution. Answers questions of the
        form "given x = 4, can y and z still be chosen?".

        Limitation: the same Boolean-coercion hazard documented on
        ``check_satisfiability`` applies here, screened against the
        residual rather than the original conjunction. Substitution is
        therefore part of the screen: a ``bool`` binding value lands in
        the residual exactly as a ``bool`` set member does and is screened
        the same way, while binding a variable to a value of the matching
        sort can retire a hazard the unsubstituted conjunction had.

        Args:
            bindings: Partial assignment substituted into the conjunction
                before the satisfiability check.
            symbol_types: Z3 sort for each identifier left free after
                substitution.
            timeout_milliseconds: Optional bound, in milliseconds, on the
                underlying Z3 solver invocation. ``None`` (the default)
                leaves the solver unbounded.

        Raises:
            MissingSymbolTypeError: If ``symbol_types`` lacks an entry for
                a free identifier of the residual expression left after
                substitution. Checked ahead of the hazard screen, so the
                precondition raises even for a residual that would
                otherwise be reported ``UNDECIDED``. Contrast a missing
                entry in ``bindings`` itself: an identifier ``bindings``
                does not cover is left free in the residual rather than
                raising, so it only raises here if ``symbol_types`` also
                fails to cover it. A missing *value* binding degrades to
                ``ConstraintOutcome.UNDECIDED`` on ``evaluate_with_bindings``;
                a missing symbol type here always raises, since the Z3
                bridge cannot proceed without a sort for every free
                identifier.
            ValueError: If ``timeout_milliseconds`` is not None and not
                positive. Checked before the empty-system and hazard
                early returns, so an inadmissible bound is rejected even
                when the outcome is decided without the solver.

        """
        validate_timeout_milliseconds(timeout_milliseconds)
        if not self.constraints:
            return ConstraintOutcome.SATISFIED
        environment = _coerce_bindings_to_environment(bindings)
        residual = self.convert_to_expression().substitute(environment)
        return _decide_satisfiability(
            residual,
            symbol_types,
            context="ConstraintSystem.check_satisfiability_with_bindings",
            timeout_milliseconds=timeout_milliseconds,
        )

    @classmethod
    @override
    def construct_from_fields(cls, fields: dict[str, Any]) -> "ConstraintSystem":
        """Route deserialized fields through the constructor for re-validation."""
        return cls(fields["constraints"])

    @override
    def __repr__(self) -> str:
        return f"ConstraintSystem({format_comma_separated_list(self.constraints)})"

    @override
    def __str__(self) -> str:
        if not self.constraints:
            return "True"
        return " and ".join(str(constraint) for constraint in self.constraints)
