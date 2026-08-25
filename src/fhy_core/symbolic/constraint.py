"""Scope-based constraints over expressions and value sets.

This module provides the three constraint kinds used by the parameter
infrastructure in ``fhy_core.symbolic.param.*``:

- ``EquationConstraint``: a Boolean expression that must hold under an
  assignment to its free identifiers.
- ``InSetConstraint``: an identifier must take a value from a permitted
  set.
- ``NotInSetConstraint``: an identifier must NOT take a value from a
  forbidden set.

A constraint's semantic identity is its *scope* -- the set of identifiers
it references (``get_free_identifiers``) -- rather than a single
designated variable. ``EquationConstraint`` is inherently multi-variable
(a dependent constraint like ``x < y`` is a first-class citizen with no
privileged side); ``InSetConstraint`` and ``NotInSetConstraint`` are
inherently unary, so their scope is always a single identifier. The one
evaluation contract is assignment-based:
``evaluate_with_bindings`` / ``is_satisfied_with_bindings`` check a
constraint against a ``Mapping[Identifier, value]``. A constraint becomes
decidable once every identifier its evaluation depends on is bound to a
literal; decidable is not decided -- binding an identifier to a
non-literal, symbolic ``Expression`` leaves a residual over the
identifiers that expression introduces, and even a fully literal
assignment can leave a residual the simplifier cannot reduce to a
literal. Either case reports ``UNDECIDED``.

``Constraint`` is a sum-type family base. Each concrete constraint is a
``@register_serializable @dataclass(frozen=True, eq=False)`` leaf whose
wrapped serialization and structural equivalence are derived from its
fields rather than hand-written. The base holds no shared state.
Instances are frozen on construction via ``FrozenMixin``. Each can be
converted to an equivalent ``Expression`` through
``Constraint.convert_to_expression``.

``SymbolicPredicate`` is the structural protocol shared by ``Constraint``
and ``ConstraintSystem``: both expose the same four-method scope/bindings
surface, so a caller can be written once against the protocol and accept
either shape.

Set-constraint members are stored with type-strict equality: ``int``,
``float``, and ``bool`` are not interchangeable, including inside nested
``tuple`` and ``frozenset`` members. The wrapping that implements this is
private; the public API accepts and returns raw values.

Set-constraint serialization is deterministic: members are emitted in
``repr``-sorted order, and the corresponding ``convert_to_expression``
leaves are emitted in the same order.

``ConstraintSystem`` is the companion set-level value object: a
canonically ordered conjunction of constraints, possibly spanning several
identifiers, with joint-satisfiability and entailment checking backed by
``fhy_core.symbolic.solver``. It is not a ``Constraint`` subclass -- joint
satisfiability is a property of a collection, not of any single
predicate. Its solver-backed entry points report ``UNDECIDED`` instead of
a decided outcome for three hazard classes the Z3 bridge cannot lower
soundly -- Boolean operands in a numeric context,
division/floor-division/modulo by a possibly-zero divisor, and an
``EQUAL``/``NOT_EQUAL`` comparison mixing an INT-sorted operand with a
float-valued literal -- because ``fhy_core.symbolic.solver`` screens the
lowered expression for those shapes before it ever reaches Z3.
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
    "SymbolicPredicate",
    "build_constraint_ordering_key",
    "create_constraint_system",
]

from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from functools import cached_property
from typing import (
    Any,
    ClassVar,
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
from fhy_core.utils import Self, format_comma_separated_list

from .expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    LiteralType,
    UnaryExpression,
    make_binary_expression,
    pformat_expression,
)
from .solver import (
    check_expression_satisfiability,
    does_expression_imply,
    simplify_expression,
    validate_timeout_milliseconds,
)
from .symbol_type import SymbolType

_LOGGER = get_logger(__name__)

ConstraintBindings: TypeAlias = Mapping[Identifier, "Expression | LiteralType"]
"""Assignment of candidate values (literals or expressions) to identifiers."""


def _validate_binding_value(identifier: Identifier, value: object) -> None:
    """Reject a binding value ``ConstraintBindings`` does not admit.

    ``ConstraintBindings`` is public and declares ``Expression |
    LiteralType``. A value in neither arm cannot be lifted into a
    substitution environment, and handing it to the expression passes
    anyway surfaces as an internal pass failure that names no identifier.
    Rejecting it here reports the caller's mistake as a domain error
    instead.

    Args:
        identifier: Identifier the value is bound to.
        value: Candidate binding value.

    Raises:
        ConstraintError: If ``value`` is neither an ``Expression`` nor a
            ``LiteralType``. The message names the identifier, the value,
            and the value's type.

    """
    if isinstance(value, (Expression, LiteralType)):
        return
    raise ConstraintError(
        f"Binding for identifier {identifier!r} must be an `Expression` or a "
        f"literal (`str`, `float`, `int`, `bool`), but got value {value!r} of "
        f"type {type(value).__name__}."
    )


def _coerce_bindings_to_environment(
    bindings: ConstraintBindings,
) -> dict[Identifier, Expression]:
    """Coerce every binding value to the ``Expression`` a substitution consumes.

    A raw ``LiteralType`` value is wrapped in a ``LiteralExpression``. An
    ``Expression`` value passes through unchanged, including a non-literal,
    symbolic one: substituting a symbolic value is supported, and the
    residual it leaves behind is what the caller's outcome is read from.

    Args:
        bindings: Mapping from identifiers to candidate values.

    Returns:
        Substitution environment binding each identifier to an
        ``Expression``.

    Raises:
        ConstraintError: If a value falls outside ``Expression |
            LiteralType``.

    """
    environment: dict[Identifier, Expression] = {}
    for identifier, value in bindings.items():
        _validate_binding_value(identifier, value)
        environment[identifier] = (
            value if isinstance(value, Expression) else LiteralExpression(value)
        )
    return environment


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


_VALUES_CODEC: FieldCodec = _make_member_set_codec("values")


@runtime_checkable
class SymbolicPredicate(Protocol):
    """Predicate over identifiers, evaluable under a partial assignment.

    The structural contract shared by ``Constraint`` and
    ``ConstraintSystem``. Both inherit this protocol as an explicit base;
    a third-party predicate may satisfy it purely structurally.
    Implementations are immutable value objects.
    """

    def get_free_identifiers(self) -> frozenset[Identifier]:
        """Return the scope: every identifier the predicate references.

        Returns:
            Frozen set of identifiers; empty for a ground predicate.

        """

    def evaluate_with_bindings(self, bindings: ConstraintBindings) -> ConstraintOutcome:
        """Return the tri-state outcome of the predicate under the bindings.

        Args:
            bindings: Mapping from identifiers to candidate values. Raw
                literal values and ``Expression`` values are both
                accepted; identifiers outside the scope are ignored.

        Returns:
            ``SATISFIED``/``VIOLATED`` when decidable under the given
            (possibly partial) bindings; ``UNDECIDED`` otherwise.

        """

    def is_satisfied_with_bindings(self, bindings: ConstraintBindings) -> bool:
        """Return whether the bindings provably satisfy the predicate.

        Both ``VIOLATED`` and ``UNDECIDED`` map to ``False`` (conservative
        rejection).

        """

    def convert_to_expression(self) -> Expression:
        """Return an ``Expression`` whose truth value matches the predicate.

        Raises:
            ConstraintError: If the predicate cannot be expressed.

        """


class Constraint(
    SymbolicPredicate,
    WrappedFamilySerializable,
    FrozenMixin,
    DerivedEquivalenceMixin,
    ABC,
):
    """A predicate over a scope of identifiers.

    Sum-type family base for the three concrete constraint kinds in this
    module (``EquationConstraint``, ``InSetConstraint``,
    ``NotInSetConstraint``). Each concrete constraint is a
    ``@register_serializable @dataclass(frozen=True, eq=False)`` leaf of
    this family; serialization and structural equivalence are derived
    from its fields. The base holds no state and designates no variable:
    a constraint's semantic identity is its scope
    (``get_free_identifiers``), and the only evaluation contract is
    assignment-based (``evaluate_with_bindings``). Instances are frozen
    at the end of construction; subsequent attribute mutation raises
    ``FrozenMutationError``.

    Subclassing contract:
        - Declare a ``@dataclass(frozen=True, eq=False)`` leaf.
        - Override ``get_free_identifiers`` to return the scope.
        - Override ``evaluate_with_bindings`` to define the tri-state
          predicate; the concrete ``is_satisfied_with_bindings`` derives
          from it.
        - Override ``convert_to_expression`` to produce an equivalent
          ``Expression``.
        - Override ``__repr__`` and ``__str__`` so the textual form
          identifies the kind and the scope.

    """

    @abstractmethod
    @override
    def get_free_identifiers(self) -> frozenset[Identifier]:
        """Return every identifier this constraint references.

        Returns:
            Frozen set of identifiers; empty for a ground constraint.

        """

    @abstractmethod
    @override
    def evaluate_with_bindings(self, bindings: ConstraintBindings) -> ConstraintOutcome:
        """Return the tri-state outcome of the constraint under the bindings.

        Args:
            bindings: Mapping from identifiers to candidate values. Raw
                ``LiteralType`` values and ``Expression`` values are both
                accepted; identifiers outside the scope are ignored.

        Returns:
            ``SATISFIED``/``VIOLATED`` when decidable under the given
            (possibly partial) bindings; ``UNDECIDED`` otherwise.

        Raises:
            ConstraintError: If a binding value falls outside
                ``Expression | LiteralType``.

        """

    @override
    def is_satisfied_with_bindings(self, bindings: ConstraintBindings) -> bool:
        """Return whether the bindings provably satisfy the constraint.

        Derived from ``evaluate_with_bindings``; both ``VIOLATED`` and the
        indeterminate ``UNDECIDED`` outcome map to ``False``, so an
        undecided check conservatively rejects the bindings.

        Args:
            bindings: Mapping from identifiers to candidate values.

        Returns:
            True if the bindings satisfy the constraint; False otherwise.

        """
        return self.evaluate_with_bindings(bindings) is ConstraintOutcome.SATISFIED

    @abstractmethod
    @override
    def convert_to_expression(self) -> Expression:
        """Return an expression equivalent to the constraint.

        Returns:
            An ``Expression`` whose truth value matches
            ``is_satisfied_with_bindings``.

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
    """Boolean-expression predicate over the expression's free identifiers.

    The constraint wraps a Boolean ``Expression``; its scope is exactly
    that expression's free identifiers (empty for a ground expression).
    ``evaluate_with_bindings`` substitutes every bound identifier
    simultaneously, simplifies the resulting expression, and reports
    ``SATISFIED`` only when the simplifier reduces it to the ``bool``
    literal ``True``.

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
          because a free identifier remains unbound, or because the
          simplifier just cannot decide). Logged at ``DEBUG`` when a
          free identifier remains in the residual (ordinary partial
          evaluation), at ``WARNING`` when none does (every identifier
          was bound yet the simplifier still could not decide).

    ``is_satisfied_with_bindings`` derives from ``evaluate_with_bindings``
    and treats both ``VIOLATED`` and ``UNDECIDED`` as ``False``, so an
    undecided check conservatively rejects the bindings.

    Attributes:
        expression: Boolean ``Expression``; the constraint's scope is
            exactly this expression's free identifiers.

    """

    expression: Expression

    def __post_init__(self) -> None:
        if not isinstance(self.expression, Expression):
            raise ConstraintError(
                f"EquationConstraint requires an `Expression` instance, but "
                f"got value {self.expression!r} of type "
                f"{type(self.expression).__name__}."
            )

    @override
    def get_free_identifiers(self) -> frozenset[Identifier]:
        """Return the expression's free identifiers."""
        return self.expression.get_free_identifiers()

    @override
    def evaluate_with_bindings(self, bindings: ConstraintBindings) -> ConstraintOutcome:
        """Substitute every bound identifier, simplify, and classify.

        Coerces each raw ``LiteralType`` binding value to a
        ``LiteralExpression``, substitutes the full multi-key environment
        through ``simplify_expression``, and reports ``SATISFIED`` for
        the ``bool`` literal ``True``, ``VIOLATED`` for any other
        literal, and ``UNDECIDED`` when no literal results. Logging on
        ``UNDECIDED``: DEBUG when the residual (substituted and
        simplified) expression still has free identifiers (expected
        partial evaluation, including the case where a symbolic binding
        introduces a new free identifier), WARNING when the residual has
        none -- every free identifier was bound yet the simplifier still
        failed to reduce it to a literal.

        Raises:
            ConstraintError: If a binding value falls outside
                ``Expression | LiteralType`` and so cannot be lifted into
                the substitution environment.
            ValueError: From ``LiteralExpression`` when a ``str`` binding
                value matches neither the integer nor the float grammar.
            PassExecutionError: Propagated from ``simplify_expression``
                when the SymPy bridge fails to lower or lift the
                substituted expression.

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
        """Return the wrapped expression."""
        return self.expression

    @override
    def __repr__(self) -> str:
        return f"EquationConstraint(expression={self.expression!r})"

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


def _evaluate_set_membership_with_bindings(
    kind_name: str,
    variable: Identifier,
    members: frozenset[_TypedMember],
    bindings: ConstraintBindings,
    *,
    satisfied_when_member: bool,
) -> ConstraintOutcome:
    """Decide type-strict membership for one set-constraint leaf under bindings.

    Shared by ``InSetConstraint`` and ``NotInSetConstraint``: the only
    difference between the two kinds is the outcome polarity a member
    decides to. Looks up ``variable`` in ``bindings``, unwraps a
    ``LiteralExpression`` binding to its raw value, and decides
    membership by type-strict comparison against ``members``. A missing
    binding or a non-literal ``Expression`` binding yields ``UNDECIDED``
    (DEBUG-logged, naming the identifier); a membership check against a
    concrete value is always decidable, so this never reports
    ``UNDECIDED`` once ``variable`` is bound to a literal.

    A value outside the declared ``Expression | LiteralType`` union
    reaches the membership check unchanged rather than being turned
    away: there is no expression to lift such a value into, so a
    hashable off-union value is simply not a member and the check stays
    decided, while an unhashable one raises ``TypeError``.

    Args:
        kind_name: Concrete leaf's class name, used to attribute the
            DEBUG log record.
        variable: The constrained identifier.
        members: Type-strict wrapped member set to decide against.
        bindings: Mapping from identifiers to candidate values.
        satisfied_when_member: True for ``InSetConstraint`` (membership
            satisfies the constraint), False for ``NotInSetConstraint``
            (membership violates it).

    Returns:
        ``SATISFIED``/``VIOLATED`` when decidable; ``UNDECIDED`` when
        ``variable`` is unbound or bound to a non-literal expression.

    Raises:
        TypeError: If the bound value is unhashable.

    """
    snapshot = dict(bindings)
    if variable not in snapshot:
        _LOGGER.debug(
            "%s.evaluate_with_bindings: no binding for variable %r; the "
            "bindings supplied %s; reporting UNDECIDED",
            kind_name,
            variable,
            format_comma_separated_list(tuple(snapshot)) or "no identifiers",
        )
        return ConstraintOutcome.UNDECIDED
    value = snapshot[variable]
    if isinstance(value, Expression):
        if not isinstance(value, LiteralExpression):
            _LOGGER.debug(
                "%s.evaluate_with_bindings: the binding for %r is the "
                "non-literal expression %r; this leaf decides against a "
                "concrete value and cannot consume a symbolic one; "
                "reporting UNDECIDED",
                kind_name,
                variable,
                value,
            )
            return ConstraintOutcome.UNDECIDED
        value = value.value
    is_member = _wrap_member(value) in members
    if is_member is satisfied_when_member:
        return ConstraintOutcome.SATISFIED
    return ConstraintOutcome.VIOLATED


@dataclass(frozen=True, eq=False)
class _SetConstraint(Constraint):
    """Shared unary-membership predicate over one identifier.

    Module-private implementing base for ``InSetConstraint`` and
    ``NotInSetConstraint``. Both kinds decide the same question --
    whether the bound value of ``variable`` is a member of ``values`` --
    and differ only in polarity, so every field, normalization step,
    cached derived state, and evaluation/conversion/rendering behavior
    lives here. A leaf contributes only:

    - ``_satisfied_when_member``: ``True`` if membership satisfies the
      constraint (``InSetConstraint``), ``False`` if it violates the
      constraint (``NotInSetConstraint``).
    - ``_comparison_operation``: the ``BinaryOperation`` comparing
      ``variable`` against one member literal (``EQUAL``/``NOT_EQUAL``).
    - ``_empty_set_literal``: the ``bool`` ``convert_to_expression``
      returns for an empty ``values`` (``False``/``True``).
    - ``_render_connective``: the word ``__str__`` renders between the
      variable and the member set (``"in"``/``"not in"``).
    - ``_combine_expressions``: folds one per-member comparison
      expression into the whole (``logical_or``/``logical_and``).

    Scope is always ``frozenset((variable,))``: both kinds are inherently
    unary. ``evaluate_with_bindings`` resolves ``variable`` from the
    bindings and decides by type-strict membership; a missing binding or
    a non-literal ``Expression`` binding is ``UNDECIDED`` (DEBUG-logged),
    while a literal binding is always decidable.

    Determinism:
        ``convert_to_expression`` emits its leaves in ``repr``-sorted
        order so structurally equivalent constraints produce
        structurally equivalent expressions. Member serialization is
        also ``repr``-sorted to match.

    Not itself ``@register_serializable``: each leaf registers its own
    type_id, and ``construct_from_fields`` here builds whichever concrete
    leaf ``cls`` names.

    """

    variable: Identifier = field(metadata=compared_as_reference())
    # Declared as the constructor-input type. ``__post_init__`` normalizes this
    # in place to a deduplicated tuple of raw, unwrapped values (the same
    # content the ``members`` property exposes) so no public attribute ever
    # yields the internal ``_TypedMember`` wrapper. Comparison uses
    # ``compared_as_value(key=_wrap_member_collection)`` rather than the
    # default ``==`` so structural/alpha equivalence stays type-strict and
    # order-independent despite the field itself being a plain tuple.
    values: MemberCollection[ConstraintMember] = field(
        metadata={
            **compared_as_value(key=_wrap_member_collection),
            "serialize_codec": _VALUES_CODEC,
        },
    )

    _satisfied_when_member: ClassVar[bool]
    _comparison_operation: ClassVar[BinaryOperation]
    _empty_set_literal: ClassVar[bool]
    _render_connective: ClassVar[str]

    def __post_init__(self) -> None:
        wrapped = _normalize_constraint_member_collection(self.values)
        object.__setattr__(
            self,
            "values",
            tuple(_unwrap_member(member) for member in wrapped),
        )
        # Seed the ``_members`` cache with the set this normalization pass has
        # already built, so no reader has to derive it a second time.
        object.__setattr__(self, "_members", wrapped)

    @property
    def members(self) -> tuple[ConstraintMember, ...]:
        """Return the members as raw values, in canonical order.

        Ordering is reproducible across processes, so a caller may iterate
        this to produce deterministic output. The ``values`` field holds
        the same members in an unspecified order.

        """
        return _order_members_canonically(self.values)

    @cached_property
    def _members(self) -> frozenset[_TypedMember]:
        """Return the type-strict member set membership is decided against.

        Held as a stored set rather than re-derived per read: ``evaluate``
        is then a constant-time frozenset lookup, and ``__repr__`` -- which
        feeds the ``ConstraintSystem`` ordering key -- costs no wrapper
        allocations. ``__post_init__`` seeds it with the set built during
        normalization; this body re-derives it from ``values`` for an
        instance that reaches a reader unseeded, so the public field stays
        the single source of truth.
        """
        return _wrap_member_collection(self.values)

    @classmethod
    @override
    def construct_from_fields(cls, fields: dict[str, Any]) -> Self:
        return cls(fields["variable"], fields["values"])

    @override
    def get_free_identifiers(self) -> frozenset[Identifier]:
        """Return the single-identifier scope."""
        return frozenset((self.variable,))

    @override
    def evaluate_with_bindings(self, bindings: ConstraintBindings) -> ConstraintOutcome:
        """Decide membership for the bound value of ``variable``.

        Missing binding or non-literal ``Expression`` binding ->
        ``UNDECIDED`` (DEBUG-logged, naming the identifier). A literal
        binding decides by type-strict membership, polarity given by
        ``_satisfied_when_member``; never ``UNDECIDED`` once bound to a
        literal.

        Raises:
            TypeError: If the bound value is unhashable.

        """
        return _evaluate_set_membership_with_bindings(
            type(self).__name__,
            self.variable,
            self._members,
            bindings,
            satisfied_when_member=self._satisfied_when_member,
        )

    @override
    def convert_to_expression(self) -> Expression:
        members = self._members
        if len(members) == 0:
            return LiteralExpression(self._empty_set_literal)
        sorted_values = sorted(members, key=repr)
        if len(sorted_values) == 1:
            return self._build_leaf_expression(sorted_values[0])
        return self._combine_expressions(
            self._build_leaf_expression(member) for member in sorted_values
        )

    @abstractmethod
    def _combine_expressions(self, expressions: Iterable[Expression]) -> Expression:
        """Fold one per-member leaf comparison expression into the whole."""

    def _build_leaf_expression(self, wrapped: _TypedMember) -> Expression:
        literal = _lift_member_to_literal_expression(_unwrap_member(wrapped))
        return make_binary_expression(
            self._comparison_operation, self.variable, literal
        )

    @override
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.variable!r}, "
            f"values={_render_member_set(self._members)})"
        )

    @override
    def __str__(self) -> str:
        return (
            f"{self.variable} {self._render_connective} "
            f"{_render_member_set_str(self._members)}"
        )


@register_serializable(type_id="in_set_constraint")
@dataclass(frozen=True, eq=False)
class InSetConstraint(_SetConstraint):
    """Permitted-set membership predicate over one identifier.

    Scope is ``frozenset((variable,))``. ``evaluate_with_bindings``
    reports ``SATISFIED`` iff the bound value is in ``values`` and
    ``VIOLATED`` otherwise, comparing by type-strict equality (so
    ``True`` and ``1`` are distinct members, and ``1`` and ``1.0`` are
    distinct members, including inside nested ``tuple`` or ``frozenset``
    members). A membership check against a concrete value is always
    decidable, so it never reports ``UNDECIDED`` once ``variable`` is
    bound to a literal.

    """

    _satisfied_when_member = True
    _comparison_operation = BinaryOperation.EQUAL
    _empty_set_literal = False
    _render_connective = "in"

    @override
    def _combine_expressions(self, expressions: Iterable[Expression]) -> Expression:
        return Expression.logical_or(*expressions)


@register_serializable(type_id="not_in_set_constraint")
@dataclass(frozen=True, eq=False)
class NotInSetConstraint(_SetConstraint):
    """Forbidden-set membership predicate over one identifier.

    Symmetric to ``InSetConstraint``: scope is ``frozenset((variable,))``,
    and ``evaluate_with_bindings`` reports ``SATISFIED`` iff the bound
    value is NOT in ``values`` and ``VIOLATED`` otherwise, comparing by
    type-strict equality. A membership check against a concrete value is
    always decidable, so it never reports ``UNDECIDED`` once ``variable``
    is bound to a literal.

    """

    _satisfied_when_member = False
    _comparison_operation = BinaryOperation.NOT_EQUAL
    _empty_set_literal = True
    _render_connective = "not in"

    @override
    def _combine_expressions(self, expressions: Iterable[Expression]) -> Expression:
        return Expression.logical_and(*expressions)


def _raise_if_missing_symbol_types(missing: frozenset[Identifier]) -> None:
    """Raise ``MissingSymbolTypeError`` naming ``missing``, or return if it is empty."""
    if not missing:
        return
    missing_names = ", ".join(sorted(identifier.name_hint for identifier in missing))
    raise MissingSymbolTypeError(
        f"symbol_types is missing an entry for free identifier(s): {missing_names}."
    )


def _validate_symbol_types_cover_free_identifiers(
    expression: Expression, symbol_types: Mapping[Identifier, SymbolType]
) -> None:
    """Raise if ``symbol_types`` lacks an entry for a free identifier of ``expression``.

    Raises:
        MissingSymbolTypeError: If one or more free identifiers of
            ``expression`` have no corresponding ``symbol_types`` entry.

    """
    _raise_if_missing_symbol_types(
        expression.get_free_identifiers() - set(symbol_types)
    )


def _validate_symbol_types_cover_both_sides(
    antecedent: Expression,
    consequent: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
) -> None:
    """Raise if ``symbol_types`` lacks an entry for a free identifier of either side.

    Raises:
        MissingSymbolTypeError: If one or more free identifiers of
            ``antecedent`` or ``consequent`` have no corresponding
            ``symbol_types`` entry.

    """
    free_identifiers = (
        antecedent.get_free_identifiers() | consequent.get_free_identifiers()
    )
    _raise_if_missing_symbol_types(free_identifiers - set(symbol_types))


def _decide_satisfiability(
    expression: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
    *,
    timeout_milliseconds: int | None = None,
) -> ConstraintOutcome:
    """Classify satisfiability of ``expression`` via the solver seam.

    Validates the caller's symbol types, then consults
    ``fhy_core.symbolic.solver.check_expression_satisfiability``. That
    seam function screens the expression for the three hazard classes
    documented on ``ConstraintSystem`` before it ever reaches Z3, so
    ``None`` from the seam -- whether from a screened hazard or an
    inconclusive solver -- maps here to ``UNDECIDED``.

    Args:
        expression: Expression to decide.
        symbol_types: Z3 sort for each free identifier of ``expression``.
        timeout_milliseconds: Optional bound, in milliseconds, on the
            solver invocation.

    Returns:
        ``SATISFIED``/``VIOLATED`` when the solver decides, ``UNDECIDED``
        when the seam screens the expression as hazardous or the solver
        is inconclusive.

    Raises:
        MissingSymbolTypeError: If ``symbol_types`` lacks an entry for a
            free identifier of ``expression``.

    """
    _validate_symbol_types_cover_free_identifiers(expression, symbol_types)
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


def build_constraint_ordering_key(constraint: Constraint) -> str:
    """Return the canonical ordering key for a constraint.

    Constant on structural-equivalence classes: two structurally
    equivalent constraints always key alike, so a system's member order
    does not depend on construction order. It is keyed on the same
    things equivalence compares -- the concrete kind, and either the
    expression tree or the variable's ``Identifier.id`` and the
    type-strict member set -- rather than on ``repr``, which neither
    separates every distinct constraint nor agrees on every equivalent
    pair. ``ConstraintSystem`` orders its members by this key, and the
    param layer orders each parameter's constraint tuple by it, so the
    two layers agree on canonical order.

    A ``Constraint`` subclass declared outside this module falls back to
    its ``repr``, which the subclassing contract requires to identify the
    kind and the scope.

    Args:
        constraint: Member to key.

    Returns:
        Textual key ordering the member within its system.

    """
    kind = type(constraint).__name__
    if isinstance(constraint, EquationConstraint):
        expression_key = _build_expression_ordering_key(constraint.expression)
        return f"{kind}|{expression_key}"
    elif isinstance(constraint, _SetConstraint):
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
class ConstraintSystem(
    SymbolicPredicate, WrappedFamilySerializable, FrozenMixin, DerivedEquivalenceMixin
):
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

    All satisfiability and implication entry points report ``UNDECIDED``
    instead of a decided outcome for three hazard classes: Boolean
    operands in numeric contexts; division/floor-division/modulo whose
    divisor is not a nonzero literal; and ``EQUAL``/``NOT_EQUAL`` mixing
    an INT-sorted operand with a float-valued literal. The screen for
    these hazards lives in ``fhy_core.symbolic.solver``, the seam every
    entry point below lowers through, and it logs a ``WARNING`` (naming
    the seam function and the offending node) before the outcome is
    reported as undecided.

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
            tuple(sorted(constraints, key=build_constraint_ordering_key)),
        )

    @override
    def get_free_identifiers(self) -> frozenset[Identifier]:
        """Return the union of every member constraint's free identifiers."""
        free: frozenset[Identifier] = frozenset()
        for constraint in self.constraints:
            free |= constraint.get_free_identifiers()
        return free

    @override
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

    @override
    def is_satisfied_with_bindings(self, bindings: ConstraintBindings) -> bool:
        """Return whether the bindings provably satisfy every constraint."""
        return self.evaluate_with_bindings(bindings) is ConstraintOutcome.SATISFIED

    @override
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

        Limitation: ``fhy_core.symbolic.solver`` screens the lowered
        conjunction for the three hazard classes documented on this class
        before the solver is consulted, and a hazardous conjunction
        returns ``UNDECIDED`` rather than a provably-wrong decided
        outcome. The Boolean-coercion hazard is a ``BoolVal`` reaching a
        numeric context -- an arithmetic operand, one side of a
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
            symbol_types: Z3 sort for each free identifier of the lowered
                conjunction. That set can be strictly smaller than
                ``get_free_identifiers()``: an empty-member
                ``InSetConstraint``/``NotInSetConstraint`` still reports
                its ``variable`` as part of the system's scope, but
                lowers to a bare ``LiteralExpression`` with no free
                identifier at all, so an unreferenced ``variable`` needs
                no entry.
            timeout_milliseconds: Optional bound, in milliseconds, on the
                underlying Z3 solver invocation. ``None`` (the default)
                leaves the solver unbounded.

        Raises:
            MissingSymbolTypeError: If ``symbol_types`` lacks an entry for
                a free identifier of the lowered conjunction. Checked
                ahead of the seam's hazard screen, so the precondition
                raises even for a conjunction that would otherwise be
                reported ``UNDECIDED``. This is a raise, not the
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

        Limitation: the same three hazard classes documented on this class
        apply here; ``fhy_core.symbolic.solver`` screens the residual
        rather than the original conjunction. Substitution is therefore
        part of the screen: a ``bool`` binding value lands in the
        residual exactly as a ``bool`` set member does and is screened
        the same way, while binding a variable to a value of the matching
        sort can retire a hazard the unsubstituted conjunction had.

        Args:
            bindings: Partial assignment substituted into the conjunction
                before the satisfiability check. Values must be
                ``Expression`` or ``LiteralType``, as
                ``ConstraintBindings`` declares.
            symbol_types: Z3 sort for each identifier left free after
                substitution.
            timeout_milliseconds: Optional bound, in milliseconds, on the
                underlying Z3 solver invocation. ``None`` (the default)
                leaves the solver unbounded.

        Raises:
            MissingSymbolTypeError: If ``symbol_types`` lacks an entry for
                a free identifier of the residual expression left after
                substitution. Checked ahead of the seam's hazard screen,
                so the precondition raises even for a residual that would
                otherwise be reported ``UNDECIDED``. Contrast a missing
                entry in ``bindings`` itself: an identifier ``bindings``
                does not cover is left free in the residual rather than
                raising, so it only raises here if ``symbol_types`` also
                fails to cover it. A missing *value* binding degrades to
                ``ConstraintOutcome.UNDECIDED`` on ``evaluate_with_bindings``;
                a missing symbol type here always raises, since the Z3
                bridge cannot proceed without a sort for every free
                identifier.
            ConstraintError: If a member cannot be converted to an
                expression, or if a ``bindings`` value falls outside
                ``Expression | LiteralType``.
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
            timeout_milliseconds=timeout_milliseconds,
        )

    def check_implication(
        self,
        other: "ConstraintSystem",
        symbol_types: Mapping[Identifier, SymbolType],
        *,
        timeout_milliseconds: int | None = None,
    ) -> ConstraintOutcome:
        """Return whether every assignment satisfying ``self`` satisfies ``other``.

        The system-level entailment seam: both sides are lowered via
        ``convert_to_expression`` and passed to
        ``fhy_core.symbolic.solver.does_expression_imply``, which screens
        both lowered sides for the three hazard classes documented on
        this class before consulting the solver. ``SATISFIED`` when
        entailment is proven, ``VIOLATED`` when a counterexample
        assignment provably exists, ``UNDECIDED`` on a screened hazard on
        either side or an inconclusive solver.

        Args:
            other: Candidate consequence system.
            symbol_types: Z3 sort for each free identifier of either
                side's lowered expression.
            timeout_milliseconds: Optional bound, in milliseconds, on the
                underlying Z3 solver invocation. ``None`` (the default)
                leaves the solver unbounded.

        Returns:
            ``SATISFIED``/``VIOLATED`` when the solver decides,
            ``UNDECIDED`` on a hazardous lowering on either side or an
            inconclusive solver.

        Raises:
            MissingSymbolTypeError: If ``symbol_types`` lacks an entry for
                a free identifier of either side's lowered expression.
                Checked ahead of the seam's hazard screens, so the
                precondition raises even for a pair that would otherwise
                be reported ``UNDECIDED``.
            ConstraintError: If a member of either side cannot be
                converted to an expression.
            ValueError: If ``timeout_milliseconds`` is not None and not
                positive. Checked before every other early return, so an
                inadmissible bound is rejected even for a hazardous pair.

        """
        validate_timeout_milliseconds(timeout_milliseconds)
        antecedent = self.convert_to_expression()
        consequent = other.convert_to_expression()
        _validate_symbol_types_cover_both_sides(antecedent, consequent, symbol_types)
        holds = does_expression_imply(
            antecedent,
            consequent,
            dict(symbol_types),
            timeout_milliseconds=timeout_milliseconds,
        )
        if holds is None:
            return ConstraintOutcome.UNDECIDED
        if holds:
            return ConstraintOutcome.SATISFIED
        return ConstraintOutcome.VIOLATED

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
