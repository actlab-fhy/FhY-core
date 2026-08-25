"""Constraint evaluation core: outcomes, bindings, and the ``Constraint`` family.

Owns the tri-state ``ConstraintOutcome`` result, the ``ConstraintBindings``
assignment type and its coercion into a substitution environment, the
``SymbolicPredicate`` protocol shared with ``ConstraintSystem``
(``fhy_core.symbolic.constraint.system``), and the ``Constraint`` sum-type
family base together with its three concrete leaves --
``EquationConstraint``, and the ``_SetConstraint``-derived
``InSetConstraint``/``NotInSetConstraint``.

A constraint's semantic identity is its *scope* -- the set of identifiers
it references (``get_free_identifiers``) -- rather than a single
designated variable. ``EquationConstraint`` is inherently multi-variable
(a dependent constraint like ``x < y`` is a first-class citizen with no
privileged side); the two set constraints are inherently unary, since
each decides membership for exactly one identifier. The one evaluation
contract is assignment-based: ``evaluate_with_bindings``/
``is_satisfied_with_bindings`` check a constraint against a
``Mapping[Identifier, value]``, reporting ``UNDECIDED`` whenever a free
identifier remains unbound or a bound value cannot be reduced to a
decision.

Each concrete leaf is a ``@register_serializable
@dataclass(frozen=True, eq=False)`` class whose wrapped serialization and
structural equivalence are derived from its fields rather than
hand-written; the ``Constraint`` base holds no shared state and is frozen
on construction via ``FrozenMixin``.
"""

__all__ = [
    "Constraint",
    "ConstraintBindings",
    "ConstraintOutcome",
    "EquationConstraint",
    "InSetConstraint",
    "NotInSetConstraint",
    "SymbolicPredicate",
]

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property
from typing import Any, ClassVar, Protocol, TypeAlias, runtime_checkable

from fhy_core.identifier import Identifier
from fhy_core.logger import get_logger
from fhy_core.serialization import WrappedFamilySerializable, register_serializable
from fhy_core.symbolic.expression import (
    BinaryOperation,
    Expression,
    LiteralExpression,
    LiteralType,
    make_binary_expression,
    pformat_expression,
)
from fhy_core.symbolic.solver import simplify_expression
from fhy_core.term import (
    DerivedEquivalenceMixin,
    compared_as_reference,
    compared_as_value,
)
from fhy_core.traits import FrozenMixin
from fhy_core.utils import Self, format_comma_separated_list
from fhy_core.utils.override import override

from .errors import ConstraintError
from .members import (
    _VALUES_CODEC,
    ConstraintMember,
    MemberCollection,
    _lift_member_to_literal_expression,
    _normalize_constraint_member_collection,
    _order_members_canonically,
    _render_member_set,
    _render_member_set_str,
    _TypedMember,
    _unwrap_member,
    _wrap_member,
    _wrap_member_collection,
)

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
