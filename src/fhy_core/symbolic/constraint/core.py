"""Constraint evaluation core: outcomes, bindings, and the ``Constraint`` family.

Owns the tri-state ``ConstraintOutcome`` result, the ``ConstraintBindings``
assignment type, the ``SymbolicPredicate`` protocol shared with
``ConstraintSystem`` (``fhy_core.symbolic.constraint.system``), and the
``Constraint`` sum-type family base together with its three concrete leaves
-- ``EquationConstraint``, ``InSetConstraint`` and ``NotInSetConstraint``.

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

The three leaves run on the Rust core (``fhy_core::constraint``): each is a
thin subclass of its ``fhy_core._rs`` class, which implements evaluation,
conversion, the canonical ordering key, structural and alpha equivalence,
freezing, pickling and the data payload, and keeps the Python objects it
was given. Evaluation asks the default solver (``get_default_solver``) and
logs its undecided outcomes on this module's logger. Members compare
type-strictly and are kept in one canonical order: by kind (``bool``,
``float``, ``frozenset``, ``int``, ``str``, ``tuple``, then ``Serializable``
members), then by value. The leaves are virtual subclasses of
``Constraint`` and ``FrozenMixin``; ``Constraint`` stays an abstract base
third parties subclass, and ``ConstraintSystem`` holds either kind.
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
from collections.abc import Mapping
from enum import Enum, auto
from typing import Protocol, TypeAlias, runtime_checkable

from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.logger import get_logger
from fhy_core.serialization import WrappedFamilySerializable, register_serializable
from fhy_core.symbolic.expression import Expression, LiteralType
from fhy_core.term import DerivedEquivalenceMixin
from fhy_core.traits import FrozenMixin
from fhy_core.utils.override import override

_LOGGER = get_logger(__name__)
"""The logger the Rust binding writes this module's records to."""

ConstraintBindings: TypeAlias = Mapping[Identifier, "Expression | LiteralType"]
"""Assignment of candidate values (literals or expressions) to identifiers."""


class ConstraintOutcome(Enum):
    """Tri-state answer to a constraint query.

    Every query that can be proven, refuted, or left open answers with
    one of these members: whether a value satisfies a constraint,
    whether a constraint system is satisfiable, whether one system
    implies another, whether a parameter has a feasible value, and
    whether one parameter's value set is a subset of another's.

    - ``SATISFIED``: the relation provably holds.
    - ``VIOLATED``: the relation provably fails.
    - ``UNDECIDED``: the checker cannot decide (for example, the
      expression simplifier could not reduce a substituted expression to
      a literal, or the solver timed out). This is neither a satisfaction
      nor a violation; it signals that the relation could not be settled.

    A member has no truth value: ``bool()`` on one raises ``TypeError``,
    so an outcome is compared against a member rather than tested for
    truthiness.
    """

    SATISFIED = auto()
    VIOLATED = auto()
    UNDECIDED = auto()

    # Folding a tri-state answer to a truth value would silently read
    # UNDECIDED as one of the decided members; refuse so every consumer
    # names the member it is folding away.
    def __bool__(self) -> bool:
        raise TypeError("ConstraintOutcome is tri-state; compare against a member.")


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
    ``NotInSetConstraint``), which run on the Rust core and are registered
    as virtual subclasses, and for constraints third parties define as
    ``@register_serializable @dataclass(frozen=True, eq=False)`` leaves of
    this family, whose serialization and structural equivalence are
    derived from their fields. The base holds no state and designates no variable:
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
        - Override ``build_ordering_key`` to key on the same things
          structural equivalence compares.
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
            ConstraintError: If a binding value is unusable by this
                constraint's own evaluation mechanism: for
                ``EquationConstraint``, a value that is neither an
                ``Expression`` nor a ``LiteralType``, or a literal value
                ``LiteralExpression`` refuses, such as a ``str`` outside
                the integer and float grammars; for a set constraint, a
                value that is neither an ``Expression`` nor a valid
                ``ConstraintMember``.
            NonBooleanLogicalOperandError: For ``EquationConstraint``, if
                the expression's root provably denotes a number, if it
                holds a provably numeric operand in a Boolean position --
                under a logical connective or as a piecewise case
                condition -- counting a binding that puts a number there,
                or if the substituted expression simplifies to a
                non-bool literal. A set constraint never raises it.

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

        Raises:
            ConstraintError: As ``evaluate_with_bindings`` raises it.
            NonBooleanLogicalOperandError: As ``evaluate_with_bindings``
                raises it.

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
            ConstraintError: If the constraint cannot be expressed (for
                example, a set member is a ``str``, or is not itself a
                ``LiteralType``).

        """

    # TODO: derive this from the field schema for third-party leaves instead
    # of overriding it per leaf. `DerivedEquivalenceMixin` already builds a
    # per-type plan that drives `is_structurally_equivalent`
    # (`fhy_core.term.derived_equivalence`), and this key is a projection of
    # that same plan. The three built-in leaves take the Rust core's key,
    # whose agreement with equivalence the core's tests pin. It is a change
    # in `fhy_core.term` affecting every `DerivedEquivalenceMixin` user, so
    # it is not in scope here.
    @abstractmethod
    def build_ordering_key(self) -> str:
        """Return the canonical ordering key for this constraint.

        Constant on structural-equivalence classes: two structurally
        equivalent constraints always key alike, so a system's member
        order does not depend on construction order. An implementation
        keys on the same things ``is_structurally_equivalent`` compares
        -- the concrete kind, plus whatever fields participate in
        equivalence -- rather than on ``repr``, which neither separates
        every distinct constraint nor agrees on every equivalent pair.

        ``ConstraintSystem`` orders its members by this key, and the
        param layer's constraint tuple inherits that order, so the two
        layers agree on canonical form.

        Returns:
            Textual key ordering the constraint within its system.

        """

    @abstractmethod
    @override
    def __repr__(self) -> str: ...

    @abstractmethod
    @override
    def __str__(self) -> str: ...


@register_serializable(type_id="equation_constraint")
class EquationConstraint(_rs.EquationConstraint, WrappedFamilySerializable):
    """Boolean-expression predicate over the expression's free identifiers.

    The constraint wraps a Boolean ``Expression``; its scope is exactly
    that expression's free identifiers (empty for a ground expression).
    ``evaluate_with_bindings`` substitutes every bound identifier in the
    scope simultaneously, simplifies the result with the default solver's
    simplifier, and reports ``SATISFIED`` only when the simplifier reduces
    it to the ``bool`` literal ``True``. A binding outside the scope is
    ignored and never inspected.

    The expression is itself a predicate, so it is screened before anything
    is substituted: a numeric root, such as ``LiteralExpression(1)`` or
    ``x + 1``, raises ``NonBooleanLogicalOperandError``, and so does a
    binding that puts a number in a Boolean position.

    Outcomes:
        - ``SATISFIED``: the substituted expression reduces to ``True``.
        - ``VIOLATED``: the substituted expression reduces to ``False``.
        - ``UNDECIDED``: the simplifier cannot reduce it to a literal, logged
          at ``DEBUG`` when a free identifier remains and at ``WARNING``
          when none does; or a binding binds a registered native constant's
          canonical identifier the expression refers to, which names a value
          rather than a variable, logged at ``WARNING``.

    A result that is a literal but not a ``bool`` raises
    ``NonBooleanLogicalOperandError``: the expression denotes a number. A
    binding value that is neither an ``Expression`` nor a literal the
    ``LiteralExpression`` constructor accepts raises ``ConstraintError``
    naming the identifier, and a simplifier failure raises the solver's
    error, ``PassExecutionError`` for the SymPy backend.

    Attributes:
        expression: The Boolean ``Expression`` given; the scope is exactly
            its free identifiers.

    """

    __slots__ = ()


@register_serializable(type_id="in_set_constraint")
class InSetConstraint(_rs.InSetConstraint, WrappedFamilySerializable):
    """Permitted-set membership predicate over one identifier.

    Scope is ``frozenset((variable,))``. ``evaluate_with_bindings`` reports
    ``SATISFIED`` iff the bound value is one of the members and ``VIOLATED``
    otherwise, comparing type-strictly: ``True``, ``1`` and ``1.0`` are
    three members, at any depth inside a ``tuple`` or ``frozenset``. A
    number whose type subclasses ``int`` or ``float``, such as an
    ``IntEnum`` member, is the exact number it denotes, both as a member and
    as a bound value, and ``-0.0`` is the member ``0.0``. A bound
    ``LiteralExpression`` is decided by its value; any other expression,
    and a missing binding, is ``UNDECIDED`` (``DEBUG``); so is any binding
    of a registered native constant's canonical identifier (``WARNING``).
    A bound value that could never be a member, or whose hash raises,
    raises ``ConstraintError``.

    Members are a ``str``, ``int``, ``float`` or ``bool``, a hashable
    ``tuple`` or ``frozenset`` of members, or a ``Serializable`` that is
    also ``Hashable``; ``None``, a NaN and other values raise
    ``ConstraintError``. ``values`` and ``members`` hold the distinct
    members in canonical order, which ``repr``, ``str``, the payload and
    ``convert_to_expression`` use too.

    Attributes:
        variable: The constrained ``Identifier``, as given.
        values: The members, in canonical order.

    """

    __slots__ = ()


@register_serializable(type_id="not_in_set_constraint")
class NotInSetConstraint(_rs.NotInSetConstraint, WrappedFamilySerializable):
    """Forbidden-set membership predicate over one identifier.

    Symmetric to ``InSetConstraint``: ``evaluate_with_bindings`` reports
    ``SATISFIED`` iff the bound value is NOT one of the members and
    ``VIOLATED`` otherwise, with the same type-strict comparison, undecided
    cases and refusals.

    Attributes:
        variable: The constrained ``Identifier``, as given.
        values: The members, in canonical order.

    """

    __slots__ = ()


# The leaves are registered, not derived: `Constraint`'s bases carry an
# instance layout a Rust-backed class cannot share, as for `Expression`.
for _leaf in (EquationConstraint, InSetConstraint, NotInSetConstraint):
    Constraint.register(_leaf)
    FrozenMixin.register(_leaf)
del _leaf
