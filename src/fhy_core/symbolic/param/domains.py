"""Value-domain strategies for parameters.

A :class:`ParamDomain` captures everything that varies between kinds of
parameter: admissibility, constraint validation, implied constraints, subset
semantics, set algebra, structural equivalence, rendering, and the optional
:class:`IntervalProfile` interval arithmetic reads. A single
:class:`~fhy_core.symbolic.param.core.Param` composes one domain rather than being
subclassed per kind.

:class:`ParamDomain` is a sum-type family base. Each concrete domain is a
``@register_serializable @dataclass(frozen=True, eq=False)`` leaf, and the
family's wrapped serialization (``{"__type__": ..., "__data__": {...}}``) is
derived. Behavior common to the numeric domains lives in the module-level
helpers below.

Subset semantics use value-space gating: two parameters are comparable for
:meth:`compute_feasibility_subset` only when their domains occupy the same value
space (the integer line for integer and interval-integer domains, the reals for
real domains, or the same finite family for ordinal, categorical, and
permutation domains). Cross-space and cross-family queries decide ``VIOLATED``.

:meth:`compute_feasibility_subset` and :meth:`has_feasible_value` answer with
the tri-state :class:`~fhy_core.symbolic.constraint.ConstraintOutcome`, so a
solver that gave up, or an enumeration a dependent constraint leaves open, is
reported as ``UNDECIDED`` rather than being folded into either decided answer.
The same holds when screening weakens the system the solver is asked about by
dropping or narrowing a constraint it cannot be posed: an answer the weaker
system still proves is kept (infeasibility, a counterexample against an exact
antecedent, an implication into an exact consequent), and an answer it does not
prove is reported as ``UNDECIDED``. Finite-set domains enumerate their value
sets and so always decide.

An ill-typed constraint, one holding a provably numeric operand in a Boolean
position, is not undecided: the ``NonBooleanLogicalOperandError`` the
constraint and solver layers raise for it propagates instead.

The six kinds and the procedures run on the Rust core (``fhy_core::param``):
each kind is a thin subclass of its ``fhy_core._rs`` class, registered as a
virtual subclass of ``ParamDomain`` and ``FrozenMixin``, and the module
functions call into the core. Values match type-strictly; ordinal values
ascend (numbers numerically across ``bool``, ``int`` and ``float``, strings
by code point, ``Serializable`` values by their own ``<``), categories keep
the constraint members' canonical order, and permutation members the order
given. ``ParamDomain`` stays an abstract base a new kind of parameter
subclasses; the core calls a Python-defined domain's methods. The
procedures' WARNING and DEBUG records are logged on this module's logger.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, TypeAlias

from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.logger import get_logger
from fhy_core.serialization import WrappedFamilySerializable, register_serializable
from fhy_core.symbolic._native_slots import copy_native_attributes
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintBindings,
    ConstraintOutcome,
    ConstraintSystem,
)
from fhy_core.symbolic.expression import Expression
from fhy_core.symbolic.symbol_type import SymbolType
from fhy_core.traits import FrozenMixin, StructuralEquivalence
from fhy_core.utils.override import override

from .values import CategoricalValue, OrdinalValue, PermutationMemberValue

__all__ = [
    "CategoricalDomain",
    "DecidedOutcome",
    "IntegerDomain",
    "IntervalIntegerDomain",
    "IntervalProfile",
    "OrdinalDomain",
    "ParamDomain",
    "PermutationDomain",
    "RealDomain",
    "compute_constraint_implication_subset",
]

# The procedures' records are logged on this module's logger, by name.
_LOGGER = get_logger(__name__)

# The outcomes an enumeration over a finite value set can report.
DecidedOutcome: TypeAlias = Literal[
    ConstraintOutcome.SATISFIED, ConstraintOutcome.VIOLATED
]


def are_all_constraints_satisfied(
    constraints: Sequence[Constraint], variable: Identifier, value: Any
) -> bool:
    """Return whether ``value`` bound to ``variable`` satisfies every constraint.

    Each constraint is evaluated alone, in order, and the check stops at the
    first one not satisfied.

    Raises:
        ConstraintError: If a constraint cannot lift ``value``, bound to
            ``variable``, into its substitution environment.
        NonBooleanLogicalOperandError: If a constraint holds a provably
            numeric operand in a Boolean position once ``value`` is
            bound; an ill-typed constraint is not reported unsatisfied.

    """
    return _rs.are_all_constraints_satisfied(constraints, variable, value)


def evaluate_system_outcome(
    system: ConstraintSystem, bindings: ConstraintBindings
) -> ConstraintOutcome:
    """Decide ``system`` under ``bindings``, degrading on an expression-pass failure.

    The members are evaluated in canonical order: the first violated member
    decides ``VIOLATED``, and otherwise an undecided member makes the
    outcome ``UNDECIDED``. Evaluation lowers through the SymPy bridge, which
    is not total: a member it cannot lower or lift raises
    ``PassExecutionError``, which makes that member undecided (logged at
    ``WARNING``) rather than escaping a parameter-level query, and the
    evaluation goes on, so a later violated member still decides.

    An ill-typed system is not degraded. A number in a Boolean position
    -- under a logical connective or as a piecewise case condition, a
    binding that puts one there included -- has a meaning under no
    backend, so ``UNDECIDED`` would invite a caller to retry a question
    that cannot succeed; the error propagates, as it does from the
    constraint and solver layers.

    Args:
        system: Constraints to decide.
        bindings: Values for the identifiers the constraints reference.

    Returns:
        The system's outcome.

    Raises:
        ConstraintError: If a member refuses the value ``bindings`` binds
            to an identifier in its scope. It is not degraded: a value that
            cannot be lifted is a caller error, not a limit of the bridge.
        NonBooleanLogicalOperandError: If a member equation holds a
            provably numeric operand in a Boolean position, counting a
            binding that puts a number there.

    """
    return _rs.evaluate_system_outcome(system, bindings)


def compute_constraint_implication_subset(
    own_domain: "ParamDomain",
    own_constraints: Sequence[Constraint],
    own_variable: Identifier,
    other_domain: "ParamDomain",
    other_constraints: Sequence[Constraint],
    other_variable: Identifier,
    symbol_type: SymbolType,
) -> ConstraintOutcome:
    """Decide whether ``own_constraints``'s admissible set is a subset of ``other``'s.

    When ``own_constraints`` contains an ``InSetConstraint``, the
    admissible values are finite and each candidate is evaluated on both
    sides with it bound: a candidate decided into ``own`` and decided out
    of ``other`` is a counterexample and decides ``VIOLATED``; ``other``
    deciding every candidate not decided out of ``own`` decides
    ``SATISFIED``; anything else, such as a candidate a dependent
    constraint leaves undecided on either side, reports ``UNDECIDED``.

    When only ``other_constraints`` is finite, the candidates ``other``
    does not decide out are enumerated and ``own`` is asked, through the
    solver, whether it provably admits a value outside them; such a value
    is a genuine counterexample, and the relation is decided ``VIOLATED``.
    No other answer is drawn from this branch.

    Otherwise the two sides' screened constraint systems are renamed onto
    one shared identifier and decided via
    ``ConstraintSystem.check_implication`` over ``symbol_type``. Screening
    only widens a side's admissible set, so a decided answer is kept
    exactly when the weakened systems still prove it: ``SATISFIED`` with
    an exact consequent, or ``VIOLATED`` with an exact antecedent. Any
    other decided answer, and a solver that gave up, reports
    ``UNDECIDED`` (logged at ``WARNING``). Over the REAL sort, a
    ``SATISFIED`` is also downgraded when the antecedent's own not-in-set
    constraint or the consequent's in-set constraint holds a lifted
    ``float`` member, which Z3 conflates with every other kind denoting the
    same number.

    Args:
        own_domain: Domain of the candidate subset parameter.
        own_constraints: Constraints of the candidate subset parameter.
        own_variable: Variable of the candidate subset parameter.
        other_domain: Domain of the candidate superset parameter.
        other_constraints: Constraints of the candidate superset parameter.
        other_variable: Variable of the candidate superset parameter.
        symbol_type: The sort used to reason about the shared variable.

    Returns:
        ``SATISFIED`` when the subset relation is decided to hold,
        ``VIOLATED`` when a counterexample is decided, and ``UNDECIDED``
        when neither the solver nor the enumeration could decide, or the
        solver decided only a weakened question.

    Raises:
        NonBooleanLogicalOperandError: If a constraint either branch
            evaluates holds a provably numeric operand in a Boolean
            position, counting an in-set candidate bound to its variable,
            or the shared variable itself when ``symbol_type`` is INT or
            REAL.

    """
    return _rs.compute_constraint_implication_subset(
        own_domain,
        own_constraints,
        own_variable,
        other_domain,
        other_constraints,
        other_variable,
        symbol_type,
    )


def is_bound_expression(expression: Expression) -> bool:
    """Return whether ``expression`` is an integer bound of the form ``x <cmp> k``.

    That is a comparison ``>=``, ``>``, ``<=`` or ``<`` of an identifier and
    an integer literal, on either side.
    """
    return _rs.is_bound_expression(expression)


@dataclass(frozen=True)
class IntervalProfile:
    """What interval arithmetic reads from a domain.

    A parameter's interval lives in its bound constraints, not in its
    domain; the domain contributes only these attributes. Interval
    arithmetic and the natural-number bound gate dispatch on a domain's
    profile rather than on its kind, so a domain takes part exactly when
    :meth:`ParamDomain.get_interval_profile` returns one.

    Attributes:
        admits_only_bounds: Whether the domain admits only bound
            constraints, so a parameter over it is an interval operand as
            it stands. A parameter over a domain that admits other
            constraints takes part only once each constraint it carries is
            checked to be a bound, and is then recast over an interval
            domain carrying its partner's ``prefer_inclusive``.
        non_negative: Whether the domain admits only non-negative values.
        zero_included: Whether the domain admits zero, given it is
            non-negative.
        prefer_inclusive: Whether bounds that arithmetic derives for a
            parameter over this domain render in inclusive form. Read only
            where ``admits_only_bounds`` holds, since a recast parameter
            renders as its partner prefers.

    """

    admits_only_bounds: bool
    non_negative: bool
    zero_included: bool
    prefer_inclusive: bool = True


class ParamDomain(WrappedFamilySerializable, FrozenMixin, StructuralEquivalence, ABC):
    """Sum-type family base describing the value space of a parameter kind.

    Concrete domains are ``@register_serializable @dataclass(frozen=True,
    eq=False)`` leaves of this family; serialization is derived by the family
    pattern (a wrapped ``{"__type__": ..., "__data__": {...}}`` envelope keyed by
    each leaf's ``type_id``). A :class:`~fhy_core.symbolic.param.core.Param` composes
    exactly one domain and delegates all kind-specific behavior to it.
    """

    _WIRE_FAMILY: ClassVar[str | None] = "param_domain"

    @property
    @abstractmethod
    def symbol_type(self) -> SymbolType | None:
        """Return this domain's numeric symbol type, or ``None`` if non-numeric.

        This is the sort used when reasoning about the domain's constraints with
        Z3.
        """

    @abstractmethod
    def is_value_admissible(self, value: Any) -> bool:
        """Return whether ``value`` lies in this domain's underlying value set."""

    @abstractmethod
    def normalize_value(self, value: Any) -> Any:
        """Return the canonical form of ``value`` used for storage and checks.

        Normalization must be idempotent: an already canonical value
        normalizes to an equal value.
        """

    @abstractmethod
    def validate_constraint(self, constraint: Constraint, variable: Identifier) -> None:
        """Raise if ``constraint`` is not permitted for this domain."""

    @abstractmethod
    def get_implied_constraints(self, variable: Identifier) -> tuple[Constraint, ...]:
        """Return constraints this domain imposes implicitly on ``variable``."""

    def get_interval_profile(self) -> IntervalProfile | None:
        """Return what interval arithmetic reads from this domain, or ``None``.

        A domain answering ``None`` takes no part in interval arithmetic or
        the natural-number bound gate. Only the integer domains override
        this.

        Returns:
            The domain's interval profile, or ``None`` if its values do not
            form an integer interval.

        """
        return None

    @abstractmethod
    def is_value_set_subset(self, other: "ParamDomain") -> bool:
        """Return whether this domain's value set is a subset of ``other``'s."""

    @abstractmethod
    def compute_feasibility_subset(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: "ParamDomain",
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
    ) -> ConstraintOutcome:
        """Decide whether this domain's constrained set is a subset of ``other``'s.

        Returns:
            ``SATISFIED`` when the subset relation holds, ``VIOLATED``
            when a counterexample is reported, and ``UNDECIDED`` when the
            solver could not decide. Finite-set domains enumerate, so
            they never report ``UNDECIDED``.

        Raises:
            NonBooleanLogicalOperandError: From a numeric domain, if a
                constraint the query evaluates holds a provably numeric
                operand in a Boolean position. A finite-set domain
                carries only set constraints and never raises it.

        """

    @abstractmethod
    def has_feasible_value(
        self, constraints: Sequence[Constraint], variable: Identifier
    ) -> ConstraintOutcome:
        """Decide whether some admissible value satisfies every constraint.

        Returns:
            ``SATISFIED`` when a satisfying value is reported, ``VIOLATED``
            when none can exist, and ``UNDECIDED`` when the solver could
            not decide. Finite-set domains enumerate, so they never report
            ``UNDECIDED``.

        Raises:
            NonBooleanLogicalOperandError: From a numeric domain, if a
                constraint the query evaluates holds a provably numeric
                operand in a Boolean position. A finite-set domain
                carries only set constraints and never raises it.

        """

    def compute_union(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: "ParamDomain",
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
        variable: Identifier,
    ) -> tuple["ParamDomain", tuple[Constraint, ...]] | None:
        """Compute the domain and constraints denoting the union of two value sets.

        Union is representable only by the kinds that override this
        method, each of which bakes both operands' effective value sets
        into a fresh member set; every other kind answers ``None``, so a
        caller can ask whether a union is representable before building
        one.

        Args:
            own_constraints: Constraints carried by the parameter owning
                this domain.
            own_variable: Variable ``own_constraints`` are scoped to.
            other: Domain of the right operand.
            other_constraints: Constraints carried by the right operand.
            other_variable: Variable ``other_constraints`` are scoped to.
            variable: Variable of the result parameter; every returned
                constraint is scoped to it.

        Returns:
            A ``(domain, constraints)`` pair denoting the union, or
            ``None`` if this kind does not represent union. An override
            bakes both operands' constraints into the member set, so its
            constraint tuple is always empty.

        Raises:
            TypeError: From an override, if ``other`` is a different kind
                or the merged ordinal members are not mutually comparable.
            ParamError: From an override, if the merged effective value
                set is empty.

        """
        del own_constraints, own_variable, other, other_constraints
        del other_variable, variable
        return None

    @abstractmethod
    def compute_intersection(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: "ParamDomain",
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
        variable: Identifier,
    ) -> tuple["ParamDomain", tuple[Constraint, ...]]:
        """Compute the domain and constraints denoting the intersection of two sets.

        Every domain kind intersects. A finite-set kind bakes the
        type-strict intersection of both operands' effective value sets
        into a fresh member set and carries no constraints; a permutation
        kind keeps its member set; a numeric kind merges the domain
        attributes conservatively. The latter two carry the conjunction of
        both operands' constraints with both operands' variables renamed
        to ``variable``.

        Args:
            own_constraints: Constraints carried by the parameter owning
                this domain.
            own_variable: Variable ``own_constraints`` are scoped to.
            other: Domain of the right operand; must be the same kind.
            other_constraints: Constraints carried by the right operand.
            other_variable: Variable ``other_constraints`` are scoped to.
            variable: Variable of the result parameter; every returned
                constraint is scoped to it.

        Returns:
            A ``(domain, constraints)`` pair denoting the intersection.

        Raises:
            TypeError: If ``other`` is a different domain kind.
            ConstraintError: If a carried constraint cannot be rescoped.
            ParamError: If the intersection is provably empty. A
                finite-set kind detects an empty member intersection
                here, and a permutation kind detects operands ranging
                over different member sets; numeric emptiness is left to
                the calling factory's feasibility query.

        """

    @abstractmethod
    @override
    def is_structurally_equivalent(self, other: object) -> bool:
        """Return whether ``other`` is a structurally identical domain."""

    @abstractmethod
    def render_set_string(self) -> str:
        """Return the ``str`` rendering of the value set (e.g. ``Z``, ``{1, 2}``)."""

    @abstractmethod
    def render_set_repr(self) -> str:
        """Return the ``repr`` fragment of the value set, or ``""`` if implicit."""


@register_serializable(type_id="integer_domain")
class IntegerDomain(_rs.IntegerDomain, WrappedFamilySerializable):
    """Integer-valued domain, optionally restricted to the natural numbers.

    ``non_negative`` restricts the domain to the natural numbers: it adds an
    implied ``>= 0`` constraint, or ``> 0`` when ``zero_included`` is
    ``False``, and admissibility, the value-set subset test and the
    domain-level feasibility, subset and set-algebra procedures all respect
    it. ``zero_included`` is stored as ``True`` unless the domain is
    non-negative, where it would mean nothing.
    """

    _WIRE_FAMILY: ClassVar[str | None] = "param_domain"

    __slots__ = ("non_negative", "zero_included")

    def __init__(self, non_negative: bool = False, zero_included: bool = True) -> None:
        copy_native_attributes(
            self, IntegerDomain, _rs.IntegerDomain, "non_negative", "zero_included"
        )


@register_serializable(type_id="real_domain")
class RealDomain(_rs.RealDomain, WrappedFamilySerializable):
    """Real-valued domain over finite floats and literal-grammar strings.

    A value is admissible exactly when it is a finite literal: a finite
    Python ``float``, or a ``str`` in the integer or float grammar
    :class:`~fhy_core.symbolic.expression.LiteralExpression` accepts, which
    denotes an exact decimal. NaN and the infinities are refused, as is a
    string that grammar refuses even where ``float()`` parses it (a sign, an
    exponent, surrounding whitespace, digit grouping, ``"nan"``, ``"inf"``).
    ``bool`` and ``int`` are not admissible.
    """

    _WIRE_FAMILY: ClassVar[str | None] = "param_domain"

    __slots__ = ()


@register_serializable(type_id="interval_integer_domain")
class IntervalIntegerDomain(_rs.IntervalIntegerDomain, WrappedFamilySerializable):
    """Integer domain whose parameters carry their interval as bound constraints.

    Admissibility accepts any strict integer within the sign restriction; the
    interval is expressed through the composing parameter's bound
    constraints. Only
    :class:`~fhy_core.symbolic.constraint.EquationConstraint` bound expressions
    are permitted, enabling interval arithmetic on the composing parameter.
    ``prefer_inclusive`` selects how arithmetic results render their bounds.
    ``non_negative`` adds the natural-number implied constraint, which the
    domain-level procedures respect as the integer domain's do.
    """

    _WIRE_FAMILY: ClassVar[str | None] = "param_domain"

    __slots__ = ("non_negative", "prefer_inclusive", "zero_included")

    def __init__(
        self,
        prefer_inclusive: bool = True,
        non_negative: bool = False,
        zero_included: bool = True,
    ) -> None:
        copy_native_attributes(
            self,
            IntervalIntegerDomain,
            _rs.IntervalIntegerDomain,
            "non_negative",
            "prefer_inclusive",
            "zero_included",
        )


@register_serializable(type_id="ordinal_domain")
class OrdinalDomain(_rs.OrdinalDomain, WrappedFamilySerializable):
    """Finite, totally-ordered set of admissible values.

    Values are stored ascending: numbers numerically across ``bool``,
    ``int`` and ``float``, strings by code point, and ``Serializable``
    values by their own ``<``; values the order cannot separate (``1`` and
    ``True``) by kind, ``bool``, ``float``, ``int``, then as given. A number
    subclass is stored as the exact ``int`` or ``float``, and ``-0.0`` as
    ``0.0``.

    A NaN value is refused: it is unequal to itself, so it could never be
    admitted, and it has no place in a total order. An infinity is kept. An
    ``Identifier`` is refused: identifiers do not order.
    """

    _WIRE_FAMILY: ClassVar[str | None] = "param_domain"

    __slots__ = ("sorted_values",)

    def __init__(self, sorted_values: Sequence[OrdinalValue]) -> None:
        copy_native_attributes(self, OrdinalDomain, _rs.OrdinalDomain, "sorted_values")


@register_serializable(type_id="categorical_domain")
class CategoricalDomain(_rs.CategoricalDomain, WrappedFamilySerializable):
    """Finite, unordered set of admissible category values.

    Categories are stored as a strict-unique tuple in the constraint
    members' canonical order: by kind (``bool``, ``frozenset``,
    ``Identifier``, ``int``, ``str``, ``tuple``, then other ``Serializable``
    values), then by value, identifiers by id, tuples and frozen sets
    element by element. Native ``frozenset`` storage
    would collapse values that compare ``==`` but are distinct kinds
    (``True`` and ``1``).
    """

    _WIRE_FAMILY: ClassVar[str | None] = "param_domain"

    __slots__ = ("categories",)

    def __init__(self, categories: Sequence[CategoricalValue]) -> None:
        copy_native_attributes(
            self, CategoricalDomain, _rs.CategoricalDomain, "categories"
        )


@register_serializable(type_id="permutation_domain")
class PermutationDomain(_rs.PermutationDomain, WrappedFamilySerializable):
    """Admissible permutations of a fixed, ordered set of members.

    A NaN member is refused: it is unequal to itself, so no permutation
    could place it and the domain would admit no value at all.
    """

    _WIRE_FAMILY: ClassVar[str | None] = "param_domain"

    __slots__ = ("ordered_members",)

    def __init__(self, ordered_members: Sequence[PermutationMemberValue]) -> None:
        copy_native_attributes(
            self, PermutationDomain, _rs.PermutationDomain, "ordered_members"
        )


# The kinds are registered, not derived: `ParamDomain`'s bases carry an
# instance layout a Rust-backed class cannot share, as for `Constraint`.
for _kind in (
    IntegerDomain,
    RealDomain,
    IntervalIntegerDomain,
    OrdinalDomain,
    CategoricalDomain,
    PermutationDomain,
):
    ParamDomain.register(_kind)
    FrozenMixin.register(_kind)
del _kind


def build_ordinal_domain(values: Sequence[OrdinalValue]) -> OrdinalDomain:
    """Validate ``values`` and build a sorted :class:`OrdinalDomain`.

    Args:
        values: The admissible ordinal values; must be non-empty, unique,
            free of NaN, and mutually comparable.

    Returns:
        The constructed domain.

    Raises:
        ParamError: If ``values`` is empty, contains duplicates, or
            contains NaN.
        TypeError: If a value is not ordinal or values are not mutually
            comparable.

    """
    return OrdinalDomain(tuple(values))


def build_categorical_domain(
    categories: Sequence[CategoricalValue],
) -> CategoricalDomain:
    """Validate ``categories`` and build a :class:`CategoricalDomain`.

    Args:
        categories: The admissible categories; must be non-empty and unique.

    Returns:
        The constructed domain.

    Raises:
        ParamError: If ``categories`` is empty or contains duplicates.
        TypeError: If a category is not a categorical value.

    """
    return CategoricalDomain(tuple(categories))


def build_permutation_domain(
    members: Sequence[PermutationMemberValue],
) -> PermutationDomain:
    """Validate ``members`` and build a :class:`PermutationDomain`.

    Args:
        members: The ordered permutation members; must be non-empty,
            unique, and free of NaN.

    Returns:
        The constructed domain.

    Raises:
        ParamError: If ``members`` is empty, contains duplicates, or
            contains NaN.
        TypeError: If a member is not a permutation member value.

    """
    return PermutationDomain(tuple(members))
