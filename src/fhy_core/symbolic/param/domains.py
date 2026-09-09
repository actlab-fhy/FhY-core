"""Value-domain strategies for parameters.

A :class:`ParamDomain` captures everything that varies between kinds of
parameter: admissibility, constraint validation, implied constraints, subset
semantics, structural equivalence, and rendering. A single
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
permutation domains). Cross-space and cross-family queries return ``False``.
"""

import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from fhy_core.identifier import Identifier
from fhy_core.logger import get_logger
from fhy_core.pass_infrastructure import PassExecutionError
from fhy_core.serialization import (
    FieldCodec,
    SerializedValue,
    WrappedFamilySerializable,
    make_field_codec,
    register_serializable,
)
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintBindings,
    ConstraintError,
    ConstraintOutcome,
    ConstraintSystem,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
    create_constraint_system,
    does_member_lift_to_expression,
)
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.symbolic.symbol_type import SymbolType
from fhy_core.traits import FrozenMixin, StructuralEquivalence
from fhy_core.utils import format_comma_separated_list, is_strict_int
from fhy_core.utils.override import override

from .values import (
    CategoricalValue,
    OrdinalValue,
    ParamError,
    PermutationMemberValue,
    deserialize_wrapped_leaf_values,
    does_collection_contain_param_value,
    is_categorical_value,
    is_ordinal_value,
    is_permutation_member_value,
    is_sequence_unique_without_set,
    is_sorted_sequence_unique,
    serialize_wrapped_leaf_value,
)

__all__ = [
    "CategoricalDomain",
    "IntegerDomain",
    "IntervalIntegerDomain",
    "OrdinalDomain",
    "ParamDomain",
    "PermutationDomain",
    "RealDomain",
    "compute_constraint_implication_subset",
]

_LOGGER = get_logger(__name__)


def are_all_constraints_satisfied(
    constraints: Sequence[Constraint], variable: Identifier, value: Any
) -> bool:
    """Return whether ``value`` bound to ``variable`` satisfies every constraint."""
    return all(
        constraint.is_satisfied_with_bindings({variable: value})
        for constraint in constraints
    )


def _is_value_valid_for(
    domain: "ParamDomain",
    constraints: Sequence[Constraint],
    variable: Identifier,
    value: Any,
) -> bool:
    return domain.is_value_admissible(value) and are_all_constraints_satisfied(
        constraints, variable, value
    )


def _compute_numeric_in_set_candidates(constraints: Sequence[Constraint]) -> list[Any]:
    """Return the type-strict intersection of every ``InSetConstraint``'s members.

    Assumes ``constraints`` contains at least one ``InSetConstraint``. Starts
    from the first one's members, intersects with every subsequent
    ``InSetConstraint``'s members, then removes every ``NotInSetConstraint``'s
    members, all under type-strict equality.
    """
    in_set_constraints = [c for c in constraints if isinstance(c, InSetConstraint)]
    not_in_set_constraints = [
        c for c in constraints if isinstance(c, NotInSetConstraint)
    ]
    candidates = list(in_set_constraints[0].members)
    for in_set_constraint in in_set_constraints[1:]:
        candidates = [
            candidate
            for candidate in candidates
            if does_collection_contain_param_value(in_set_constraint.members, candidate)
        ]
    for not_in_set_constraint in not_in_set_constraints:
        candidates = [
            candidate
            for candidate in candidates
            if not does_collection_contain_param_value(
                not_in_set_constraint.members, candidate
            )
        ]
    return candidates


def _build_equation_constraint_system(
    constraints: Sequence[Constraint],
) -> ConstraintSystem:
    """Build a system from every ``EquationConstraint`` member of ``constraints``."""
    return create_constraint_system(
        *(c for c in constraints if isinstance(c, EquationConstraint))
    )


def evaluate_system_outcome(
    system: ConstraintSystem, bindings: ConstraintBindings
) -> ConstraintOutcome:
    """Decide ``system`` under ``bindings``, degrading on an expression-pass failure.

    Evaluation lowers through the SymPy bridge, which is not total: a
    constraint it cannot lower or lift raises ``PassExecutionError``.
    That is the backend failing to answer rather than the parameter being
    invalid, so it degrades to ``UNDECIDED`` (logged at ``WARNING``) the
    way every other undecidable outcome here does. Every parameter-level
    entry point returns ``bool``, so a bridge failure must not escape one
    as an exception.

    Args:
        system: Constraints to decide.
        bindings: Values for the identifiers the constraints reference.

    Returns:
        The system's outcome, or ``UNDECIDED`` when the bridge failed.

    """
    try:
        return system.evaluate_with_bindings(bindings)
    except PassExecutionError:
        _LOGGER.warning(
            "evaluate_system_outcome: the expression bridge could not evaluate "
            "%r under bindings for %s; reporting UNDECIDED.",
            system,
            format_comma_separated_list(tuple(bindings)) or "no identifiers",
        )
        return ConstraintOutcome.UNDECIDED


def _enumerate_feasible_in_set_candidates(
    domain: "ParamDomain", constraints: Sequence[Constraint], variable: Identifier
) -> list[Any]:
    """Return the in-set candidates not disproven by the domain or equation constraints.

    A candidate is included when it is domain-admissible and its outcome
    against the conjunction of ``constraints``'s equation constraints is
    ``SATISFIED`` or ``UNDECIDED`` (the documented optimistic default for
    an undecided candidate, e.g. one a dependent constraint leaves
    unresolved, logged at ``WARNING``); a ``VIOLATED`` candidate is
    excluded. Assumes ``constraints`` contains at least one
    ``InSetConstraint``.

    """
    equation_system = _build_equation_constraint_system(constraints)
    feasible: list[Any] = []
    for candidate in _compute_numeric_in_set_candidates(constraints):
        if not domain.is_value_admissible(candidate):
            continue
        outcome = evaluate_system_outcome(equation_system, {variable: candidate})
        if outcome is ConstraintOutcome.VIOLATED:
            continue
        if outcome is ConstraintOutcome.UNDECIDED:
            _LOGGER.warning(
                "_enumerate_feasible_in_set_candidates: equation constraints "
                "could not decide candidate %r for variable %r; optimistically "
                "treating it as feasible.",
                candidate,
                variable,
            )
        feasible.append(candidate)
    return feasible


def _is_candidate_accepted_by_other_side(
    other_domain: "ParamDomain",
    other_constraints: Sequence[Constraint],
    other_variable: Identifier,
    candidate: Any,
) -> bool:
    """Return whether ``other``'s domain and constraints admit ``candidate``.

    Set-constraint membership checks are type-strict. A ``VIOLATED``
    equation-constraint outcome rejects the candidate; ``UNDECIDED``
    follows this module's optimistic convention and is treated as
    accepted (logged at ``WARNING``).

    """
    if not other_domain.is_value_admissible(candidate):
        return False
    for constraint in other_constraints:
        if isinstance(constraint, InSetConstraint):
            if not does_collection_contain_param_value(constraint.members, candidate):
                return False
        elif isinstance(constraint, NotInSetConstraint):
            if does_collection_contain_param_value(constraint.members, candidate):
                return False
    equation_system = _build_equation_constraint_system(other_constraints)
    outcome = evaluate_system_outcome(equation_system, {other_variable: candidate})
    if outcome is ConstraintOutcome.UNDECIDED:
        _LOGGER.warning(
            "_is_candidate_accepted_by_other_side: equation constraints could "
            "not decide candidate %r for variable %r; optimistically treating "
            "it as accepted.",
            candidate,
            other_variable,
        )
    return outcome is not ConstraintOutcome.VIOLATED


def _split_not_in_set_members_by_liftability(
    constraint: NotInSetConstraint,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Split ``constraint``'s members into liftable and non-liftable groups.

    A member is liftable when a singleton ``NotInSetConstraint`` built
    from it alone converts to an expression without raising.

    Args:
        constraint: The not-in-set constraint whose members to split.

    Returns:
        A ``(liftable, excluded)`` pair of member tuples, in the order
        ``constraint.members`` iterates them.

    """
    liftable: list[Any] = []
    excluded: list[Any] = []
    for member in constraint.members:
        if does_member_lift_to_expression(member):
            liftable.append(member)
        else:
            excluded.append(member)
    return tuple(liftable), tuple(excluded)


def _log_set_constraint_scope_exclusion(
    constraint: InSetConstraint | NotInSetConstraint, variable: Identifier
) -> None:
    """Log a WARNING that constraint is excluded for being scoped elsewhere."""
    _LOGGER.warning(
        "_build_screened_constraint_system: excluding %r from the "
        "screened system for variable %r; it is scoped to %r "
        "instead.",
        constraint,
        variable,
        constraint.variable,
    )


def _screen_equation_constraint(
    constraint: EquationConstraint, variable: Identifier
) -> EquationConstraint | None:
    """Return constraint if its scope is exactly ``{variable}``, else None.

    A dependent constraint whose scope reaches beyond ``variable`` is
    excluded (logged at ``WARNING``) rather than raising, so the caller
    degrades to the optimistic default instead of crashing on a foreign
    identifier.

    """
    if constraint.get_free_identifiers() == frozenset((variable,)):
        return constraint
    _LOGGER.warning(
        "_build_screened_constraint_system: excluding dependent "
        "constraint %r from the screened system for variable %r; "
        "its scope %r reaches beyond %r.",
        constraint,
        variable,
        constraint.get_free_identifiers(),
        variable,
    )
    return None


def _screen_in_set_constraint(
    constraint: InSetConstraint, variable: Identifier
) -> InSetConstraint | None:
    """Return constraint if it is scoped to ``variable`` and every member lifts.

    An ``InSetConstraint``'s members combine with ``OR``, so narrowing
    around a member that cannot lift (``convert_to_expression`` raises
    ``ConstraintError``, e.g. for a string or container member) would
    only shrink the admissible set; the whole constraint is excluded
    instead (logged at ``WARNING``) whenever it is scoped elsewhere or
    any member fails to lift.

    """
    if constraint.variable != variable:
        _log_set_constraint_scope_exclusion(constraint, variable)
        return None
    try:
        constraint.convert_to_expression()
    except ConstraintError as error:
        _LOGGER.warning(
            "_build_screened_constraint_system: excluding %r for "
            "variable %r; it does not lift to an expression (%s).",
            constraint,
            variable,
            error,
        )
        return None
    return constraint


def _screen_not_in_set_constraint(
    constraint: NotInSetConstraint, variable: Identifier
) -> NotInSetConstraint | None:
    """Return constraint narrowed to its liftable members, or None.

    A ``NotInSetConstraint``'s members combine with ``AND``, so dropping
    a non-liftable member only widens the admissible set: the constraint
    is narrowed to its liftable members (logged at ``WARNING`` when any
    member is excluded), and dropped entirely (also logged at
    ``WARNING``) when it is scoped elsewhere or no member lifts.

    """
    if constraint.variable != variable:
        _log_set_constraint_scope_exclusion(constraint, variable)
        return None
    liftable, excluded = _split_not_in_set_members_by_liftability(constraint)
    if not liftable:
        _LOGGER.warning(
            "_build_screened_constraint_system: excluding %r for "
            "variable %r; none of its members lift to an expression.",
            constraint,
            variable,
        )
        return None
    if excluded:
        _LOGGER.warning(
            "_build_screened_constraint_system: narrowing %r for "
            "variable %r to its liftable member(s) %r; excluded "
            "non-liftable member(s) %r.",
            constraint,
            variable,
            liftable,
            excluded,
        )
    return NotInSetConstraint(variable, liftable)


def _build_screened_constraint_system(
    constraints: Sequence[Constraint], variable: Identifier
) -> ConstraintSystem:
    """Build the decidable-without-enumeration constraint system for ``variable``.

    Keeps an ``EquationConstraint`` scoped to exactly ``{variable}`` (see
    ``_screen_equation_constraint``), an ``InSetConstraint`` scoped to
    ``variable`` with every member liftable (see
    ``_screen_in_set_constraint``), and a ``NotInSetConstraint`` scoped
    to ``variable`` with at least one liftable member, narrowed to those
    members (see ``_screen_not_in_set_constraint``). Every exclusion and
    narrowing is logged at ``WARNING``, so the caller degrades to the
    documented optimistic default instead of crashing or silently losing
    constraints.

    """
    system, _ = _build_screened_constraint_system_with_fidelity(constraints, variable)
    return system


def _build_screened_constraint_system_with_fidelity(
    constraints: Sequence[Constraint], variable: Identifier
) -> tuple[ConstraintSystem, bool]:
    """Build the screened system and report whether it lost nothing.

    The fidelity flag says the screened system denotes exactly the same
    value set as ``constraints``: no constraint was excluded and none was
    narrowed. A caller may only read a decided satisfying assignment as a
    genuine witness about the original constraints when this holds, since
    screening can only weaken a system, and a weakened system admits
    values the original forbids.

    Args:
        constraints: Constraints to screen.
        variable: Variable the system is built for.

    Returns:
        The screened system paired with whether it is exact.

    """
    members: list[Constraint] = []
    is_exact = True
    for constraint in constraints:
        screened: Constraint | None
        if isinstance(constraint, EquationConstraint):
            screened = _screen_equation_constraint(constraint, variable)
        elif isinstance(constraint, InSetConstraint):
            screened = _screen_in_set_constraint(constraint, variable)
        elif isinstance(constraint, NotInSetConstraint):
            screened = _screen_not_in_set_constraint(constraint, variable)
        else:
            screened = None
        if screened is None:
            is_exact = False
            continue
        if screened is not constraint:
            is_exact = False
        members.append(screened)
    return create_constraint_system(*members), is_exact


def _rename_constraint_variable(
    constraint: Constraint, old_variable: Identifier, new_variable: Identifier
) -> Constraint:
    """Return ``constraint`` with ``old_variable`` renamed to ``new_variable``.

    Handles the two constraint shapes ``_build_screened_constraint_system``
    produces: an ``EquationConstraint``'s expression is substituted (a
    no-op wherever ``old_variable`` is not actually free in it), and an
    ``InSetConstraint``/``NotInSetConstraint``'s ``variable`` field is
    replaced after confirming it is actually ``old_variable``, since
    unlike substitution, replacing that field is not self-correcting.

    Args:
        constraint: The constraint to rename.
        old_variable: The identifier expected to be renamed.
        new_variable: The identifier to rename it to.

    Returns:
        An equivalent constraint scoped to ``new_variable`` in place of
        ``old_variable``.

    Raises:
        ConstraintError: If ``constraint`` is an ``InSetConstraint``/
            ``NotInSetConstraint`` not scoped to ``old_variable``, or if
            ``constraint`` is neither an ``EquationConstraint`` nor a set
            constraint.

    """
    if isinstance(constraint, EquationConstraint):
        return EquationConstraint(
            constraint.expression.substitute(
                {old_variable: IdentifierExpression(new_variable)}
            )
        )
    if isinstance(constraint, (InSetConstraint, NotInSetConstraint)):
        if constraint.variable != old_variable:
            raise ConstraintError(
                f"Cannot rename {constraint!r} from {old_variable!r} to "
                f"{new_variable!r}: it is scoped to {constraint.variable!r}, "
                f"not {old_variable!r}."
            )
        return type(constraint)(new_variable, constraint.values)
    raise ConstraintError(  # pragma: no cover
        f"Cannot rename an unexpected constraint kind: {type(constraint).__name__}."
    )


def _rename_constraint_system_variable(
    system: ConstraintSystem, old_variable: Identifier, new_variable: Identifier
) -> ConstraintSystem:
    """Return a system equivalent to ``system`` with its variable renamed."""
    return create_constraint_system(
        *(
            _rename_constraint_variable(constraint, old_variable, new_variable)
            for constraint in system.constraints
        )
    )


def _does_own_admit_a_value_outside(
    own_domain: "ParamDomain",
    own_constraints: Sequence[Constraint],
    own_variable: Identifier,
    permitted_values: Sequence[Any],
    symbol_type: SymbolType,
) -> bool:
    """Return whether ``own`` provably admits a value outside ``permitted_values``.

    Decides only the negative direction of the subset relation, and only
    from proof. Requires the screened system to be exact, since a
    weakened system admits values the original forbids, and requires the
    solver to decide ``SATISFIED``, since an undecided outcome is not a
    witness. Both conditions failing simply means no counterexample was
    proven, not that none exists.

    Args:
        own_domain: Domain of the candidate subset parameter.
        own_constraints: Constraints of the candidate subset parameter.
        own_variable: Variable of the candidate subset parameter.
        permitted_values: Values the other side admits, over-approximated.
        symbol_type: Z3 sort used to reason about the variable.

    Returns:
        True only when a value satisfying every one of ``own``'s
        constraints provably lies outside ``permitted_values``.

    """
    own_system, is_exact = _build_screened_constraint_system_with_fidelity(
        own_constraints, own_variable
    )
    if not is_exact:
        return False
    common_variable = Identifier("var")
    renamed = _rename_constraint_system_variable(
        own_system, own_variable, common_variable
    )
    try:
        exclusion = NotInSetConstraint(common_variable, tuple(permitted_values))
        exclusion.convert_to_expression()
    except ConstraintError:
        # The permitted values do not lift to an expression, so the
        # exclusion cannot be posed to the solver at all.
        return False
    witness_system = create_constraint_system(*renamed.constraints, exclusion)
    outcome = witness_system.check_satisfiability({common_variable: symbol_type})
    if outcome is not ConstraintOutcome.SATISFIED:
        return False
    _LOGGER.debug(
        "_does_own_admit_a_value_outside: %r provably admits a value outside "
        "the %d value(s) the other side permits.",
        own_variable,
        len(permitted_values),
    )
    return True


def compute_constraint_implication_subset(
    own_domain: "ParamDomain",
    own_constraints: Sequence[Constraint],
    own_variable: Identifier,
    other_domain: "ParamDomain",
    other_constraints: Sequence[Constraint],
    other_variable: Identifier,
    symbol_type: SymbolType,
) -> bool:
    """Return whether ``own_constraints``'s admissible set is a subset of ``other``'s.

    When ``own_constraints`` contains an ``InSetConstraint``, the
    admissible values are finite: every surviving candidate (see
    ``_enumerate_feasible_in_set_candidates``) must be accepted by
    ``other_domain``/``other_constraints`` (type-strict set membership,
    domain admissibility, and equation constraints decided per candidate).

    When only ``other_constraints`` is finite, its admissible values are
    enumerated and ``own`` is asked, through the solver, whether it
    provably admits a value outside them (see
    ``_does_own_admit_a_value_outside``); such a value is a genuine
    counterexample, since the enumeration over-approximates what the
    other side admits, and the relation is decided ``False``.

    Otherwise the two sides' variable-only constraint systems (see
    ``_build_screened_constraint_system``) are renamed onto one shared
    identifier and decided via ``ConstraintSystem.check_implication`` over
    ``symbol_type``. An outcome the checks cannot disprove -- a solver
    ``UNDECIDED`` result, or a constraint excluded for reaching outside
    either parameter's own variable -- is treated as "not a
    counterexample" (each logged at ``WARNING``), so the subset relation
    holds; a ``True`` result therefore means "not disproven", not
    "proven". A ``False`` result from this branch is likewise not a
    proof, since screening weakens the antecedent and can manufacture a
    counterexample; only the two enumeration-backed branches decide
    ``False`` from proof.

    Args:
        own_domain: Domain of the candidate subset parameter.
        own_constraints: Constraints of the candidate subset parameter.
        own_variable: Variable of the candidate subset parameter.
        other_domain: Domain of the candidate superset parameter.
        other_constraints: Constraints of the candidate superset parameter.
        other_variable: Variable of the candidate superset parameter.
        symbol_type: The Z3 sort used to reason about the shared variable.

    Returns:
        Whether the subset relation holds.

    """
    if any(isinstance(c, InSetConstraint) for c in own_constraints):
        own_candidates = _enumerate_feasible_in_set_candidates(
            own_domain, own_constraints, own_variable
        )
        return all(
            _is_candidate_accepted_by_other_side(
                other_domain, other_constraints, other_variable, candidate
            )
            for candidate in own_candidates
        )
    if any(isinstance(c, InSetConstraint) for c in other_constraints):
        other_candidates = _enumerate_feasible_in_set_candidates(
            other_domain, other_constraints, other_variable
        )
        if _does_own_admit_a_value_outside(
            own_domain, own_constraints, own_variable, other_candidates, symbol_type
        ):
            return False
    common_variable = Identifier("var")
    own_system = _rename_constraint_system_variable(
        _build_screened_constraint_system(own_constraints, own_variable),
        own_variable,
        common_variable,
    )
    other_system = _rename_constraint_system_variable(
        _build_screened_constraint_system(other_constraints, other_variable),
        other_variable,
        common_variable,
    )
    outcome = own_system.check_implication(other_system, {common_variable: symbol_type})
    if outcome is ConstraintOutcome.UNDECIDED:
        _LOGGER.warning(
            "compute_constraint_implication_subset: the solver could not "
            "decide whether %r implies %r; optimistically treating %r as a "
            "subset of %r.",
            own_variable,
            other_variable,
            own_variable,
            other_variable,
        )
    return outcome is not ConstraintOutcome.VIOLATED


class ParamDomain(WrappedFamilySerializable, FrozenMixin, StructuralEquivalence, ABC):
    """Sum-type family base describing the value space of a parameter kind.

    Concrete domains are ``@register_serializable @dataclass(frozen=True,
    eq=False)`` leaves of this family; serialization is derived by the family
    pattern (a wrapped ``{"__type__": ..., "__data__": {...}}`` envelope keyed by
    each leaf's ``type_id``). A :class:`~fhy_core.symbolic.param.core.Param` composes
    exactly one domain and delegates all kind-specific behavior to it.
    """

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
        """Return the canonical form of ``value`` used for storage and checks."""

    @abstractmethod
    def validate_constraint(self, constraint: Constraint, variable: Identifier) -> None:
        """Raise if ``constraint`` is not permitted for this domain."""

    @abstractmethod
    def get_implied_constraints(self, variable: Identifier) -> tuple[Constraint, ...]:
        """Return constraints this domain imposes implicitly on ``variable``."""

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
    ) -> bool:
        """Return whether this domain's constrained set is a subset of ``other``'s."""

    @abstractmethod
    def has_feasible_value(
        self, constraints: Sequence[Constraint], variable: Identifier
    ) -> bool:
        """Return whether some admissible value satisfies every constraint."""

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


def _encode_wrapped_leaf_sequence(values: Any) -> SerializedValue:
    """Encode a sequence of leaf values into a list of wrapped registry dicts."""
    return [serialize_wrapped_leaf_value(value) for value in values]


def _make_wrapped_leaf_sequence_codec(
    value_type_guard: Any, expected_description: str
) -> FieldCodec:
    """Build a per-field codec for a finite domain's wrapped-leaf value sequence.

    The value field holds a union leaf type that the engine cannot infer, so the
    finite domains supply this codec explicitly. Encoding wraps each element with
    :func:`serialize_wrapped_leaf_value`; decoding validates and unwraps the list
    with :func:`deserialize_wrapped_leaf_values`.
    """

    def _decode(data: Any) -> list[Any]:
        if not isinstance(data, list):
            raise TypeError("Expected a list of wrapped leaf values.")
        return deserialize_wrapped_leaf_values(
            ParamDomain, data, value_type_guard, expected_description
        )

    return make_field_codec(_encode_wrapped_leaf_sequence, _decode)


_ORDINAL_VALUES_CODEC: FieldCodec = _make_wrapped_leaf_sequence_codec(
    is_ordinal_value, "a list of orderable serializable values or primitive values"
)
_CATEGORICAL_VALUES_CODEC: FieldCodec = _make_wrapped_leaf_sequence_codec(
    is_categorical_value, "a list of equal serializable values or primitive values"
)
_PERMUTATION_MEMBERS_CODEC: FieldCodec = _make_wrapped_leaf_sequence_codec(
    is_permutation_member_value,
    "a list of equal serializable values or primitive values",
)


# ---------------------------------------------------------------------------
# Numeric domains
# ---------------------------------------------------------------------------


def _is_numeric_value_set_subset(
    own_symbol_type: SymbolType | None, other: ParamDomain
) -> bool:
    return own_symbol_type is not None and other.symbol_type == own_symbol_type


def _build_non_negative_implied_constraints(
    variable: Identifier, *, non_negative: bool, zero_included: bool
) -> tuple[Constraint, ...]:
    """Return the sign bound a non-negative integer domain implies.

    Shared by the integer domains, whose implied constraints differ only
    in the flags they hold rather than in how those flags map to a bound.

    Args:
        variable: Identifier the bound constrains.
        non_negative: Whether the domain admits only non-negative values.
        zero_included: Whether the domain admits zero, given it is
            non-negative.

    Returns:
        A single lower-bound constraint for a non-negative domain, and an
        empty tuple otherwise.

    """
    if not non_negative:
        return ()
    variable_expression = IdentifierExpression(variable)
    if zero_included:
        return (EquationConstraint(variable_expression >= 0),)
    return (EquationConstraint(variable_expression > 0),)


def _compute_numeric_feasibility_subset(
    own: ParamDomain,
    own_constraints: Sequence[Constraint],
    own_variable: Identifier,
    other: ParamDomain,
    other_constraints: Sequence[Constraint],
    other_variable: Identifier,
) -> bool:
    if own.symbol_type is None or other.symbol_type != own.symbol_type:
        return False
    return compute_constraint_implication_subset(
        own,
        own_constraints,
        own_variable,
        other,
        other_constraints,
        other_variable,
        own.symbol_type,
    )


def _numeric_has_feasible_value(
    domain: ParamDomain,
    symbol_type: SymbolType,
    constraints: Sequence[Constraint],
    variable: Identifier,
) -> bool:
    """Return whether some domain-admissible value satisfies every constraint.

    Routes through enumeration when an ``InSetConstraint`` makes the
    admissible values finite (see
    ``_enumerate_feasible_in_set_candidates``); otherwise decides the
    screened ``ConstraintSystem`` built from ``variable``-only equation
    constraints and ``NotInSetConstraint``s narrowed to their liftable
    members (see ``_build_screened_constraint_system``), with dependent
    constraints, foreign-scoped constraints, and a solver ``UNDECIDED``
    result all degrading to the documented optimistic default (``True``,
    logged at ``WARNING``).

    """
    if any(isinstance(c, InSetConstraint) for c in constraints):
        return bool(
            _enumerate_feasible_in_set_candidates(domain, constraints, variable)
        )
    system = _build_screened_constraint_system(constraints, variable)
    outcome = system.check_satisfiability({variable: symbol_type})
    if outcome is ConstraintOutcome.UNDECIDED:
        _LOGGER.warning(
            "_numeric_has_feasible_value: the solver could not decide "
            "satisfiability for variable %r; optimistically treating it as "
            "feasible.",
            variable,
        )
    return outcome is not ConstraintOutcome.VIOLATED


@register_serializable(type_id="integer_domain")
@dataclass(frozen=True, eq=False)
class IntegerDomain(ParamDomain):
    """Integer-valued domain, optionally restricted to the natural numbers.

    ``non_negative`` does not change admissibility (any strict integer is
    admissible); it adds an implied ``>= 0`` constraint, or ``> 0`` when
    ``zero_included`` is ``False``.
    """

    non_negative: bool = False
    zero_included: bool = True

    def __post_init__(self) -> None:
        # ``zero_included`` is only meaningful for a natural-number domain. When
        # the domain is not restricted to non-negatives the field is inert, so
        # canonicalize it: two otherwise-equal domains then never differ solely
        # in this dead field.
        if not self.non_negative:
            object.__setattr__(self, "zero_included", True)

    @property
    @override
    def symbol_type(self) -> SymbolType | None:
        return SymbolType.INT

    @override
    def is_value_admissible(self, value: Any) -> bool:
        return is_strict_int(value)

    @override
    def normalize_value(self, value: Any) -> Any:
        return value

    @override
    def validate_constraint(self, constraint: Constraint, variable: Identifier) -> None:
        del constraint, variable

    @override
    def get_implied_constraints(self, variable: Identifier) -> tuple[Constraint, ...]:
        return _build_non_negative_implied_constraints(
            variable,
            non_negative=self.non_negative,
            zero_included=self.zero_included,
        )

    @override
    def is_value_set_subset(self, other: ParamDomain) -> bool:
        return _is_numeric_value_set_subset(self.symbol_type, other)

    @override
    def compute_feasibility_subset(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
    ) -> bool:
        return _compute_numeric_feasibility_subset(
            self,
            own_constraints,
            own_variable,
            other,
            other_constraints,
            other_variable,
        )

    @override
    def has_feasible_value(
        self, constraints: Sequence[Constraint], variable: Identifier
    ) -> bool:
        return _numeric_has_feasible_value(self, SymbolType.INT, constraints, variable)

    @override
    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, IntegerDomain)
            and self.non_negative == other.non_negative
            and self.zero_included == other.zero_included
        )

    @override
    def render_set_string(self) -> str:
        return "Z"

    @override
    def render_set_repr(self) -> str:
        return ""


@register_serializable(type_id="real_domain")
@dataclass(frozen=True, eq=False)
class RealDomain(ParamDomain):
    """Real-valued domain (floats and float-parseable strings)."""

    @property
    @override
    def symbol_type(self) -> SymbolType | None:
        return SymbolType.REAL

    @override
    def is_value_admissible(self, value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, float):
            return True
        if isinstance(value, str):
            try:
                float(value)
            except ValueError:
                return False
            return True
        return False

    @override
    def normalize_value(self, value: Any) -> Any:
        return value

    @override
    def validate_constraint(self, constraint: Constraint, variable: Identifier) -> None:
        del constraint, variable

    @override
    def get_implied_constraints(self, variable: Identifier) -> tuple[Constraint, ...]:
        del variable
        return ()

    @override
    def is_value_set_subset(self, other: ParamDomain) -> bool:
        return _is_numeric_value_set_subset(self.symbol_type, other)

    @override
    def compute_feasibility_subset(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
    ) -> bool:
        return _compute_numeric_feasibility_subset(
            self,
            own_constraints,
            own_variable,
            other,
            other_constraints,
            other_variable,
        )

    @override
    def has_feasible_value(
        self, constraints: Sequence[Constraint], variable: Identifier
    ) -> bool:
        return _numeric_has_feasible_value(self, SymbolType.REAL, constraints, variable)

    @override
    def is_structurally_equivalent(self, other: object) -> bool:
        return isinstance(other, RealDomain)

    @override
    def render_set_string(self) -> str:
        return "R"

    @override
    def render_set_repr(self) -> str:
        return ""


def is_bound_expression(expression: Expression) -> bool:
    """Return whether ``expression`` is an integer bound of the form ``x <cmp> k``."""
    if not isinstance(expression, BinaryExpression):
        return False
    if expression.operation not in (
        BinaryOperation.GREATER_EQUAL,
        BinaryOperation.GREATER,
        BinaryOperation.LESS_EQUAL,
        BinaryOperation.LESS,
    ):
        return False
    if not (
        (
            isinstance(expression.left, IdentifierExpression)
            or isinstance(expression.right, IdentifierExpression)
        )
        and (
            isinstance(expression.left, LiteralExpression)
            or isinstance(expression.right, LiteralExpression)
        )
    ):
        return False
    if isinstance(expression.left, LiteralExpression):
        literal_expression = expression.left
    elif isinstance(expression.right, LiteralExpression):
        literal_expression = expression.right
    else:  # pragma: no cover
        raise RuntimeError("Somehow failed to find LiteralExpression in bound.")
    return isinstance(literal_expression.value, int)


@register_serializable(type_id="interval_integer_domain")
@dataclass(frozen=True, eq=False)
class IntervalIntegerDomain(ParamDomain):
    """Integer domain whose parameters carry their interval as bound constraints.

    Admissibility accepts any strict integer; the interval is expressed through
    the composing parameter's bound constraints, not through this domain's
    admissibility. Only :class:`~fhy_core.symbolic.constraint.EquationConstraint` bound
    expressions are permitted, enabling interval arithmetic on the composing
    parameter. ``prefer_inclusive`` selects how arithmetic results render their
    bounds. ``non_negative`` adds the natural-number implied constraint.
    """

    prefer_inclusive: bool = True
    non_negative: bool = False
    zero_included: bool = True

    def __post_init__(self) -> None:
        # See ``IntegerDomain.__post_init__``: canonicalize the inert
        # ``zero_included`` field when the domain is not restricted to
        # non-negatives.
        if not self.non_negative:
            object.__setattr__(self, "zero_included", True)

    @property
    @override
    def symbol_type(self) -> SymbolType | None:
        return SymbolType.INT

    @override
    def is_value_admissible(self, value: Any) -> bool:
        return is_strict_int(value)

    @override
    def normalize_value(self, value: Any) -> Any:
        return value

    @override
    def validate_constraint(self, constraint: Constraint, variable: Identifier) -> None:
        del variable
        if not isinstance(constraint, EquationConstraint):
            raise TypeError(
                "Interval integer parameters only support EquationConstraint "
                "constraints."
            )
        if not is_bound_expression(constraint.convert_to_expression()):
            raise ParamError(
                "Interval integer parameters only support bound expressions of "
                'the form "x >= k", "x > k", "x <= k", or "x < k" where k is an '
                "integer."
            )

    @override
    def get_implied_constraints(self, variable: Identifier) -> tuple[Constraint, ...]:
        return _build_non_negative_implied_constraints(
            variable,
            non_negative=self.non_negative,
            zero_included=self.zero_included,
        )

    @override
    def is_value_set_subset(self, other: ParamDomain) -> bool:
        return _is_numeric_value_set_subset(self.symbol_type, other)

    @override
    def compute_feasibility_subset(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
    ) -> bool:
        return _compute_numeric_feasibility_subset(
            self,
            own_constraints,
            own_variable,
            other,
            other_constraints,
            other_variable,
        )

    @override
    def has_feasible_value(
        self, constraints: Sequence[Constraint], variable: Identifier
    ) -> bool:
        return _numeric_has_feasible_value(self, SymbolType.INT, constraints, variable)

    @override
    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, IntervalIntegerDomain)
            and self.prefer_inclusive == other.prefer_inclusive
            and self.non_negative == other.non_negative
            and self.zero_included == other.zero_included
        )

    @override
    def render_set_string(self) -> str:
        return "Z"

    @override
    def render_set_repr(self) -> str:
        return ""


# ---------------------------------------------------------------------------
# Finite-set domains
# ---------------------------------------------------------------------------


def _validate_finite_set_constraint(constraint: Constraint, kind: str) -> None:
    if not isinstance(constraint, (InSetConstraint, NotInSetConstraint)):
        raise ParamError(
            f"Only in-set and not-in-set constraints are allowed for {kind} parameters."
        )


@register_serializable(type_id="ordinal_domain")
@dataclass(frozen=True, eq=False)
class OrdinalDomain(ParamDomain):
    """Finite, totally-ordered set of admissible values."""

    sorted_values: tuple[OrdinalValue, ...] = field(
        metadata={"serialize_codec": _ORDINAL_VALUES_CODEC}
    )

    def __post_init__(self) -> None:
        values = tuple(self.sorted_values)
        if not values:
            raise ParamError("Values must be non-empty.")
        for value in values:
            if not is_ordinal_value(value):
                raise TypeError(
                    "Ordinal values must satisfy orderable semantics and be "
                    "serializable, or be primitive bool/int/float/str values."
                )
        try:
            canonical = tuple(sorted(values))
        except TypeError as exc:
            raise TypeError(
                "Ordinal values must be mutually comparable for sorting."
            ) from exc
        if not is_sorted_sequence_unique(canonical):
            raise ParamError("Values must be unique.")
        object.__setattr__(self, "sorted_values", canonical)

    @property
    @override
    def symbol_type(self) -> SymbolType | None:
        return None

    @override
    def is_value_admissible(self, value: Any) -> bool:
        return is_ordinal_value(value) and does_collection_contain_param_value(
            self.sorted_values, value
        )

    @override
    def normalize_value(self, value: Any) -> Any:
        return value

    @override
    def validate_constraint(self, constraint: Constraint, variable: Identifier) -> None:
        del variable
        _validate_finite_set_constraint(constraint, "ordinal")

    @override
    def get_implied_constraints(self, variable: Identifier) -> tuple[Constraint, ...]:
        del variable
        return ()

    @override
    def is_value_set_subset(self, other: ParamDomain) -> bool:
        if not isinstance(other, OrdinalDomain):
            return False
        return all(
            does_collection_contain_param_value(other.sorted_values, value)
            for value in self.sorted_values
        )

    @override
    def compute_feasibility_subset(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
    ) -> bool:
        if not isinstance(other, OrdinalDomain):
            return False
        for value in self.sorted_values:
            if not _is_value_valid_for(self, own_constraints, own_variable, value):
                continue
            if not _is_value_valid_for(other, other_constraints, other_variable, value):
                return False
        return True

    @override
    def has_feasible_value(
        self, constraints: Sequence[Constraint], variable: Identifier
    ) -> bool:
        return any(
            _is_value_valid_for(self, constraints, variable, value)
            for value in self.sorted_values
        )

    @override
    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, OrdinalDomain)
            and self.sorted_values == other.sorted_values
        )

    @override
    def render_set_string(self) -> str:
        return f"{{{format_comma_separated_list(self.sorted_values, str_func=str)}}}"

    @override
    def render_set_repr(self) -> str:
        return f"{{{format_comma_separated_list(self.sorted_values)}}}"

    @classmethod
    @override
    def construct_from_fields(cls, fields: dict[str, Any]) -> "OrdinalDomain":
        return build_ordinal_domain(fields["sorted_values"])


@register_serializable(type_id="categorical_domain")
@dataclass(frozen=True, eq=False)
class CategoricalDomain(ParamDomain):
    """Finite, unordered set of admissible category values.

    Categories are stored as a strict-unique, ``repr``-canonicalized tuple.
    Native ``frozenset`` storage would collapse values that compare ``==`` but
    are distinct kinds (``True`` and ``1``), so the tuple preserves them while
    keeping a deterministic order for serialization and rendering.
    """

    categories: tuple[CategoricalValue, ...] = field(
        metadata={"serialize_codec": _CATEGORICAL_VALUES_CODEC}
    )

    def __post_init__(self) -> None:
        values = tuple(self.categories)
        if not values:
            raise ParamError("Categories must be non-empty.")
        for category in values:
            if not is_categorical_value(category):
                raise TypeError(
                    "Categorical values must satisfy equal semantics and be "
                    "serializable, or be primitive bool/int/str values."
                )
        if not is_sequence_unique_without_set(values):
            raise ParamError("Values must be unique.")
        # Categories are unordered; canonicalize by ``repr`` for a deterministic
        # storage order (categorical values are not necessarily mutually
        # orderable).
        object.__setattr__(self, "categories", tuple(sorted(values, key=repr)))

    @property
    @override
    def symbol_type(self) -> SymbolType | None:
        return None

    @override
    def is_value_admissible(self, value: Any) -> bool:
        return is_categorical_value(value) and does_collection_contain_param_value(
            self.categories, value
        )

    @override
    def normalize_value(self, value: Any) -> Any:
        return value

    @override
    def validate_constraint(self, constraint: Constraint, variable: Identifier) -> None:
        del variable
        _validate_finite_set_constraint(constraint, "categorical")

    @override
    def get_implied_constraints(self, variable: Identifier) -> tuple[Constraint, ...]:
        del variable
        return ()

    @override
    def is_value_set_subset(self, other: ParamDomain) -> bool:
        if not isinstance(other, CategoricalDomain):
            return False
        return all(
            does_collection_contain_param_value(other.categories, category)
            for category in self.categories
        )

    @override
    def compute_feasibility_subset(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
    ) -> bool:
        if not isinstance(other, CategoricalDomain):
            return False
        for category in self.categories:
            if not _is_value_valid_for(self, own_constraints, own_variable, category):
                continue
            if not _is_value_valid_for(
                other, other_constraints, other_variable, category
            ):
                return False
        return True

    @override
    def has_feasible_value(
        self, constraints: Sequence[Constraint], variable: Identifier
    ) -> bool:
        return any(
            _is_value_valid_for(self, constraints, variable, category)
            for category in self.categories
        )

    @override
    def is_structurally_equivalent(self, other: object) -> bool:
        # Categories are unordered, so compare order-independently with the
        # strict value predicate. Native ``tuple ==`` would wrongly equate
        # ``(True,)`` and ``(1,)`` because ``True == 1``.
        if not isinstance(other, CategoricalDomain):
            return False
        if len(self.categories) != len(other.categories):
            return False
        return all(
            does_collection_contain_param_value(other.categories, category)
            for category in self.categories
        )

    @override
    def render_set_string(self) -> str:
        return f"{{{format_comma_separated_list(self.categories, str_func=str)}}}"

    @override
    def render_set_repr(self) -> str:
        return f"{{{format_comma_separated_list(self.categories)}}}"

    @classmethod
    @override
    def construct_from_fields(cls, fields: dict[str, Any]) -> "CategoricalDomain":
        return build_categorical_domain(tuple(fields["categories"]))


@register_serializable(type_id="permutation_domain")
@dataclass(frozen=True, eq=False)
class PermutationDomain(ParamDomain):
    """Admissible permutations of a fixed, ordered set of members."""

    ordered_members: tuple[PermutationMemberValue, ...] = field(
        metadata={"serialize_codec": _PERMUTATION_MEMBERS_CODEC}
    )

    def __post_init__(self) -> None:
        values = tuple(self.ordered_members)
        if not values:
            raise ParamError("Members must be non-empty.")
        for value in values:
            if not is_permutation_member_value(value):
                raise TypeError(
                    "Permutation members must satisfy equal semantics and be "
                    "serializable, or be primitive bool/int/float/str values."
                )
        if not is_sequence_unique_without_set(values):
            raise ParamError("Values must be unique.")
        object.__setattr__(self, "ordered_members", values)

    @property
    @override
    def symbol_type(self) -> SymbolType | None:
        return None

    @override
    def is_value_admissible(self, value: Any) -> bool:
        return (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
            and self._is_valid_permutation(value)
        )

    def _is_valid_permutation(self, value: Sequence[Any]) -> bool:
        return (
            all(
                is_permutation_member_value(element)
                and does_collection_contain_param_value(self.ordered_members, element)
                for element in value
            )
            and len(value) == len(self.ordered_members)
            and is_sequence_unique_without_set(value)
        )

    @override
    def normalize_value(self, value: Any) -> Any:
        return tuple(value)

    @override
    def validate_constraint(self, constraint: Constraint, variable: Identifier) -> None:
        del variable
        _validate_finite_set_constraint(constraint, "permutation")

    @override
    def get_implied_constraints(self, variable: Identifier) -> tuple[Constraint, ...]:
        del variable
        return ()

    @override
    def is_value_set_subset(self, other: ParamDomain) -> bool:
        if not isinstance(other, PermutationDomain):
            return False
        if len(self.ordered_members) != len(other.ordered_members):
            return False
        return all(
            does_collection_contain_param_value(other.ordered_members, member)
            for member in self.ordered_members
        )

    @override
    def compute_feasibility_subset(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
    ) -> bool:
        if not isinstance(other, PermutationDomain):
            return False
        if len(self.ordered_members) != len(other.ordered_members):
            return False
        for permutation in itertools.permutations(self.ordered_members):
            if not _is_value_valid_for(
                self, own_constraints, own_variable, permutation
            ):
                continue
            if not _is_value_valid_for(
                other, other_constraints, other_variable, permutation
            ):
                return False
        return True

    @override
    def has_feasible_value(
        self, constraints: Sequence[Constraint], variable: Identifier
    ) -> bool:
        return any(
            _is_value_valid_for(self, constraints, variable, permutation)
            for permutation in itertools.permutations(self.ordered_members)
        )

    @override
    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, PermutationDomain)
            and self.ordered_members == other.ordered_members
        )

    @override
    def render_set_string(self) -> str:
        return f"{{{format_comma_separated_list(self.ordered_members, str_func=str)}}}"

    @override
    def render_set_repr(self) -> str:
        return f"{{{format_comma_separated_list(self.ordered_members)}}}"

    @classmethod
    @override
    def construct_from_fields(cls, fields: dict[str, Any]) -> "PermutationDomain":
        return build_permutation_domain(fields["ordered_members"])


def build_ordinal_domain(values: Sequence[OrdinalValue]) -> OrdinalDomain:
    """Validate ``values`` and build a sorted :class:`OrdinalDomain`.

    Args:
        values: The admissible ordinal values; must be non-empty, unique, and
            mutually comparable.

    Returns:
        The constructed domain.

    Raises:
        ParamError: If ``values`` is empty or contains duplicates.
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
        members: The ordered permutation members; must be non-empty and unique.

    Returns:
        The constructed domain.

    Raises:
        ParamError: If ``members`` is empty or contains duplicates.
        TypeError: If a member is not a permutation member value.

    """
    return PermutationDomain(tuple(members))
