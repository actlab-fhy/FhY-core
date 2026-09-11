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
"""

import itertools
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

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
    do_ordered_param_values_match,
    does_collection_contain_param_value,
    is_categorical_value,
    is_ordinal_value,
    is_permutation_member_value,
    is_sequence_unique_without_set,
    serialize_wrapped_leaf_value,
)

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

_LOGGER = get_logger(__name__)

# The outcomes an enumeration over a finite value set can report.
DecidedOutcome: TypeAlias = Literal[
    ConstraintOutcome.SATISFIED, ConstraintOutcome.VIOLATED
]


def are_all_constraints_satisfied(
    constraints: Sequence[Constraint], variable: Identifier, value: Any
) -> bool:
    """Return whether ``value`` bound to ``variable`` satisfies every constraint.

    Raises:
        ConstraintError: If a constraint cannot lift ``value``, bound to
            ``variable``, into its substitution environment.
        NonBooleanLogicalOperandError: If a constraint holds a provably
            numeric operand in a Boolean position once ``value`` is
            bound; an ill-typed constraint is not reported unsatisfied.

    """
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


def _decide_from_enumeration(is_holding: bool) -> DecidedOutcome:
    """Map a decided boolean onto ``SATISFIED`` or ``VIOLATED``.

    Only enumeration over a finite value set may call this: every member
    of such a set is decided, so the boolean is a proof and leaves no
    room for ``UNDECIDED``.

    Args:
        is_holding: Whether the question was decided affirmatively.

    Returns:
        ``SATISFIED`` when ``is_holding``, ``VIOLATED`` otherwise.

    """
    return ConstraintOutcome.SATISFIED if is_holding else ConstraintOutcome.VIOLATED


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
    constraint it cannot lower or lift raises ``PassExecutionError``. A
    bridge failure is an undecided answer, not an invalid parameter, so
    it degrades to ``UNDECIDED`` (logged at ``WARNING``) rather than
    escaping a parameter-level query as an exception.

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
        The system's outcome, or ``UNDECIDED`` when the bridge failed.

    Raises:
        ConstraintError: If a member refuses the value ``bindings`` binds
            to an identifier in its scope, as
            ``ConstraintSystem.evaluate_with_bindings`` raises it. It is
            not degraded: a value that cannot be lifted is a caller error,
            not a limit of the bridge.
        NonBooleanLogicalOperandError: If a member equation holds a
            provably numeric operand in a Boolean position, counting a
            binding that puts a number there.

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


def _evaluate_in_set_candidates(
    domain: "ParamDomain", constraints: Sequence[Constraint], variable: Identifier
) -> Iterator[tuple[Any, ConstraintOutcome]]:
    """Yield each in-set candidate paired with its outcome under ``constraints``.

    A candidate the domain does not admit is ``VIOLATED`` outright.
    Otherwise its outcome is that of the conjunction of ``constraints``'s
    equation constraints with the candidate bound to ``variable``, so a
    dependent constraint the binding leaves unresolved yields
    ``UNDECIDED`` rather than a decided answer. Assumes ``constraints``
    contains at least one ``InSetConstraint``.

    """
    equation_system = _build_equation_constraint_system(constraints)
    for candidate in _compute_numeric_in_set_candidates(constraints):
        if not domain.is_value_admissible(candidate):
            yield candidate, ConstraintOutcome.VIOLATED
            continue
        yield candidate, evaluate_system_outcome(equation_system, {variable: candidate})


def _decide_feasibility_by_enumeration(
    domain: "ParamDomain", constraints: Sequence[Constraint], variable: Identifier
) -> ConstraintOutcome:
    """Decide feasibility from each in-set candidate's outcome.

    ``SATISFIED`` when some candidate is decided ``SATISFIED``;
    ``VIOLATED`` when every candidate is decided ``VIOLATED``, which
    includes there being no candidate at all; ``UNDECIDED`` otherwise,
    logged at ``WARNING`` naming the undecided candidates. Assumes
    ``constraints`` contains at least one ``InSetConstraint``.

    """
    undecided: list[Any] = []
    for candidate, outcome in _evaluate_in_set_candidates(
        domain, constraints, variable
    ):
        if outcome is ConstraintOutcome.SATISFIED:
            return ConstraintOutcome.SATISFIED
        if outcome is ConstraintOutcome.UNDECIDED:
            undecided.append(candidate)
    if not undecided:
        return ConstraintOutcome.VIOLATED
    _LOGGER.warning(
        "_decide_feasibility_by_enumeration: equation constraints could not "
        "decide candidate(s) %s for variable %r and decided none feasible; "
        "reporting UNDECIDED.",
        format_comma_separated_list(undecided),
        variable,
    )
    return ConstraintOutcome.UNDECIDED


def _evaluate_candidate_against_other_side(
    other_domain: "ParamDomain",
    other_constraints: Sequence[Constraint],
    other_variable: Identifier,
    candidate: Any,
) -> ConstraintOutcome:
    """Return ``other``'s outcome for ``candidate`` bound to ``other_variable``.

    ``VIOLATED`` when ``other_domain`` does not admit the candidate or a
    set constraint rejects it under type-strict membership. Otherwise the
    outcome of ``other_constraints``'s equation constraints with the
    candidate bound, which is ``UNDECIDED`` when a dependent constraint
    leaves it unresolved.

    """
    if not other_domain.is_value_admissible(candidate):
        return ConstraintOutcome.VIOLATED
    for constraint in other_constraints:
        if isinstance(constraint, InSetConstraint):
            if not does_collection_contain_param_value(constraint.members, candidate):
                return ConstraintOutcome.VIOLATED
        elif isinstance(constraint, NotInSetConstraint):
            if does_collection_contain_param_value(constraint.members, candidate):
                return ConstraintOutcome.VIOLATED
    equation_system = _build_equation_constraint_system(other_constraints)
    return evaluate_system_outcome(equation_system, {other_variable: candidate})


def _decide_subset_by_enumerating_own(
    own_domain: "ParamDomain",
    own_constraints: Sequence[Constraint],
    own_variable: Identifier,
    other_domain: "ParamDomain",
    other_constraints: Sequence[Constraint],
    other_variable: Identifier,
) -> ConstraintOutcome:
    """Decide the subset relation from ``own``'s in-set candidates.

    A candidate ``own`` decides ``VIOLATED`` is skipped. ``VIOLATED`` when
    a candidate ``own`` decides ``SATISFIED`` is decided ``VIOLATED`` by
    ``other``: that is a counterexample. ``SATISFIED`` when ``other``
    decides every remaining candidate ``SATISFIED``, since a candidate
    ``other`` accepts cannot break the relation whether or not it lies in
    ``own``. ``UNDECIDED`` otherwise, logged at ``WARNING`` naming the
    candidates; a candidate ``own`` leaves undecided and ``other``
    rejects is not a counterexample, since it may not lie in ``own`` at
    all. Assumes ``own_constraints`` contains at least one
    ``InSetConstraint``.

    """
    undecided: list[Any] = []
    for candidate, own_outcome in _evaluate_in_set_candidates(
        own_domain, own_constraints, own_variable
    ):
        if own_outcome is ConstraintOutcome.VIOLATED:
            continue
        other_outcome = _evaluate_candidate_against_other_side(
            other_domain, other_constraints, other_variable, candidate
        )
        if other_outcome is ConstraintOutcome.SATISFIED:
            continue
        if (
            own_outcome is ConstraintOutcome.SATISFIED
            and other_outcome is ConstraintOutcome.VIOLATED
        ):
            return ConstraintOutcome.VIOLATED
        undecided.append(candidate)
    if not undecided:
        return ConstraintOutcome.SATISFIED
    _LOGGER.warning(
        "_decide_subset_by_enumerating_own: candidate(s) %s of variable %r "
        "could not be decided against variable %r on both sides; reporting "
        "UNDECIDED.",
        format_comma_separated_list(undecided),
        own_variable,
        other_variable,
    )
    return ConstraintOutcome.UNDECIDED


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
    poses a weakened system to the solver instead of crashing on a
    foreign identifier; the exclusion marks that system inexact.

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
    narrowing is logged at ``WARNING`` and weakens the system; this
    convenience discards whether that happened, which
    :func:`_build_screened_constraint_system_with_fidelity` reports.

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


def _rescope_constraints_to_variable(
    constraints: Sequence[Constraint],
    old_variable: Identifier,
    new_variable: Identifier,
) -> tuple[Constraint, ...]:
    """Return ``constraints`` each rescoped from ``old_variable`` to ``new_variable``.

    Args:
        constraints: Constraints to rescope, in the order to keep.
        old_variable: Variable the constraints are currently scoped to.
        new_variable: Variable the returned constraints are scoped to.

    Returns:
        Equivalent constraints scoped to ``new_variable``.

    Raises:
        ConstraintError: If a constraint cannot be rescoped (propagated
            from :func:`_rename_constraint_variable`).

    """
    return tuple(
        _rename_constraint_variable(constraint, old_variable, new_variable)
        for constraint in constraints
    )


def _substitute_operand_variable(
    constraints: Sequence[Constraint],
    operand_variable: Identifier,
    variable: Identifier,
) -> tuple[Constraint, ...]:
    """Return ``constraints`` with ``operand_variable`` replaced by ``variable``.

    Only an ``EquationConstraint`` can mention an identifier beyond the
    one it is scoped to, so only its expression is substituted (a no-op
    where ``operand_variable`` is not free in it); a set constraint is
    returned as is.

    """
    return tuple(
        EquationConstraint(
            constraint.expression.substitute(
                {operand_variable: IdentifierExpression(variable)}
            )
        )
        if isinstance(constraint, EquationConstraint)
        else constraint
        for constraint in constraints
    )


def _merge_intersection_constraints(
    own_constraints: Sequence[Constraint],
    own_variable: Identifier,
    other_constraints: Sequence[Constraint],
    other_variable: Identifier,
    variable: Identifier,
) -> tuple[Constraint, ...]:
    """Return both operands' constraints rescoped to ``variable``, own side first.

    ``variable`` stands for both operands' quantities, so each side is
    rescoped from its own variable and then has the other operand's
    variable substituted by ``variable`` as well (see
    ``_substitute_operand_variable``); a constraint relating the two
    operands becomes a constraint on ``variable`` alone. An identifier
    that is neither operand's variable stays free.

    Raises:
        ConstraintError: If a constraint cannot be rescoped (propagated
            from :func:`_rename_constraint_variable`).

    """
    own = _substitute_operand_variable(
        _rescope_constraints_to_variable(own_constraints, own_variable, variable),
        other_variable,
        variable,
    )
    other = _substitute_operand_variable(
        _rescope_constraints_to_variable(other_constraints, other_variable, variable),
        own_variable,
        variable,
    )
    return own + other


def _collect_effective_finite_values(
    domain: "ParamDomain",
    constraints: Sequence[Constraint],
    variable: Identifier,
    values: Sequence[Any],
) -> tuple[Any, ...]:
    """Return the members of ``values`` a domain's own constraints leave valid."""
    return tuple(
        value
        for value in values
        if _is_value_valid_for(domain, constraints, variable, value)
    )


def _combine_finite_set_values(
    own_domain: "ParamDomain",
    own_constraints: Sequence[Constraint],
    own_variable: Identifier,
    own_values: Sequence[Any],
    other_domain: "ParamDomain",
    other_constraints: Sequence[Constraint],
    other_variable: Identifier,
    other_values: Sequence[Any],
    combine: Callable[[Sequence[Any], Sequence[Any]], tuple[Any, ...]],
) -> tuple[Any, ...]:
    """Return ``combine`` applied to both operands' effective finite value sets.

    Each side's declared members are filtered by that side's own
    constraints first, so a set-algebra result folds both operands'
    constraints into the member set it bakes.

    Args:
        own_domain: Domain of the left operand.
        own_constraints: Constraints carried by the left operand.
        own_variable: Variable ``own_constraints`` are scoped to.
        own_values: The left operand's declared members.
        other_domain: Domain of the right operand.
        other_constraints: Constraints carried by the right operand.
        other_variable: Variable ``other_constraints`` are scoped to.
        other_values: The right operand's declared members.
        combine: Type-strict set operation over the two effective sets.

    Returns:
        The combined member sequence, which may be empty.

    """
    own_effective = _collect_effective_finite_values(
        own_domain, own_constraints, own_variable, own_values
    )
    other_effective = _collect_effective_finite_values(
        other_domain, other_constraints, other_variable, other_values
    )
    return combine(own_effective, other_effective)


def _merge_finite_values(own: Sequence[Any], other: Sequence[Any]) -> tuple[Any, ...]:
    """Return the type-strict union of two finite value sequences.

    ``own`` is kept in full; a value from ``other`` is appended only when
    no value already collected matches it under the type-strict
    membership predicate, so ``True`` never absorbs ``1``.
    """
    merged = list(own)
    for value in other:
        if not does_collection_contain_param_value(merged, value):
            merged.append(value)
    return tuple(merged)


def _intersect_finite_values(
    own: Sequence[Any], other: Sequence[Any]
) -> tuple[Any, ...]:
    """Return the type-strict intersection of two finite value sequences."""
    return tuple(
        value for value in own if does_collection_contain_param_value(other, value)
    )


def _merge_non_negative_attributes(
    left_non_negative: bool,
    left_zero_included: bool,
    right_non_negative: bool,
    right_zero_included: bool,
) -> tuple[bool, bool]:
    """Return the ``(non_negative, zero_included)`` pair an intersection inherits.

    ``non_negative`` is the disjunction of both operands': either operand
    ruling out negative values rules them out of the intersection too.
    ``zero_included`` tightens to ``False`` as soon as a non-negative
    operand excludes zero.

    Args:
        left_non_negative: Whether the left operand admits no negatives.
        left_zero_included: Whether the left operand admits zero, given it
            is non-negative.
        right_non_negative: Whether the right operand admits no negatives.
        right_zero_included: Whether the right operand admits zero, given
            it is non-negative.

    Returns:
        The merged pair.

    """
    non_negative = left_non_negative or right_non_negative
    zero_included = not (
        (left_non_negative and not left_zero_included)
        or (right_non_negative and not right_zero_included)
    )
    return non_negative, zero_included


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

    Raises:
        NonBooleanLogicalOperandError: If ``own``'s screened system
            holds a provably numeric operand in a Boolean position,
            counting ``own_variable`` itself when ``symbol_type`` is INT
            or REAL.

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


def _does_set_constraint_hold_a_float_member(
    constraint: InSetConstraint | NotInSetConstraint, variable: Identifier
) -> bool:
    """Return whether constraint is scoped to variable and holds a float member.

    A lifted ``float`` member is the one kind Z3's REAL sort conflates
    with every other kind denoting the same number (a decimal-grammar
    ``str``, in particular): type-strict membership treats them as
    distinct members, but the sort lowers them all to one rational. Reads
    the public ``members``, scoped to ``variable`` so a constraint scoped
    elsewhere never triggers a kind-conflation downgrade for it.

    """
    return constraint.variable == variable and any(
        isinstance(member, float) for member in constraint.members
    )


def _downgrade_unproven_implication(
    outcome: ConstraintOutcome,
    is_own_exact: bool,
    is_other_exact: bool,
    own_constraints: Sequence[Constraint],
    own_variable: Identifier,
    other_constraints: Sequence[Constraint],
    other_variable: Identifier,
    symbol_type: SymbolType,
) -> ConstraintOutcome:
    """Return ``outcome`` unless it rests on an unproven side, then ``UNDECIDED``.

    Screening only widens a side's admissible set. A ``VIOLATED`` rests on
    a value inside the antecedent and outside the consequent, which an
    inexact antecedent may not actually admit; a ``SATISFIED`` rests on
    every antecedent value lying inside the consequent, which an inexact
    consequent may not actually admit. Over the REAL sort, Z3 also
    conflates a lifted ``float`` member with every other kind denoting the
    same number: a not-in-set antecedent holding one excludes more than
    type-strict membership does (the true antecedent is wider), and an
    in-set consequent holding one admits more than type-strict membership
    does (the consequent is wider than it should be); either makes a
    ``SATISFIED`` unproven for the same reason an inexact consequent does.
    A ``VIOLATED`` is not downgraded for this reason, since its
    counterexample can take the member's own kind. Every downgrade is
    logged at ``WARNING``.

    """
    is_own_narrowed_by_kind_conflation = symbol_type is SymbolType.REAL and any(
        isinstance(constraint, NotInSetConstraint)
        and _does_set_constraint_hold_a_float_member(constraint, own_variable)
        for constraint in own_constraints
    )
    is_other_widened_by_kind_conflation = symbol_type is SymbolType.REAL and any(
        isinstance(constraint, InSetConstraint)
        and _does_set_constraint_hold_a_float_member(constraint, other_variable)
        for constraint in other_constraints
    )
    is_unproven = (outcome is ConstraintOutcome.VIOLATED and not is_own_exact) or (
        outcome is ConstraintOutcome.SATISFIED
        and (
            not is_other_exact
            or is_own_narrowed_by_kind_conflation
            or is_other_widened_by_kind_conflation
        )
    )
    if not is_unproven:
        return outcome
    _LOGGER.warning(
        "compute_constraint_implication_subset: the solver's %s answer to "
        "whether %r implies %r rests on constraints screening dropped or "
        "narrowed, or on a REAL-sort member Z3 conflates with another kind; "
        "reporting UNDECIDED.",
        outcome.name,
        own_variable,
        other_variable,
    )
    return ConstraintOutcome.UNDECIDED


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
    sides with it bound (see ``_decide_subset_by_enumerating_own``): a
    candidate decided into ``own`` and decided out of ``other`` is a
    counterexample and decides ``VIOLATED``; ``other`` deciding every
    candidate not decided out of ``own`` decides ``SATISFIED``; anything
    else, such as a candidate a dependent constraint leaves undecided on
    either side, reports ``UNDECIDED``.

    When only ``other_constraints`` is finite, the candidates ``other``
    does not decide out are enumerated and ``own`` is asked, through the
    solver, whether it provably admits a value outside them (see
    ``_does_own_admit_a_value_outside``); such a value is a genuine
    counterexample, since the enumeration over-approximates what the
    other side admits, and the relation is decided ``VIOLATED``. No other
    answer is drawn from this branch.

    Otherwise the two sides' screened constraint systems (see
    ``_build_screened_constraint_system_with_fidelity``) are renamed onto
    one shared identifier and decided via
    ``ConstraintSystem.check_implication`` over ``symbol_type``. Screening
    only widens a side's admissible set, so a decided answer is kept
    exactly when the weakened systems still prove it: ``SATISFIED`` with
    an exact consequent, or ``VIOLATED`` with an exact antecedent. A
    ``VIOLATED`` from an inexact antecedent (a counterexample the dropped
    constraints might forbid) is reported ``UNDECIDED`` (logged at
    ``WARNING``), as is a solver that gave up.

    A ``SATISFIED`` is also downgraded to ``UNDECIDED`` when the
    consequent is inexact (an implication the dropped constraints might
    break), and, over the REAL sort, when the antecedent's own not-in-set
    constraint or the consequent's in-set constraint holds a lifted
    ``float`` member: Z3 conflates that member with every other kind
    denoting the same number, narrowing the antecedent or widening the
    consequent beyond what type-strict membership says (see
    ``_downgrade_unproven_implication``). A ``VIOLATED`` is not
    downgraded for the REAL-sort case, since its counterexample can take
    the member's own kind.

    Args:
        own_domain: Domain of the candidate subset parameter.
        own_constraints: Constraints of the candidate subset parameter.
        own_variable: Variable of the candidate subset parameter.
        other_domain: Domain of the candidate superset parameter.
        other_constraints: Constraints of the candidate superset parameter.
        other_variable: Variable of the candidate superset parameter.
        symbol_type: The Z3 sort used to reason about the shared variable.

    Returns:
        ``SATISFIED`` when the subset relation is decided to hold,
        ``VIOLATED`` when a counterexample is decided, and ``UNDECIDED``
        when neither the solver nor the enumeration could decide, or the
        solver decided only a weakened question.

    Raises:
        NonBooleanLogicalOperandError: If a constraint either branch
            evaluates holds a provably numeric operand in a Boolean
            position -- under a logical connective or as a piecewise
            case condition -- counting an in-set candidate bound to its
            variable, or the shared variable itself when ``symbol_type``
            is INT or REAL. Such a constraint is ill-typed rather than
            undecided, so it raises instead of reporting ``UNDECIDED``.

    """
    if any(isinstance(c, InSetConstraint) for c in own_constraints):
        return _decide_subset_by_enumerating_own(
            own_domain,
            own_constraints,
            own_variable,
            other_domain,
            other_constraints,
            other_variable,
        )
    if any(isinstance(c, InSetConstraint) for c in other_constraints):
        # A candidate ``other`` leaves undecided may still be admitted, so
        # it stays permitted: over-approximating what ``other`` admits is
        # what keeps a value found outside it a genuine counterexample.
        permitted_values = [
            candidate
            for candidate, outcome in _evaluate_in_set_candidates(
                other_domain, other_constraints, other_variable
            )
            if outcome is not ConstraintOutcome.VIOLATED
        ]
        if _does_own_admit_a_value_outside(
            own_domain, own_constraints, own_variable, permitted_values, symbol_type
        ):
            return ConstraintOutcome.VIOLATED
    common_variable = Identifier("var")
    own_screened, is_own_exact = _build_screened_constraint_system_with_fidelity(
        own_constraints, own_variable
    )
    other_screened, is_other_exact = _build_screened_constraint_system_with_fidelity(
        other_constraints, other_variable
    )
    own_system = _rename_constraint_system_variable(
        own_screened, own_variable, common_variable
    )
    other_system = _rename_constraint_system_variable(
        other_screened, other_variable, common_variable
    )
    outcome = own_system.check_implication(other_system, {common_variable: symbol_type})
    if outcome is ConstraintOutcome.UNDECIDED:
        _LOGGER.warning(
            "compute_constraint_implication_subset: the solver could not "
            "decide whether %r implies %r; reporting UNDECIDED.",
            own_variable,
            other_variable,
        )
    return _downgrade_unproven_implication(
        outcome,
        is_own_exact,
        is_other_exact,
        own_constraints,
        own_variable,
        other_constraints,
        other_variable,
        symbol_type,
    )


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
) -> ConstraintOutcome:
    if own.symbol_type is None or other.symbol_type != own.symbol_type:
        return ConstraintOutcome.VIOLATED
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
) -> ConstraintOutcome:
    """Decide whether some domain-admissible value satisfies every constraint.

    Routes through enumeration when an ``InSetConstraint`` makes the
    admissible values finite (see ``_decide_feasibility_by_enumeration``):
    a candidate decided to satisfy every constraint decides
    ``SATISFIED``, every candidate decided to violate one decides
    ``VIOLATED``, and an undecided candidate with no decided-feasible
    sibling reports ``UNDECIDED``. Otherwise decides the screened
    ``ConstraintSystem`` built from ``variable``-only equation
    constraints and ``NotInSetConstraint``s narrowed to their liftable
    members (see ``_build_screened_constraint_system_with_fidelity``).
    Screening only widens the admissible set, so ``SATISFIED`` is
    reported only when that system is exact: when a dependent or
    foreign-scoped constraint was dropped or narrowed (logged at
    ``WARNING``), the satisfying value may violate it, and ``UNDECIDED``
    is reported instead (also logged at ``WARNING``).

    ``VIOLATED`` on the screened system is reported as it stands, except
    over the REAL sort when some not-in-set constraint on ``variable``
    holds a lifted ``float`` member: Z3 conflates that member with every
    other kind denoting the same number, so it excludes more than
    type-strict membership does, and the ``VIOLATED`` is downgraded to
    ``UNDECIDED`` too (logged at ``WARNING``, naming ``variable``). A
    solver that gives up reports ``UNDECIDED``.

    Raises:
        NonBooleanLogicalOperandError: If a constraint the enumeration
            or the solver evaluates holds a provably numeric operand in a
            Boolean position, counting an in-set candidate bound to
            ``variable``, or ``variable`` itself when ``symbol_type`` is
            INT or REAL.

    """
    if any(isinstance(c, InSetConstraint) for c in constraints):
        return _decide_feasibility_by_enumeration(domain, constraints, variable)
    system, is_exact = _build_screened_constraint_system_with_fidelity(
        constraints, variable
    )
    outcome = system.check_satisfiability({variable: symbol_type})
    if outcome is ConstraintOutcome.SATISFIED and not is_exact:
        _LOGGER.warning(
            "_numeric_has_feasible_value: the solver's SATISFIED answer for "
            "variable %r rests on constraints screening dropped or narrowed; "
            "reporting UNDECIDED.",
            variable,
        )
        return ConstraintOutcome.UNDECIDED
    is_narrowed_by_kind_conflation = symbol_type is SymbolType.REAL and any(
        isinstance(constraint, NotInSetConstraint)
        and _does_set_constraint_hold_a_float_member(constraint, variable)
        for constraint in constraints
    )
    if outcome is ConstraintOutcome.VIOLATED and is_narrowed_by_kind_conflation:
        _LOGGER.warning(
            "_numeric_has_feasible_value: the solver's VIOLATED answer for "
            "variable %r rests on a not-in-set constraint whose float "
            "member the REAL sort conflates with another kind denoting the "
            "same number; reporting UNDECIDED.",
            variable,
        )
        return ConstraintOutcome.UNDECIDED
    if outcome is ConstraintOutcome.UNDECIDED:
        _LOGGER.warning(
            "_numeric_has_feasible_value: the solver could not decide "
            "satisfiability for variable %r; reporting UNDECIDED.",
            variable,
        )
    return outcome


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
    def get_interval_profile(self) -> IntervalProfile:
        """Return a profile that admits constraints other than bounds.

        A parameter over this domain may carry any constraint, so it takes
        part in interval arithmetic only once its constraints are checked
        to be bounds. Its sign restriction feeds the natural-number bound
        gate as it stands.
        """
        return IntervalProfile(
            admits_only_bounds=False,
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
    ) -> ConstraintOutcome:
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
    ) -> ConstraintOutcome:
        return _numeric_has_feasible_value(self, SymbolType.INT, constraints, variable)

    @override
    def compute_intersection(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
        variable: Identifier,
    ) -> tuple[ParamDomain, tuple[Constraint, ...]]:
        if not isinstance(other, IntegerDomain):
            raise TypeError(
                "Cannot intersect an IntegerDomain with a domain of type "
                f"{type(other).__name__}."
            )
        non_negative, zero_included = _merge_non_negative_attributes(
            self.non_negative,
            self.zero_included,
            other.non_negative,
            other.zero_included,
        )
        return IntegerDomain(
            non_negative=non_negative, zero_included=zero_included
        ), _merge_intersection_constraints(
            own_constraints, own_variable, other_constraints, other_variable, variable
        )

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


def _is_literal_grammar_string(value: str) -> bool:
    """Return whether ``value`` is a string ``LiteralExpression`` accepts.

    Asks :class:`~fhy_core.symbolic.expression.LiteralExpression` itself
    rather than restating its integer and float grammar, so admissibility
    cannot drift from the literal a constraint evaluation lifts a bound
    value into.
    """
    try:
        LiteralExpression(value)
    except ValueError:
        return False
    return True


@register_serializable(type_id="real_domain")
@dataclass(frozen=True, eq=False)
class RealDomain(ParamDomain):
    """Real-valued domain over finite floats and literal-grammar strings.

    A value is admissible exactly when it is a finite literal: a finite
    Python ``float``, or a ``str`` in the integer or float grammar
    :class:`~fhy_core.symbolic.expression.LiteralExpression` accepts, which
    denotes an exact decimal. NaN and the infinities are refused, as is a
    string that grammar refuses even where ``float()`` parses it (a sign, an
    exponent, surrounding whitespace, digit grouping, ``"nan"``, ``"inf"``).
    Constraint evaluation lifts the candidate into a literal, so no
    admissible value makes a validator raise. ``bool`` and ``int`` are not
    admissible.
    """

    @property
    @override
    def symbol_type(self) -> SymbolType | None:
        return SymbolType.REAL

    @override
    def is_value_admissible(self, value: Any) -> bool:
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, str):
            return _is_literal_grammar_string(value)
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
    ) -> ConstraintOutcome:
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
    ) -> ConstraintOutcome:
        return _numeric_has_feasible_value(self, SymbolType.REAL, constraints, variable)

    @override
    def compute_intersection(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
        variable: Identifier,
    ) -> tuple[ParamDomain, tuple[Constraint, ...]]:
        if not isinstance(other, RealDomain):
            raise TypeError(
                "Cannot intersect a RealDomain with a domain of type "
                f"{type(other).__name__}."
            )
        return RealDomain(), _merge_intersection_constraints(
            own_constraints, own_variable, other_constraints, other_variable, variable
        )

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
    def get_interval_profile(self) -> IntervalProfile:
        """Return a profile that admits only bounds, with this rendering preference.

        :meth:`validate_constraint` refuses every constraint but a bound, so
        a parameter over this domain is an interval operand as it stands.
        """
        return IntervalProfile(
            admits_only_bounds=True,
            non_negative=self.non_negative,
            zero_included=self.zero_included,
            prefer_inclusive=self.prefer_inclusive,
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
    ) -> ConstraintOutcome:
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
    ) -> ConstraintOutcome:
        return _numeric_has_feasible_value(self, SymbolType.INT, constraints, variable)

    @override
    def compute_intersection(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
        variable: Identifier,
    ) -> tuple[ParamDomain, tuple[Constraint, ...]]:
        """Intersect, carrying this operand's ``prefer_inclusive`` onto the result.

        The merged domain takes ``prefer_inclusive`` from ``self`` rather
        than reconciling it with ``other``'s. The carried constraints are
        both operands' own bounds verbatim, so the value set is the same
        either way; the flag only selects how arithmetic on the result
        renders the bounds it derives.
        """
        if not isinstance(other, IntervalIntegerDomain):
            raise TypeError(
                "Cannot intersect an IntervalIntegerDomain with a domain of "
                f"type {type(other).__name__}."
            )
        non_negative, zero_included = _merge_non_negative_attributes(
            self.non_negative,
            self.zero_included,
            other.non_negative,
            other.zero_included,
        )
        return IntervalIntegerDomain(
            prefer_inclusive=self.prefer_inclusive,
            non_negative=non_negative,
            zero_included=zero_included,
        ), _merge_intersection_constraints(
            own_constraints, own_variable, other_constraints, other_variable, variable
        )

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


def _raise_if_any_member_is_nan(values: Sequence[Any], kind: str) -> None:
    """Raise if any of ``values`` is a NaN float.

    NaN is unequal to itself, so a NaN member could never match a
    candidate: the domain would silently lack that member, and the
    uniqueness check could not tell two NaNs apart. An infinity equals
    itself and orders against every float, so it stays an admissible
    member.

    Args:
        values: The domain's members, already checked to be leaf values.
        kind: How the error message names the members, such as
            ``"Ordinal values"``.

    Raises:
        ParamError: If any of ``values`` is a NaN float.

    """
    if any(isinstance(value, float) and math.isnan(value) for value in values):
        raise ParamError(
            f"{kind} must not include NaN: NaN is unequal to itself, so a NaN "
            "member could never be admitted."
        )


def _order_finite_values_by_repr(values: Sequence[Any]) -> list[Any]:
    """Return ``values`` ordered by ``repr``.

    Every admissible leaf value has a ``repr``, and a value's ``repr`` depends
    only on the value, so the order is total and identical in every process.
    That makes it usable both as the sole order of an unordered value set and as
    the tiebreak between values whose own comparison cannot separate them (``1``
    and ``True`` compare equal, yet render as ``"1"`` and ``"True"``).
    """
    return sorted(values, key=repr)


@register_serializable(type_id="ordinal_domain")
@dataclass(frozen=True, eq=False)
class OrdinalDomain(ParamDomain):
    """Finite, totally-ordered set of admissible values.

    Values are stored as a strict-unique tuple in ascending order, with ``repr``
    breaking ties between values the order cannot separate (``1`` and ``True``
    compare equal). The stored order therefore depends only on the value set, not
    on the order the values were given in.

    A NaN value is refused: it is unequal to itself, so it could never be
    admitted, and it has no place in a total order. An infinity is kept.
    """

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
        _raise_if_any_member_is_nan(values, "Ordinal values")
        # Sorting is stable, so pre-ordering by ``repr`` decides the position of
        # values the ascending sort leaves tied (``1`` and ``True``).
        repr_ordered_values = _order_finite_values_by_repr(values)
        try:
            canonical = tuple(sorted(repr_ordered_values))
        except TypeError as exc:
            raise TypeError(
                "Ordinal values must be mutually comparable for sorting."
            ) from exc
        if not is_sequence_unique_without_set(canonical):
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
    ) -> DecidedOutcome:
        if not isinstance(other, OrdinalDomain):
            return ConstraintOutcome.VIOLATED
        for value in self.sorted_values:
            if not _is_value_valid_for(self, own_constraints, own_variable, value):
                continue
            if not _is_value_valid_for(other, other_constraints, other_variable, value):
                return ConstraintOutcome.VIOLATED
        return ConstraintOutcome.SATISFIED

    @override
    def has_feasible_value(
        self, constraints: Sequence[Constraint], variable: Identifier
    ) -> DecidedOutcome:
        return _decide_from_enumeration(
            any(
                _is_value_valid_for(self, constraints, variable, value)
                for value in self.sorted_values
            )
        )

    @override
    def compute_union(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
        variable: Identifier,
    ) -> tuple[ParamDomain, tuple[Constraint, ...]]:
        del variable
        if not isinstance(other, OrdinalDomain):
            raise TypeError(
                "Cannot union an OrdinalDomain with a domain of type "
                f"{type(other).__name__}."
            )
        merged = _combine_finite_set_values(
            self,
            own_constraints,
            own_variable,
            self.sorted_values,
            other,
            other_constraints,
            other_variable,
            other.sorted_values,
            _merge_finite_values,
        )
        if not merged:
            raise ParamError("Union of ordinal value sets is empty.")
        return build_ordinal_domain(merged), ()

    @override
    def compute_intersection(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
        variable: Identifier,
    ) -> tuple[ParamDomain, tuple[Constraint, ...]]:
        del variable
        if not isinstance(other, OrdinalDomain):
            raise TypeError(
                "Cannot intersect an OrdinalDomain with a domain of type "
                f"{type(other).__name__}."
            )
        intersected = _combine_finite_set_values(
            self,
            own_constraints,
            own_variable,
            self.sorted_values,
            other,
            other_constraints,
            other_variable,
            other.sorted_values,
            _intersect_finite_values,
        )
        if not intersected:
            raise ParamError("Intersection of ordinal value sets is empty.")
        return build_ordinal_domain(intersected), ()

    @override
    def is_structurally_equivalent(self, other: object) -> bool:
        # Values are stored in a canonical order, so compare them index-wise with
        # the strict value predicate. Native ``tuple ==`` would wrongly equate
        # ``(1,)`` and ``(True,)`` because ``True == 1``.
        return isinstance(other, OrdinalDomain) and do_ordered_param_values_match(
            self.sorted_values, other.sorted_values
        )

    @override
    def render_set_string(self) -> str:
        return f"{{{format_comma_separated_list(self.sorted_values, str_func=str)}}}"

    @override
    def render_set_repr(self) -> str:
        return f"{{{format_comma_separated_list(self.sorted_values)}}}"

    @classmethod
    @override
    def construct_from_fields(cls, fields: Mapping[str, Any]) -> "OrdinalDomain":
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
        object.__setattr__(
            self, "categories", tuple(_order_finite_values_by_repr(values))
        )

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
    ) -> DecidedOutcome:
        if not isinstance(other, CategoricalDomain):
            return ConstraintOutcome.VIOLATED
        for category in self.categories:
            if not _is_value_valid_for(self, own_constraints, own_variable, category):
                continue
            if not _is_value_valid_for(
                other, other_constraints, other_variable, category
            ):
                return ConstraintOutcome.VIOLATED
        return ConstraintOutcome.SATISFIED

    @override
    def has_feasible_value(
        self, constraints: Sequence[Constraint], variable: Identifier
    ) -> DecidedOutcome:
        return _decide_from_enumeration(
            any(
                _is_value_valid_for(self, constraints, variable, category)
                for category in self.categories
            )
        )

    @override
    def compute_union(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
        variable: Identifier,
    ) -> tuple[ParamDomain, tuple[Constraint, ...]]:
        del variable
        if not isinstance(other, CategoricalDomain):
            raise TypeError(
                "Cannot union a CategoricalDomain with a domain of type "
                f"{type(other).__name__}."
            )
        merged = _combine_finite_set_values(
            self,
            own_constraints,
            own_variable,
            self.categories,
            other,
            other_constraints,
            other_variable,
            other.categories,
            _merge_finite_values,
        )
        if not merged:
            raise ParamError("Union of categorical value sets is empty.")
        return build_categorical_domain(merged), ()

    @override
    def compute_intersection(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
        variable: Identifier,
    ) -> tuple[ParamDomain, tuple[Constraint, ...]]:
        del variable
        if not isinstance(other, CategoricalDomain):
            raise TypeError(
                "Cannot intersect a CategoricalDomain with a domain of type "
                f"{type(other).__name__}."
            )
        intersected = _combine_finite_set_values(
            self,
            own_constraints,
            own_variable,
            self.categories,
            other,
            other_constraints,
            other_variable,
            other.categories,
            _intersect_finite_values,
        )
        if not intersected:
            raise ParamError("Intersection of categorical value sets is empty.")
        return build_categorical_domain(intersected), ()

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
    def construct_from_fields(cls, fields: Mapping[str, Any]) -> "CategoricalDomain":
        return build_categorical_domain(tuple(fields["categories"]))


@register_serializable(type_id="permutation_domain")
@dataclass(frozen=True, eq=False)
class PermutationDomain(ParamDomain):
    """Admissible permutations of a fixed, ordered set of members.

    A NaN member is refused: it is unequal to itself, so no permutation
    could place it and the domain would admit no value at all.
    """

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
        _raise_if_any_member_is_nan(values, "Permutation members")
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
    ) -> DecidedOutcome:
        if not isinstance(other, PermutationDomain):
            return ConstraintOutcome.VIOLATED
        if len(self.ordered_members) != len(other.ordered_members):
            return ConstraintOutcome.VIOLATED
        for permutation in itertools.permutations(self.ordered_members):
            if not _is_value_valid_for(
                self, own_constraints, own_variable, permutation
            ):
                continue
            if not _is_value_valid_for(
                other, other_constraints, other_variable, permutation
            ):
                return ConstraintOutcome.VIOLATED
        return ConstraintOutcome.SATISFIED

    @override
    def has_feasible_value(
        self, constraints: Sequence[Constraint], variable: Identifier
    ) -> DecidedOutcome:
        return _decide_from_enumeration(
            any(
                _is_value_valid_for(self, constraints, variable, permutation)
                for permutation in itertools.permutations(self.ordered_members)
            )
        )

    @override
    def compute_intersection(
        self,
        own_constraints: Sequence[Constraint],
        own_variable: Identifier,
        other: ParamDomain,
        other_constraints: Sequence[Constraint],
        other_variable: Identifier,
        variable: Identifier,
    ) -> tuple[ParamDomain, tuple[Constraint, ...]]:
        """Intersect, keeping the member set and conjoining both constraint sets.

        Permutations are not enumerated here: two permutation domains
        either range over the same members, in which case the intersection
        is that same member set narrowed by both operands' constraints, or
        they do not, in which case no value is admissible to both.
        """
        if not isinstance(other, PermutationDomain):
            raise TypeError(
                "Cannot intersect a PermutationDomain with a domain of type "
                f"{type(other).__name__}."
            )
        if not (self.is_value_set_subset(other) and other.is_value_set_subset(self)):
            raise ParamError(
                "Intersection of permutation domains with different member "
                "sets is empty."
            )
        return self, _merge_intersection_constraints(
            own_constraints, own_variable, other_constraints, other_variable, variable
        )

    @override
    def is_structurally_equivalent(self, other: object) -> bool:
        # Member position is part of a permutation domain's identity, so compare
        # the members index-wise with the strict value predicate. Native
        # ``tuple ==`` would wrongly equate ``(1,)`` and ``(True,)`` because
        # ``True == 1``.
        return isinstance(other, PermutationDomain) and do_ordered_param_values_match(
            self.ordered_members, other.ordered_members
        )

    @override
    def render_set_string(self) -> str:
        return f"{{{format_comma_separated_list(self.ordered_members, str_func=str)}}}"

    @override
    def render_set_repr(self) -> str:
        return f"{{{format_comma_separated_list(self.ordered_members)}}}"

    @classmethod
    @override
    def construct_from_fields(cls, fields: Mapping[str, Any]) -> "PermutationDomain":
        return build_permutation_domain(fields["ordered_members"])


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
