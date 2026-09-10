"""Solver-backed conjunction of constraints: ``ConstraintSystem``.

``create_constraint_system`` and ``ConstraintSystem`` are the companion
set-level value object to the leaf constraints in
``fhy_core.symbolic.constraint.core``: a canonically ordered conjunction
of constraints, possibly spanning several identifiers, with
joint-satisfiability and entailment checking backed by
``fhy_core.symbolic.solver``. See ``ConstraintSystem`` for the hazard
classes its solver-backed entry points screen for before consulting Z3.

The module also owns the shared ``symbol_types``-coverage validation
(``_validate_symbol_types_cover_free_identifiers``,
``_validate_symbol_types_cover_both_sides``,
``_validate_symbol_types_cover_residual``) and the classification
helper (``_classify_solver_answer``) that every solver-backed entry
point routes its seam answer through, so the rule that an undecided seam
answer stays undecided has one owner.
"""

__all__ = [
    "ConstraintSystem",
    "create_constraint_system",
]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fhy_core.identifier import Identifier
from fhy_core.logger import get_logger
from fhy_core.serialization import WrappedFamilySerializable, register_serializable
from fhy_core.symbolic.expression import (
    Expression,
    LiteralExpression,
    try_get_native_constant_for_identifier,
    validate_predicate,
)
from fhy_core.symbolic.solver import (
    check_expression_satisfiability,
    does_expression_imply,
    validate_timeout_milliseconds,
)
from fhy_core.symbolic.symbol_type import SymbolType
from fhy_core.term import DerivedEquivalenceMixin
from fhy_core.traits import FrozenMixin
from fhy_core.utils import format_comma_separated_list
from fhy_core.utils.override import override

from .core import (
    Constraint,
    ConstraintBindings,
    ConstraintOutcome,
    InSetConstraint,
    NotInSetConstraint,
    SymbolicPredicate,
    _coerce_bindings_to_environment,
    _find_bound_native_constants,
)
from .errors import ConstraintError, MissingSymbolTypeError

_LOGGER = get_logger(__name__)


def _raise_if_missing_symbol_types(
    identifiers: frozenset[Identifier], symbol_types: Mapping[Identifier, SymbolType]
) -> None:
    """Raise ``MissingSymbolTypeError`` naming each of ``identifiers`` with no sort.

    A registered native constant's canonical identifier is exempt: it
    names a value rather than a variable, and the solver seam refuses it
    without reading a sort.
    """
    missing = [
        identifier
        for identifier in identifiers - set(symbol_types)
        if try_get_native_constant_for_identifier(identifier) is None
    ]
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
    _raise_if_missing_symbol_types(expression.get_free_identifiers(), symbol_types)


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
    _raise_if_missing_symbol_types(
        antecedent.get_free_identifiers() | consequent.get_free_identifiers(),
        symbol_types,
    )


def _validate_symbol_types_cover_residual(
    expression: Expression,
    environment: Mapping[Identifier, Expression],
    symbol_types: Mapping[Identifier, SymbolType],
) -> None:
    """Raise if a residual free identifier has no ``symbol_types`` entry.

    Reads the identifiers ``expression.substitute(environment)`` would
    leave free without substituting, so the precondition can be checked
    ahead of a substitution that would raise on its own: each free
    identifier ``environment`` binds is replaced by its value's free
    identifiers, and every other free identifier stays.

    Raises:
        MissingSymbolTypeError: If one or more identifiers left free by
            substituting ``environment`` into ``expression`` have no
            corresponding ``symbol_types`` entry.

    """
    residual: frozenset[Identifier] = frozenset()
    for identifier in expression.get_free_identifiers():
        bound = environment.get(identifier)
        residual |= (
            frozenset({identifier}) if bound is None else bound.get_free_identifiers()
        )
    _raise_if_missing_symbol_types(residual, symbol_types)


def _classify_solver_answer(answer: bool | None) -> ConstraintOutcome:
    """Map a solver seam's tri-state answer onto a ``ConstraintOutcome``.

    Every solver-backed entry point classifies through here, so the rule
    that an undecided seam answer stays undecided -- rather than
    collapsing to a decided outcome -- has exactly one owner.

    Args:
        answer: Seam result, where ``None`` reports that the question was
            screened as hazardous or left inconclusive by the solver.

    Returns:
        ``UNDECIDED`` for ``None``, ``SATISFIED`` for ``True``, and
        ``VIOLATED`` for ``False``.

    """
    if answer is None:
        return ConstraintOutcome.UNDECIDED
    if answer:
        return ConstraintOutcome.SATISFIED
    return ConstraintOutcome.VIOLATED


def _decide_satisfiability(
    expression: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
    *,
    members: Sequence[Expression] = (),
    timeout_milliseconds: int | None = None,
) -> ConstraintOutcome:
    """Classify satisfiability of ``expression`` via the solver seam.

    Validates the caller's symbol types, then, when ``members`` is given,
    screens each one with ``validate_predicate`` before consulting
    ``fhy_core.symbolic.solver.check_expression_satisfiability``, so a
    numeric-rooted member is refused naming its own expression rather
    than the synthetic conjunction ``expression`` lowers. That seam
    function screens the expression for the hazard classes documented on
    ``ConstraintSystem`` before it ever reaches Z3, so ``None`` from the
    seam -- whether from a screened hazard or an inconclusive solver --
    maps here to ``UNDECIDED``.

    Args:
        expression: Expression to decide; the lowered conjunction of
            ``members`` when the caller has individual members, or an
            already-substituted residual otherwise.
        symbol_types: Z3 sort for each free identifier of ``expression``.
        members: Individual member expressions to screen with
            ``validate_predicate`` ahead of the seam call. Empty when the
            caller has already screened its members itself, or has none
            to screen separately from ``expression``.
        timeout_milliseconds: Optional bound, in milliseconds, on the
            solver invocation.

    Returns:
        ``SATISFIED``/``VIOLATED`` when the solver decides, ``UNDECIDED``
        when the seam screens the expression as hazardous or the solver
        is inconclusive.

    Raises:
        MissingSymbolTypeError: If ``symbol_types`` lacks an entry for a
            free identifier of ``expression``.
        NonBooleanLogicalOperandError: If a member of ``members``
            provably denotes a number, or is otherwise ill-typed as a
            predicate.

    """
    _validate_symbol_types_cover_free_identifiers(expression, symbol_types)
    for member in members:
        validate_predicate(member, symbol_types=symbol_types)
    return _classify_solver_answer(
        check_expression_satisfiability(
            expression,
            dict(symbol_types),
            timeout_milliseconds=timeout_milliseconds,
        )
    )


def _convert_members_to_conjunction(members: Sequence[Constraint]) -> Expression:
    """Return the conjunction of ``members``' own expression forms.

    Mirrors ``ConstraintSystem.convert_to_expression``, but over a
    caller-chosen subset of a system's members rather than the whole
    system: an empty sequence yields ``LiteralExpression(True)``, a
    single member yields that member's expression unwrapped, and
    otherwise a ``logical_and`` over the members in the given order.

    Args:
        members: Constraints to conjoin.

    Returns:
        An ``Expression`` whose truth value matches the conjunction of
        ``members``.

    Raises:
        ConstraintError: If any member cannot be converted to an
            expression.

    """
    if not members:
        return LiteralExpression(True)
    expressions = [member.convert_to_expression() for member in members]
    if len(expressions) == 1:
        return expressions[0]
    return Expression.logical_and(*expressions)


def _partition_decided_set_leaves(
    constraints: tuple[Constraint, ...],
    environment: Mapping[Identifier, Expression],
) -> tuple[list[InSetConstraint | NotInSetConstraint], list[Constraint]]:
    """Split ``constraints`` into leaves a concrete binding decides and the rest.

    A decided leaf is an ``InSetConstraint``/``NotInSetConstraint`` whose
    variable ``environment`` binds to a ``LiteralExpression`` -- the
    coerced form of both a raw value and an already-literal binding.
    Such a leaf is never lowered to Z3, where type-strict membership
    cannot be expressed; instead it is decided directly by its own
    ``evaluate_with_bindings``, the same mechanism
    ``ConstraintSystem.evaluate_with_bindings`` uses, so the two entry
    points agree by construction. A variable left unbound, or bound to a
    symbolic (non-literal) ``Expression``, still goes to the solver.

    Args:
        constraints: The system's members, in canonical order.
        environment: Substitution environment coerced from the caller's
            bindings.

    Returns:
        The decided leaves, then the remaining members, each in the
        given relative order.

    """
    decided_leaves: list[InSetConstraint | NotInSetConstraint] = []
    rest: list[Constraint] = []
    for constraint in constraints:
        if isinstance(constraint, (InSetConstraint, NotInSetConstraint)) and isinstance(
            environment.get(constraint.variable), LiteralExpression
        ):
            decided_leaves.append(constraint)
        else:
            rest.append(constraint)
    return decided_leaves, rest


def _decide_leaves_with_bindings(
    leaves: Sequence[InSetConstraint | NotInSetConstraint],
    bindings: ConstraintBindings,
) -> ConstraintOutcome:
    """Fold each decided leaf's own outcome into one outcome for the group.

    Mirrors how ``ConstraintSystem.evaluate_with_bindings`` folds member
    outcomes: a ``VIOLATED`` leaf outranks every other leaf, an
    ``UNDECIDED`` leaf otherwise carries the group to ``UNDECIDED``, and
    a group whose every leaf is ``SATISFIED`` is itself ``SATISFIED``.

    Args:
        leaves: Set-constraint leaves decided directly, without Z3.
        bindings: Original bindings passed to each leaf's own
            ``evaluate_with_bindings``, so a raw value stays raw and only
            a ``LiteralExpression`` binding is normalized.

    Returns:
        The folded outcome of every leaf in ``leaves``.

    """
    saw_undecided = False
    for leaf in leaves:
        outcome = leaf.evaluate_with_bindings(bindings)
        if outcome is ConstraintOutcome.VIOLATED:
            return ConstraintOutcome.VIOLATED
        if outcome is ConstraintOutcome.UNDECIDED:
            saw_undecided = True
    return ConstraintOutcome.UNDECIDED if saw_undecided else ConstraintOutcome.SATISFIED


def _combine_satisfiability_outcomes(
    leaves_outcome: ConstraintOutcome, residual_outcome: ConstraintOutcome
) -> ConstraintOutcome:
    """Fold a decided-leaves outcome and a residual outcome into one outcome.

    A ``VIOLATED`` side outranks the other; otherwise an ``UNDECIDED``
    side carries the result to ``UNDECIDED``; two ``SATISFIED`` sides
    give ``SATISFIED``.

    """
    if ConstraintOutcome.VIOLATED in (leaves_outcome, residual_outcome):
        return ConstraintOutcome.VIOLATED
    if ConstraintOutcome.UNDECIDED in (leaves_outcome, residual_outcome):
        return ConstraintOutcome.UNDECIDED
    return ConstraintOutcome.SATISFIED


def create_constraint_system(*constraints: Constraint) -> "ConstraintSystem":
    """Create a constraint system from the given constraints.

    The door every caller builds a system through. ``ConstraintSystem``
    holds its members as a ``tuple`` and is annotated as taking one, so
    an iterable of a different shape is unpacked here rather than passed
    to the constructor under a type suppression: ``*sequence`` for a
    sequence already in hand, ``*generator`` for a lazy one, which the
    call itself materializes.

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

    Semantically the logical AND of its member constraints. Members are
    taken as a ``tuple`` -- ``create_constraint_system`` is the door that
    normalizes any other argument shape into one -- and are normalized
    into canonical order, keyed on the same things structural
    equivalence compares, so structurally equivalent systems built from
    differently ordered inputs are structurally equivalent and serialize
    identically. Duplicate constraints are retained (conjunction is
    idempotent). Instances are frozen; mutation raises
    ``FrozenMutationError``.

    ``ConstraintSystem`` is declared ``@dataclass(frozen=True, eq=False)``,
    so ``__eq__`` and ``__hash__`` fall back to object identity rather than
    comparing the ``constraints`` tuple. Two structurally equivalent
    systems are therefore **distinct dict keys** and **distinct set
    members**: use ``is_structurally_equivalent`` for value-equality
    semantics, and avoid using ``ConstraintSystem`` instances as dict keys
    when you expect value-based lookups.

    All satisfiability and implication entry points report ``UNDECIDED``
    instead of a decided outcome for four hazard classes:

    - a reference to a registered native constant, which Z3 has no term
      for and could only lower as a variable free to take any value;
    - a Boolean operand in a numeric context;
    - a partial arithmetic operation off the domain its lowering is sound
      on -- true division without a finite nonzero literal divisor and a
      REAL-sorted operand, floor division or modulo without a finite
      strictly positive literal divisor, or exponentiation without a
      literal integer exponent of at least one;
    - an ``EQUAL``/``NOT_EQUAL`` mixing an INT and a REAL sort, in either
      arrangement.

    The screen for these hazards lives in ``fhy_core.symbolic.solver``,
    the seam every entry point below lowers through, and it logs a
    ``WARNING`` (naming the seam function and the offending node) before
    the outcome is reported as undecided. It covers the questions this
    class asks the solver; ``Constraint.evaluate_with_bindings`` decides
    an assignment through the expression bridge instead, which these
    screens do not cover, so the two can disagree on a system the screens
    refuse but substitution decides.

    """

    constraints: tuple[Constraint, ...]

    def __post_init__(self) -> None:
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
            tuple(sorted(constraints, key=lambda member: member.build_ordering_key())),
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

        Raises:
            ConstraintError: If a member refuses the value bound to an
                identifier in its scope, as that member's
                ``evaluate_with_bindings`` documents. Raised by the first
                such member reached in canonical order.
            NonBooleanLogicalOperandError: If a member equation holds a
                provably numeric operand in a Boolean position -- under
                a logical connective or as a piecewise case condition --
                counting a binding that puts a number there. Raised by
                the first such member reached in canonical order.

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
        """Return whether the bindings provably satisfy every constraint.

        Raises:
            ConstraintError: As ``evaluate_with_bindings`` raises it.
            NonBooleanLogicalOperandError: As ``evaluate_with_bindings``
                raises it.

        """
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
        conjunction for the hazard classes documented on this class
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
                no entry. A registered native constant's canonical
                identifier needs none either: it names a value rather
                than a variable.
            timeout_milliseconds: Optional bound, in milliseconds, on the
                underlying Z3 solver invocation. ``None`` (the default)
                leaves the solver unbounded.

        Raises:
            MissingSymbolTypeError: If ``symbol_types`` lacks an entry for
                a free identifier of the lowered conjunction. Checked
                ahead of ill-typedness and of the seam's hazard screen,
                the order the solver seam uses, so the precondition
                raises even for a conjunction that is also ill-typed or
                would otherwise be reported ``UNDECIDED``. This is a
                raise, not the
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
            NonBooleanLogicalOperandError: If a member's own expression
                provably denotes a number, or otherwise holds a provably
                numeric operand in a Boolean position -- under a logical
                connective or as a piecewise case condition -- counting a
                variable ``symbol_types`` declares INT or REAL. Such a
                member is ill-typed rather than undecidable, so it raises
                instead of reporting ``UNDECIDED``, naming the offending
                member's own expression rather than the synthetic
                conjunction. Checked after the symbol-type precondition
                and ahead of the seam's hazard screen, so it is reported
                even where the screen would also refuse the conjunction.

        """
        validate_timeout_milliseconds(timeout_milliseconds)
        if not self.constraints:
            return ConstraintOutcome.SATISFIED
        return _decide_satisfiability(
            self.convert_to_expression(),
            symbol_types,
            members=[
                constraint.convert_to_expression() for constraint in self.constraints
            ],
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

        A member this system holds as an ``InSetConstraint``/
        ``NotInSetConstraint`` whose variable is bound to a concrete
        value -- a raw value or a ``LiteralExpression`` -- is a decided
        leaf: it is decided directly by its own ``evaluate_with_bindings``
        rather than lowered to Z3. Z3's numeric equality cannot express
        this package's type-strict membership (it would, for example,
        equate the ``bool`` ``True`` with the ``int`` ``1``, or a binary
        float with the decimal string denoting the same number), so
        deciding a concrete-bound leaf this way is both sound and the
        only way to agree with ``evaluate_with_bindings`` by construction.
        Every other member -- an equation, or a set leaf left unbound or
        bound to a symbolic (non-literal) expression -- is the residual:
        it is substituted and decided over the remaining free identifiers
        via the z3 bridge, exactly as today. ``symbol_types`` needs
        entries only for the identifiers the residual leaves free.
        Answers questions of the form "given x = 4, can y and z still be
        chosen?".

        A decided leaf that is not a member makes the whole system
        ``VIOLATED``, outranking a satisfiable residual. Otherwise, an
        ``UNDECIDED`` decided leaf (an unusable binding aside, this only
        happens for a bound registered native constant) carries the
        system to ``UNDECIDED`` unless a later decided leaf or the
        residual is ``VIOLATED``. A residual-free system -- every member
        was a decided leaf -- is decided from the leaves alone, without
        consulting the solver at all.

        Limitation: the same hazard classes documented on this class
        apply to the residual; ``fhy_core.symbolic.solver`` screens the
        substituted residual rather than the original conjunction.
        Substitution is therefore part of the screen: a ``bool`` binding
        value lands in the residual exactly as a ``bool`` set member does
        and is screened the same way, while binding a variable to a value
        of the matching sort can retire a hazard the unsubstituted
        residual had.

        A binding for a registered native constant's canonical identifier
        is refused rather than substituted: the identifier names a value
        rather than a variable, and substituting it would answer for a
        world where the constant has the bound value. This refusal
        covers a constant referenced by the residual or bound as a
        decided leaf's own variable. A binding for a constant the system
        references reports ``UNDECIDED`` with a ``WARNING``, after every
        check listed under ``Raises``, exactly as ``evaluate_with_bindings``
        reports it; one for a constant the system does not reference is
        ignored.

        Args:
            bindings: Partial assignment consulted for the satisfiability
                check. Values must be ``Expression`` or ``LiteralType``,
                as ``ConstraintBindings`` declares.
            symbol_types: Z3 sort for each identifier the residual leaves
                free.
            timeout_milliseconds: Optional bound, in milliseconds, on the
                underlying Z3 solver invocation. ``None`` (the default)
                leaves the solver unbounded.

        Raises:
            MissingSymbolTypeError: If ``symbol_types`` lacks an entry for
                a free identifier of the residual expression left after
                substitution. Checked before anything is substituted,
                against the identifiers substitution will leave free, and
                ahead of ill-typedness and of the seam's hazard screen --
                the order ``check_satisfiability`` and the solver seam
                use -- so the precondition raises even for bindings that
                are also ill-typed or a residual that would otherwise be
                reported ``UNDECIDED``. Contrast a missing
                entry in ``bindings`` itself: an identifier ``bindings``
                does not cover is left free in the residual rather than
                raising, so it only raises here if ``symbol_types`` also
                fails to cover it. A missing *value* binding degrades to
                ``ConstraintOutcome.UNDECIDED`` on ``evaluate_with_bindings``;
                a missing symbol type here always raises, since the Z3
                bridge cannot proceed without a sort for every free
                identifier.
            ConstraintError: If a ``bindings`` value cannot be lifted into
                the substitution environment: it falls outside
                ``Expression | LiteralType``, or ``LiteralExpression``
                refuses it, as it refuses a ``str`` matching neither the
                integer nor the float grammar. Checked against every
                binding, whether or not the identifier it names ends up
                deciding a leaf directly, so an empty system returns
                ``SATISFIED`` without inspecting ``bindings`` at all, but
                a non-empty one always does. Also raised if a residual
                member cannot be converted to an expression; a decided
                leaf is never lowered, so a member unusable that way --
                for example a categorical string member, which membership
                compares type-strictly but Z3 could only lower by
                canonicalizing against numeric members -- does not raise
                once its variable is concretely bound.
            ValueError: If ``timeout_milliseconds`` is not None and not
                positive. Checked before the empty-system and hazard
                early returns, so an inadmissible bound is rejected even
                when the outcome is decided without the solver.
            NonBooleanLogicalOperandError: If a residual member's own
                expression provably denotes a number, or otherwise holds
                a provably numeric operand in a Boolean position -- under
                a logical connective or as a piecewise case condition --
                counting an identifier ``bindings`` binds to a number, or
                an unbound one ``symbol_types`` declares INT or REAL. Each
                residual member is screened on its own, so the error
                names the offending member's own expression rather than
                the synthetic conjunction. Checked against the bindings
                before they are substituted, since substituting a number
                into a case condition would build a piecewise that
                refuses its own condition, but after the symbol-type
                precondition and ahead of the seam's hazard screen, as in
                ``check_satisfiability``. ``evaluate_with_bindings``
                refuses the same bindings with the same error.

        """
        validate_timeout_milliseconds(timeout_milliseconds)
        if not self.constraints:
            return ConstraintOutcome.SATISFIED
        environment = _coerce_bindings_to_environment(bindings)
        decided_leaves, rest = _partition_decided_set_leaves(
            self.constraints, environment
        )
        residual_expression = _convert_members_to_conjunction(rest)
        _validate_symbol_types_cover_residual(
            residual_expression, environment, symbol_types
        )
        for constraint in rest:
            validate_predicate(
                constraint.convert_to_expression(),
                environment,
                symbol_types=symbol_types,
            )
        scope = residual_expression.get_free_identifiers() | {
            leaf.variable for leaf in decided_leaves
        }
        captured = _find_bound_native_constants(scope, environment)
        if captured:
            _LOGGER.warning(
                "ConstraintSystem.check_satisfiability_with_bindings: "
                "identifier(s) %s are the canonical identifiers of registered "
                "native constants, which name values rather than variables, so "
                "the supplied binding cannot be honored; reporting UNDECIDED "
                "rather than deciding for a world where the constant has the "
                "bound value",
                format_comma_separated_list(tuple(captured)),
            )
            return ConstraintOutcome.UNDECIDED
        leaves_outcome = _decide_leaves_with_bindings(decided_leaves, bindings)
        if leaves_outcome is ConstraintOutcome.VIOLATED or not rest:
            return leaves_outcome
        residual = residual_expression.substitute(environment)
        residual_outcome = _decide_satisfiability(
            residual,
            symbol_types,
            timeout_milliseconds=timeout_milliseconds,
        )
        return _combine_satisfiability_outcomes(leaves_outcome, residual_outcome)

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
        both lowered sides for the hazard classes documented on
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
            NonBooleanLogicalOperandError: If a member of either side
                provably denotes a number, or otherwise holds a provably
                numeric operand in a Boolean position -- under a logical
                connective or as a piecewise case condition -- counting a
                variable ``symbol_types`` declares INT or REAL. Such a
                member is ill-typed rather than undecidable, so it raises
                instead of reporting ``UNDECIDED``, naming the offending
                member's own expression rather than either side's
                synthetic conjunction. Every member of ``self`` is
                checked before any member of ``other``.

        """
        validate_timeout_milliseconds(timeout_milliseconds)
        antecedent = self.convert_to_expression()
        consequent = other.convert_to_expression()
        _validate_symbol_types_cover_both_sides(antecedent, consequent, symbol_types)
        for constraint in (*self.constraints, *other.constraints):
            validate_predicate(
                constraint.convert_to_expression(), symbol_types=symbol_types
            )
        return _classify_solver_answer(
            does_expression_imply(
                antecedent,
                consequent,
                dict(symbol_types),
                timeout_milliseconds=timeout_milliseconds,
            )
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
