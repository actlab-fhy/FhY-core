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
``_validate_symbol_types_cover_both_sides``) and the classification
helper (``_decide_satisfiability``) that every solver-backed entry point
routes through.
"""

__all__ = [
    "ConstraintSystem",
    "create_constraint_system",
]

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fhy_core.identifier import Identifier
from fhy_core.logger import get_logger
from fhy_core.serialization import WrappedFamilySerializable, register_serializable
from fhy_core.symbolic.expression import Expression, LiteralExpression
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
    SymbolicPredicate,
    _coerce_bindings_to_environment,
)
from .errors import ConstraintError, MissingSymbolTypeError
from .ordering import build_constraint_ordering_key

_LOGGER = get_logger(__name__)


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
