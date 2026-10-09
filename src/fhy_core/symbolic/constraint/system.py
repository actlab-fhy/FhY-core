"""Solver-backed conjunction of constraints: ``ConstraintSystem``.

``create_constraint_system`` and ``ConstraintSystem`` are the companion
set-level value object to the leaf constraints in
``fhy_core.symbolic.constraint.core``: a canonically ordered conjunction
of constraints, possibly spanning several identifiers, with
joint-satisfiability and entailment checking backed by
``fhy_core.symbolic.solver``. The system runs on the Rust core
(``fhy_core::constraint::ConstraintSystem``); a member of a kind defined in
Python is driven through its own methods. See ``ConstraintSystem`` for the
hazard classes its solver-backed entry points screen for before consulting
the solver.
"""

__all__ = [
    "ConstraintSystem",
    "create_constraint_system",
]

from typing import ClassVar

from fhy_core import _rs
from fhy_core.logger import get_logger
from fhy_core.serialization import WrappedFamilySerializable, register_serializable
from fhy_core.symbolic._native_slots import copy_native_attributes
from fhy_core.traits import FrozenMixin

from .core import Constraint

_LOGGER = get_logger(__name__)
"""The logger the Rust binding writes this module's records to."""


def create_constraint_system(*constraints: Constraint) -> "ConstraintSystem":
    """Create a constraint system from the given constraints.

    The door every caller builds a system through. ``ConstraintSystem``
    holds its members as a ``tuple``, so an iterable of a different shape
    is unpacked here: ``*sequence`` for a sequence already in hand,
    ``*generator`` for a lazy one, which the call itself materializes.

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
class ConstraintSystem(_rs.ConstraintSystem, WrappedFamilySerializable):
    """An ordered conjunction of constraints over shared identifiers.

    Semantically the logical AND of its member constraints. Members are
    normalized into canonical order, sorted stably by
    ``build_ordering_key``, so structurally equivalent systems built from
    differently ordered inputs are structurally equivalent and serialize
    identically. Duplicate constraints are retained (conjunction is
    idempotent). ``constraints`` returns the member objects given. A member
    must be a ``Constraint``; one of a kind defined in Python is called
    through its own methods. Instances are frozen; mutation raises
    ``FrozenMutationError``.

    ``==`` and ``hash`` are object identity, so two structurally equivalent
    systems are **distinct dict keys** and **distinct set members**: use
    ``is_structurally_equivalent`` for value-equality semantics.

    ``evaluate_with_bindings`` evaluates the members in order under a
    snapshot of the bindings: ``VIOLATED`` at the first violated member,
    ``SATISFIED`` if every member is, and ``UNDECIDED`` otherwise, each
    undecided member logged at ``DEBUG`` on this module's logger.

    The solver-backed entry points ask the default solver
    (``get_default_solver``) and check, in order: ``timeout_milliseconds``
    (``ValueError``); an empty system is ``SATISFIED``; the members'
    conversions (``ConstraintError``); ``symbol_types`` covering every
    identifier the question leaves free, native constants aside
    (``MissingSymbolTypeError``); each member's expression as a predicate
    (``NonBooleanLogicalOperandError``, naming the member's own
    expression); then the question. ``check_satisfiability_with_bindings``
    also lifts every binding into a literal or keeps it as an expression
    (``ConstraintError``), decides a set member whose variable is bound to
    a literal by itself, substitutes the rest, and reports a bound native
    constant the system refers to as ``UNDECIDED`` with a ``WARNING``.
    ``check_implication`` checks both sides, this system's members first.

    Every question reports ``UNDECIDED`` instead of a decided outcome when
    the solver answers ``unknown``, and for the hazard classes the solver
    screens, logging a ``WARNING`` on ``fhy_core.symbolic.solver`` naming
    the node:

    - a reference to a registered native constant, which the solver has no
      term for and could only lower as a variable free to take any value;
    - a Boolean operand in a numeric context;
    - a partial arithmetic operation off the domain its lowering is sound
      on -- true division without a finite nonzero literal divisor and a
      REAL-sorted operand, floor division or modulo without a finite
      strictly positive literal divisor, or exponentiation without a
      literal integer exponent of at least one;
    - an ``InSetConstraint`` or ``NotInSetConstraint`` member of the other
      numeric kind than its variable, such as ``2.0`` for an INT variable:
      membership is type-strict, while the solver would compare the two by
      value. An ``EquationConstraint``'s equality of an int with a float is
      answered by value, as its evaluation compares them.

    ``Constraint.evaluate_with_bindings`` decides an assignment through the
    simplifier instead, which these screens do not cover, so the two can
    disagree on a system the screens refuse but substitution decides.
    """

    _WIRE_FAMILY: ClassVar[str | None] = "constraint_system"

    # The members are copied into a slot on construction, so reading them
    # costs a slot read rather than a call into the extension.
    __slots__ = ("constraints",)

    def __init__(self, constraints: tuple[Constraint, ...]) -> None:
        copy_native_attributes(
            self, ConstraintSystem, _rs.ConstraintSystem, "constraints"
        )


FrozenMixin.register(ConstraintSystem)
