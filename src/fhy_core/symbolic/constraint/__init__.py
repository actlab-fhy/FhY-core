"""Scope-based constraints over expressions and value sets, and their conjunctions.

Provides the three constraint kinds used by the parameter infrastructure
in ``fhy_core.symbolic.param.*``:

- ``EquationConstraint``: a Boolean expression that must hold under an
  assignment to its free identifiers.
- ``InSetConstraint``: an identifier must take a value from a permitted
  set.
- ``NotInSetConstraint``: an identifier must NOT take a value from a
  forbidden set.

Each is a ``Constraint`` sum-type leaf sharing the assignment-based
``evaluate_with_bindings``/``is_satisfied_with_bindings`` contract, and
each can be converted to an equivalent ``Expression`` through
``convert_to_expression``. ``ConstraintSystem`` is the companion
set-level value object: a canonically ordered conjunction of constraints
with joint-satisfiability and entailment checking backed by
``fhy_core.symbolic.solver``. ``SymbolicPredicate`` is the structural
protocol shared by ``Constraint`` and ``ConstraintSystem``.

The package is organized by concern:

- ``errors``: ``ConstraintError``, ``MissingSymbolTypeError``.
- ``members``: constraint-member validation, type-strict wrapping,
  canonical ordering, and the set-constraint member codec.
- ``core``: ``ConstraintOutcome``, bindings coercion, the
  ``SymbolicPredicate`` protocol, and the ``Constraint`` family.
- ``ordering``: ``build_constraint_ordering_key``.
- ``system``: ``create_constraint_system`` and ``ConstraintSystem``.
"""

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

from .core import (
    Constraint,
    ConstraintBindings,
    ConstraintOutcome,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
    SymbolicPredicate,
)
from .errors import ConstraintError, MissingSymbolTypeError
from .members import ConstraintMember, MemberCollection
from .ordering import build_constraint_ordering_key
from .system import ConstraintSystem, create_constraint_system
