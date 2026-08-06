"""Backend-agnostic entry point for symbolic queries over expressions.

Simplification is answered by the SymPy bridge; satisfiability,
implication, and universal validity by the Z3 bridge. Backend
selection is explicit: asking a backend for a query kind it cannot
answer raises ``SolverCapabilityError``. Each query kind currently has
exactly one capable backend.

Known divergences: the Z3 and SymPy bridges disagree with each other
and with the type checker on logical-not, ``Rational`` lifting, integer
division, floor-division/modulo Euclidean semantics, and inf/nan
lifting. This module routes to each bridge unchanged; it does not
reconcile that math.
"""

__all__ = [
    "SolverBackend",
    "SolverCapabilityError",
    "SolverQueryKind",
    "assert_expression_implies",
    "assert_holds_for_all_free_assignments",
    "check_expression_satisfiability",
    "does_expression_imply",
    "get_backend_capabilities",
    "holds_for_all_free_assignments",
    "simplify_expression",
]

from collections.abc import Set as AbstractSet

from immutabledict import immutabledict

from fhy_core.error import register_error
from fhy_core.identifier import Identifier
from fhy_core.utils import StrEnum

from .expression import Expression, LiteralExpression
from .expression.passes.sympy import simplify_expression as _sympy_simplify_expression
from .expression.passes.z3 import (
    assert_expression_implies as _z3_assert_expression_implies,
)
from .expression.passes.z3 import (
    assert_holds_for_all_free_assignments as _z3_assert_holds_for_all_free_assignments,
)
from .expression.passes.z3 import does_expression_imply as _z3_does_expression_imply
from .expression.passes.z3 import (
    holds_for_all_free_assignments as _z3_holds_for_all_free_assignments,
)
from .symbol_type import SymbolType


class SolverBackend(StrEnum):
    """Symbolic engine selectable for a solver query."""

    SYMPY = "sympy"
    Z3 = "z3"


class SolverQueryKind(StrEnum):
    """Kind of question a solver backend can be asked."""

    SIMPLIFICATION = "simplification"
    SATISFIABILITY = "satisfiability"
    IMPLICATION = "implication"
    UNIVERSAL_VALIDITY = "universal_validity"


@register_error
class SolverCapabilityError(ValueError):
    """Raised when the requested backend cannot answer the requested query kind."""


_BACKEND_CAPABILITIES: immutabledict[SolverBackend, frozenset[SolverQueryKind]] = (
    immutabledict(
        {
            SolverBackend.SYMPY: frozenset({SolverQueryKind.SIMPLIFICATION}),
            SolverBackend.Z3: frozenset(
                {
                    SolverQueryKind.SATISFIABILITY,
                    SolverQueryKind.IMPLICATION,
                    SolverQueryKind.UNIVERSAL_VALIDITY,
                }
            ),
        }
    )
)


def get_backend_capabilities(backend: SolverBackend) -> frozenset[SolverQueryKind]:
    """Return the query kinds the given backend can answer.

    Returns:
        ``{SIMPLIFICATION}`` for SYMPY; ``{SATISFIABILITY, IMPLICATION,
        UNIVERSAL_VALIDITY}`` for Z3.

    """
    return _BACKEND_CAPABILITIES[backend]


def _validate_backend_capability(
    backend: SolverBackend, query_kind: SolverQueryKind
) -> None:
    if query_kind not in get_backend_capabilities(backend):
        raise SolverCapabilityError(
            f"Backend {backend!r} cannot answer {query_kind!r} queries; "
            f"it supports {sorted(get_backend_capabilities(backend))}."
        )


def _validate_timeout_milliseconds(timeout_milliseconds: int | None) -> None:
    if timeout_milliseconds is not None and timeout_milliseconds <= 0:
        raise ValueError(
            "timeout_milliseconds must be None or a positive integer, but got "
            f"{timeout_milliseconds!r}."
        )


def simplify_expression(
    expression: Expression,
    environment: dict[Identifier, Expression] | None = None,
    *,
    backend: SolverBackend = SolverBackend.SYMPY,
) -> Expression:
    """Simplify an expression, optionally substituting an environment first.

    With an environment binding every free identifier, simplification is
    evaluation: the result is a ``LiteralExpression`` whenever the backend
    can decide the value. Delegates to the SymPy bridge, which performs
    all conversion and simplification math.

    Args:
        expression: Expression to simplify.
        environment: Environment to substitute into the expression before
            simplifying. Defaults to ``None``.
        backend: Solver backend to route the query to. Defaults to
            ``SolverBackend.SYMPY``, the only SIMPLIFICATION-capable
            backend today.

    Returns:
        Simplified expression.

    Raises:
        SolverCapabilityError: If ``backend`` is not SIMPLIFICATION-capable
            (currently: any backend other than SYMPY).

    """
    _validate_backend_capability(backend, SolverQueryKind.SIMPLIFICATION)
    # INVARIANT: _BACKEND_CAPABILITIES grants SIMPLIFICATION to exactly one
    # backend (SYMPY), so delegating straight to the SymPy bridge is valid
    # without dispatching on `backend`. Adding a second SIMPLIFICATION-capable
    # backend requires real dispatch here.
    return _sympy_simplify_expression(expression, environment)


def check_expression_satisfiability(
    expression: Expression,
    symbol_types: dict[Identifier, SymbolType],
    *,
    backend: SolverBackend = SolverBackend.Z3,
    timeout_milliseconds: int | None = None,
) -> bool | None:
    """Return whether some assignment to the free identifiers satisfies the expression.

    True if a satisfying assignment provably exists; False if provably
    none exists; None if the solver returns unknown. Implemented as the
    inversion of ``does_expression_imply(expression, false)``; callers
    such as ``param.domains`` and ``constraint.ConstraintSystem`` route
    their satisfiability checks through this one construction.

    Args:
        expression: Expression to check.
        symbol_types: Z3 sort to use for each free identifier of
            ``expression``.
        backend: Solver backend to route the query to. Defaults to
            ``SolverBackend.Z3``, the only SATISFIABILITY-capable backend
            today.
        timeout_milliseconds: Optional bound, in milliseconds, on the
            underlying Z3 solver invocation. ``None`` (the default)
            leaves the solver unbounded.

    Returns:
        True, False, or None per the truth table above.

    Raises:
        SolverCapabilityError: If ``backend`` is not SATISFIABILITY-capable.
        KeyError: If ``symbol_types`` lacks an entry for a free identifier.
        ValueError: If ``timeout_milliseconds`` is not None and not positive.

    """
    _validate_backend_capability(backend, SolverQueryKind.SATISFIABILITY)
    _validate_timeout_milliseconds(timeout_milliseconds)
    # INVARIANT: _BACKEND_CAPABILITIES grants SATISFIABILITY to exactly one
    # backend (Z3), so delegating straight to the Z3 bridge is valid without
    # dispatching on `backend`. Adding a second SATISFIABILITY-capable
    # backend requires real dispatch here.
    implies_false = _z3_does_expression_imply(
        expression,
        LiteralExpression(False),
        symbol_types,
        timeout_milliseconds=timeout_milliseconds,
    )
    if implies_false is None:
        return None
    return not implies_false


def does_expression_imply(
    antecedent: Expression,
    consequent: Expression,
    symbol_types: dict[Identifier, SymbolType],
    *,
    backend: SolverBackend = SolverBackend.Z3,
    timeout_milliseconds: int | None = None,
) -> bool | None:
    """Return whether the antecedent logically implies the consequent.

    Delegates to the Z3 bridge, which performs all conversion and
    decision math; None means the solver returned unknown.

    Args:
        antecedent: The premise expression.
        consequent: The conclusion expression.
        symbol_types: Z3 sort to use for each identifier referenced by
            either expression.
        backend: Solver backend to route the query to. Defaults to
            ``SolverBackend.Z3``, the only IMPLICATION-capable backend
            today.
        timeout_milliseconds: Optional bound, in milliseconds, on the
            underlying Z3 solver invocation. ``None`` (the default)
            leaves the solver unbounded.

    Returns:
        True if the implication holds for every assignment; False if a
        counterexample exists; None if the solver returns unknown.

    Raises:
        SolverCapabilityError: If ``backend`` is not IMPLICATION-capable.
        KeyError: If ``symbol_types`` lacks an entry for a free identifier.
        ValueError: If ``timeout_milliseconds`` is not None and not positive.

    """
    _validate_backend_capability(backend, SolverQueryKind.IMPLICATION)
    _validate_timeout_milliseconds(timeout_milliseconds)
    # INVARIANT: _BACKEND_CAPABILITIES grants IMPLICATION to exactly one
    # backend (Z3), so delegating straight to the Z3 bridge is valid without
    # dispatching on `backend`. Adding a second IMPLICATION-capable backend
    # requires real dispatch here.
    return _z3_does_expression_imply(
        antecedent,
        consequent,
        symbol_types,
        timeout_milliseconds=timeout_milliseconds,
    )


def holds_for_all_free_assignments(
    considered_identifiers: AbstractSet[Identifier],
    expression: Expression,
    symbol_types: dict[Identifier, SymbolType],
    *,
    backend: SolverBackend = SolverBackend.Z3,
    timeout_milliseconds: int | None = None,
) -> bool | None:
    """Return whether the expression holds for every free assignment.

    Args:
        considered_identifiers: Identifiers existentially quantified by
            the check; identifiers in the expression but not in this set
            are treated as free (universally quantified).
        expression: Expression to check.
        symbol_types: Z3 sort to use for each identifier appearing in
            the expression.
        backend: Solver backend to route the query to. Defaults to
            ``SolverBackend.Z3``, the only UNIVERSAL_VALIDITY-capable
            backend today.
        timeout_milliseconds: Optional bound, in milliseconds, on the
            underlying Z3 solver invocation. ``None`` (the default)
            leaves the solver unbounded.

    Returns:
        True if the expression has a witness for every free assignment;
        False if some free assignment has no witness; None if the solver
        returns unknown.

    Raises:
        SolverCapabilityError: If ``backend`` is not UNIVERSAL_VALIDITY-capable.
        ValueError: If ``timeout_milliseconds`` is not None and not positive.

    """
    _validate_backend_capability(backend, SolverQueryKind.UNIVERSAL_VALIDITY)
    _validate_timeout_milliseconds(timeout_milliseconds)
    # INVARIANT: _BACKEND_CAPABILITIES grants UNIVERSAL_VALIDITY to exactly
    # one backend (Z3), so delegating straight to the Z3 bridge is valid
    # without dispatching on `backend`. Adding a second
    # UNIVERSAL_VALIDITY-capable backend requires real dispatch here.
    return _z3_holds_for_all_free_assignments(
        considered_identifiers,
        expression,
        symbol_types,
        timeout_milliseconds=timeout_milliseconds,
    )


def assert_holds_for_all_free_assignments(
    considered_identifiers: AbstractSet[Identifier],
    expression: Expression,
    symbol_types: dict[Identifier, SymbolType],
    *,
    backend: SolverBackend = SolverBackend.Z3,
    timeout_milliseconds: int | None = None,
) -> bool:
    """Check universal validity, raising ``UndecidableError`` on ``unknown``.

    Args:
        considered_identifiers: As for
            :func:`holds_for_all_free_assignments`.
        expression: As for :func:`holds_for_all_free_assignments`.
        symbol_types: As for :func:`holds_for_all_free_assignments`.
        backend: As for :func:`holds_for_all_free_assignments`.
        timeout_milliseconds: As for :func:`holds_for_all_free_assignments`.

    Returns:
        The decided ``bool`` result.

    Raises:
        SolverCapabilityError: If ``backend`` is not UNIVERSAL_VALIDITY-capable.
        UndecidableError: When the solver returns unknown.
        ValueError: If ``timeout_milliseconds`` is not None and not positive.

    """
    _validate_backend_capability(backend, SolverQueryKind.UNIVERSAL_VALIDITY)
    _validate_timeout_milliseconds(timeout_milliseconds)
    # INVARIANT: _BACKEND_CAPABILITIES grants UNIVERSAL_VALIDITY to exactly
    # one backend (Z3), so delegating straight to the Z3 bridge is valid
    # without dispatching on `backend`. Adding a second
    # UNIVERSAL_VALIDITY-capable backend requires real dispatch here.
    return _z3_assert_holds_for_all_free_assignments(
        considered_identifiers,
        expression,
        symbol_types,
        timeout_milliseconds=timeout_milliseconds,
    )


def assert_expression_implies(
    antecedent: Expression,
    consequent: Expression,
    symbol_types: dict[Identifier, SymbolType],
    *,
    backend: SolverBackend = SolverBackend.Z3,
    timeout_milliseconds: int | None = None,
) -> bool:
    """Check the implication, raising ``UndecidableError`` on ``unknown``.

    Raises ``UndecidableError`` instead of returning ``None`` when the
    solver's result is unknown.

    Args:
        antecedent: As for :func:`does_expression_imply`.
        consequent: As for :func:`does_expression_imply`.
        symbol_types: As for :func:`does_expression_imply`.
        backend: As for :func:`does_expression_imply`.
        timeout_milliseconds: As for :func:`does_expression_imply`.

    Returns:
        The decided ``bool`` result.

    Raises:
        SolverCapabilityError: If ``backend`` is not IMPLICATION-capable.
        UndecidableError: When the solver returns unknown.
        ValueError: If ``timeout_milliseconds`` is not None and not positive.

    """
    _validate_backend_capability(backend, SolverQueryKind.IMPLICATION)
    _validate_timeout_milliseconds(timeout_milliseconds)
    # INVARIANT: _BACKEND_CAPABILITIES grants IMPLICATION to exactly one
    # backend (Z3), so delegating straight to the Z3 bridge is valid without
    # dispatching on `backend`. Adding a second IMPLICATION-capable backend
    # requires real dispatch here.
    return _z3_assert_expression_implies(
        antecedent,
        consequent,
        symbol_types,
        timeout_milliseconds=timeout_milliseconds,
    )
