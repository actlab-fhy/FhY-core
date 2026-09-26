"""Questions about expressions, answered by pluggable solver backends.

The solver answers four kinds of question: what an expression simplifies
to, whether some assignment satisfies an expression, whether one expression
implies another, and whether an expression holds for every assignment of its
free identifiers. The query logic is the Rust core's
(``fhy_core::solver``): each question is checked, screened, and encoded as
one SMT-LIB2 script in Rust, and its backend is called once.

Backends are pluggable. A :class:`Solver` holds an :class:`SmtSolver` for the
three logical questions and a :class:`Simplifier` for simplification:

- ``SolverBackend.Z3`` is the z3-solver adapter,
  :class:`~fhy_core.symbolic.expression.passes.z3.Z3Solver`, which needs the
  ``z3-solver`` package (``pip install fhy_core[z3]``);
- ``SolverBackend.SYMPY`` is the sympy adapter,
  :class:`~fhy_core.symbolic.expression.passes.sympy.SympySimplifier`, which
  needs the ``sympy`` package (``pip install fhy_core[sympy]``);
- :class:`SmtLib2ProcessSolver` drives any SMT-LIB2 executable, such as
  ``z3 -in`` or ``cvc5 --lang=smt2``, from Rust;
- a Python subclass of :class:`SmtSolver` or :class:`Simplifier` plugs in any
  other backend.

The module functions below take ``backend``: ``None`` (the default) asks the
default solver, which :func:`get_default_solver` returns and
:func:`set_default_solver` replaces; the constraints and params ask it too.
Its initial value holds the z3-solver and sympy adapters. A
:class:`SolverBackend` member asks its adapter. Neither package is imported
until a question needs it; a question whose backend's package is missing
raises :class:`SolverBackendUnavailableError`, never a degraded answer.

The order of checks of a logical question, which every entry point shares:
the backend's capability (:class:`SolverCapabilityError`), then
``timeout_milliseconds`` (``ValueError``), then the symbol types of every
identifier the question mentions, except a native constant's (``KeyError``),
then each expression's use as a predicate
(``NonBooleanLogicalOperandError``), then the hazard screen of each
expression on its own. The screen refuses what SMT-LIB2 arithmetic cannot
state in this package's semantics: a native constant, a non-finite float, a
Boolean in a numeric context, a partial operation off its safe domain
(division by anything but a finite nonzero literal with a real operand,
floor division or modulo by anything but a finite positive literal, a power
without an integer literal exponent of at least one), and an equality of a
numeric literal with an operand of another or an unknown int/real kind. A
refused question is logged at WARNING on this module's logger and answers
``None``; the strict ``assert_*`` companions raise ``UndecidableError`` with
the reason ``"hazard_screen"`` instead, and with the backend's reason for an
``unknown`` answer.

The lowering keeps this package's semantics: division is exact, floor
division and modulo round toward negative infinity, and a float or decimal
literal is its exact rational. Simplification runs in the backend's own
arithmetic: sympy evaluates binary-float arithmetic in binary floating point,
while the logical questions reason in exact rationals, so a ground
comparison doing float arithmetic can come out differently, for example
``(1e16 + 1.0) == 1e16`` simplifies to ``True`` while the logical questions
find it ``False``.
"""

__all__ = [
    "SatResult",
    "SatStatus",
    "Simplifier",
    "SmtLib2ProcessSolver",
    "SmtScript",
    "SmtSolver",
    "Solver",
    "SolverBackend",
    "SolverBackendError",
    "SolverBackendUnavailableError",
    "SolverCapabilityError",
    "SolverQueryKind",
    "assert_expression_implies",
    "assert_holds_for_all_free_assignments",
    "check_expression_satisfiability",
    "convert_expression_to_smtlib2",
    "does_expression_imply",
    "get_backend_capabilities",
    "get_default_solver",
    "holds_for_all_free_assignments",
    "is_backend_available",
    "set_default_solver",
    "simplify_expression",
    "validate_timeout_milliseconds",
]

from abc import ABC, abstractmethod
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from functools import cache

from immutabledict import immutabledict

from fhy_core import _rs
from fhy_core.error import register_error
from fhy_core.identifier import Identifier
from fhy_core.utils import StrEnum, is_strict_int
from fhy_core.utils.override import override

from .expression import Expression
from .symbol_type import SymbolType

Solver = _rs.Solver
SmtScript = _rs.SmtScript
SatResult = _rs.SatResult
SmtLib2ProcessSolver = _rs.SmtLib2ProcessSolver
get_default_solver = _rs.get_default_solver
set_default_solver = _rs.set_default_solver


class SolverBackend(StrEnum):
    """A shipped solver backend a question can name."""

    SYMPY = "sympy"
    Z3 = "z3"


class SolverQueryKind(StrEnum):
    """Kind of question a solver backend can be asked."""

    SIMPLIFICATION = "simplification"
    SATISFIABILITY = "satisfiability"
    IMPLICATION = "implication"
    UNIVERSAL_VALIDITY = "universal_validity"


class SatStatus(StrEnum):
    """The answer of an SMT-LIB2 ``check-sat``, as :attr:`SatResult.status`."""

    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"


@register_error
class SolverCapabilityError(ValueError):
    """Raised when the requested backend cannot answer the requested query kind."""


@register_error
class SolverBackendError(RuntimeError):
    """Raised when a backend implemented in Rust fails, naming the backend."""


@register_error
class SolverBackendUnavailableError(ImportError):
    """Raised when a question needs a backend whose package is not installed.

    The message names the package and the extra that installs it. It is
    raised when a question needs the backend, never when the package is
    imported, and constraints and params let it propagate: a missing package
    is a configuration error, not an undecided question.
    """


class SmtSolver(_rs.SmtSolverBase, ABC):
    """A backend that decides SMT-LIB2 scripts.

    Subclasses implement :meth:`check`, and may override :attr:`name`. A
    :class:`Solver` calls ``check`` from Rust, once per question, after its
    checks and the hazard screen passed. :class:`SmtLib2ProcessSolver`, a
    backend implemented in Rust, is registered as a virtual subclass.
    """

    @abstractmethod
    def check(
        self, script: SmtScript, *, timeout_milliseconds: int | None
    ) -> SatResult:
        """Decide whether the assertions of ``script`` can hold together.

        Args:
            script: The script, whose ``text`` is the whole SMT-LIB2 script
                without ``check-sat``.
            timeout_milliseconds: The bound of the check, or ``None`` for an
                unbounded one. A backend that runs out of time answers
                ``SatResult.unknown("timeout")``.

        Returns:
            ``SatResult.SAT``, ``SatResult.UNSAT``, or
            ``SatResult.unknown(reason)``.

        """

    @property
    def name(self) -> str:
        """Return the backend's name, which errors and warnings name it by."""
        return type(self).__name__


SmtSolver.register(SmtLib2ProcessSolver)


class Simplifier(_rs.SimplifierBase, ABC):
    """A backend that simplifies expressions.

    Subclasses implement :meth:`simplify`, and may override :attr:`name`. A
    :class:`Solver` screens the expression and substitutes the environment
    before it calls ``simplify`` from Rust, once per question.
    """

    @abstractmethod
    def simplify(self, expression: Expression) -> Expression:
        """Return a simpler expression equal to ``expression``.

        Simplification is best-effort: returning ``expression`` itself means
        nothing simpler was found.

        Args:
            expression: The expression, with the environment already
                substituted.

        Returns:
            The simplified expression.

        """

    @property
    def name(self) -> str:
        """Return the backend's name, which errors and warnings name it by."""
        return type(self).__name__


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
    """Return the query kinds the given kind of backend can answer.

    Args:
        backend: Backend to look up.

    Returns:
        The backend's supported query kinds, or an empty set for a
        backend with no capability table entry -- so an unrecognized
        backend is reported by the caller's capability check as a
        ``SolverCapabilityError`` rather than escaping as a ``KeyError``.

    """
    return _BACKEND_CAPABILITIES.get(backend, frozenset())


def _validate_backend_capability(
    backend: SolverBackend, query_kind: SolverQueryKind
) -> None:
    if query_kind not in get_backend_capabilities(backend):
        raise SolverCapabilityError(
            f"Backend {backend!r} cannot answer {query_kind!r} queries; "
            f"it supports {sorted(get_backend_capabilities(backend))}."
        )


def validate_timeout_milliseconds(timeout_milliseconds: int | None) -> None:
    """Raise unless ``timeout_milliseconds`` is ``None`` or a positive integer.

    Public so a caller that decides an outcome without reaching the solver
    -- and therefore never passes the value on -- can still hold up the
    same precondition the solver entry points enforce.

    Args:
        timeout_milliseconds: Candidate bound, in milliseconds.

    Raises:
        ValueError: If the value is not ``None`` and not a positive
            integer. A ``bool`` is refused even though it subclasses
            ``int``, and so is a ``float``: neither is the unsigned
            integer a backend's timeout takes.

    """
    if timeout_milliseconds is not None and (
        not is_strict_int(timeout_milliseconds) or timeout_milliseconds <= 0
    ):
        raise ValueError(
            "timeout_milliseconds must be None or a positive integer, but got "
            f"{timeout_milliseconds!r}."
        )


@cache
def _resolve_adapter(backend: SolverBackend) -> SmtSolver | Simplifier:
    """Return the adapter of ``backend``, created on first use and reused.

    Raises:
        SolverBackendUnavailableError: If the backend's package is not
            installed.

    """
    if backend is SolverBackend.Z3:
        from .expression.passes.z3 import Z3Solver  # noqa: PLC0415

        return Z3Solver()
    from .expression.passes.sympy import SympySimplifier  # noqa: PLC0415

    return SympySimplifier()


@cache
def _solver_of(backend: SolverBackend) -> Solver:
    """Return the solver holding the adapter of ``backend``, reused."""
    adapter = _resolve_adapter(backend)
    if isinstance(adapter, Simplifier):
        return Solver(simplifier=adapter)
    return Solver(smt_solver=adapter)


def _select_solver(
    backend: SolverBackend | None, query_kind: SolverQueryKind
) -> Solver:
    """Return the default solver for ``None``, or the solver of ``backend``.

    Raises:
        SolverCapabilityError: If ``backend`` cannot answer ``query_kind``.
        SolverBackendUnavailableError: If ``backend``'s package is not
            installed.

    """
    if backend is None:
        return get_default_solver()
    _validate_backend_capability(backend, query_kind)
    return _solver_of(SolverBackend(backend))


def is_backend_available(backend: SolverBackend) -> bool:
    """Return whether the package of ``backend`` is installed and imports.

    Args:
        backend: Backend to look up.

    Returns:
        True if a question can use the backend.

    """
    try:
        _resolve_adapter(SolverBackend(backend))
    except SolverBackendUnavailableError:
        return False
    return True


class _DeferredSmtSolver(SmtSolver):
    """The adapter of a shipped SMT backend, resolved on the first question.

    The default solver holds one, so importing this module imports no
    backend package, and a missing package is reported when a question
    needs it.
    """

    _backend: SolverBackend

    def __init__(self, backend: SolverBackend) -> None:
        super().__init__()
        self._backend = backend

    @property
    @override
    def name(self) -> str:
        return self._backend.value

    @override
    def check(
        self, script: SmtScript, *, timeout_milliseconds: int | None
    ) -> SatResult:
        adapter = _resolve_adapter(self._backend)
        assert isinstance(adapter, SmtSolver)
        return adapter.check(script, timeout_milliseconds=timeout_milliseconds)


class _DeferredSimplifier(Simplifier):
    """The adapter of a shipped simplifier, resolved on the first question."""

    _backend: SolverBackend

    def __init__(self, backend: SolverBackend) -> None:
        super().__init__()
        self._backend = backend

    @property
    @override
    def name(self) -> str:
        return self._backend.value

    @override
    def simplify(self, expression: Expression) -> Expression:
        adapter = _resolve_adapter(self._backend)
        assert isinstance(adapter, Simplifier)
        return adapter.simplify(expression)


set_default_solver(
    Solver(
        smt_solver=_DeferredSmtSolver(SolverBackend.Z3),
        simplifier=_DeferredSimplifier(SolverBackend.SYMPY),
    )
)


def convert_expression_to_smtlib2(
    expression: Expression,
    symbol_types: Mapping[Identifier, SymbolType] | None = None,
) -> str:
    """Return the SMT-LIB2 script of ``expression``.

    A Boolean expression is the script's one assertion; any other is named
    by the constant ``value``, which the script asserts equal to it.

    Args:
        expression: Expression to lower.
        symbol_types: Sort of each identifier of ``expression``. A native
            constant's canonical identifier needs none, and is refused.

    Returns:
        The script's text, without ``check-sat``.

    Raises:
        KeyError: If ``symbol_types`` misses an identifier.
        NonBooleanLogicalOperandError: If a Boolean position of
            ``expression`` provably holds a number.
        NativeConstantLoweringError: If ``expression`` refers to a native
            constant.
        TypeError: For a node SMT-LIB2 has no term for: a call, a Boolean
            meeting a number, a non-finite float, or a power without an
            integer literal exponent of at least one.

    """
    return SmtScript.lower(expression, symbol_types).text


def simplify_expression(
    expression: Expression,
    environment: Mapping[Identifier, Expression] | None = None,
    *,
    backend: SolverBackend | None = None,
) -> Expression:
    """Simplify an expression, optionally substituting an environment first.

    With an environment binding every free identifier, simplification is
    evaluation: the result is a ``LiteralExpression`` whenever the backend
    can decide the value. The sympy backend is best-effort: where
    ``sympy.simplify`` raises ``PrecisionExhausted``, drops a piecewise's
    otherwise branch because the case conditions before it hold for every
    real, or cannot compare a piecewise that has a Boolean identifier in a
    case condition and a branch that is not a real number, the expression
    comes back with ``environment`` substituted but unsimplified.

    Args:
        expression: Expression to simplify.
        environment: Environment to substitute into the expression before
            simplifying. Defaults to ``None``.
        backend: Backend to route the query to, or ``None`` (the default)
            for the default solver's simplifier.

    Returns:
        Simplified expression, or the substituted but unsimplified expression
        in the cases above.

    Raises:
        SolverCapabilityError: If ``backend`` cannot simplify.
        SolverBackendUnavailableError: If the backend's package is not
            installed.
        NativeConstantBindingError: If ``environment`` binds a registered
            native constant's canonical identifier that ``expression``
            references, whose value is fixed.
        NonBooleanLogicalOperandError: If an operand of a ``LogicalExpression``
            or ``LOGICAL_NOT`` node, or a piecewise case condition, provably
            denotes a number, counting an operand ``environment`` binds to
            one.

    """
    return _select_solver(backend, SolverQueryKind.SIMPLIFICATION).simplify_expression(
        expression, environment
    )


def check_expression_satisfiability(
    expression: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
    *,
    backend: SolverBackend | None = None,
    timeout_milliseconds: int | None = None,
) -> bool | None:
    """Return whether some assignment to the free identifiers satisfies the expression.

    True if a satisfying assignment provably exists; False if provably none
    exists; None if the backend answers unknown, or if ``expression`` is
    refused by the hazard screen documented on the module docstring (logged
    at WARNING, naming this function and the offending node).

    Args:
        expression: Expression to check.
        symbol_types: Sort of each free identifier of ``expression``.
        backend: Backend to route the query to, or ``None`` (the default)
            for the default solver's SMT backend.
        timeout_milliseconds: Optional bound, in milliseconds, on the
            backend's check. ``None`` (the default) leaves it unbounded.

    Returns:
        True, False, or None per the truth table above.

    Raises:
        SolverCapabilityError: If ``backend`` is not SATISFIABILITY-capable.
        SolverBackendUnavailableError: If the backend's package is not
            installed.
        ValueError: If ``timeout_milliseconds`` is not None and not positive.
        KeyError: If ``symbol_types`` lacks an entry for a free identifier
            other than a native constant's canonical identifier. Checked
            ahead of the hazard screen.
        NonBooleanLogicalOperandError: If ``expression``'s root, an operand
            of a ``LogicalExpression`` or ``LOGICAL_NOT`` node, a piecewise
            case condition, or a branch of a piecewise in a Boolean position
            provably denotes a number, counting an identifier
            ``symbol_types`` declares INT or REAL. Checked after the
            ``symbol_types`` precondition and ahead of the hazard screen.
        TypeError: If ``expression`` holds a call, which has to be inlined
            first.

    """
    return _select_solver(
        backend, SolverQueryKind.SATISFIABILITY
    ).check_expression_satisfiability(
        expression, symbol_types, timeout_milliseconds=timeout_milliseconds
    )


def does_expression_imply(
    antecedent: Expression,
    consequent: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
    *,
    backend: SolverBackend | None = None,
    timeout_milliseconds: int | None = None,
) -> bool | None:
    """Return whether the antecedent logically implies the consequent.

    None means the backend answered unknown, or that either side was
    refused by the hazard screen documented on the module docstring (logged
    at WARNING, naming this function and the offending node; the antecedent
    is screened before the consequent).

    Args:
        antecedent: The premise expression.
        consequent: The conclusion expression.
        symbol_types: Sort of each identifier referenced by either
            expression.
        backend: Backend to route the query to, or ``None`` (the default)
            for the default solver's SMT backend.
        timeout_milliseconds: Optional bound, in milliseconds, on the
            backend's check. ``None`` (the default) leaves it unbounded.

    Returns:
        True if the implication holds for every assignment; False if a
        counterexample exists; None if the backend answers unknown or
        either side is screened out.

    Raises:
        SolverCapabilityError: If ``backend`` is not IMPLICATION-capable.
        SolverBackendUnavailableError: If the backend's package is not
            installed.
        ValueError: If ``timeout_milliseconds`` is not None and not positive.
        KeyError: If ``symbol_types`` lacks an entry for a free identifier
            of either expression, other than a native constant's canonical
            identifier.
        NonBooleanLogicalOperandError: If either expression cannot be a
            predicate, as for :func:`check_expression_satisfiability`. Both
            are checked before either is screened.
        TypeError: If either expression holds a call.

    """
    return _select_solver(backend, SolverQueryKind.IMPLICATION).does_expression_imply(
        antecedent, consequent, symbol_types, timeout_milliseconds=timeout_milliseconds
    )


def holds_for_all_free_assignments(
    considered_identifiers: AbstractSet[Identifier],
    expression: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
    *,
    backend: SolverBackend | None = None,
    timeout_milliseconds: int | None = None,
) -> bool | None:
    """Return whether the expression holds for every free assignment.

    True iff ``forall <free identifiers>. exists <considered_identifiers>.
    expression``: for every assignment to the identifiers not in
    ``considered_identifiers``, some assignment to the considered ones
    satisfies the expression. With nothing considered, the check is
    universal validity.

    Args:
        considered_identifiers: Identifiers existentially quantified by
            the check; identifiers in the expression but not in this set
            are treated as free (universally quantified). One the
            expression does not mention quantifies nothing, but still needs
            a ``symbol_types`` entry.
        expression: Expression to check.
        symbol_types: Sort of each identifier appearing in the expression
            or considered.
        backend: Backend to route the query to, or ``None`` (the default)
            for the default solver's SMT backend.
        timeout_milliseconds: Optional bound, in milliseconds, on the
            backend's check. ``None`` (the default) leaves it unbounded.

    Returns:
        True if the expression has a witness for every free assignment;
        False if some free assignment has none; None if the backend answers
        unknown, or ``expression`` is refused by the hazard screen
        documented on the module docstring (logged at WARNING, naming this
        function and the offending node).

    Raises:
        SolverCapabilityError: If ``backend`` is not UNIVERSAL_VALIDITY-capable.
        SolverBackendUnavailableError: If the backend's package is not
            installed.
        ValueError: If ``timeout_milliseconds`` is not None and not positive.
        KeyError: If ``symbol_types`` lacks an entry for a free or
            considered identifier, other than a native constant's canonical
            identifier.
        NonBooleanLogicalOperandError: If ``expression`` cannot be a
            predicate, as for :func:`check_expression_satisfiability`.
        TypeError: If ``expression`` holds a call.

    """
    return _select_solver(
        backend, SolverQueryKind.UNIVERSAL_VALIDITY
    ).holds_for_all_free_assignments(
        considered_identifiers,
        expression,
        symbol_types,
        timeout_milliseconds=timeout_milliseconds,
    )


def assert_holds_for_all_free_assignments(
    considered_identifiers: AbstractSet[Identifier],
    expression: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
    *,
    backend: SolverBackend | None = None,
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
        UndecidableError: When the backend answers unknown, with its reason
            (for example ``"timeout"``), or when ``expression`` is refused
            by the hazard screen, with the reason ``"hazard_screen"``
            (logged at WARNING, naming this function and the offending
            node).
        SolverCapabilityError: As for
            :func:`holds_for_all_free_assignments`.
        SolverBackendUnavailableError: As for
            :func:`holds_for_all_free_assignments`.
        ValueError: As for :func:`holds_for_all_free_assignments`.
        KeyError: As for :func:`holds_for_all_free_assignments`.
        NonBooleanLogicalOperandError: As for
            :func:`holds_for_all_free_assignments`; reported as its own
            error rather than as ``UndecidableError``, since no
            ``timeout_milliseconds`` makes an ill-typed expression
            decidable.
        TypeError: As for :func:`holds_for_all_free_assignments`.

    """
    return _select_solver(
        backend, SolverQueryKind.UNIVERSAL_VALIDITY
    ).assert_holds_for_all_free_assignments(
        considered_identifiers,
        expression,
        symbol_types,
        timeout_milliseconds=timeout_milliseconds,
    )


def assert_expression_implies(
    antecedent: Expression,
    consequent: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
    *,
    backend: SolverBackend | None = None,
    timeout_milliseconds: int | None = None,
) -> bool:
    """Check the implication, raising ``UndecidableError`` on ``unknown``.

    Args:
        antecedent: As for :func:`does_expression_imply`.
        consequent: As for :func:`does_expression_imply`.
        symbol_types: As for :func:`does_expression_imply`.
        backend: As for :func:`does_expression_imply`.
        timeout_milliseconds: As for :func:`does_expression_imply`.

    Returns:
        The decided ``bool`` result.

    Raises:
        UndecidableError: When the backend answers unknown, with its reason,
            or when either expression is refused by the hazard screen, with
            the reason ``"hazard_screen"``.
        SolverCapabilityError: As for :func:`does_expression_imply`.
        SolverBackendUnavailableError: As for :func:`does_expression_imply`.
        ValueError: As for :func:`does_expression_imply`.
        KeyError: As for :func:`does_expression_imply`.
        NonBooleanLogicalOperandError: As for :func:`does_expression_imply`;
            reported as its own error rather than as ``UndecidableError``.
        TypeError: As for :func:`does_expression_imply`.

    """
    return _select_solver(
        backend, SolverQueryKind.IMPLICATION
    ).assert_expression_implies(
        antecedent, consequent, symbol_types, timeout_milliseconds=timeout_milliseconds
    )
