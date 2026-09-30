"""The z3-solver backend of the solver, and the lowering of expressions to z3.

:class:`Z3Solver` is the :class:`~fhy_core.symbolic.solver.SmtSolver` that
``SolverBackend.Z3`` names: it decides a script with the ``z3-solver``
package, which releases the interpreter during its calls into z3.
:func:`convert_expression_to_z3_expression` parses the Rust lowering of an
expression into z3 terms.

Importing this module imports ``z3``; without the ``z3-solver`` package it
raises :class:`~fhy_core.symbolic.solver.SolverBackendUnavailableError`,
naming the extra to install.
"""

__all__ = ["Z3Solver", "convert_expression_to_z3_expression"]

from collections.abc import Mapping

from immutabledict import immutabledict

from fhy_core.identifier import Identifier
from fhy_core.symbolic.solver import (
    SatResult,
    SmtScript,
    SmtSolver,
    SolverBackendUnavailableError,
)
from fhy_core.symbolic.symbol_type import SymbolType
from fhy_core.utils.override import override

from ..core import Expression

try:
    import z3  # type: ignore[import-untyped]
except ImportError as error:
    raise SolverBackendUnavailableError(
        "The z3 solver backend needs the z3-solver package, which is not "
        "installed; install it with `pip install fhy_core[z3]`, or "
        "`pip install fhy_core[solvers]` for every solver backend."
    ) from error


class Z3Solver(SmtSolver):
    """The SMT backend of the ``z3-solver`` package.

    Each check creates a ``z3.SimpleSolver``, applies the timeout with
    ``set(timeout=...)``, reads the script with ``from_string``, and checks
    it; a check that runs out of time answers ``unknown`` with the reason
    ``"timeout"``. A one-shot check needs none of the incremental front end of
    ``z3.Solver()``, which costs milliseconds to set up. It holds no state,
    so one object serves every question.
    """

    @property
    @override
    def name(self) -> str:
        return "z3"

    @override
    def check(
        self, script: SmtScript, *, timeout_milliseconds: int | None
    ) -> SatResult:
        solver = z3.SimpleSolver()
        if timeout_milliseconds is not None:
            solver.set(timeout=timeout_milliseconds)
        solver.from_string(script.text)
        result = solver.check()
        if result == z3.sat:
            return SatResult.SAT
        elif result == z3.unsat:
            return SatResult.UNSAT
        elif result == z3.unknown:
            reason = solver.reason_unknown()
            # The simple solver stops at its timeout by canceling the check;
            # a backend that runs out of time answers "timeout".
            if reason == "canceled" and timeout_milliseconds is not None:
                reason = "timeout"
            return SatResult.unknown(reason)
        raise RuntimeError(f"Unexpected Z3 result: {result!r}.")


def _build_constant(symbol: str, sort: SymbolType) -> z3.ExprRef:
    if sort is SymbolType.INT:
        return z3.Int(symbol)
    if sort is SymbolType.REAL:
        return z3.Real(symbol)
    return z3.Bool(symbol)


def convert_expression_to_z3_expression(
    expression: Expression,
    symbol_types: Mapping[Identifier, SymbolType] | None = None,
) -> tuple[z3.ExprRef, immutabledict[Identifier, z3.ExprRef]]:
    """Convert an expression to a z3 expression.

    The expression is lowered to SMT-LIB2 in Rust, with this package's
    semantics (exact division, floor division and modulo toward negative
    infinity, exact rationals for float and decimal literals), and the
    script is parsed with ``z3.parse_smt2_string``. Each identifier becomes
    the z3 constant ``<name_hint>_<id>`` of its sort.

    Args:
        expression: Expression to convert.
        symbol_types: Sort of each identifier. A native constant's canonical
            identifier needs no entry, and is refused.

    Returns:
        The z3 expression and the z3 constant of each identifier.

    Raises:
        KeyError: If ``symbol_types`` misses an identifier other than a
            native constant's canonical identifier.
        NonBooleanLogicalOperandError: If an operand of a
            ``LogicalExpression`` or ``LOGICAL_NOT`` node, or a piecewise
            case condition, provably denotes a number. Checked after the
            ``symbol_types`` precondition.
        NativeConstantLoweringError: If ``expression`` references a
            registered native constant's canonical identifier. Checked after
            both errors above.
        TypeError: For a node SMT-LIB2 has no term for: a call, a Boolean
            meeting a number, a non-finite float, or a power without an
            integer literal exponent of at least one.

    """
    script = SmtScript.lower(expression, symbol_types)
    (assertion,) = z3.parse_smt2_string(script.text)
    term = assertion if script.value_sort is None else assertion.arg(1)
    constants = {
        identifier: _build_constant(symbol, sort)
        for identifier, symbol, sort in script.declarations
    }
    return term, immutabledict(constants)
