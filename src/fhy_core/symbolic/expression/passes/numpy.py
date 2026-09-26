"""Evaluate an expression over NumPy values.

The evaluation runs in the Rust core (``fhy_core::expression::evaluate``,
S9 of ``docs/design/python-switch.md``), over NumPy arrays read through
rust-numpy. :func:`evaluate_expression_with_numpy` inlines composed
built-ins and registered functions, converts the environment's bindings
of the identifiers the inlined tree refers to, and walks the tree once,
broadcasting as NumPy does.

The evaluation computes in three domains: ``bool``, ``int64`` and
``float64``. A binding of another Boolean, integer or floating-point dtype
is converted to one of them (``float32`` becomes ``float64``, a
``uint64`` above the signed range raises ``OverflowError``), and object,
complex, string and other dtypes are refused with ``TypeError``.

- Integer arithmetic is checked: an overflow raises ``OverflowError``, and
  an integer floor division or remainder by zero ``ZeroDivisionError``.
  ``/`` is the real quotient even of two integers; ``//`` rounds toward
  negative infinity and ``%`` has the sign of the divisor. An integer
  raised to a negative integer power raises ``ValueError``. Mixed integer
  and real operands compute in reals.
- Real arithmetic is IEEE's: ``sqrt(-1)``, ``log(0)`` and division by zero
  give ``nan`` or ``inf``, without a warning.
- A Boolean compares with ``==`` and ``!=`` and is refused (``TypeError``)
  in arithmetic and orderings; a connective operand or piecewise condition
  that is a number is refused with :class:`NonBooleanLogicalOperandError`.
- An ``INT``-sorted native (``round``, ``floor``, ``ceil``) needs a finite
  value in the ``int64`` range: ``NonFiniteCastError`` or
  ``OverflowError`` otherwise. ``erf`` and ``gelu`` are computed.
- A failure belongs to its element: a piecewise raises it only for an
  element whose selected branch fails, and ``&&`` or ``||`` only where no
  other operand decides the element, so ``{floor(sqrt(x)) if x >= 0; 0
  otherwise}`` and ``(y != 0) && (x // y > 1)`` evaluate everywhere.

A 0-d result is a NumPy scalar (``numpy.bool_``, ``numpy.int64`` or
``numpy.float64``); any other result is a new C-contiguous array, never a
binding. Every error is raised directly, under its own class.

NumPy is an optional dependency: importing this module never imports
NumPy, and without it the evaluation raises :class:`ImportError` with the
extra to install.
"""

__all__ = [
    "evaluate_expression_with_numpy",
]

from typing import TYPE_CHECKING, Any, TypeAlias

from immutabledict import immutabledict

from fhy_core import _rs
from fhy_core.pass_infrastructure import (
    CompilerPass,
    PassExecutionError,
    register_pass,
)
from fhy_core.utils.override import override

from ..core import Expression

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    import numpy.typing as npt

    from fhy_core.identifier import Identifier

    NumpyEnvironment: TypeAlias = Mapping[Identifier, npt.ArrayLike]
    """Binding of each free identifier to a NumPy-consumable value."""

    NumpyResult: TypeAlias = npt.NDArray[Any] | np.generic
    """Concrete value produced by NumPy evaluation: an array or a scalar."""


@register_pass(
    "fhy_core.symbolic.expression.evaluate_with_numpy",
    "Evaluate a fully-bound expression tree to concrete NumPy values.",
)
class NumpyExpressionEvaluator(CompilerPass[Expression, "NumpyResult"]):
    """Pass evaluating an expression over NumPy values.

    A run evaluates its input over the environment given at construction,
    which is copied then, as :func:`evaluate_expression_with_numpy` does. A
    failure fails the run with ``PassExecutionError``, whose ``__cause__``
    is the error the function raises.
    """

    _environment: "immutabledict[Identifier, npt.ArrayLike]"

    def __init__(self, environment: "NumpyEnvironment") -> None:
        super().__init__()
        self._environment = immutabledict(environment)

    @override
    def run_pass(self, ir: Expression) -> "NumpyResult":
        result: NumpyResult = _rs.evaluate_expression_with_numpy(ir, self._environment)
        return result

    @override
    def did_change(self, input_ir: Expression, output: "NumpyResult") -> bool:
        """Report that evaluation always produces a new value."""
        _ = (input_ir, output)
        return True

    @override
    def get_noop_output(self, ir: Expression) -> "NumpyResult":
        raise PassExecutionError(
            f'Pass "{self.get_pass_name()}" does not define noop output.'
        )


def evaluate_expression_with_numpy(
    expression: Expression,
    environment: "NumpyEnvironment",
) -> "NumpyResult":
    """Evaluate ``expression`` to a NumPy value over ``environment``.

    Args:
        expression: Expression tree to evaluate. Every free identifier
            (after inlining) must be bound in ``environment`` or be a
            constant's canonical identifier (``pi``, ``e``, ``inf``,
            ``nan``, or a registered constant).
        environment: Value for each free identifier, as anything NumPy
            accepts (``ndarray``, scalar, or nested sequence). Bindings of
            identifiers the inlined expression does not refer to are
            ignored.

    Returns:
        A NumPy scalar when every binding is a scalar, and otherwise a new
        array of the bindings' broadcast shape.

    Raises:
        ImportError: If NumPy is not installed.
        EntryLookupError: A call names an unregistered function.
        FunctionArityError: A call's argument count is wrong, or its
            target is a constant.
        RecursionError: A registered function is recursive.
        NonBooleanLogicalOperandError: A connective operand, a negation's
            operand, or a piecewise condition is a number.
        NativeConstantBindingError: ``environment`` binds a constant's
            canonical identifier the expression refers to.
        UnboundVariableError: A free identifier is neither bound nor a
            constant.
        StringLiteralPrecisionError: A decimal literal has no exact binary
            float.
        UnsupportedNumpyLoweringError: A call of a registered native
            function, which has no implementation the evaluator can run.
        NonFiniteCastError: An ``INT``-sorted native's selected result is
            ``nan`` or infinite.
        OverflowError: An integer overflows ``int64``, or a binding does
            not fit it.
        ZeroDivisionError: An integer is floor-divided by zero, or its
            remainder taken.
        ValueError: Shapes do not broadcast, or an integer is raised to a
            negative integer power.
        TypeError: A Boolean is used as a number, a piecewise mixes
            Booleans and numbers, or a binding's dtype is unsupported.

    """
    result: NumpyResult = _rs.evaluate_expression_with_numpy(expression, environment)
    return result
