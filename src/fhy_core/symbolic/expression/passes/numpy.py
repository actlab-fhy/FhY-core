"""Evaluate a fully-bound expression tree to concrete NumPy values.

This pass is the fast path for computing a function authored in the
expression vocabulary over data. Given an expression and an environment
binding every free identifier to a NumPy-consumable value, it walks the
tree once and applies a vectorized NumPy operation at each node, producing
a NumPy array (or scalar) rather than another :class:`Expression`.

Walking the tree once with whole arrays bound to the variables issues the
same sequence of NumPy calls a user would write by hand, so throughput
approaches native NumPy for an element-wise transform over a large array:
the tree walk is ``O(tree_size)`` Python calls, each dispatching one
C-level operation over all elements. Contrast the existing paths, which are
unsuited to per-value computation over arrays: ``evaluate_expression``
folds only all-literal native calls and returns an ``Expression``, and
``simplify_expression`` runs symbolic SymPy algebra on scalars.

A native call's result is cast to the dtype of its declared result sort,
so ``floor``/``round``/``ceil`` (declared ``INT``) return an integer array
rather than NumPy's floating-point default, agreeing with the declared
sort and with ``evaluate_expression``. Floating-point domain conditions
follow NumPy: ``sqrt(-1)``, ``log(0)``, and division by zero produce
``nan``/``inf`` (with NumPy's usual warning) rather than raising -- unlike
``evaluate_expression``, whose scalar native implementations raise. NumPy
does still raise for operations it rejects outright rather than folding to
``nan``/``inf`` -- most notably an integer base raised to a negative integer
power (``numpy.power``) -- and such a ``ValueError`` surfaces wrapped in
``PassExecutionError``.

A piecewise expression lowers to a right-folded chain of ``numpy.where``
calls, one per case. This is not lazy: every case value and ``otherwise``
are evaluated for every element before selection. A domain error in an
*unselected* case still produces its ``nan``/``inf`` (and warning); the
conditions do not guard their sibling cases from evaluation.

Expression-bodied built-ins (``relu``, ``sigmoid``, ``clamp``, ...) are
inlined automatically before the walk via ``inline_functions``, so the
caller does not pre-inline them. ``erf`` (and therefore ``gelu``) has no
NumPy ufunc and is unsupported.

NumPy is an optional dependency: importing this module never imports
NumPy. The runtime import happens inside
:func:`evaluate_expression_with_numpy`; when NumPy is not installed it
raises :class:`ImportError` with guidance to install the ``numpy`` extra.
"""

__all__ = [
    "evaluate_expression_with_numpy",
]

from typing import TYPE_CHECKING, Any, TypeAlias

from immutabledict import immutabledict

from fhy_core.pass_infrastructure import (
    PassExecutionError,
    VisitablePass,
    register_pass,
)
from fhy_core.utils.override import override

from ..core import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
)
from ..errors import (
    EntryLookupError,
    UnboundVariableError,
    UnsupportedNumpyLoweringError,
)
from ..registry import NativeFunction, get_registered_entry
from ..sort import FunctionSort
from .inline import inline_functions
from .native_lowering import coerce_literal_value, try_get_native_constant_value

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    import numpy.typing as npt

    from fhy_core.identifier import Identifier

    NumpyEnvironment: TypeAlias = Mapping[Identifier, npt.ArrayLike]
    """Binding of each free identifier to a NumPy-consumable value."""

    NumpyResult: TypeAlias = npt.NDArray[Any] | np.generic | bool | int | float
    """Concrete value produced by NumPy evaluation: an array or a scalar."""


# Binary operation -> attribute name of the NumPy ufunc that lowers it.
_BINARY_UFUNC_NAMES: immutabledict[BinaryOperation, str] = immutabledict(
    {
        BinaryOperation.ADD: "add",
        BinaryOperation.SUBTRACT: "subtract",
        BinaryOperation.MULTIPLY: "multiply",
        BinaryOperation.DIVIDE: "true_divide",
        BinaryOperation.FLOOR_DIVIDE: "floor_divide",
        BinaryOperation.MODULO: "mod",
        BinaryOperation.POWER: "power",
        BinaryOperation.LOGICAL_AND: "logical_and",
        BinaryOperation.LOGICAL_OR: "logical_or",
        BinaryOperation.EQUAL: "equal",
        BinaryOperation.NOT_EQUAL: "not_equal",
        BinaryOperation.LESS: "less",
        BinaryOperation.LESS_EQUAL: "less_equal",
        BinaryOperation.GREATER: "greater",
        BinaryOperation.GREATER_EQUAL: "greater_equal",
    }
)

# Unary operation -> attribute name of the NumPy ufunc that lowers it.
_UNARY_UFUNC_NAMES: immutabledict[UnaryOperation, str] = immutabledict(
    {
        UnaryOperation.NEGATE: "negative",
        UnaryOperation.POSITIVE: "positive",
        UnaryOperation.LOGICAL_NOT: "logical_not",
    }
)

# Native-function name -> attribute name of the NumPy function that lowers it.
# Most are ufuncs; ``round`` resolves to ``numpy.round`` (an array function).
# ``erf`` is intentionally absent: NumPy has no vectorized ``erf``.
_NATIVE_FUNCTION_UFUNC_NAMES: immutabledict[str, str] = immutabledict(
    {
        "exp": "exp",
        "exp2": "exp2",
        "log": "log",
        "log2": "log2",
        "log10": "log10",
        "sqrt": "sqrt",
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "arcsin": "arcsin",
        "arccos": "arccos",
        "arctan": "arctan",
        "sinh": "sinh",
        "cosh": "cosh",
        "tanh": "tanh",
        "round": "round",
        "floor": "floor",
        "ceil": "ceil",
    }
)

# Result sort -> attribute name of the NumPy dtype a native-call result is
# cast to, so the result conforms to the call's declared result sort.
# ``REAL`` is intentionally absent: real-sorted results are already
# floating-point and pass through with their width preserved (e.g. a
# ``float32`` input stays ``float32``).
_SORT_CAST_DTYPE_NAMES: immutabledict[FunctionSort, str] = immutabledict(
    {
        FunctionSort.BOOL: "bool_",
        FunctionSort.NAT: "int64",
        FunctionSort.INT: "int64",
    }
)


def _import_numpy() -> Any:
    """Import and return NumPy, or raise a guiding ``ImportError``."""
    try:
        import numpy  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "NumPy is required for `evaluate_expression_with_numpy`; install it "
            "with `pip install fhy_core[numpy]`."
        ) from error
    return numpy


@register_pass(
    "fhy_core.symbolic.expression.evaluate_with_numpy",
    "Evaluate a fully-bound expression tree to concrete NumPy values.",
)
class NumpyExpressionEvaluator(VisitablePass[Expression, "NumpyResult"]):
    """Bottom-up evaluator lowering an expression tree to NumPy values.

    Each node is lowered to a vectorized NumPy operation over its
    already-evaluated children. Identifiers resolve against the caller's
    environment (coerced with ``numpy.asarray``) or the native-constant
    registry; every other free identifier raises
    :class:`UnboundVariableError`. A native call with no NumPy lowering
    (``erf``, or a non-built-in native) raises
    :class:`UnsupportedNumpyLoweringError`.

    Expression-bodied calls are expected to have been inlined before the
    walk; :func:`evaluate_expression_with_numpy` runs ``inline_functions``
    first, so only native calls reach :meth:`visit_call_expression`.
    """

    _environment: "NumpyEnvironment"
    # NumPy is optional, so its module and result values are typed ``Any``
    # rather than referencing NumPy types at import time.
    _numpy: Any

    def __init__(self, environment: "NumpyEnvironment", numpy_module: Any) -> None:
        super().__init__()
        self._environment = environment
        self._numpy = numpy_module

    def visit_literal_expression(self, expression: LiteralExpression) -> Any:
        """Return the literal's value, coercing string-form numerics."""
        return coerce_literal_value(expression.value)

    def visit_identifier_expression(self, expression: IdentifierExpression) -> Any:
        """Resolve an identifier to its bound array or native-constant value."""
        identifier = expression.identifier
        if identifier in self._environment:
            return self._numpy.asarray(self._environment[identifier])
        constant_value = try_get_native_constant_value(identifier.name_hint)
        if constant_value is not None:
            return constant_value
        raise UnboundVariableError(self._describe_unbound_identifier(identifier))

    def _describe_unbound_identifier(self, identifier: "Identifier") -> str:
        """Explain why a bare identifier could not be resolved to a value.

        A name that resolves to a registered function (not a constant) is
        the likely result of dropping a call, so the message points the
        caller at that instead of the generic native-constant hint.
        """
        name_hint = identifier.name_hint
        try:
            get_registered_entry(name_hint)
        except EntryLookupError:
            return (
                f"identifier {name_hint!r} is not bound in the environment "
                f"and does not match a registered native constant."
            )
        return (
            f"identifier {name_hint!r} names a registered function, not a "
            f"value; call it as {name_hint}(...) or bind it in the environment."
        )

    def visit_unary_expression(self, expression: UnaryExpression) -> Any:
        """Apply the NumPy ufunc for a unary operation to its operand."""
        operand = self.visit(expression.operand)
        ufunc = getattr(self._numpy, _UNARY_UFUNC_NAMES[expression.operation])
        return ufunc(operand)

    def visit_binary_expression(self, expression: BinaryExpression) -> Any:
        """Apply the NumPy ufunc for a binary operation to its operands."""
        left = self.visit(expression.left)
        right = self.visit(expression.right)
        ufunc = getattr(self._numpy, _BINARY_UFUNC_NAMES[expression.operation])
        return ufunc(left, right)

    def visit_piecewise_expression(self, expression: PiecewiseExpression) -> Any:
        """Select elementwise via a right-folded chain of ``numpy.where`` calls.

        Every case value and ``otherwise`` are evaluated for every
        element before selection (``numpy.where`` is not lazy), so a
        domain error (division by zero, ``log`` of a non-positive) in an
        unselected case still produces its ``nan``/``inf`` and NumPy
        warning -- the conditions do not guard their sibling cases from
        evaluation. The chain is right-folded from ``otherwise``, so the
        first case's ``numpy.where`` is outermost and first-match-wins
        holds.
        """
        case_values = [
            (self.visit(condition), self.visit(value))
            for condition, value in expression.get_cases()
        ]
        result = self.visit(expression.otherwise)
        for condition_value, value_value in reversed(case_values):
            result = self._numpy.where(condition_value, value_value, result)
        return result

    def visit_call_expression(self, expression: CallExpression) -> Any:
        """Apply a native call's NumPy operation, then cast to its result sort.

        The registered entry is resolved once and dispatched by type --
        mirroring ``evaluate_expression`` and ``inline_functions`` -- so a
        name that is not a :class:`NativeFunction` with a ufunc mapping
        (``erf``, or a caller-registered native) raises
        :class:`UnsupportedNumpyLoweringError` before any NumPy work.
        """
        function_name = expression.function_name
        entry = get_registered_entry(function_name)
        ufunc_name = _NATIVE_FUNCTION_UFUNC_NAMES.get(function_name)
        if not isinstance(entry, NativeFunction) or ufunc_name is None:
            raise UnsupportedNumpyLoweringError(
                f"native function {function_name!r} has no NumPy lowering."
            )
        ufunc = getattr(self._numpy, ufunc_name)
        arguments = [self.visit(argument) for argument in expression.arguments]
        return self._cast_to_result_sort(ufunc(*arguments), entry.result_sort)

    def _cast_to_result_sort(self, result: Any, result_sort: FunctionSort) -> Any:
        """Cast a native-call result to the dtype of its declared result sort.

        Integer- and boolean-sorted natives (``floor``, ``round``, ...)
        would otherwise return NumPy's floating-point default; casting
        makes the NumPy path agree with the declared sort and with
        ``evaluate_expression``. Real-sorted results are already
        floating-point and pass through unchanged, preserving their width.
        A non-finite (``nan``/``inf``) result of an integer-sorted native
        casts to a platform-defined sentinel per NumPy's own ``astype``
        semantics (no exception is raised).
        """
        dtype_name = _SORT_CAST_DTYPE_NAMES.get(result_sort)
        if dtype_name is None:
            return result
        return result.astype(getattr(self._numpy, dtype_name))

    @override
    def did_change(self, input_ir: Expression, output: "NumpyResult") -> bool:
        """Report that evaluation always produces a new value.

        The default change detection compares ``input_ir`` against
        ``output`` with ``!=``. Here ``output`` is a NumPy array, so that
        comparison would trigger NumPy's elementwise machinery over the
        whole array -- an ``O(n_elements)`` cost on every call. Evaluation
        never returns the input tree, so this reports ``True`` directly.
        """
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
    """Evaluate ``expression`` to a concrete NumPy value.

    Expression-bodied function calls are inlined first, then the tree is
    walked once: each node applies the corresponding vectorized NumPy
    operation to its already-evaluated children. Every free identifier
    must be resolvable -- bound in ``environment`` or matching a
    registered native constant (``pi``, ``e``, ``inf``, ``nan``).

    Args:
        expression: Expression tree to evaluate. Every free identifier
            (after inlining) must be bound in ``environment`` or be a
            registered native constant.
        environment: Value for each free identifier, as anything NumPy
            accepts (``ndarray``, scalar, or nested sequence). Bindings
            for identifiers that are not free in ``expression`` are
            ignored.

    Returns:
        The evaluated value: a NumPy array when any bound variable is
        array-valued. For a fully scalar environment the result is a NumPy
        or Python scalar, except that a piecewise- or bare-identifier-rooted
        expression returns a rank-0 ``ndarray``. A native call's result is
        cast to the dtype of its declared result sort (so
        ``floor``/``round``/``ceil`` yield integers); every other node's
        dtype follows NumPy's type-promotion rules. Floating-point domain
        conditions (``sqrt(-1)``, ``log(0)``, division by zero) follow
        NumPy and yield ``nan``/``inf`` rather than raising -- unlike
        ``evaluate_expression``, whose scalar native implementations raise.
        A piecewise expression evaluates every case value and ``otherwise``
        for every element (it lowers to a chain of ``numpy.where`` calls),
        so a domain error in an unselected case is not suppressed by its
        condition.

    Raises:
        ImportError: If NumPy is not installed. Raised directly, before
            any evaluation, with guidance to install the ``numpy`` extra.
        PassExecutionError: Wraps each domain failure below, with the
            underlying typed error attached as ``__cause__`` (matching the
            sibling expression passes). The underlying errors are:

            - :class:`UnboundVariableError`: a free identifier is neither
              bound in ``environment`` nor a registered native constant.
            - :class:`UnsupportedNumpyLoweringError`: a node has no NumPy
              lowering (``erf``/``gelu``, or a non-built-in native
              function).
            - :class:`StringLiteralPrecisionError`: a float-grammar
              string literal cannot be coerced to a binary ``float``
              without precision loss.
            - :class:`EntryLookupError`: a call references an
              unregistered function name.
            - :class:`FunctionArityError`: a call's argument count does
              not match its registered arity, or the call target is a
              native constant.
            - ``RecursionError``: a registered function is transitively
              recursive and cannot be inlined.
            - ``ValueError``: raised by NumPy for an operation it rejects
              on the given dtypes rather than folding to ``nan``/``inf``
              -- most notably an integer base raised to a negative integer
              power (``numpy.power``).

    """
    numpy_module = _import_numpy()
    inlined_expression = inline_functions(expression)
    evaluator = NumpyExpressionEvaluator(environment, numpy_module)
    return evaluator(inlined_expression)
