"""Evaluate a fully-bound expression tree to concrete NumPy values.

This pass is the fast path for computing a function authored in the
expression vocabulary over data. Given an expression and an environment
binding every free identifier to a NumPy-consumable value, it walks the
tree once and applies a vectorized NumPy ufunc at each node, producing a
NumPy array (or scalar) rather than another :class:`Expression`.

Walking the tree once with whole arrays bound to the variables issues the
same sequence of NumPy calls a user would write by hand, so throughput
approaches native NumPy for an element-wise transform over a large array:
the tree walk is ``O(tree_size)`` Python calls, each dispatching one
C-level ufunc over all elements. Contrast the existing paths, which are
unsuited to per-value computation over arrays: ``evaluate_expression``
folds only all-literal native calls and returns an ``Expression``, and
``simplify_expression`` runs symbolic SymPy algebra on scalars.

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

from typing import TYPE_CHECKING, Any, TypeAlias, cast

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
    TernaryExpression,
    UnaryExpression,
    UnaryOperation,
)
from ..errors import UnboundVariableError, UnsupportedNumpyLoweringError
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

# Native-function name -> attribute name of the NumPy ufunc that lowers it.
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
    "fhy_core.expression.evaluate_with_numpy",
    "Evaluate a fully-bound expression tree to concrete NumPy values.",
)
class NumpyExpressionEvaluator(VisitablePass[Expression, Any]):
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
        raise UnboundVariableError(
            f"identifier {identifier.name_hint!r} is not bound in the "
            f"environment and does not match a registered native constant."
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

    def visit_ternary_expression(self, expression: TernaryExpression) -> Any:
        """Select elementwise between the branches with ``numpy.where``."""
        condition = self.visit(expression.condition)
        true_value = self.visit(expression.true_value)
        false_value = self.visit(expression.false_value)
        return self._numpy.where(condition, true_value, false_value)

    def visit_call_expression(self, expression: CallExpression) -> Any:
        """Apply the NumPy ufunc for a native-function call to its arguments."""
        ufunc_name = _NATIVE_FUNCTION_UFUNC_NAMES.get(expression.function_name)
        if ufunc_name is None:
            raise UnsupportedNumpyLoweringError(
                f"native function {expression.function_name!r} has no NumPy lowering."
            )
        ufunc = getattr(self._numpy, ufunc_name)
        arguments = [self.visit(argument) for argument in expression.arguments]
        return ufunc(*arguments)

    @override
    def did_change(self, input_ir: Expression, output: Any) -> bool:
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
    def get_noop_output(self, ir: Expression) -> Any:
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
        array-valued, otherwise a NumPy or Python scalar. Result dtype
        follows NumPy's type-promotion rules; declared result sorts are
        not enforced.

    Raises:
        ImportError: If NumPy is not installed. Raised directly, before
            any evaluation, with guidance to install the ``numpy`` extra.
        PassExecutionError: For every domain failure, with the underlying
            typed error attached as ``__cause__`` (matching the sibling
            expression passes). The underlying errors are:

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

    """
    numpy_module = _import_numpy()
    inlined_expression = inline_functions(expression)
    evaluator = NumpyExpressionEvaluator(environment, numpy_module)
    return cast("NumpyResult", evaluator(inlined_expression))
