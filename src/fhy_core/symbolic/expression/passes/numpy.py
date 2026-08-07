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
follow NumPy at the point of computation: ``sqrt(-1)``, ``log(0)``, and
division by zero produce ``nan``/``inf`` (with NumPy's usual warning)
rather than raising -- unlike ``evaluate_expression``, whose scalar native
implementations raise immediately. A non-finite value that reaches an
integer- or boolean-sorted cast (for example ``round(nan)``) raises
``NonFiniteCastError`` instead of being cast to a platform-defined
sentinel. NumPy does still raise for operations it rejects outright
rather than folding to ``nan``/``inf`` -- most notably an integer base
raised to a negative integer power (``numpy.power``) -- and such a
``ValueError`` surfaces wrapped in ``PassExecutionError``.

A piecewise expression lowers to a right-folded chain of ``numpy.where``
calls, one per case. This is not lazy: every case value and ``otherwise``
are evaluated for every element before selection. A domain error in an
*unselected* case still produces its ``nan``/``inf`` (and warning); the
conditions do not guard their sibling cases from evaluation. They do
guard the integer-sorted cast of the result, though: a non-finite value
arising inside a branch raises ``NonFiniteCastError`` only for elements
the selected branch actually returns, so guarding a domain error with a
condition -- ``{floor(sqrt(x)) if x >= 0; 0 otherwise}`` -- yields the
guarded value rather than an error. Each condition must be
boolean-dtyped: ``numpy.where`` would otherwise silently treat a nonzero
numeric condition as true, so a non-boolean condition raises
``TypeError``.

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
    NonFiniteCastError,
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
    # Non-``None`` only while a piecewise branch is being evaluated, where a
    # non-finite cast accumulates its offending elements into this mask
    # instead of raising; see :meth:`_visit_branch_deferring_non_finite`.
    _deferred_non_finite: Any | None

    def __init__(self, environment: "NumpyEnvironment", numpy_module: Any) -> None:
        super().__init__()
        self._environment = environment
        self._numpy = numpy_module
        self._deferred_non_finite = None

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
        evaluation. They do, however, guard the integer-sorted cast of
        those values: a non-finite value produced inside a branch is
        recorded per element rather than raising, and the recorded masks
        are folded through the same ``numpy.where`` chain as the values,
        so :class:`NonFiniteCastError` is raised only for an element the
        selected branch actually returns. Guarding a domain error with a
        condition therefore works. A nested piecewise passes its own
        surviving mask up instead of raising, so the outermost node --
        the only one whose selection the caller actually receives -- is
        what decides.

        The chain is right-folded from ``otherwise``, so the first case's
        ``numpy.where`` is outermost and first-match-wins holds. Each
        condition must be boolean-dtyped: ``numpy.where`` would otherwise
        silently treat a nonzero numeric condition as true, so a
        non-boolean condition raises ``TypeError`` instead.
        """
        lowered_cases: list[tuple[Any, Any, Any]] = []
        for index, (condition, value) in enumerate(expression.get_cases()):
            condition_value = self.visit(condition)
            if not self._is_boolean_condition_value(condition_value):
                raise TypeError(
                    f"piecewise case {index} condition must be boolean-dtyped, "
                    f"but got dtype "
                    f"{getattr(condition_value, 'dtype', type(condition_value))}"
                )
            case_value, case_non_finite = self._visit_branch_deferring_non_finite(value)
            lowered_cases.append((condition_value, case_value, case_non_finite))
        result, non_finite = self._visit_branch_deferring_non_finite(
            expression.otherwise
        )
        for condition_value, case_value, case_non_finite in reversed(lowered_cases):
            result = self._numpy.where(condition_value, case_value, result)
            non_finite = self._numpy.where(condition_value, case_non_finite, non_finite)
        if self._numpy.any(non_finite):
            if self._deferred_non_finite is None:
                raise NonFiniteCastError(
                    "cannot cast a non-finite value to an integer- or boolean-sorted "
                    "dtype: the branch selected for at least one element produced a "
                    "nan/inf value with no faithful representation there."
                )
            # Nested piecewise: this node's own selection is poisoned, but an
            # enclosing condition may still discard the element. Pass the mask
            # up rather than deciding here. The poisoned lanes of ``result``
            # already carry zero, substituted at the cast.
            self._deferred_non_finite = self._numpy.logical_or(
                self._deferred_non_finite, non_finite
            )
        return result

    def _visit_branch_deferring_non_finite(self, node: Expression) -> tuple[Any, Any]:
        """Evaluate a piecewise branch, returning its value and non-finite mask.

        While the branch is being walked, an integer- or boolean-sorted
        cast of a non-finite value adds the offending elements to the mask
        instead of raising, so the caller can discard the ones its
        condition does not select. The mask starts out as a scalar
        ``False`` and broadcasts against every cast that adds to it.
        """
        outer = self._deferred_non_finite
        self._deferred_non_finite = self._numpy.zeros((), dtype=self._numpy.bool_)
        try:
            value = self.visit(node)
            non_finite = self._deferred_non_finite
        finally:
            self._deferred_non_finite = outer
        return value, non_finite

    def _is_boolean_condition_value(self, value: Any) -> bool:
        """Return whether a lowered piecewise condition value is boolean-dtyped.

        A raw Python ``bool`` (from a boolean ``LiteralExpression``, the
        only literal value a piecewise condition may hold) has no
        ``dtype`` attribute and is checked directly; every other lowered
        condition is NumPy-typed and is checked by dtype.
        """
        if isinstance(value, bool):
            return True
        return bool(getattr(value, "dtype", None) == self._numpy.bool_)

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
        A non-finite (``nan``/``inf``) result raises ``NonFiniteCastError``
        rather than casting, since neither value has a faithful integer
        or boolean representation. Inside a piecewise branch the offending
        elements are recorded instead, and
        :meth:`visit_piecewise_expression` raises only if the selected
        branch returns one; the non-finite elements are replaced with zero
        before the cast so the discarded lanes carry no platform sentinel.
        """
        dtype_name = _SORT_CAST_DTYPE_NAMES.get(result_sort)
        if dtype_name is None:
            return result
        non_finite = self._numpy.logical_not(self._numpy.isfinite(result))
        if self._numpy.any(non_finite):
            if self._deferred_non_finite is None:
                raise NonFiniteCastError(
                    f"cannot cast a non-finite value to the {result_sort.value}-sorted "
                    f"dtype {dtype_name!r}: the result contains a nan/inf value with "
                    "no faithful representation there."
                )
            self._deferred_non_finite = self._numpy.logical_or(
                self._deferred_non_finite, non_finite
            )
            result = self._numpy.where(non_finite, 0, result)
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
        NumPy and yield ``nan``/``inf`` rather than raising at the point of
        computation -- unlike ``evaluate_expression``, whose scalar native
        implementations raise immediately. A piecewise expression evaluates
        every case value and ``otherwise`` for every element (it lowers to
        a chain of ``numpy.where`` calls), so a domain error in an
        unselected case is not suppressed by its condition -- but the
        resulting ``nan``/``inf`` is discarded with its lane, so an
        integer-sorted branch guarded by a condition does not raise.

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
            - :class:`NonFiniteCastError`: a ``nan``/``inf`` value reaches
              a ``BOOL``/``NAT``/``INT``-sorted cast, which has no
              faithful representation for it. Inside a piecewise, only an
              element the selected branch returns raises; one produced by
              an unselected branch is discarded with its lane.
            - ``TypeError``: a piecewise condition is not boolean-dtyped.
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
