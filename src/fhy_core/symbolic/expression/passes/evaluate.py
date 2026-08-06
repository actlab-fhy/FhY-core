"""Bottom-up evaluator for native call sites and constant references.

The evaluator performs two narrow rewrites:

1. A ``CallExpression`` whose registered target is a
   :class:`NativeFunction` and whose arguments are all
   :class:`LiteralExpression` (after recursive evaluation) becomes
   ``LiteralExpression(implementation(*values))``.
2. An ``IdentifierExpression`` whose identifier name matches a
   registered :class:`NativeConstant` becomes
   ``LiteralExpression(constant.value)``.

Every other node is preserved (with its children recursively evaluated)
by the :class:`RewritablePass` base class. Literal arithmetic is not
folded; that remains a :func:`simplify_expression` (sympy) job.

Argument coercion at the native boundary is exact-only: an integer-grammar
string literal converts via ``int``, but a float-grammar string literal is
refused with :class:`StringLiteralPrecisionError` rather than collapsed to a
lossy binary ``float``.

Call-site preconditions: the evaluator distinguishes four cases for
``visit_call_expression``. A call to an unregistered name raises
:class:`EntryLookupError` (typo / unknown function). A call whose
target is a :class:`NativeConstant` raises :class:`FunctionArityError`
(constants are not callable). A call whose target is a
:class:`RegisteredFunction` (expression-bodied) emits a
``WARNING``-level diagnostic recommending ``inline_functions`` and
leaves the node unchanged. A call to a :class:`NativeFunction` with
all-literal arguments folds, with the implementation's return value
validated against the declared ``result_sort`` (raises
:class:`NativeResultSortError` on mismatch).
"""

__all__ = [
    "ExpressionEvaluator",
    "evaluate_expression",
]

from fhy_core.diagnostic import DiagnosticLevel
from fhy_core.pass_infrastructure import RewritablePass, register_pass

from ..core import (
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
)
from ..errors import NativeResultSortError
from ..registry import (
    NativeConstant,
    RegisteredFunction,
    get_registered_entry,
)
from ..sort import is_python_value_compatible_with_sort
from .inline import FunctionArityError
from .native_lowering import coerce_literal_value, try_get_native_constant_value


def _try_extract_native_argument_values(
    arguments: tuple[Expression, ...],
) -> tuple[bool | int | float, ...] | None:
    """Return native-callable argument values, or ``None`` if any are non-literal."""
    values: list[bool | int | float] = []
    for argument in arguments:
        if not isinstance(argument, LiteralExpression):
            return None
        values.append(coerce_literal_value(argument.value))
    return tuple(values)


def _build_literal_expression(
    value: bool | int | float,
) -> LiteralExpression:
    """Wrap a native-result value in a :class:`LiteralExpression`."""
    return LiteralExpression(value)


@register_pass(
    "fhy_core.symbolic.expression.evaluate",
    "Fold native-function calls with all-literal arguments and resolve "
    "native-constant references.",
)
class ExpressionEvaluator(RewritablePass[Expression]):
    """Bottom-up evaluator over the expression IR.

    Folds literal-argument calls to native functions and substitutes
    native-constant references; other nodes are preserved by the
    :class:`RewritablePass` base.

    Raises:
        PassExecutionError: Via the standard pass-framework guard, with
            the underlying error attached as ``__cause__``. The
            underlying errors are:

            - :class:`EntryLookupError` when a ``CallExpression``
              references an unregistered function name.
            - :class:`FunctionArityError` when the call target is a
              registered :class:`NativeConstant` (not callable).
            - :class:`NativeResultSortError` when a
              :class:`NativeFunction` implementation returns a value
              whose Python runtime type is not compatible with its
              declared ``result_sort``.
            - :class:`StringLiteralPrecisionError` when a native call
              argument is a float-grammar string literal, whose exact
              decimal form cannot be collapsed to a binary ``float``
              without precision loss.
            - ``TypeError`` when a :class:`NativeFunction` call's
              all-literal argument count does not match what its
              ``implementation`` accepts (e.g. ``sin(1.0, 2.0)``). Unlike
              :class:`FunctionInliner`, this evaluator never validates a
              native call's arity against its declared
              ``parameter_sorts`` before invoking ``implementation``;
              running :func:`inline_functions` first, which does
              validate arity and raises :class:`FunctionArityError`
              instead, is the only way to route around this bare
              ``TypeError``.
            - ``ValueError``, ``OverflowError``, or
              ``ZeroDivisionError`` from a native implementation's
              own raises (e.g. ``math.sqrt(-1)``).

    Notes:
        Calls to an expression-bodied :class:`RegisteredFunction` emit
        a ``WARNING``-level diagnostic via the pass framework and the
        node is preserved unchanged; run :func:`inline_functions`
        before :func:`evaluate_expression` to fold those calls.

    """

    def visit_identifier_expression(
        self, expression: IdentifierExpression
    ) -> Expression | None:
        """Resolve a native-constant reference to its literal value."""
        constant_value = try_get_native_constant_value(expression.identifier.name_hint)
        if constant_value is None:
            return None
        return _build_literal_expression(constant_value)

    def visit_call_expression(self, expression: CallExpression) -> Expression | None:
        """Fold a native-function call with all-literal arguments to a literal."""
        entry = get_registered_entry(expression.function_name)
        if isinstance(entry, NativeConstant):
            raise FunctionArityError(
                f"call to {expression.function_name!r} is to a registered "
                f"native constant, which is not callable"
            )
        if isinstance(entry, RegisteredFunction):
            self.report(
                DiagnosticLevel.WARNING,
                f"call to {expression.function_name!r} was not inlined; "
                f"run `inline_functions` before `evaluate_expression`",
            )
            return None
        # entry is NativeFunction
        argument_values = _try_extract_native_argument_values(expression.arguments)
        if argument_values is None:
            return None
        result = entry.implementation(*argument_values)
        if not is_python_value_compatible_with_sort(result, entry.result_sort):
            raise NativeResultSortError(
                f"NativeFunction {entry.name!r} returned {result!r} "
                f"(Python type {type(result).__name__}), which is not "
                f"compatible with the declared result sort {entry.result_sort}."
            )
        return _build_literal_expression(result)


def evaluate_expression(expression: Expression) -> Expression:
    """Fold native call sites and resolve native-constant references.

    Args:
        expression: Expression tree to evaluate.

    Returns:
        An expression tree where every literal-argument native call is
        folded to a ``LiteralExpression`` and every identifier reference
        matching a registered native constant is replaced with its
        literal value. Other nodes are preserved.

    Raises:
        PassExecutionError: As for :class:`ExpressionEvaluator`.

    """
    return ExpressionEvaluator()(expression)
