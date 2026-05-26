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
"""

__all__ = [
    "ExpressionEvaluator",
    "evaluate_expression",
]

from fhy_core.pass_infrastructure import RewritablePass, register_pass

from ..core import (
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
)
from ..registry import (
    FunctionLookupError,
    NativeConstant,
    NativeFunction,
    get_registered_function,
)


def _try_get_native_constant_value(
    name: str,
) -> bool | int | float | complex | None:
    """Return the constant value bound to ``name``, or ``None`` if absent."""
    try:
        entry = get_registered_function(name)
    except FunctionLookupError:
        return None
    if isinstance(entry, NativeConstant):
        return entry.value
    return None


def _try_get_native_function(name: str) -> NativeFunction | None:
    """Return the native function bound to ``name``, or ``None`` if absent."""
    try:
        entry = get_registered_function(name)
    except FunctionLookupError:
        return None
    if isinstance(entry, NativeFunction):
        return entry
    return None


def _try_extract_native_argument_values(
    arguments: tuple[Expression, ...],
) -> tuple[bool | int | float, ...] | None:
    """Return native-callable argument values, or ``None`` if any are non-literal."""
    values: list[bool | int | float] = []
    for argument in arguments:
        if not isinstance(argument, LiteralExpression):
            return None
        values.append(_coerce_literal_value_for_native(argument.value))
    return tuple(values)


def _coerce_literal_value_for_native(
    value: bool | int | float | str,
) -> bool | int | float:
    """Coerce a ``LiteralExpression`` value to a native-callable Python value.

    Numeric tokens from :func:`parse_expression` are stored as ``str``
    on ``LiteralExpression``; this helper converts them to the narrowest
    matching numeric type before they reach a native math callable.
    """
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        return float(value)


def _build_literal_expression(
    value: bool | int | float | complex,
) -> LiteralExpression:
    """Wrap a native-result value in a :class:`LiteralExpression`.

    Raises ``TypeError`` for ``complex`` values because
    ``LiteralExpression`` does not support complex literals.
    """
    if isinstance(value, complex) and not isinstance(value, (bool, int, float)):
        raise TypeError(
            f"Native function returned complex value {value!r}; "
            "`LiteralExpression` does not support complex values yet."
        )
    return LiteralExpression(value)


@register_pass(
    "fhy_core.expression.evaluate",
    "Fold native-function calls with all-literal arguments and resolve "
    "native-constant references.",
)
class ExpressionEvaluator(RewritablePass[Expression]):
    """Bottom-up evaluator over the expression IR.

    Folds literal-argument calls to native functions and substitutes
    native-constant references; other nodes are preserved by the
    :class:`RewritablePass` base.

    Raises:
        PassExecutionError: Via the standard pass-framework guard if
            the native implementation raises during evaluation; the
            underlying exception (typically ``ValueError``,
            ``OverflowError``, or ``ZeroDivisionError`` from ``math``)
            is attached as ``__cause__``.

    """

    def visit_identifier_expression(
        self, expression: IdentifierExpression
    ) -> Expression | None:
        constant_value = _try_get_native_constant_value(expression.identifier.name_hint)
        if constant_value is None:
            return None
        return _build_literal_expression(constant_value)

    def visit_call_expression(self, expression: CallExpression) -> Expression | None:
        native = _try_get_native_function(expression.function_name)
        if native is None:
            return None
        argument_values = _try_extract_native_argument_values(expression.arguments)
        if argument_values is None:
            return None
        return _build_literal_expression(native.implementation(*argument_values))


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
