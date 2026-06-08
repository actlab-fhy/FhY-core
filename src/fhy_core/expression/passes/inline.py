"""Inline expression-bodied function calls.

Inlining is a bottom-up rewrite over the expression IR. For each
``CallExpression``, the inliner resolves the call target in the
registry and dispatches by entry kind:

- :class:`RegisteredFunction`: expression-bodied. The call is inlined
  by substituting the call's argument expressions for the function's
  parameter identifiers in the body. Self-recursive and mutually
  recursive calls are detected via a per-pass ``_in_progress`` set and
  raise :class:`RecursionError`.
- :class:`NativeFunction`: Python-backed. Native calls pass through
  the inliner unchanged; folding to a literal is handled later by
  :class:`ExpressionEvaluator`. The inliner still validates declared
  arity against the call site's argument count.
- :class:`NativeConstant`: not callable. Reaching a call site whose
  target resolves to a constant raises :class:`FunctionArityError`.

Calls to unregistered names raise :class:`EntryLookupError`. The
public entry point is :func:`inline_functions`; the :class:`FunctionInliner`
class exists so callers running the inliner inside a pass manager can
collect diagnostics and ``PassResult`` metadata.
"""

__all__ = [
    "FunctionArityError",
    "FunctionInliner",
    "inline_functions",
]

from fhy_core.error import register_error
from fhy_core.pass_infrastructure import RewritablePass, register_pass

from ..core import CallExpression, Expression
from ..registry import (
    NativeConstant,
    NativeFunction,
    RegisteredFunction,
    get_registered_entry,
)


@register_error
class FunctionArityError(ValueError):
    """Argument count does not match the registered function's parameters."""


def _check_call_arity(
    name: str, expected_count: int, arguments: tuple[Expression, ...]
) -> None:
    """Raise ``FunctionArityError`` when the actual argument count differs."""
    if len(arguments) != expected_count:
        raise FunctionArityError(
            f"Function {name!r} expects {expected_count} argument(s), "
            f"but got {len(arguments)}."
        )


@register_pass(
    "fhy_core.expression.inline_functions",
    "Replace expression-bodied CallExpression nodes with their inlined bodies.",
)
class FunctionInliner(RewritablePass[Expression]):
    """Inline registered-function call expressions.

    For a ``CallExpression`` whose target is a :class:`RegisteredFunction`,
    the body is fetched and its parameters substituted with the
    already-inlined argument expressions; nested calls are inlined
    bottom-up; transitively-recursive registrations are rejected.

    Native function calls (:class:`NativeFunction`) are opaque to the
    inliner and pass through with their arguments inlined; the
    evaluator folds them when the arguments are literals. Calls whose
    target is a :class:`NativeConstant` are rejected as a usage error.

    Direct calls to ``transform`` surface the underlying domain errors
    listed below. Invoking the pass through ``__call__`` / ``execute``
    (the standard pass-framework path used by ``inline_functions``)
    wraps each of these in ``PassExecutionError`` with the original
    attached as ``__cause__`` and named in the wrapper message.

    Raises:
        EntryLookupError: If a call references an unregistered name.
        FunctionArityError: If a call's argument count does not match
            the registered function's parameter count, or if the call
            target is a registered constant.
        RecursionError: If a registered function (transitively) calls
            itself.

    """

    _in_progress: set[str]

    def __init__(self) -> None:
        super().__init__()
        self._in_progress = set()

    def visit_call_expression(self, expression: CallExpression) -> Expression | None:
        """Inline an expression-bodied call by substituting its arguments."""
        name = expression.function_name
        self._reject_if_recursive(name)
        registered = get_registered_entry(name)
        if isinstance(registered, NativeConstant):
            raise FunctionArityError(
                f"{name!r} is a registered constant, not a function; reference "
                f"it as an identifier instead of calling it."
            )
        if isinstance(registered, NativeFunction):
            _check_call_arity(
                name, len(registered.parameter_sorts), expression.arguments
            )
            return None
        return self._inline_expression_bodied_call(name, registered, expression)

    def _inline_expression_bodied_call(
        self,
        name: str,
        registered: RegisteredFunction,
        expression: CallExpression,
    ) -> Expression:
        _check_call_arity(name, len(registered.parameters), expression.arguments)
        substitutions = dict(zip(registered.parameters, expression.arguments))
        substituted_body = registered.body.substitute(substitutions)
        self._in_progress.add(name)
        try:
            return self.transform(substituted_body)
        finally:
            self._in_progress.discard(name)

    def _reject_if_recursive(self, name: str) -> None:
        if name in self._in_progress:
            raise RecursionError(
                f"Function {name!r} is (transitively) recursive and cannot be inlined."
            )


def inline_functions(expression: Expression) -> Expression:
    """Inline every expression-bodied ``CallExpression`` in ``expression``.

    Args:
        expression: Expression tree to inline.

    Returns:
        An expression tree where every expression-bodied call has been
        replaced by its body (with parameters substituted). Native
        function call nodes remain in the tree with their arguments
        recursively inlined; the evaluator handles them.

    Raises:
        PassExecutionError: If a call references an unregistered
            function (``EntryLookupError`` as cause), if a call's
            argument count does not match the registered function's
            parameter count or the call target is a constant
            (``FunctionArityError`` as cause), or if a registered
            function transitively calls itself (``RecursionError`` as
            cause).

    """
    return FunctionInliner()(expression)
