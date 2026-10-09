"""Inline expression-bodied function calls.

Inlining replaces every call of a composed built-in or of a registered
:class:`RegisteredFunction` by the function's body with the call's
arguments substituted for its parameters, and inlines the result in turn.
It runs in the Rust core (``FunctionRegistry::inline``), over a snapshot of
the registry:

- Arguments are inlined before they are substituted, and a node that
  occurs in several places is inlined once, so the work is linear in the
  distinct nodes: nesting a call inside its own argument, as in
  ``relu(relu(...(x)))``, costs one step per level. A tree of any depth
  inlines.
- A call of a native built-in or of a :class:`NativeFunction` is kept,
  with its arguments inlined, once its argument count is checked; folding
  it to a literal is :class:`ExpressionEvaluator`'s job.
- The result is the input object itself when it calls nothing to inline,
  and otherwise shares every subtree object that has no such call.

The refusals carry the core's text: a call of an unregistered name raises
:class:`EntryLookupError`, a wrong argument count or a call of a
registered constant :class:`FunctionArityError`, and a function reached
again inside its own body :class:`RecursionError`. The public entry point
is :func:`inline_functions`; the :class:`FunctionInliner` class exists so
callers running the inliner inside a pass manager can collect diagnostics
and ``PassResult`` metadata.
"""

__all__ = [
    "FunctionArityError",
    "FunctionInliner",
    "inline_functions",
]

from fhy_core import _rs
from fhy_core.error import register_error
from fhy_core.pass_infrastructure import CompilerPass, register_pass
from fhy_core.utils.override import override

from ..core import Expression


@register_error
class FunctionArityError(ValueError):
    """A call's argument count is wrong, or its target is not callable at all.

    Raised both when a call's argument count does not match the
    function's parameters, and when the call target resolves to a
    registered constant, which cannot be called at any arity.
    """


@register_pass(
    "fhy_core.symbolic.expression.inline_functions",
    "Replace expression-bodied CallExpression nodes with their inlined bodies.",
)
class FunctionInliner(CompilerPass[Expression, Expression]):
    """Inline composed built-in and registered-function calls.

    A run inlines its input as :func:`inline_functions` describes, in the
    Rust core. It changed the IR exactly when its output is not its input
    object. A refusal fails the run with ``PassExecutionError``, whose
    ``__cause__`` is the refusal: :class:`EntryLookupError` for an
    unregistered name, :class:`FunctionArityError` for a wrong argument
    count or a call of a constant, and :class:`RecursionError` for a
    (transitively) recursive function.
    """

    @override
    def get_noop_output(self, ir: Expression) -> Expression:
        return ir

    @override
    def run_pass(self, ir: Expression) -> Expression:
        return _rs.inline_functions(ir)

    @override
    def did_change(self, input_ir: Expression, output: Expression) -> bool:
        return output is not input_ir


def inline_functions(expression: Expression) -> Expression:
    """Inline every expression-bodied ``CallExpression`` in ``expression``.

    Args:
        expression: Expression tree to inline.

    Returns:
        An expression tree where every call of a composed built-in or a
        registered function has been replaced by its body (with the
        inlined arguments substituted), ``expression`` itself when there
        is none. Native function call nodes remain in the tree with their
        arguments inlined; the evaluator handles them.

    Raises:
        PassExecutionError: If a call references an unregistered
            function (``EntryLookupError`` as cause), if a call's
            argument count does not match the function's parameter count
            or the call target is a constant (``FunctionArityError`` as
            cause), or if a registered function transitively calls itself
            (``RecursionError`` as cause).

    """
    return FunctionInliner()(expression)
