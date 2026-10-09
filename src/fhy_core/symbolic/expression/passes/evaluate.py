"""Fold native call sites and constant references.

The fold runs in the Rust core (``Evaluator::fold``), over a snapshot of the
function registry. It performs two narrow rewrites, bottom-up, once per
distinct node:

1. A ``CallExpression`` of a native function whose arguments are all
   :class:`LiteralExpression` becomes the literal it computes. A native
   built-in is computed by the core (IEEE results: ``sqrt(-1.0)`` is
   ``nan``; ``round`` rounds half to even; ``round``, ``floor`` and
   ``ceil`` give the exact ``int``), and a registered
   :class:`NativeFunction` by its ``implementation``, whose result must
   have the declared result sort.
2. An ``IdentifierExpression`` carrying the canonical identifier of a
   constant becomes the constant's value. Recognition is by identifier
   identity, so an identifier that merely shares a constant's
   ``name_hint`` is left alone as the free variable it is.

Every other node is kept, with its children folded; literal arithmetic
is not folded. A call of a function with a body (a composed built-in or a
:class:`RegisteredFunction`) is kept and reported with a ``WARNING``
recommending ``inline_functions``. A folded call is checked first: its
argument count and argument sorts, and a decimal argument must equal a
binary float exactly.
"""

__all__ = [
    "ExpressionEvaluator",
    "evaluate_expression",
]

from fhy_core import _rs
from fhy_core.diagnostic import DiagnosticLevel
from fhy_core.pass_infrastructure import CompilerPass, register_pass
from fhy_core.utils.override import override

from ..core import Expression


@register_pass(
    "fhy_core.symbolic.expression.evaluate",
    "Fold native-function calls with all-literal arguments and resolve "
    "native-constant references.",
)
class ExpressionEvaluator(CompilerPass[Expression, Expression]):
    """Fold native calls with literal arguments and constant references.

    A run folds its input as :func:`evaluate_expression` describes, in the
    Rust core. It changed the IR exactly when its output is not its input
    object.

    Raises:
        PassExecutionError: With the underlying error as ``__cause__``:

            - :class:`EntryLookupError` for a call of an unregistered name;
            - :class:`FunctionArityError` for a call of a constant, or a
              folded call's wrong argument count;
            - ``TypeError`` for a folded call's argument of the wrong sort;
            - :class:`StringLiteralPrecisionError` for a decimal argument
              no binary float equals;
            - :class:`NonFiniteCastError` for an ``INT``-sorted built-in
              whose value is ``nan`` or infinite;
            - :class:`NativeResultSortError` for a native implementation's
              result of the wrong sort;
            - the exception a native implementation raises, unchanged.

    """

    @override
    def get_noop_output(self, ir: Expression) -> Expression:
        return ir

    @override
    def run_pass(self, ir: Expression) -> Expression:
        output, not_inlined = _rs.fold_expression(ir)
        for function_name in not_inlined:
            self.report(
                DiagnosticLevel.WARNING,
                f"call to {function_name!r} was not inlined; "
                f"run `inline_functions` before `evaluate_expression`",
            )
        return output

    @override
    def did_change(self, input_ir: Expression, output: Expression) -> bool:
        return output is not input_ir


def evaluate_expression(expression: Expression) -> Expression:
    """Fold native call sites and resolve native-constant references.

    Args:
        expression: Expression tree to evaluate.

    Returns:
        An expression tree where every literal-argument native call is
        folded to a ``LiteralExpression`` and every reference to a
        constant's canonical identifier is replaced with its literal
        value, ``expression`` itself when nothing folds. Other nodes are
        kept.

    Raises:
        PassExecutionError: As for :class:`ExpressionEvaluator`.

    """
    return ExpressionEvaluator()(expression)
