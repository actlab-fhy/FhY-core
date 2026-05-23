"""Inline registered-function call expressions."""

__all__ = [
    "FunctionArityError",
    "FunctionInliner",
    "inline_functions",
]

from fhy_core.error import register_error
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import VisitablePass, register_pass

from ..core import (
    BinaryExpression,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    TernaryExpression,
    UnaryExpression,
)
from ..registry import get_registered_function
from .basic import substitute_identifiers


@register_error
class FunctionArityError(ValueError):
    """Argument count does not match the registered function's parameters."""


@register_pass(
    "fhy_core.expression.inline_functions",
    "Replace registered-function CallExpression nodes with their inlined bodies.",
)
class FunctionInliner(VisitablePass[Expression, Expression]):
    """Inline registered functions in an expression tree.

    For each ``CallExpression``, the registered function body is fetched
    and its parameters substituted with the (already-inlined) argument
    expressions. Substitution is structural and uses the existing
    ``substitute_identifiers`` machinery; nested calls are inlined
    bottom-up. Recursive registrations are detected on the call stack
    and rejected.

    Direct calls to ``visit`` surface the underlying domain errors
    listed below. Invoking the pass through ``__call__`` / ``execute``
    (the standard pass-framework path used by ``inline_functions``)
    wraps each of these in ``PassExecutionError`` with the original
    attached as ``__cause__`` and named in the wrapper message.

    Raises:
        FunctionLookupError: If a call references a name with no
            registered function.
        FunctionArityError: If a call's argument count does not match
            the registered function's parameter count.
        RecursionError: If a registered function (transitively) calls
            itself.

    """

    _in_progress: set[str]

    def __init__(self) -> None:
        super().__init__()
        self._in_progress = set()

    def visit_literal_expression(
        self, literal_expression: LiteralExpression
    ) -> Expression:
        return literal_expression

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> Expression:
        return identifier_expression

    def visit_unary_expression(self, unary_expression: UnaryExpression) -> Expression:
        return UnaryExpression(
            unary_expression.operation, self.visit(unary_expression.operand)
        )

    def visit_binary_expression(
        self, binary_expression: BinaryExpression
    ) -> Expression:
        return BinaryExpression(
            binary_expression.operation,
            self.visit(binary_expression.left),
            self.visit(binary_expression.right),
        )

    def visit_ternary_expression(
        self, ternary_expression: TernaryExpression
    ) -> Expression:
        return TernaryExpression(
            self.visit(ternary_expression.condition),
            self.visit(ternary_expression.true_value),
            self.visit(ternary_expression.false_value),
        )

    def visit_call_expression(self, call_expression: CallExpression) -> Expression:
        name = call_expression.function_name
        self._reject_if_recursive(name)

        inlined_arguments = tuple(
            self.visit(argument) for argument in call_expression.arguments
        )
        registered = get_registered_function(name)
        self._check_arity(name, registered.parameters, inlined_arguments)

        substitutions = dict(zip(registered.parameters, inlined_arguments))
        substituted_body = substitute_identifiers(registered.body, substitutions)

        self._in_progress.add(name)
        try:
            return self.visit(substituted_body)
        finally:
            self._in_progress.discard(name)

    def _reject_if_recursive(self, name: str) -> None:
        if name in self._in_progress:
            raise RecursionError(
                f"Function {name!r} is (transitively) recursive and cannot be inlined."
            )

    @staticmethod
    def _check_arity(
        name: str,
        parameters: tuple[Identifier, ...],
        arguments: tuple[Expression, ...],
    ) -> None:
        if len(arguments) != len(parameters):
            raise FunctionArityError(
                f"Function {name!r} expects {len(parameters)} argument(s), "
                f"but got {len(arguments)}."
            )

    def get_noop_output(self, ir: Expression) -> Expression:
        return ir


def inline_functions(expression: Expression) -> Expression:
    """Inline every ``CallExpression`` in ``expression``.

    Args:
        expression: Expression tree to inline.

    Returns:
        An expression tree containing no ``CallExpression`` nodes.

    Raises:
        PassExecutionError: If a call references an unregistered
            function (``FunctionLookupError`` as cause), if a call's
            argument count does not match the registered function's
            parameter count (``FunctionArityError`` as cause), or if a
            registered function transitively calls itself
            (``RecursionError`` as cause). The original exception is
            attached via ``__cause__``; the wrapper message names the
            inner type so callers can match on it.

    """
    return FunctionInliner()(expression)
