"""Pretty-printer for expressions."""

from fhy_core.utils.override import override

__all__ = ["pformat_expression"]

from fhy_core.pass_infrastructure import PassExecutionError, VisitablePass

from .core import (
    BINARY_OPERATION_SYMBOLS,
    UNARY_OPERATION_SYMBOLS,
    BinaryExpression,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    PiecewiseExpression,
    UnaryExpression,
)


class ExpressionPrettyFormatter(VisitablePass[Expression, str]):
    """Pretty-formatter for expressions."""

    _is_id_shown: bool
    _is_printed_functional: bool

    def __init__(
        self, is_id_shown: bool = False, is_printed_functional: bool = False
    ) -> None:
        super().__init__()
        self._is_id_shown = is_id_shown
        self._is_printed_functional = is_printed_functional

    @override
    def __call__(self, expression: Expression) -> str:
        formatted_expression = super().__call__(expression)
        if not isinstance(formatted_expression, str):
            raise TypeError(
                f"Invalid formatted expression type: {type(formatted_expression)}"
            )
        return formatted_expression

    def visit_unary_expression(self, unary_expression: UnaryExpression) -> str:
        if self._is_printed_functional:
            return (
                f"({unary_expression.operation.value} "
                f"{self.visit(unary_expression.operand)})"
            )
        else:
            return (
                f"({UNARY_OPERATION_SYMBOLS[unary_expression.operation]}"
                f"{self.visit(unary_expression.operand)})"
            )

    def visit_binary_expression(self, binary_expression: BinaryExpression) -> str:
        left = self.visit(binary_expression.left)
        right = self.visit(binary_expression.right)
        if self._is_printed_functional:
            return f"({binary_expression.operation.value} {left} {right})"
        else:
            return (
                f"({left} "
                f"{BINARY_OPERATION_SYMBOLS[binary_expression.operation]} "
                f"{right})"
            )

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> str:
        identifier = identifier_expression.identifier
        if not self._is_id_shown:
            return identifier.name_hint
        else:
            return repr(identifier)

    def visit_literal_expression(self, literal_expression: LiteralExpression) -> str:
        return str(literal_expression.value)

    def visit_piecewise_expression(
        self, piecewise_expression: PiecewiseExpression
    ) -> str:
        if self._is_printed_functional:
            parts: list[str] = []
            for condition, value in piecewise_expression.get_cases():
                parts.append(self.visit(condition))
                parts.append(self.visit(value))
            parts.append(self.visit(piecewise_expression.otherwise))
            return f"(piecewise {' '.join(parts)})"
        case_clauses = [
            f"{self.visit(value)} if {self.visit(condition)}"
            for condition, value in piecewise_expression.get_cases()
        ]
        otherwise = self.visit(piecewise_expression.otherwise)
        return "{" + "; ".join([*case_clauses, f"{otherwise} otherwise"]) + "}"

    def visit_call_expression(self, call_expression: CallExpression) -> str:
        rendered_arguments = [
            self.visit(argument) for argument in call_expression.arguments
        ]
        name = call_expression.function_name
        if self._is_printed_functional:
            if rendered_arguments:
                return f"({name} {' '.join(rendered_arguments)})"
            return f"({name})"
        return f"{name}({', '.join(rendered_arguments)})"

    @override
    def get_noop_output(self, ir: Expression) -> str:
        raise PassExecutionError(
            f'Pass "{self.get_pass_name()}" does not define noop output.'
        )


def pformat_expression(
    expression: Expression, show_id: bool = False, functional: bool = False
) -> str:
    """Pretty-format an expression.

    Args:
        expression: Expression to pretty-format.
        show_id: Whether to show the identifier ID.
        functional: Whether to use functional notation.

    Returns:
        Pretty-formatted expression.

    """
    return ExpressionPrettyFormatter(
        is_id_shown=show_id, is_printed_functional=functional
    )(expression)
