"""Expression passes that interface with Z3."""

import operator
from typing import Any, Callable

import z3  # type: ignore
from frozendict import frozendict

from fhy_core.expression.core import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    SymbolType,
    UnaryExpression,
    UnaryOperation,
)
from fhy_core.expression.visitor import (
    ExpressionBasePass,
)
from fhy_core.identifier import Identifier


def _z3_floor_divide(left: z3.ExprRef, right: z3.ExprRef) -> z3.ExprRef:
    expr: z3.ArithRef = left / right
    if expr.is_real():
        return z3.ToInt(expr)
    elif expr.is_int():
        return expr
    else:
        raise ValueError(f"Unsupported floor divide expression type: {expr}")


class ExpressionToZ3Converter(ExpressionBasePass):
    """Transforms an expression into a Z3 expression."""

    _UNARY_OPERATION_Z3_OPERATORS: frozendict[UnaryOperation, Callable[[Any], Any]] = (
        frozendict(
            {
                UnaryOperation.NEGATE: operator.neg,
                UnaryOperation.POSITIVE: operator.pos,
                UnaryOperation.LOGICAL_NOT: lambda operand: z3.Not(operand),
            }
        )
    )
    _BINARY_OPERATION_Z3_OPERATORS: frozendict[
        BinaryOperation, Callable[[Any, Any], Any]
    ] = frozendict(
        {
            BinaryOperation.ADD: operator.add,
            BinaryOperation.SUBTRACT: operator.sub,
            BinaryOperation.MULTIPLY: operator.mul,
            BinaryOperation.DIVIDE: operator.truediv,
            BinaryOperation.FLOOR_DIVIDE: _z3_floor_divide,
            BinaryOperation.MODULO: operator.mod,
            BinaryOperation.POWER: operator.pow,
            BinaryOperation.LOGICAL_AND: operator.and_,
            BinaryOperation.LOGICAL_OR: operator.or_,
            BinaryOperation.EQUAL: operator.eq,
            BinaryOperation.NOT_EQUAL: operator.ne,
            BinaryOperation.LESS: operator.lt,
            BinaryOperation.LESS_EQUAL: operator.le,
            BinaryOperation.GREATER: operator.gt,
            BinaryOperation.GREATER_EQUAL: operator.ge,
        }
    )

    _symbol_types: dict[Identifier, SymbolType]
    _identifier_to_z3_expression: dict[Identifier, z3.ExprRef]

    def __init__(self, symbol_types: dict[Identifier, SymbolType]) -> None:
        self._symbol_types = symbol_types
        self._identifier_to_z3_expression = {}

    @property
    def identifier_to_z3_expression(self) -> dict[Identifier, z3.ExprRef]:
        return self._identifier_to_z3_expression

    def visit_binary_expression(
        self, binary_expression: BinaryExpression
    ) -> z3.ExprRef:
        left = self.visit(binary_expression.left)
        right = self.visit(binary_expression.right)
        operator = self._BINARY_OPERATION_Z3_OPERATORS[binary_expression.operation]
        return operator(left, right)

    def visit_unary_expression(self, unary_expression: UnaryExpression) -> z3.ExprRef:
        operand = self.visit(unary_expression.operand)
        operator = self._UNARY_OPERATION_Z3_OPERATORS[unary_expression.operation]
        return operator(operand)

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> z3.ExprRef:
        identifier_name = self.format_identifier(identifier_expression.identifier)
        identifier_type = self._symbol_types[identifier_expression.identifier]
        if identifier_type == SymbolType.REAL:
            result = z3.Real(identifier_name)
        elif identifier_type == SymbolType.INT:
            result = z3.Int(identifier_name)
        elif identifier_type == SymbolType.BOOL:
            result = z3.Bool(identifier_name)
        else:
            raise ValueError(f"Unsupported identifier type: {identifier_type}.")
        self._identifier_to_z3_expression[identifier_expression.identifier] = result
        return result

    def visit_literal_expression(
        self, literal_expression: LiteralExpression
    ) -> z3.ExprRef:
        if isinstance(literal_expression.value, bool):
            return z3.BoolVal(literal_expression.value)
        elif isinstance(literal_expression.value, int):
            return z3.IntVal(literal_expression.value)
        elif isinstance(literal_expression.value, float):
            return z3.RealVal(literal_expression.value)
        elif isinstance(literal_expression.value, str):
            if literal_expression.value == "True":
                return z3.BoolVal(True)
            elif literal_expression.value == "False":
                return z3.BoolVal(False)
            else:
                return z3.RealVal(literal_expression.value)
        else:
            raise TypeError(
                f"Unsupported literal type: {type(literal_expression.value)}"
            )

    @staticmethod
    def format_identifier(identifier: Identifier) -> str:
        return f"{identifier.name_hint}_{identifier.id}"


def convert_expression_to_z3_expression(
    expression: Expression, symbol_types: dict[Identifier, SymbolType] | None = None
) -> tuple[z3.ExprRef, dict[Identifier, z3.ExprRef]]:
    """Convert an expression to a Z3 expression.

    Args:
        expression: Expression to convert.
        symbol_types: Symbol types.

    Returns:
        Z3 expression and mapping of identifiers to Z3 expressions.

    """
    converter = ExpressionToZ3Converter(symbol_types or {})
    z3_expression = converter.visit(expression)
    return z3_expression, converter.identifier_to_z3_expression


def is_satisfiable(
    considered_identifiers: set[Identifier],
    expression: Expression,
    symbol_types: dict[Identifier, SymbolType],
) -> bool | None:
    """Check if the expression is satisfiable.

    Args:
        considered_identifiers: Considered identifiers.
        expression: Expression to check.
        symbol_types: Symbol types.

    Returns:
        True if the expression is satisfiable; False if the expression is unsatisfiable;
        None if the satisfiability is unknown.

    """
    z3_expression, identifier_to_z3_expression = convert_expression_to_z3_expression(
        expression, symbol_types
    )
    z3_expression = z3.Not(z3_expression)
    z3_expression = (
        z3.ForAll(
            [
                identifier_to_z3_expression[identifier]
                for identifier in considered_identifiers
            ],
            z3_expression,
        )
        if considered_identifiers
        else z3_expression
    )

    solver = z3.Solver()
    solver.add(z3_expression)

    result = solver.check()
    if result == z3.unsat:
        return True
    elif result == z3.sat:
        return False
    elif result == z3.unknown:
        return None
    else:
        raise RuntimeError("Unexpected Z3 result.")
