"""Expression passes that interface with SymPy."""

__all__ = [
    "convert_expression_to_sympy_expression",
    "convert_sympy_expression_to_expression",
    "simplify_expression",
    "substitute_sympy_expression_variables",
]

import operator
from typing import Any, Callable

import sympy  # type: ignore
import sympy.logic  # type: ignore
import sympy.logic.boolalg  # type: ignore
from frozendict import frozendict

from fhy_core.expression.core import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy_core.expression.visitor import (
    ExpressionBasePass,
)
from fhy_core.identifier import Identifier


class ExpressionToSympyConverter(ExpressionBasePass):
    """Transforms an expression to SymPy expression."""

    _UNARY_OPERATION_SYMPY_OPERATORS: frozendict[
        UnaryOperation, Callable[[Any], Any]
    ] = frozendict(
        {
            UnaryOperation.NEGATE: operator.neg,
            UnaryOperation.POSITIVE: operator.pos,
            UnaryOperation.LOGICAL_NOT: operator.not_,
        }
    )
    _BINARY_OPERATION_SYMPY_OPERATORS: frozendict[
        BinaryOperation, Callable[[Any, Any], Any]
    ] = frozendict(
        {
            BinaryOperation.ADD: operator.add,
            BinaryOperation.SUBTRACT: operator.sub,
            BinaryOperation.MULTIPLY: operator.mul,
            BinaryOperation.DIVIDE: operator.truediv,
            BinaryOperation.FLOOR_DIVIDE: lambda x, y: sympy.floor(x / y),
            BinaryOperation.MODULO: operator.mod,
            BinaryOperation.POWER: operator.pow,
            BinaryOperation.LOGICAL_AND: operator.and_,
            BinaryOperation.LOGICAL_OR: operator.or_,
            BinaryOperation.EQUAL: lambda x, y: sympy.Eq(x, y),
            BinaryOperation.NOT_EQUAL: lambda x, y: sympy.Ne(x, y),
            BinaryOperation.LESS: operator.lt,
            BinaryOperation.LESS_EQUAL: operator.le,
            BinaryOperation.GREATER: operator.gt,
            BinaryOperation.GREATER_EQUAL: operator.ge,
        }
    )

    def visit_binary_expression(
        self, binary_expression: BinaryExpression
    ) -> sympy.Expr | sympy.logic.boolalg.Boolean:
        left = self.visit(binary_expression.left)
        right = self.visit(binary_expression.right)
        return self._BINARY_OPERATION_SYMPY_OPERATORS[binary_expression.operation](
            left, right
        )

    def visit_unary_expression(
        self, unary_expression: UnaryExpression
    ) -> sympy.Expr | sympy.logic.boolalg.Boolean:
        operand = self.visit(unary_expression.operand)
        return self._UNARY_OPERATION_SYMPY_OPERATORS[unary_expression.operation](
            operand
        )

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> sympy.Expr | sympy.logic.boolalg.Boolean:
        identifier = identifier_expression.identifier
        return sympy.Symbol(self.format_identifier(identifier))

    def visit_literal_expression(
        self, literal_expression: LiteralExpression
    ) -> sympy.Expr | sympy.logic.boolalg.Boolean:
        if isinstance(literal_expression.value, bool):
            if literal_expression.value:
                return sympy.true
            else:
                return sympy.false
        elif isinstance(literal_expression.value, int):
            return sympy.Integer(literal_expression.value)
        elif isinstance(literal_expression.value, float):
            return sympy.Float(literal_expression.value)
        elif isinstance(literal_expression.value, str):
            if literal_expression.value == "True":
                return sympy.true
            elif literal_expression.value == "False":
                return sympy.false
            else:
                return sympy.parse_expr(literal_expression.value)
        else:
            raise TypeError(
                f"Unsupported literal type: {type(literal_expression.value)}"
            )

    @staticmethod
    def format_identifier(identifier: Identifier) -> str:
        return f"{identifier.name_hint}_{identifier.id}"


def convert_expression_to_sympy_expression(
    expression: Expression,
) -> sympy.Expr | sympy.logic.boolalg.Boolean:
    """Convert an expression to a SymPy expression.

    Args:
        expression: Expression to convert.

    Returns:
        SymPy expression.

    """
    converter = ExpressionToSympyConverter()
    return converter(expression)


def substitute_sympy_expression_variables(
    sympy_expression: sympy.Expr | sympy.logic.boolalg.Boolean,
    environment: dict[Identifier, Expression],
) -> sympy.Expr | sympy.logic.boolalg.Boolean:
    """Substitute variables in a SymPy expression.

    Args:
        sympy_expression: SymPy expression to substitute variables in.
        environment: Environment to substitute variables from.

    Returns:
        SymPy expression with substituted variables.

    """
    # TODO: Figure out why this is necessary...
    #       The sympy expression should only take on the two types in the type hint,
    #       but it seems that it can also be a bool.
    if isinstance(sympy_expression, bool):
        return sympy_expression
    return sympy_expression.subs(
        {
            ExpressionToSympyConverter.format_identifier(
                k
            ): convert_expression_to_sympy_expression(v)
            for k, v in environment.items()
        }
    )


class SymPyToExpressionConverter:
    """Converts a SymPy expression to an expression tree."""

    def __call__(
        self, sympy_expression: sympy.Expr | sympy.logic.boolalg.Boolean
    ) -> Expression:
        return self.convert(sympy_expression)

    def convert(self, node: sympy.Expr | sympy.logic.boolalg.Boolean) -> Expression:
        """Convert a SymPy node.

        Args:
            node: SymPy node to convert.

        Returns:
            Expression tree.

        """
        if isinstance(node, sympy.Expr):
            return self.convert_expr(node)
        elif isinstance(node, sympy.logic.boolalg.Boolean):
            return self.convert_bool(node)
        else:
            raise TypeError(f"Unsupported node type: {type(node)}")

    def convert_expr(self, expr: sympy.Expr) -> Expression:
        if isinstance(expr, sympy.Add):
            return self.convert_Add(expr)
        elif isinstance(expr, sympy.Mul):
            return self.convert_Mul(expr)
        elif isinstance(expr, sympy.Mod):
            return self.convert_Mod(expr)
        elif isinstance(expr, sympy.Pow):
            return self.convert_Pow(expr)
        elif isinstance(expr, sympy.Symbol):
            return self.convert_Symbol(expr)
        elif isinstance(expr, sympy.Integer):
            return self.convert_Integer(expr)
        elif isinstance(expr, sympy.Float):
            return self.convert_Float(expr)
        else:
            raise TypeError(f"Unsupported expression type: {type(expr)}")

    def convert_bool(
        self, boolean_expression: sympy.logic.boolalg.Boolean
    ) -> Expression:
        if isinstance(boolean_expression, sympy.logic.boolalg.Not):
            return self.convert_Not(boolean_expression)
        elif isinstance(boolean_expression, sympy.logic.boolalg.And):
            return self.convert_And(boolean_expression)
        elif isinstance(boolean_expression, sympy.logic.boolalg.Or):
            return self.convert_Or(boolean_expression)
        elif isinstance(boolean_expression, sympy.logic.boolalg.Xor):
            return self.convert_Xor(boolean_expression)
        elif isinstance(boolean_expression, sympy.logic.boolalg.Nor):
            return self.convert_Nor(boolean_expression)
        elif isinstance(boolean_expression, sympy.logic.boolalg.Nand):
            return self.convert_Nand(boolean_expression)
        elif isinstance(boolean_expression, sympy.core.relational.Relational):
            return self.convert_relational(boolean_expression)
        elif isinstance(boolean_expression, sympy.logic.boolalg.Implies):
            return self.convert_Implies(boolean_expression)
        elif isinstance(boolean_expression, sympy.logic.boolalg.BooleanTrue):
            return LiteralExpression(True)
        elif isinstance(boolean_expression, sympy.logic.boolalg.BooleanFalse):
            return LiteralExpression(False)
        else:
            raise TypeError(
                f"Unsupported boolean expression type: {type(boolean_expression)}"
            )

    def convert_relational(
        self, relational: sympy.core.relational.Relational
    ) -> Expression:
        """Convert a SymPy relational node to an expression.

        Args:
            relational: SymPy relational node to convert.

        Returns:
            Expression.

        """
        if isinstance(relational, sympy.Equality):
            return self.convert_Equality(relational)
        elif isinstance(relational, sympy.Unequality):
            return self.convert_Unequality(relational)
        elif isinstance(relational, sympy.StrictLessThan):
            return self.convert_StrictLessThan(relational)
        elif isinstance(relational, sympy.LessThan):
            return self.convert_LessThan(relational)
        elif isinstance(relational, sympy.StrictGreaterThan):
            return self.convert_StrictGreaterThan(relational)
        elif isinstance(relational, sympy.GreaterThan):
            return self.convert_GreaterThan(relational)
        else:
            raise TypeError(f"Unsupported relational type: {type(relational)}")

    def convert_Add(self, add: sympy.Add) -> Expression:
        """Convert a SymPy Add node to an expression.

        Args:
            add: SymPy Add node to convert.

        Returns:
            Expression.

        """
        if len(add.args) == 0:
            return LiteralExpression(0)
        elif len(add.args) == 1:
            return self.convert(add.args[0])
        else:
            return self._convert_commutative_and_associative_binary_operation(
                BinaryOperation.ADD, add
            )

    def convert_Mul(self, mul: sympy.Mul) -> Expression:
        """Convert a SymPy Mul node to an expression.

        Args:
            mul: SymPy Mul node to convert.

        Returns:
            expression.

        """
        if len(mul.args) == 0:
            return LiteralExpression(1)
        elif len(mul.args) == 1:
            return self.convert(mul.args[0])
        else:
            return self._convert_commutative_and_associative_binary_operation(
                BinaryOperation.MULTIPLY, mul
            )

    def convert_Mod(self, mod: sympy.Mod) -> BinaryExpression:
        """Convert a SymPy Mod node to an expression.

        Args:
            mod: SymPy Mod node to convert.

        Returns:
            Binary expression.

        """
        return self._convert_two_argument_binary_operation(BinaryOperation.MODULO, mod)

    def convert_Pow(self, pow_: sympy.Pow) -> BinaryExpression:
        """Convert a SymPy Pow node to an expression.

        Args:
            pow_: SymPy Pow node to convert.

        Returns:
            Binary expression.

        """
        return self._convert_two_argument_binary_operation(BinaryOperation.POWER, pow_)

    def convert_Not(self, not_: sympy.logic.boolalg.Not) -> UnaryExpression:
        """Convert a SymPy Not node to an expression.

        Args:
            not_: SymPy Not node to convert.

        Returns:
            Unary expression.

        """
        operand = self.convert(not_.args[0])
        return UnaryExpression(UnaryOperation.LOGICAL_NOT, operand)

    def convert_And(self, and_: sympy.logic.boolalg.And) -> BinaryExpression:
        """Convert a SymPy And node to an expression.

        Args:
            and_: SymPy And node to convert.

        Returns:
            Binary expression.

        """
        return self._convert_commutative_and_associative_binary_operation(
            BinaryOperation.LOGICAL_AND, and_
        )

    def convert_Or(self, or_: sympy.logic.boolalg.Or) -> BinaryExpression:
        """Convert a SymPy Or node to an expression.

        Args:
            or_: SymPy Or node to convert.

        Returns:
            Binary expression.

        """
        return self._convert_commutative_and_associative_binary_operation(
            BinaryOperation.LOGICAL_OR, or_
        )

    def convert_Xor(self, xor: sympy.logic.boolalg.Xor) -> BinaryExpression:
        """Convert a SymPy Xor node to an expression.

        Args:
            xor: SymPy Xor node to convert.

        Returns:
            Binary expression.

        """
        left = self.convert(xor.args[0])
        right = self.convert(sympy.Xor(*xor.args[1:], evaluate=False))
        return BinaryExpression(
            BinaryOperation.LOGICAL_AND,
            BinaryExpression(BinaryOperation.LOGICAL_OR, left, right),
            UnaryExpression(
                UnaryOperation.LOGICAL_NOT,
                BinaryExpression(BinaryOperation.LOGICAL_AND, left, right),
            ),
        )

    def convert_Nor(self, nor: sympy.logic.boolalg.Nor) -> Expression:
        """Convert a SymPy Nor node to an expression.

        Args:
            nor: SymPy Nor node to convert.

        Returns:
            Expression.

        """
        or_statement = self._convert_commutative_and_associative_binary_operation(
            BinaryOperation.LOGICAL_OR, nor
        )
        return UnaryExpression(UnaryOperation.LOGICAL_NOT, or_statement)

    def convert_Nand(self, nand: sympy.logic.boolalg.Nand) -> Expression:
        """Convert a SymPy Nand node to an expression.

        Args:
            nand: SymPy Nand node to convert.

        Returns:
            Expression.

        """
        and_statement = self._convert_commutative_and_associative_binary_operation(
            BinaryOperation.LOGICAL_AND, nand
        )
        return UnaryExpression(UnaryOperation.LOGICAL_NOT, and_statement)

    def convert_Equality(self, equivalent: sympy.Equality) -> BinaryExpression:
        """Convert a SymPy Equality node to an expression.

        Args:
            equivalent: SymPy Equality node to convert.

        Returns:
            Binary expression.

        """
        return self._convert_two_argument_binary_operation(
            BinaryOperation.EQUAL, equivalent
        )

    def convert_Unequality(self, unequality: sympy.Unequality) -> BinaryExpression:
        """Convert a SymPy Unequality node to an expression.

        Args:
            unequality: SymPy Unequality node to convert.

        Returns:
            Binary expression.

        """
        return self._convert_two_argument_binary_operation(
            BinaryOperation.NOT_EQUAL, unequality
        )

    def convert_StrictLessThan(
        self, strict_less_than: sympy.StrictLessThan
    ) -> BinaryExpression:
        """Convert a SymPy StrictLessThan node to an expression.

        Args:
            strict_less_than: SymPy StrictLessThan node to convert.

        Returns:
            Binary expression.

        """
        return self._convert_two_argument_binary_operation(
            BinaryOperation.LESS, strict_less_than
        )

    def convert_LessThan(self, less_than: sympy.LessThan) -> BinaryExpression:
        """Convert a SymPy LessThan node to an expression.

        Args:
            less_than: SymPy LessThan node to convert.

        Returns:
            Binary expression.

        """
        return self._convert_two_argument_binary_operation(
            BinaryOperation.LESS_EQUAL, less_than
        )

    def convert_StrictGreaterThan(
        self, strict_greater_than: sympy.StrictGreaterThan
    ) -> BinaryExpression:
        """Convert a SymPy StrictGreaterThan node to an expression.

        Args:
            strict_greater_than: SymPy StrictGreaterThan node to convert.

        Returns:
            Binary expression.

        """
        return self._convert_two_argument_binary_operation(
            BinaryOperation.GREATER, strict_greater_than
        )

    def convert_GreaterThan(self, greater_than: sympy.GreaterThan) -> BinaryExpression:
        """Convert a SymPy GreaterThan node to an expression.

        Args:
            greater_than: SymPy GreaterThan node to convert.

        Returns:
            Binary expression.

        """
        return self._convert_two_argument_binary_operation(
            BinaryOperation.GREATER_EQUAL, greater_than
        )

    def convert_Implies(self, implies: sympy.logic.boolalg.Implies) -> BinaryExpression:
        """Convert a SymPy Implies node to an expression.

        Args:
            implies: SymPy Implies node to convert.

        Returns:
            Binary expression.

        """
        raise NotImplementedError("Implies is not supported.")

    def _convert_commutative_and_associative_binary_operation(
        self,
        operation: BinaryOperation,
        sympy_operation: sympy.Expr | sympy.logic.boolalg.Boolean,
    ) -> BinaryExpression:
        left = self.convert(sympy_operation.args[0])
        right = self.convert(
            sympy_operation.func(*sympy_operation.args[1:], evaluate=False)
        )
        return BinaryExpression(operation, left, right)

    def _convert_two_argument_binary_operation(
        self,
        operation: BinaryOperation,
        sympy_operation: sympy.Expr | sympy.logic.boolalg.Boolean,
    ) -> BinaryExpression:
        NUM_REQUIRED_ARGS = 2
        if len(sympy_operation.args) != NUM_REQUIRED_ARGS:
            raise ValueError(
                "Expected a binary operation to have exactly two arguments."
            )
        left = self.convert(sympy_operation.args[0])
        right = self.convert(sympy_operation.args[1])
        return BinaryExpression(operation, left, right)

    def convert_Symbol(self, symbol: sympy.Symbol) -> IdentifierExpression:
        symbol_name = symbol.name
        last_underscore_index = symbol_name.rfind("_")
        if last_underscore_index == -1:
            raise RuntimeError(
                "When converting a symbol from SymPy to an identifier, the "
                "symbol did not contain an underscore. This typically means "
                "that the symbol was not generated by the "
                "SymPyToExpressionConverter."
            )
        identifier_id = int(symbol_name[last_underscore_index + 1 :])
        identifier_name_hint = symbol_name[:last_underscore_index]
        identifier = Identifier(identifier_name_hint)
        identifier._id = identifier_id
        return IdentifierExpression(identifier)

    def convert_Integer(self, int_: sympy.Integer) -> LiteralExpression:
        return LiteralExpression(int(int_))

    def convert_Float(self, float_: sympy.Float) -> LiteralExpression:
        return LiteralExpression(float(float_))


def convert_sympy_expression_to_expression(
    sympy_expression: sympy.Expr | sympy.logic.boolalg.Boolean,
) -> Expression:
    """Convert a SymPy expression to an expression.

    Args:
        sympy_expression: SymPy expression to convert.

    Returns:
        Expression.

    """
    converter = SymPyToExpressionConverter()
    return converter(sympy_expression)


def simplify_expression(
    expression: Expression, environment: dict[Identifier, Expression] | None = None
) -> Expression:
    """Simplify an expression.

    Args:
        expression: Expression to simplify.
        environment: Environment to simplify the expression in. Defaults to None.

    Returns:
        Result of the expression.

    """
    sympy_expression = convert_expression_to_sympy_expression(expression)
    if environment is not None:
        sympy_expression = substitute_sympy_expression_variables(
            sympy_expression, environment
        )
    result = sympy.simplify(sympy_expression)
    return convert_sympy_expression_to_expression(result)
