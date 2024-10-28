"""Expression passes that interface with SymPy."""

import sympy
import sympy.logic
import sympy.logic.boolalg  # type: ignore

from fhy_core.expression.core import (
    BINARY_OPERATION_OPERATORS,
    UNARY_OPERATION_OPERATORS,
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

    def visit_binary_expression(
        self, binary_expression: BinaryExpression
    ) -> sympy.Expr | sympy.logic.boolalg.Boolean:
        left = self.visit(binary_expression.left)
        right = self.visit(binary_expression.right)
        return BINARY_OPERATION_OPERATORS[binary_expression.operation](left, right)

    def visit_unary_expression(
        self, unary_expression: UnaryExpression
    ) -> sympy.Expr | sympy.logic.boolalg.Boolean:
        operand = self.visit(unary_expression.operand)
        return UNARY_OPERATION_OPERATORS[unary_expression.operation](operand)

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> sympy.Expr | sympy.logic.boolalg.Boolean:
        identifier = identifier_expression.identifier
        return sympy.Symbol(self.format_identifier(identifier))

    def visit_literal_expression(
        self, literal_expression: LiteralExpression
    ) -> sympy.Expr | sympy.logic.boolalg.Boolean:
        if isinstance(literal_expression.value, int):
            return sympy.Integer(literal_expression.value)
        elif isinstance(literal_expression.value, float):
            return sympy.Float(literal_expression.value)
        elif isinstance(literal_expression.value, complex):
            raise NotImplementedError()
        elif isinstance(literal_expression.value, bool):
            if literal_expression.value:
                return sympy.true
            else:
                return sympy.false
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
        elif isinstance(boolean_expression, sympy.logic.boolalg.Equivalent):
            return self.convert_Equivalent(boolean_expression)
        elif isinstance(boolean_expression, sympy.logic.boolalg.Implies):
            return self.convert_Implies(boolean_expression)
        elif isinstance(boolean_expression, sympy.logic.boolalg.Nor):
            return self.convert_Nor(boolean_expression)
        elif isinstance(boolean_expression, sympy.logic.boolalg.Nand):
            return self.convert_Nand(boolean_expression)
        elif isinstance(boolean_expression, sympy.logic.boolalg.BooleanTrue):
            return LiteralExpression(True)
        elif isinstance(boolean_expression, sympy.logic.boolalg.BooleanFalse):
            return LiteralExpression(False)
        else:
            raise TypeError(
                f"Unsupported boolean expression type: {type(boolean_expression)}"
            )

    def convert_Add(self, add: sympy.Add) -> BinaryExpression:
        """Convert a SymPy Add node to an expression.

        Args:
            add: SymPy Add node to convert.

        Returns:
            Binary expression.

        """
        if len(add.args) == 0:
            return LiteralExpression(0)
        elif len(add.args) == 1:
            return self.convert(add.args[0])
        else:
            return self._convert_commutative_and_associative_binary_operation(
                BinaryOperation.ADD, add
            )

    def convert_Mul(self, mul: sympy.Mul) -> BinaryExpression:
        """Convert a SymPy Mul node to an expression.

        Args:
            mul: SymPy Mul node to convert.

        Returns:
            Binary expression.

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
        left = self.convert(mod.args[0])
        right = self.convert(mod.args[1])
        return BinaryExpression(BinaryOperation.MODULO, left, right)

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
        raise NotImplementedError("Xor is not supported.")

    def convert_Equivalent(
        self, equivalent: sympy.logic.boolalg.Equivalent
    ) -> BinaryExpression:
        """Convert a SymPy Equivalent node to an expression.

        Args:
            equivalent: SymPy Equivalent node to convert.

        Returns:
            Binary expression.

        """
        raise NotImplementedError("Equivalent is not supported.")

    def convert_Implies(self, implies: sympy.logic.boolalg.Implies) -> BinaryExpression:
        """Convert a SymPy Implies node to an expression.

        Args:
            implies: SymPy Implies node to convert.

        Returns:
            Binary expression.

        """
        raise NotImplementedError("Implies is not supported.")

    def convert_Nor(self, nor: sympy.logic.boolalg.Nor) -> BinaryExpression:
        """Convert a SymPy Nor node to an expression.

        Args:
            nor: SymPy Nor node to convert.

        Returns:
            Binary expression.

        """
        raise NotImplementedError("Nor is not supported.")

    def convert_Nand(self, nand: sympy.logic.boolalg.Nand) -> BinaryExpression:
        """Convert a SymPy Nand node to an expression.

        Args:
            nand: SymPy Nand node to convert.

        Returns:
            Binary expression.

        """
        raise NotImplementedError("Nand is not supported.")

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
        return LiteralExpression(int_.p)

    def convert_Float(self, float_: sympy.Float) -> LiteralExpression:
        return LiteralExpression(float_.p)


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
