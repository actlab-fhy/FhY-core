"""Expression passes that interface with SymPy."""

__all__ = [
    "convert_expression_to_sympy_expression",
    "convert_sympy_expression_to_expression",
    "simplify_expression",
    "substitute_sympy_expression_variables",
]

import operator
from typing import Any, Callable, ClassVar

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
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import (
    CompilerPass,
    PassExecutionError,
    VisitablePass,
    register_pass,
)


@register_pass(
    "fhy_core.expression.to_sympy",
    "Lower expression IR into an equivalent SymPy expression.",
)
class ExpressionToSympyConverter(VisitablePass[Expression, Any]):
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
            BinaryOperation.EQUAL: sympy.Eq,
            BinaryOperation.NOT_EQUAL: sympy.Ne,
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
                return sympy.Float(literal_expression.value)
        else:
            raise TypeError(
                f"Unsupported literal type: {type(literal_expression.value)}"
            )

    @staticmethod
    def format_identifier(identifier: Identifier) -> str:
        return f"{identifier.name_hint}_{identifier.id}"

    def get_noop_output(self, ir: Expression) -> Any:
        raise PassExecutionError(
            f'Pass "{self.get_pass_name()}" does not define noop output.'
        )


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
    # SymPy can fold boolean-valued subexpressions to plain Python `bool`
    # instances (notably ``True``/``False`` after simplification of a
    # ``sympy.logic.boolalg.Boolean``). These instances lack ``.subs`` and
    # also have nothing to substitute, so we short-circuit the no-op case.
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


@register_pass(
    "fhy_core.expression.from_sympy",
    "Lift a SymPy expression into the FhY expression IR.",
)
class SymPyToExpressionConverter(
    CompilerPass[sympy.Expr | sympy.logic.boolalg.Boolean, Expression]
):
    """Converts a SymPy expression to an expression tree."""

    _EXPR_DISPATCH: ClassVar[tuple[tuple[type, str], ...]] = (
        (sympy.Add, "_convert_add"),
        (sympy.Mul, "_convert_mul"),
        (sympy.Mod, "_convert_mod"),
        (sympy.Pow, "_convert_pow"),
        (sympy.Symbol, "_convert_symbol"),
        (sympy.Integer, "_convert_integer"),
        (sympy.Float, "_convert_float"),
    )
    _BOOL_DISPATCH: ClassVar[tuple[tuple[type, str], ...]] = (
        (sympy.logic.boolalg.Not, "_convert_not"),
        (sympy.logic.boolalg.And, "_convert_and"),
        (sympy.logic.boolalg.Or, "_convert_or"),
        (sympy.logic.boolalg.Xor, "_convert_xor"),
        (sympy.logic.boolalg.Nor, "_convert_nor"),
        (sympy.logic.boolalg.Nand, "_convert_nand"),
        (sympy.core.relational.Relational, "convert_relational"),
        (sympy.logic.boolalg.Implies, "_convert_implies"),
        (sympy.logic.boolalg.BooleanTrue, "_convert_boolean_true"),
        (sympy.logic.boolalg.BooleanFalse, "_convert_boolean_false"),
    )
    _RELATIONAL_DISPATCH: ClassVar[tuple[tuple[type, str], ...]] = (
        (sympy.Equality, "_convert_equality"),
        (sympy.Unequality, "_convert_unequality"),
        (sympy.StrictLessThan, "_convert_strict_less_than"),
        (sympy.LessThan, "_convert_less_than"),
        (sympy.StrictGreaterThan, "_convert_strict_greater_than"),
        (sympy.GreaterThan, "_convert_greater_than"),
    )

    def run_pass(self, ir: sympy.Expr | sympy.logic.boolalg.Boolean) -> Expression:
        return self.convert(ir)

    def get_noop_output(
        self, ir: sympy.Expr | sympy.logic.boolalg.Boolean
    ) -> Expression:
        raise PassExecutionError(
            f'Pass "{self.get_pass_name()}" does not define noop output.'
        )

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
        return self._dispatch(expr, self._EXPR_DISPATCH, "Unsupported expression type")

    def convert_bool(
        self, boolean_expression: sympy.logic.boolalg.Boolean
    ) -> Expression:
        return self._dispatch(
            boolean_expression,
            self._BOOL_DISPATCH,
            "Unsupported boolean expression type",
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
        return self._dispatch(
            relational, self._RELATIONAL_DISPATCH, "Unsupported relational type"
        )

    def _dispatch(
        self,
        node: Any,
        table: tuple[tuple[type, str], ...],
        error_label: str,
    ) -> Expression:
        for sympy_type, method_name in table:
            if isinstance(node, sympy_type):
                method = getattr(self, method_name)
                return method(node)  # type: ignore[no-any-return]
        raise TypeError(f"{error_label}: {type(node)}")

    def _convert_add(self, add: sympy.Add) -> Expression:
        if len(add.args) == 0:
            return LiteralExpression(0)
        elif len(add.args) == 1:
            return self.convert(add.args[0])
        else:
            return self._convert_commutative_and_associative_binary_operation(
                BinaryOperation.ADD, add
            )

    def _convert_mul(self, mul: sympy.Mul) -> Expression:
        if len(mul.args) == 0:
            return LiteralExpression(1)
        elif len(mul.args) == 1:
            return self.convert(mul.args[0])
        else:
            return self._convert_commutative_and_associative_binary_operation(
                BinaryOperation.MULTIPLY, mul
            )

    def _convert_mod(self, mod: sympy.Mod) -> BinaryExpression:
        return self._convert_two_argument_binary_operation(BinaryOperation.MODULO, mod)

    def _convert_pow(self, pow_: sympy.Pow) -> BinaryExpression:
        return self._convert_two_argument_binary_operation(BinaryOperation.POWER, pow_)

    def _convert_not(self, not_: sympy.logic.boolalg.Not) -> UnaryExpression:
        operand = self.convert(not_.args[0])
        return UnaryExpression(UnaryOperation.LOGICAL_NOT, operand)

    def _convert_and(self, and_: sympy.logic.boolalg.And) -> BinaryExpression:
        return self._convert_commutative_and_associative_binary_operation(
            BinaryOperation.LOGICAL_AND, and_
        )

    def _convert_or(self, or_: sympy.logic.boolalg.Or) -> BinaryExpression:
        return self._convert_commutative_and_associative_binary_operation(
            BinaryOperation.LOGICAL_OR, or_
        )

    def _convert_xor(self, xor: sympy.logic.boolalg.Xor) -> BinaryExpression:
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

    def _convert_nor(self, nor: sympy.logic.boolalg.Nor) -> Expression:
        or_statement = self._convert_commutative_and_associative_binary_operation(
            BinaryOperation.LOGICAL_OR, nor
        )
        return UnaryExpression(UnaryOperation.LOGICAL_NOT, or_statement)

    def _convert_nand(self, nand: sympy.logic.boolalg.Nand) -> Expression:
        and_statement = self._convert_commutative_and_associative_binary_operation(
            BinaryOperation.LOGICAL_AND, nand
        )
        return UnaryExpression(UnaryOperation.LOGICAL_NOT, and_statement)

    def _convert_equality(self, equivalent: sympy.Equality) -> BinaryExpression:
        return self._convert_two_argument_binary_operation(
            BinaryOperation.EQUAL, equivalent
        )

    def _convert_unequality(self, unequality: sympy.Unequality) -> BinaryExpression:
        return self._convert_two_argument_binary_operation(
            BinaryOperation.NOT_EQUAL, unequality
        )

    def _convert_strict_less_than(
        self, strict_less_than: sympy.StrictLessThan
    ) -> BinaryExpression:
        return self._convert_two_argument_binary_operation(
            BinaryOperation.LESS, strict_less_than
        )

    def _convert_less_than(self, less_than: sympy.LessThan) -> BinaryExpression:
        return self._convert_two_argument_binary_operation(
            BinaryOperation.LESS_EQUAL, less_than
        )

    def _convert_strict_greater_than(
        self, strict_greater_than: sympy.StrictGreaterThan
    ) -> BinaryExpression:
        return self._convert_two_argument_binary_operation(
            BinaryOperation.GREATER, strict_greater_than
        )

    def _convert_greater_than(
        self, greater_than: sympy.GreaterThan
    ) -> BinaryExpression:
        return self._convert_two_argument_binary_operation(
            BinaryOperation.GREATER_EQUAL, greater_than
        )

    def _convert_implies(
        self, implies: sympy.logic.boolalg.Implies
    ) -> BinaryExpression:
        _ = implies
        raise NotImplementedError("Implies is not supported.")

    def _convert_boolean_true(
        self, node: sympy.logic.boolalg.BooleanTrue
    ) -> LiteralExpression:
        _ = node
        return LiteralExpression(True)

    def _convert_boolean_false(
        self, node: sympy.logic.boolalg.BooleanFalse
    ) -> LiteralExpression:
        _ = node
        return LiteralExpression(False)

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

    def _convert_symbol(self, symbol: sympy.Symbol) -> IdentifierExpression:
        symbol_name = symbol.name
        last_underscore_index = symbol_name.rfind("_")
        if last_underscore_index == -1:
            raise RuntimeError(
                "When converting a symbol from SymPy to an identifier, the "
                "symbol did not contain an underscore. This typically means "
                "that the symbol was not produced by the "
                "ExpressionToSympyConverter, whose `format_identifier` "
                "encodes identifiers as '<name_hint>_<id>'."
            )
        identifier_id = int(symbol_name[last_underscore_index + 1 :])
        identifier_name_hint = symbol_name[:last_underscore_index]
        identifier = Identifier.deserialize_from_dict(
            {"id": identifier_id, "name_hint": identifier_name_hint}
        )
        return IdentifierExpression(identifier)

    def _convert_integer(self, int_: sympy.Integer) -> LiteralExpression:
        return LiteralExpression(int(int_))

    def _convert_float(self, float_: sympy.Float) -> LiteralExpression:
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
        Simplified expression.

    """
    sympy_expression = convert_expression_to_sympy_expression(expression)
    if environment is not None:
        sympy_expression = substitute_sympy_expression_variables(
            sympy_expression, environment
        )
    result = sympy.simplify(sympy_expression)
    return convert_sympy_expression_to_expression(result)
