"""Parser from strings to expression trees."""

__all__ = ["parse_expression", "tokenize_expression"]

import re
from typing import Callable, TypeVar

from frozendict import frozendict

from fhy_core.identifier import Identifier

from .core import (
    BINARY_SYMBOL_OPERATIONS,
    UNARY_SYMBOL_OPERATIONS,
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
)


def _build_token_pattern(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(f"{pattern}" for pattern in patterns))


_FLOAT_PATTERN = re.compile(r"\d+\.\d+")
_INTEGER_PATTERN = re.compile(r"\d+")
_IDENTIFIER_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_TOKEN_PATTERN = _build_token_pattern(
    _FLOAT_PATTERN.pattern,
    _INTEGER_PATTERN.pattern,
    _IDENTIFIER_PATTERN.pattern,
    r"\*\*",
    r"&&",
    r"\|\|",
    r"<=|>=|==|!=",
    r"//",
    r"[-!+*/%<>()]",
)


_OperationType = TypeVar("_OperationType")


def _get_symbol_to_operation_subdict(
    parent_dict: frozendict[str, _OperationType], *symbols: str
) -> frozendict[str, _OperationType]:
    return frozendict(
        {
            symbol: operation
            for symbol, operation in parent_dict.items()
            if symbol in symbols
        }
    )


_LOGICAL_OR_SYMBOL_OPERATIONS: frozendict[str, BinaryOperation] = (
    _get_symbol_to_operation_subdict(BINARY_SYMBOL_OPERATIONS, "||")
)
_LOGICAL_AND_SYMBOL_OPERATIONS: frozendict[str, BinaryOperation] = (
    _get_symbol_to_operation_subdict(BINARY_SYMBOL_OPERATIONS, "&&")
)
_EQUALITY_SYMBOL_OPERATIONS: frozendict[str, BinaryOperation] = (
    _get_symbol_to_operation_subdict(BINARY_SYMBOL_OPERATIONS, "==", "!=")
)
_COMPARISON_SYMBOL_OPERATIONS: frozendict[str, BinaryOperation] = (
    _get_symbol_to_operation_subdict(BINARY_SYMBOL_OPERATIONS, "<", "<=", ">", ">=")
)
_ADDITION_SYMBOL_OPERATIONS: frozendict[str, BinaryOperation] = (
    _get_symbol_to_operation_subdict(BINARY_SYMBOL_OPERATIONS, "+", "-")
)
_MULTIPLICATION_SYMBOL_OPERATIONS: frozendict[str, BinaryOperation] = (
    _get_symbol_to_operation_subdict(BINARY_SYMBOL_OPERATIONS, "*", "/", "//", "%")
)
_EXPONENTIATION_SYMBOL_OPERATIONS: frozendict[str, BinaryOperation] = (
    _get_symbol_to_operation_subdict(BINARY_SYMBOL_OPERATIONS, "**")
)


class ExpressionParser:
    """Parser for expressions.

    Args:
        tokens: List of tokens.

    TODO: Document the BNF grammar for these expressions here.

    """

    _tokens: list[str]
    _current: int
    _idenfitiers: dict[str, Identifier]

    def __init__(self, tokens: list[str]):
        self._tokens = tokens
        self._current = 0
        self._idenfitiers = {}

    def parse(self) -> Expression:
        """Return the parsed expression as an expression tree."""
        return self._expression()

    def _expression(self) -> Expression:
        return self._addition()

    def _binary_operation(
        self,
        operations: frozendict[str, BinaryOperation],
        next_precedence: Callable[[], Expression],
    ) -> Expression:
        expression = next_precedence()
        while self._match(*operations.keys()):
            op = self._get_previous_token()
            right = next_precedence()
            expression = BinaryExpression(operations[op], expression, right)
        return expression

    def _logical_or(self) -> Expression:
        return self._binary_operation(_LOGICAL_OR_SYMBOL_OPERATIONS, self._logical_and)

    def _logical_and(self) -> Expression:
        return self._binary_operation(_LOGICAL_AND_SYMBOL_OPERATIONS, self._comparison)

    def _equality(self) -> Expression:
        return self._binary_operation(_EQUALITY_SYMBOL_OPERATIONS, self._comparison)

    def _comparison(self) -> Expression:
        return self._binary_operation(_COMPARISON_SYMBOL_OPERATIONS, self._addition)

    def _addition(self) -> Expression:
        return self._binary_operation(_ADDITION_SYMBOL_OPERATIONS, self._multiplication)

    def _multiplication(self) -> Expression:
        return self._binary_operation(_MULTIPLICATION_SYMBOL_OPERATIONS, self._unary)

    def _unary(self) -> Expression:
        if self._match("-", "+", "!"):
            op = self._get_previous_token()
            right = self._unary()
            return UnaryExpression(UNARY_SYMBOL_OPERATIONS[op], right)
        return self._exponentiation()

    def _exponentiation(self) -> Expression:
        return self._binary_operation(_EXPONENTIATION_SYMBOL_OPERATIONS, self._primary)

    def _primary(self) -> Expression:
        if self._match_number():
            return LiteralExpression(self._get_previous_token())
        elif self._match_identifier():
            previous_token = self._get_previous_token()
            if previous_token == "True":
                return LiteralExpression(True)
            elif previous_token == "False":
                return LiteralExpression(False)
            else:
                identifier = self._get_identifier_from_symbol(previous_token)
                return IdentifierExpression(identifier)
        elif self._match("("):
            expression = self._expression()
            self._consume_token(
                ")",
                'Expected ")" after expression, but got '
                f"{self._peek_at_current_token()}.",
            )
            return expression
        raise RuntimeError(
            f'Unexpected token "{self._peek_at_current_token()}" at position '
            f"{self._current} while parsing an expression."
        )

    def _get_identifier_from_symbol(self, symbol: str) -> Identifier:
        if symbol not in self._idenfitiers:
            self._idenfitiers[symbol] = Identifier(symbol)
        return self._idenfitiers[symbol]

    def _match(self, *types: str) -> bool:
        if self._peek_at_current_token() in types:
            self._advance_to_next_token()
            return True
        return False

    def _match_number(self) -> bool:
        pattern = _build_token_pattern(_FLOAT_PATTERN.pattern, _INTEGER_PATTERN.pattern)
        current_token = self._peek_at_current_token()
        if current_token and re.match(pattern, current_token):
            self._advance_to_next_token()
            return True
        return False

    def _match_identifier(self) -> bool:
        current_token = self._peek_at_current_token()
        if current_token and re.match(_IDENTIFIER_PATTERN, current_token):
            self._advance_to_next_token()
            return True
        return False

    def _consume_token(self, token: str, error_message: str) -> str:
        if self._peek_at_current_token() == token:
            return self._advance_to_next_token()
        else:
            raise RuntimeError(error_message)

    def _peek_at_current_token(self) -> str | None:
        """Return the current token without advancing; if no tokens, return None."""
        if self._current < len(self._tokens):
            return self._tokens[self._current]
        return None

    def _advance_to_next_token(self) -> str:
        """Return the current token and advance to the next one."""
        self._current += 1
        return self._tokens[self._current - 1]

    def _get_previous_token(self) -> str:
        """Return the previous token."""
        return self._tokens[self._current - 1]


def tokenize_expression(expression_str: str) -> list[str]:
    """Tokenize the expression string.

    Args:
        expression_str: Expression string.

    Returns:
        List of tokens.

    """
    tokens = re.findall(_TOKEN_PATTERN, expression_str)
    return [token for token in tokens if token.strip()]


def parse_expression(expression_str: str) -> Expression:
    """Parse an expression string.

    Args:
        expression_str: Expression string.

    Returns:
        Expression tree.

    """
    tokens = tokenize_expression(expression_str)
    parser = ExpressionParser(tokens)
    return parser.parse()
