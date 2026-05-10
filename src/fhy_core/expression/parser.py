"""Parser from strings to expression trees."""

__all__ = ["ParseError", "parse_expression", "tokenize_expression"]

import re
from typing import Callable

from frozendict import frozendict

from fhy_core.error import register_error
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


@register_error
class ParseError(RuntimeError):
    """Raised when the expression parser cannot consume its input.

    Subclasses ``RuntimeError`` so existing ``except RuntimeError`` callers
    keep working.
    """


def _build_token_pattern(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(f"{pattern}" for pattern in patterns))


_FLOAT_PATTERN = re.compile(r"\d+\.\d*|\.\d+")
_INTEGER_PATTERN = re.compile(r"\d+")
_NUMERIC_PATTERN = _build_token_pattern(
    _FLOAT_PATTERN.pattern, _INTEGER_PATTERN.pattern
)
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


def _select_binary_symbol_operations(
    *symbols: str,
) -> frozendict[str, BinaryOperation]:
    return frozendict({symbol: BINARY_SYMBOL_OPERATIONS[symbol] for symbol in symbols})


_LOGICAL_OR_SYMBOL_OPERATIONS = _select_binary_symbol_operations("||")
_LOGICAL_AND_SYMBOL_OPERATIONS = _select_binary_symbol_operations("&&")
_EQUALITY_SYMBOL_OPERATIONS = _select_binary_symbol_operations("==", "!=")
_COMPARISON_SYMBOL_OPERATIONS = _select_binary_symbol_operations("<", "<=", ">", ">=")
_ADDITION_SYMBOL_OPERATIONS = _select_binary_symbol_operations("+", "-")
_MULTIPLICATION_SYMBOL_OPERATIONS = _select_binary_symbol_operations(
    "*", "/", "//", "%"
)
_EXPONENTIATION_SYMBOL_OPERATIONS = _select_binary_symbol_operations("**")


class ExpressionParser:
    """Parser for expressions.

    Args:
        tokens: List of tokens.

    Grammar (BNF, lowest-to-highest precedence)::

        expression     ::= logical_or
        logical_or     ::= logical_and ( "||" logical_and )*
        logical_and    ::= equality ( "&&" equality )*
        equality       ::= comparison ( ( "==" | "!=" ) comparison )*
        comparison     ::= addition ( ( "<" | "<=" | ">" | ">=" ) addition )*
        addition       ::= multiplication ( ( "+" | "-" ) multiplication )*
        multiplication ::= unary ( ( "*" | "/" | "//" | "%" ) unary )*
        unary          ::= ( "-" | "+" | "!" ) unary | exponentiation
        exponentiation ::= primary ( "**" primary )*
        primary        ::= number
                         | "True" | "False"
                         | identifier
                         | "(" expression ")"

    Binary operators are left-associative except ``**`` which is
    right-associative in typical usage but parsed here left-to-right at the
    exponentiation precedence level.
    """

    _tokens: list[str]
    _current: int
    _identifiers: dict[str, Identifier]
    _is_consumed: bool

    def __init__(self, tokens: list[str]):
        self._tokens = tokens
        self._current = 0
        self._identifiers = {}
        self._is_consumed = False

    def parse(self) -> Expression:
        """Return the parsed expression as an expression tree.

        Each ``ExpressionParser`` instance is single-use: a second call to
        ``parse()`` raises ``ParseError``. Use a fresh parser (or call
        ``parse_expression``) to parse another input.

        Raises:
            ParseError: On malformed input, trailing tokens, unexpected end
                of input, or a second call on the same parser.

        """
        if self._is_consumed:
            raise ParseError(
                "ExpressionParser already consumed; create a new instance "
                "to parse again."
            )
        self._is_consumed = True
        expression = self._expression()
        if self._current < len(self._tokens):
            trailing_token = self._tokens[self._current]
            raise ParseError(
                f'Unexpected trailing token "{trailing_token}" at position '
                f"{self._current} after parsing an expression."
            )
        return expression

    def _expression(self) -> Expression:
        return self._logical_or()

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
        return self._binary_operation(_LOGICAL_AND_SYMBOL_OPERATIONS, self._equality)

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
            self._consume_token(")", 'Expected ")" after expression')
            return expression
        current_token = self._peek_at_current_token()
        if current_token is None:
            raise ParseError(
                f"Unexpected end of input at position {self._current} "
                "while parsing an expression."
            )
        raise ParseError(
            f'Unexpected token "{current_token}" at position '
            f"{self._current} while parsing an expression."
        )

    def _get_identifier_from_symbol(self, symbol: str) -> Identifier:
        if symbol not in self._identifiers:
            self._identifiers[symbol] = Identifier(symbol)
        return self._identifiers[symbol]

    def _match(self, *types: str) -> bool:
        if self._peek_at_current_token() in types:
            self._advance_to_next_token()
            return True
        return False

    def _match_number(self) -> bool:
        current_token = self._peek_at_current_token()
        if current_token and re.match(_NUMERIC_PATTERN, current_token):
            self._advance_to_next_token()
            return True
        return False

    def _match_identifier(self) -> bool:
        current_token = self._peek_at_current_token()
        if current_token and re.match(_IDENTIFIER_PATTERN, current_token):
            self._advance_to_next_token()
            return True
        return False

    def _consume_token(self, token: str, error_message_prefix: str) -> str:
        current_token = self._peek_at_current_token()
        if current_token == token:
            return self._advance_to_next_token()
        if current_token is None:
            raise ParseError(
                f"{error_message_prefix} but reached end of input at "
                f"position {self._current}."
            )
        raise ParseError(
            f'{error_message_prefix} but got "{current_token}" at position '
            f"{self._current}."
        )

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

    Walks the input left-to-right, collecting tokens. Whitespace is skipped.
    Any character that is neither whitespace nor part of a valid token is
    collected and reported in a single ``ParseError``.

    Args:
        expression_str: Expression string.

    Returns:
        List of tokens.

    Raises:
        ParseError: If the input contains any character that cannot be part
            of a valid token (e.g. ``@``, non-ASCII characters).

    """
    tokens: list[str] = []
    unrecognized: list[tuple[int, str]] = []
    position = 0
    length = len(expression_str)
    while position < length:
        char = expression_str[position]
        if char.isspace():
            position += 1
            continue
        match = _TOKEN_PATTERN.match(expression_str, position)
        if match is not None:
            tokens.append(match.group(0))
            position = match.end()
            continue
        unrecognized.append((position, char))
        position += 1
    if unrecognized:
        positions = [str(pos) for pos, _ in unrecognized]
        characters = "".join(char for _, char in unrecognized)
        raise ParseError(
            f"Unrecognized characters in expression: {characters!r} at "
            f"positions [{', '.join(positions)}]."
        )
    return tokens


def parse_expression(expression_str: str) -> Expression:
    """Parse an expression string.

    Each call mints fresh ``Identifier`` instances for the symbols it sees.
    Repeated identifier symbols *within* one call share an ``Identifier``;
    repeated symbols *across* calls do not. Callers wanting cross-parse
    interning must do their own symbol resolution.

    The parser is recursive-descent; deeply-nested input (long unary or
    parenthesis chains) is bounded by Python's recursion limit. For typical
    compiler input this is not a concern; very large generated input may
    require raising the limit before calling.

    Args:
        expression_str: Expression string.

    Returns:
        Expression tree.

    Raises:
        ParseError: For malformed input, unknown characters, trailing
            tokens, unexpected end of input, or unclosed parentheses.

    """
    tokens = tokenize_expression(expression_str)
    parser = ExpressionParser(tokens)
    return parser.parse()
