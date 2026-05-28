"""Declarative pattern matching over the expression IR.

This module exposes a `Pattern` hierarchy for describing shapes of
expressions (with optional capture variables, predicates, and
alternatives), a `MatchBindings` value object that records captured
sub-expressions, and the `match_pattern` / `does_pattern_match` free
functions that drive a one-shot, root-level match. Rule-based
rewriting on top of these primitives lives in the sibling
``rewrite.py`` module.
"""

__all__ = [
    "AlternativesPattern",
    "BinaryExpressionPattern",
    "CallExpressionPattern",
    "CapturePattern",
    "IdentifierPattern",
    "LiteralPattern",
    "MatchBindings",
    "Pattern",
    "PredicatePattern",
    "TernaryExpressionPattern",
    "UnaryExpressionPattern",
    "WildcardPattern",
    "does_pattern_match",
    "match_pattern",
]

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import final

from frozendict import frozendict

from fhy_core.identifier import Identifier
from fhy_core.trait import FrozenMixin

from ..core import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    LiteralType,
    TernaryExpression,
    UnaryExpression,
    UnaryOperation,
)


@final
@dataclass(frozen=True)
class MatchBindings(FrozenMixin):
    """Immutable map from capture identifier to matched expression.

    A `MatchBindings` is produced by `Pattern.match` (or returned
    unchanged by patterns that capture nothing). The only mutation
    seam is `try_bind`, which honors the "repeated capture identifier
    must map to a structurally equivalent expression" rule.
    """

    bindings: frozendict[Identifier, Expression] = field(
        default_factory=frozendict[Identifier, Expression]
    )

    @classmethod
    def empty(cls) -> "MatchBindings":
        """Return the bindings used by `Pattern.match` at the root.

        Returns:
            An empty `MatchBindings`.

        """
        return cls()

    def is_empty(self) -> bool:
        """Return whether no identifiers are bound.

        Returns:
            ``True`` when no capture identifiers are bound; otherwise
            ``False``.

        """
        return len(self.bindings) == 0

    def names(self) -> frozenset[Identifier]:
        """Return the set of currently-bound capture identifiers.

        Returns:
            The frozen set of capture identifiers.

        """
        return frozenset(self.bindings.keys())

    def get(self, name: Identifier | str) -> Expression:
        """Return the expression bound to ``name``.

        Args:
            name: Capture identifier to look up. A ``str`` is
                resolved to the unique bound `Identifier` whose
                ``name_hint`` matches.

        Returns:
            The bound expression.

        Raises:
            KeyError: If ``name`` is not bound, or if ``name`` is a
                ``str`` that matches no bound identifier.
            ValueError: If ``name`` is a ``str`` matching more than one
                bound identifier (ambiguous lookup).

        """
        return self.bindings[self._resolve_name(name)]

    def has(self, name: Identifier | str) -> bool:
        """Return whether ``name`` is bound.

        Args:
            name: Capture identifier to check. A ``str`` matches when
                any bound `Identifier`'s ``name_hint`` equals it.

        Returns:
            ``True`` when ``name`` is bound; otherwise ``False``.

        """
        if isinstance(name, Identifier):
            return name in self.bindings
        return any(identifier.name_hint == name for identifier in self.bindings)

    def _resolve_name(self, name: Identifier | str) -> Identifier:
        """Resolve a name to the unique bound `Identifier`.

        Args:
            name: Capture identifier or its ``name_hint``.

        Returns:
            The `Identifier` to look up.

        Raises:
            KeyError: If ``name`` is a ``str`` that matches no bound
                identifier.
            ValueError: If ``name`` is a ``str`` matching more than one
                bound identifier.

        """
        if isinstance(name, Identifier):
            return name
        matches = [
            identifier for identifier in self.bindings if identifier.name_hint == name
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(name)
        raise ValueError(
            f"Capture name {name!r} is ambiguous: matches "
            f"{len(matches)} bound identifiers."
        )

    def try_bind(
        self, name: Identifier, expression: Expression
    ) -> "MatchBindings | None":
        """Attempt to add or reconfirm a capture binding.

        When ``name`` is not bound, returns a new `MatchBindings`
        with ``name`` mapped to ``expression``. When ``name`` is
        already bound to a structurally equivalent expression,
        returns the receiver itself. Otherwise returns ``None`` ---
        the caller's match must fail.

        Args:
            name: Capture identifier to bind.
            expression: Expression to bind ``name`` to.

        Returns:
            A new `MatchBindings`, the receiver itself, or ``None``.

        """
        existing = self.bindings.get(name)
        if existing is None:
            return MatchBindings(self.bindings.set(name, expression))
        elif existing.is_structurally_equivalent(expression):
            return self
        else:
            return None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MatchBindings):
            return NotImplemented
        elif self.bindings.keys() != other.bindings.keys():
            return False
        else:
            return all(
                self.bindings[name].is_structurally_equivalent(other.bindings[name])
                for name in self.bindings
            )

    def __hash__(self) -> int:
        return hash(frozenset(self.bindings.keys()))


class Pattern(ABC):
    """Abstract base class for expression-tree patterns.

    Subclasses describe a shape that an `Expression` may or may not
    match. Successful matches return a `MatchBindings`; failed
    matches return ``None``. The framework never raises on a
    structural mismatch; raises are reserved for predicate callbacks
    and rule rewrite callbacks supplied by the caller.
    """

    def match(self, expression: Expression) -> MatchBindings | None:
        """Attempt to match ``expression`` against this pattern.

        Args:
            expression: Expression to match against.

        Returns:
            The resulting `MatchBindings` on success, or ``None`` on
            failure.

        """
        return self.match_under(expression, MatchBindings.empty())

    @abstractmethod
    def match_under(
        self, expression: Expression, bindings: MatchBindings
    ) -> MatchBindings | None:
        """Attempt to match ``expression`` while threading ``bindings``.

        Composition seam for compound patterns: sub-patterns are
        matched with the accumulator returned by the previous
        sub-pattern's match, so that repeated capture names stay
        consistent across positions.

        Args:
            expression: Expression to match against.
            bindings: Bindings accumulated so far.

        Returns:
            The updated `MatchBindings` on success, or ``None`` on
            failure.

        """


@final
@dataclass(frozen=True)
class WildcardPattern(Pattern):
    """Match any expression; capture nothing.

    The "don't care" filler used inside compound patterns. Matching
    always succeeds; the input bindings are returned unchanged.
    """

    def match_under(
        self, expression: Expression, bindings: MatchBindings
    ) -> MatchBindings | None:
        del expression
        return bindings


@final
@dataclass(frozen=True, init=False)
class CapturePattern(Pattern):
    """Match a sub-expression and bind it to a capture identifier.

    The constructor accepts either an `Identifier` or a ``str``;
    strings are wrapped in a fresh `Identifier` so callers writing
    patterns by hand do not have to construct identifiers themselves.
    Each ``str``-form construction produces a distinct `Identifier`;
    reuse a shared `Identifier` (or the same `CapturePattern`
    instance) across positions when the same capture must appear
    twice in one pattern.

    Attributes:
        name: Capture identifier under which the matched expression
            is recorded in `MatchBindings`.
        sub_pattern: Inner pattern the expression must match before
            capture. Use `WildcardPattern()` for "any expression."

    """

    name: Identifier
    sub_pattern: Pattern

    def __init__(self, name: Identifier | str, sub_pattern: Pattern) -> None:
        coerced = Identifier(name) if isinstance(name, str) else name
        object.__setattr__(self, "name", coerced)
        object.__setattr__(self, "sub_pattern", sub_pattern)

    def match_under(
        self, expression: Expression, bindings: MatchBindings
    ) -> MatchBindings | None:
        inner = self.sub_pattern.match_under(expression, bindings)
        if inner is None:
            return None
        else:
            return inner.try_bind(self.name, expression)


@final
@dataclass(frozen=True)
class LiteralPattern(Pattern):
    """Match a `LiteralExpression`.

    Attributes:
        value: Required literal value. ``None`` means "any literal
            value." When a value is supplied, the candidate's value
            must satisfy ``type(self.value) is type(other.value)``
            and ``self.value == other.value``.

    """

    value: LiteralType | None = None

    def match_under(
        self, expression: Expression, bindings: MatchBindings
    ) -> MatchBindings | None:
        if not isinstance(expression, LiteralExpression):
            return None
        elif self.value is None:
            return bindings
        elif (
            type(self.value) is type(expression.value)
            and self.value == expression.value
        ):
            return bindings
        else:
            return None


@final
@dataclass(frozen=True)
class IdentifierPattern(Pattern):
    """Match an `IdentifierExpression`.

    Attributes:
        identifier: Required identifier. ``None`` means "any
            identifier." When supplied, the candidate's identifier
            must be the same `Identifier` object (process-global
            identity).

    """

    identifier: Identifier | None = None

    def match_under(
        self, expression: Expression, bindings: MatchBindings
    ) -> MatchBindings | None:
        if not isinstance(expression, IdentifierExpression):
            return None
        elif self.identifier is None:
            return bindings
        elif self.identifier == expression.identifier:
            return bindings
        else:
            return None


@final
@dataclass(frozen=True)
class UnaryExpressionPattern(Pattern):
    """Match a `UnaryExpression`.

    Attributes:
        operation: Required unary operation. ``None`` means "any
            unary operation."
        operand: Pattern the operand must match.

    """

    operation: UnaryOperation | None
    operand: Pattern

    def match_under(
        self, expression: Expression, bindings: MatchBindings
    ) -> MatchBindings | None:
        if not isinstance(expression, UnaryExpression):
            return None
        elif self.operation is not None and self.operation != expression.operation:
            return None
        else:
            return self.operand.match_under(expression.operand, bindings)


@final
@dataclass(frozen=True)
class BinaryExpressionPattern(Pattern):
    """Match a `BinaryExpression`.

    Attributes:
        operation: Required binary operation. ``None`` means "any
            binary operation."
        left: Pattern the left operand must match.
        right: Pattern the right operand must match.

    """

    operation: BinaryOperation | None
    left: Pattern
    right: Pattern

    def match_under(
        self, expression: Expression, bindings: MatchBindings
    ) -> MatchBindings | None:
        if not isinstance(expression, BinaryExpression):
            return None
        elif self.operation is not None and self.operation != expression.operation:
            return None
        after_left = self.left.match_under(expression.left, bindings)
        if after_left is None:
            return None
        else:
            return self.right.match_under(expression.right, after_left)


@final
@dataclass(frozen=True)
class TernaryExpressionPattern(Pattern):
    """Match a `TernaryExpression`.

    Attributes:
        condition: Pattern the condition must match.
        true_value: Pattern the true-branch value must match.
        false_value: Pattern the false-branch value must match.

    """

    condition: Pattern
    true_value: Pattern
    false_value: Pattern

    def match_under(
        self, expression: Expression, bindings: MatchBindings
    ) -> MatchBindings | None:
        if not isinstance(expression, TernaryExpression):
            return None
        after_condition = self.condition.match_under(expression.condition, bindings)
        if after_condition is None:
            return None
        after_true = self.true_value.match_under(expression.true_value, after_condition)
        if after_true is None:
            return None
        else:
            return self.false_value.match_under(expression.false_value, after_true)


@final
@dataclass(frozen=True)
class CallExpressionPattern(Pattern):
    """Match a `CallExpression`.

    Attributes:
        function_name: Required function name. ``None`` means "any
            function name."
        arguments: Tuple of patterns the arguments must match
            position-wise. ``None`` means "any number of arguments,
            any pattern." When supplied, the pattern matches only
            when ``len(self.arguments) == len(call.arguments)``.

    """

    function_name: str | None
    arguments: tuple[Pattern, ...] | None

    def match_under(
        self, expression: Expression, bindings: MatchBindings
    ) -> MatchBindings | None:
        if not isinstance(expression, CallExpression):
            return None
        elif (
            self.function_name is not None
            and self.function_name != expression.function_name
        ):
            return None
        elif self.arguments is None:
            return bindings
        elif len(self.arguments) != len(expression.arguments):
            return None
        accumulator = bindings
        for argument_pattern, argument in zip(self.arguments, expression.arguments):
            next_bindings = argument_pattern.match_under(argument, accumulator)
            if next_bindings is None:
                return None
            accumulator = next_bindings
        return accumulator


@final
@dataclass(frozen=True)
class PredicatePattern(Pattern):
    """Match when a Python predicate over the expression returns ``True``.

    Captures nothing. Predicate exceptions propagate; the framework
    does not coerce them to a non-match.

    Attributes:
        predicate: Callable taking the candidate `Expression` and
            returning a ``bool``.

    """

    predicate: Callable[[Expression], bool]

    def match_under(
        self, expression: Expression, bindings: MatchBindings
    ) -> MatchBindings | None:
        if self.predicate(expression):
            return bindings
        else:
            return None


@final
@dataclass(frozen=True)
class AlternativesPattern(Pattern):
    """Match when any of several sub-patterns matches.

    Sub-patterns are tried left-to-right; the bindings from the first
    successful sub-pattern are returned. Each attempt starts from
    the input bindings, so a failed attempt does not taint the
    accumulator for the next.

    Attributes:
        alternatives: Tuple of sub-patterns. Must be non-empty.

    """

    alternatives: tuple[Pattern, ...]

    def __post_init__(self) -> None:
        if not self.alternatives:
            raise ValueError("AlternativesPattern.alternatives must be non-empty.")

    def match_under(
        self, expression: Expression, bindings: MatchBindings
    ) -> MatchBindings | None:
        for alternative in self.alternatives:
            result = alternative.match_under(expression, bindings)
            if result is not None:
                return result
        return None


def match_pattern(pattern: Pattern, expression: Expression) -> MatchBindings | None:
    """Match ``pattern`` against ``expression`` at the root.

    Args:
        pattern: Pattern to match.
        expression: Expression to match against.

    Returns:
        The resulting `MatchBindings` on success, or ``None`` on
        failure.

    """
    return pattern.match(expression)


def does_pattern_match(pattern: Pattern, expression: Expression) -> bool:
    """Report whether ``pattern`` matches ``expression`` at the root.

    Args:
        pattern: Pattern to match.
        expression: Expression to match against.

    Returns:
        ``True`` when the pattern matches; otherwise ``False``.

    """
    return pattern.match(expression) is not None
