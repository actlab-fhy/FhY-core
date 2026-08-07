"""Declarative pattern matching over the expression IR.

This module exposes a `Pattern` hierarchy for describing shapes of
expressions (with optional capture variables, predicates, and
alternatives), a `MatchBindings` value object that records captured
sub-expressions, and the `match_pattern` / `does_pattern_match` free
functions that drive a one-shot, root-level match. Rule-based
rewriting on top of these primitives lives in the sibling
``rewrite.py`` module.
"""

from fhy_core.utils.override import override

__all__ = [
    "AlternativesPattern",
    "BinaryExpressionPattern",
    "CallExpressionPattern",
    "CapturePattern",
    "IdentifierPattern",
    "LiteralPattern",
    "MatchBindings",
    "Pattern",
    "PiecewiseExpressionPattern",
    "PredicatePattern",
    "UnaryExpressionPattern",
    "WildcardPattern",
    "does_pattern_match",
    "match_pattern",
]

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import final

from immutabledict import immutabledict

from fhy_core.identifier import Identifier
from fhy_core.traits import FrozenMixin

from ..core import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    LiteralType,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
)


@final
@dataclass(frozen=True)
class MatchBindings(FrozenMixin):
    """Immutable map from capture name to matched expression.

    Capture names are plain strings: ``CapturePattern("x", ...)``
    used in two positions of a compound pattern shares the same
    capture and triggers the repeated-capture rule below.

    A `MatchBindings` is produced by `Pattern.match` (or returned
    unchanged by patterns that capture nothing). The only mutation
    seam is `try_bind`, which honors the "repeated capture name
    must map to a structurally equivalent expression" rule.

    Equality and hash semantics. Two `MatchBindings` are equal when
    their bound capture-name sets match and each pair of values is
    structurally equivalent. The hash is computed over the bound
    capture-name set only; two instances with identical key sets
    but different bound expressions collide in hash and are
    distinguished by `__eq__`.
    """

    bindings: immutabledict[str, Expression] = field(default_factory=immutabledict)

    def __post_init__(self) -> None:
        for key, value in self.bindings.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"MatchBindings keys must be str; got {type(key).__name__}."
                )
            if not isinstance(value, Expression):
                raise TypeError(
                    f"MatchBindings values must be Expression; "
                    f"got {type(value).__name__} for key {key!r}."
                )

    @classmethod
    def empty(cls) -> "MatchBindings":
        """Return an empty `MatchBindings`.

        Returns:
            An empty `MatchBindings`.

        """
        return cls()

    def is_empty(self) -> bool:
        """Return whether no capture names are bound.

        Returns:
            ``True`` when no capture names are bound; otherwise
            ``False``.

        """
        return len(self.bindings) == 0

    def names(self) -> frozenset[str]:
        """Return the set of currently-bound capture names.

        Returns:
            The frozen set of capture names.

        """
        return frozenset(self.bindings.keys())

    def get(self, name: str) -> Expression:
        """Return the expression bound to ``name``.

        Args:
            name: Capture name to look up.

        Returns:
            The bound expression.

        Raises:
            KeyError: If ``name`` is not bound.

        """
        return self.bindings[name]

    def has(self, name: str) -> bool:
        """Return whether ``name`` is bound.

        Args:
            name: Capture name to check.

        Returns:
            ``True`` when ``name`` is bound; otherwise ``False``.

        """
        return name in self.bindings

    def try_bind(self, name: str, expression: Expression) -> "MatchBindings | None":
        """Attempt to add or reconfirm a capture binding.

        When ``name`` is not bound, returns a new `MatchBindings`
        with ``name`` mapped to ``expression``. When ``name`` is
        already bound to a structurally equivalent expression,
        returns the receiver itself; the originally-bound expression
        is retained and the new ``expression`` is discarded, so
        downstream consumers see the first-bound instance. Otherwise
        returns ``None`` --- the caller's match must fail.

        Args:
            name: Capture name to bind.
            expression: Expression to bind ``name`` to.

        Returns:
            A new `MatchBindings`, the receiver itself, or ``None``.

        """
        if name not in self.bindings:
            return MatchBindings(self.bindings.set(name, expression))
        if self.bindings[name].is_structurally_equivalent(expression):
            return self
        return None

    @override
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

    @override
    def __hash__(self) -> int:
        return hash(frozenset(self.bindings.keys()))


class Pattern(FrozenMixin, ABC):
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


def _validate_pattern_instance(value: object, context: str) -> None:
    """Validate that ``value`` is a `Pattern` instance.

    Args:
        value: Candidate to check.
        context: Description of the field or position being validated,
            used verbatim at the front of the error message.

    Raises:
        ValueError: If ``value`` is not a `Pattern` instance.

    """
    if not isinstance(value, Pattern):
        raise ValueError(
            f"{context} must be a Pattern instance, but got value {value!r} "
            f"of type {type(value).__name__}."
        )


@final
@dataclass(frozen=True)
class WildcardPattern(Pattern):
    """Match any expression; capture nothing.

    The "don't care" filler used inside compound patterns. Matching
    always succeeds; the input bindings are returned unchanged.
    """

    @override
    def match_under(
        self, expression: Expression, bindings: MatchBindings
    ) -> MatchBindings | None:
        del expression
        return bindings


@final
@dataclass(frozen=True)
class CapturePattern(Pattern):
    """Match a sub-expression and bind it to a capture name.

    Capture names are plain strings: writing ``CapturePattern("x", …)``
    in two positions of the same compound pattern shares one capture,
    so the matched sub-expressions must be structurally equivalent for
    the overall pattern to succeed.

    Attributes:
        name: Capture name under which the matched expression is
            recorded in `MatchBindings`.
        sub_pattern: Inner pattern the expression must match before
            capture. Use `WildcardPattern()` for "any expression."

    """

    name: str
    sub_pattern: Pattern

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError(
                f"CapturePattern name must be a str, but got value {self.name!r} "
                f"of type {type(self.name).__name__}."
            )
        if not self.name:
            raise ValueError("CapturePattern name must not be empty.")
        _validate_pattern_instance(self.sub_pattern, "CapturePattern sub_pattern")

    @override
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

    @override
    def match_under(
        self, expression: Expression, bindings: MatchBindings
    ) -> MatchBindings | None:
        if not isinstance(expression, LiteralExpression):
            return None
        elif self.value is None:
            return bindings
        elif (
            # bool is a subclass of int: a type-identity check prevents
            # LiteralPattern(value=1) from matching LiteralExpression(True).
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
            identity; see :class:`Identifier` for the equality
            contract).

    """

    identifier: Identifier | None = None

    @override
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

    def __post_init__(self) -> None:
        _validate_pattern_instance(self.operand, "UnaryExpressionPattern operand")

    @override
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

    def __post_init__(self) -> None:
        _validate_pattern_instance(self.left, "BinaryExpressionPattern left")
        _validate_pattern_instance(self.right, "BinaryExpressionPattern right")

    @override
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
class PiecewiseExpressionPattern(Pattern):
    """Match a `PiecewiseExpression`.

    Attributes:
        cases: `(condition_pattern, value_pattern)` pairs matched
            position-wise against the expression's cases; the case
            count must match exactly. Must be non-empty when
            supplied; `None` means "any cases."
        otherwise: Pattern the `otherwise` expression must match.

    Raises:
        ValueError: If `cases` is supplied empty, if any element of
            `cases` is not a two-element `(condition_pattern,
            value_pattern)` pair, if any condition or value pattern
            within `cases` is not a `Pattern` instance, or if
            `otherwise` is not a `Pattern` instance.

    """

    cases: tuple[tuple[Pattern, Pattern], ...] | None
    otherwise: Pattern

    def __post_init__(self) -> None:
        cases = None if self.cases is None else tuple(self.cases)
        if cases is not None:
            if not cases:
                raise ValueError(
                    "PiecewiseExpressionPattern.cases must be non-empty when "
                    "supplied; use `None` to match any case count."
                )
            coerced_cases: list[tuple[Pattern, Pattern]] = []
            for case in cases:
                try:
                    condition_pattern, value_pattern = case
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "PiecewiseExpressionPattern.cases elements must be "
                        "(condition_pattern, value_pattern) pairs, but got "
                        f"value {case!r} of type {type(case).__name__}."
                    ) from error
                _validate_pattern_instance(
                    condition_pattern,
                    "PiecewiseExpressionPattern.cases condition pattern",
                )
                _validate_pattern_instance(
                    value_pattern, "PiecewiseExpressionPattern.cases value pattern"
                )
                coerced_cases.append((condition_pattern, value_pattern))
            cases = tuple(coerced_cases)
        _validate_pattern_instance(
            self.otherwise, "PiecewiseExpressionPattern.otherwise"
        )
        object.__setattr__(self, "cases", cases)

    @override
    def match_under(
        self, expression: Expression, bindings: MatchBindings
    ) -> MatchBindings | None:
        if not isinstance(expression, PiecewiseExpression):
            return None
        accumulator = bindings
        if self.cases is not None:
            expression_cases = expression.get_cases()
            if len(self.cases) != len(expression_cases):
                return None
            for (condition_pattern, value_pattern), (condition, value) in zip(
                self.cases, expression_cases, strict=True
            ):
                after_condition = condition_pattern.match_under(condition, accumulator)
                if after_condition is None:
                    return None
                after_value = value_pattern.match_under(value, after_condition)
                if after_value is None:
                    return None
                accumulator = after_value
        return self.otherwise.match_under(expression.otherwise, accumulator)


@final
@dataclass(frozen=True)
class CallExpressionPattern(Pattern):
    """Match a `CallExpression`.

    Attributes:
        function_name: Required function name. ``None`` means "any
            function name."
        arguments: Tuple of patterns the arguments must match
            position-wise. The empty tuple ``()`` matches only
            zero-argument calls; ``None`` means "any arity, any
            arguments." When a non-``None`` tuple is supplied, the
            pattern matches only when
            ``len(self.arguments) == len(call.arguments)``.

    Raises:
        ValueError: If `arguments` is supplied and any element is
            not a `Pattern` instance.

    """

    function_name: str | None
    arguments: tuple[Pattern, ...] | None

    def __post_init__(self) -> None:
        arguments = None if self.arguments is None else tuple(self.arguments)
        if arguments is not None:
            for argument in arguments:
                _validate_pattern_instance(
                    argument, "CallExpressionPattern.arguments element"
                )
        object.__setattr__(self, "arguments", arguments)

    @override
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
        for argument_pattern, argument in zip(
            self.arguments, expression.arguments, strict=True
        ):
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

    @override
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
    accumulator for the next; captures from a failed alternative
    are discarded. Captures inside the chosen alternative are
    visible to subsequent siblings in the enclosing compound
    pattern.

    Attributes:
        alternatives: Tuple of sub-patterns. Must be non-empty.

    Raises:
        ValueError: If `alternatives` is empty, or if any element is
            not a `Pattern` instance.

    """

    alternatives: tuple[Pattern, ...]

    def __post_init__(self) -> None:
        alternatives = tuple(self.alternatives)
        if not alternatives:
            raise ValueError("AlternativesPattern.alternatives must be non-empty.")
        for alternative in alternatives:
            _validate_pattern_instance(
                alternative, "AlternativesPattern.alternatives element"
            )
        object.__setattr__(self, "alternatives", alternatives)

    @override
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
