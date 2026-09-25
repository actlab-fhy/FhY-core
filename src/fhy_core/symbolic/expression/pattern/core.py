"""Declarative pattern matching over the expression IR.

This module exposes the `Capture` handle, a closed `Pattern` hierarchy for
describing shapes of expressions (with captures, predicates, and
alternatives), the `MatchBindings` a match returns, and the
`match_pattern` / `does_pattern_match` free functions that drive a
one-shot, root-level match. Rule-based rewriting on top of these
primitives lives in the sibling ``rewrite.py`` module.

The classes are backed by the Rust implementation (``fhy_core._rs``),
with the Rust core's semantics:

- A `Capture` is an identity handle: the same capture object in two
  positions of a pattern means "equal expressions", and two captures made
  with one name are independent.
- `MatchBindings` are keyed by captures, and only a match produces
  bindings that bind one.
- The pattern kinds are closed: a Python subclass of `Pattern` that is
  none of them cannot be built, and a subclass of one of them is that
  kind.
- A degenerate pattern, such as a logical pattern of fewer than two
  operand patterns, a piecewise pattern of no case, or alternatives of
  none, builds and matches nothing.

Patterns are immutable. They compare, hash, print, and pickle by their
fields, as dataclasses do; a capture compares by identity and a predicate
by the callable's ``==``. A match runs in Rust; a predicate is called with
the node object of the expression being matched, and its result is read
by truthiness.
"""

__all__ = [
    "AlternativesPattern",
    "BinaryExpressionPattern",
    "CallExpressionPattern",
    "Capture",
    "CapturePattern",
    "IdentifierPattern",
    "LiteralPattern",
    "LogicalExpressionPattern",
    "MatchBindings",
    "Pattern",
    "PiecewiseExpressionPattern",
    "PredicatePattern",
    "UnaryExpressionPattern",
    "WildcardPattern",
    "does_pattern_match",
    "match_pattern",
]

from typing import final

from fhy_core import _rs
from fhy_core.traits import FrozenMixin

from ..core import (
    Expression,
)


@final
class Capture(_rs.Capture):
    """A handle a `CapturePattern` binds a matched expression to.

    Identity decides equality: a capture equals only itself, and hashes by
    identity, so ``Capture("x")`` made twice gives two independent
    captures. Using one capture object in two positions of a pattern
    requires the expressions matched there to be structurally equal. The
    name serves only ``str``, ``repr`` and messages; any ``str`` is a valid
    name, the empty one included, and anything else raises ``TypeError``.

    A capture pickles as a call of its class with its name, so each load
    makes a new capture; a capture shared within one pickle stays shared.

    Attributes:
        name: The name the capture was made with.

    """

    __slots__ = ()


FrozenMixin.register(Capture)
Capture._register_public_class()


@final
class MatchBindings(_rs.MatchBindings):
    """The captures of a successful match, in binding order.

    Only a match produces bindings that bind a capture:
    ``MatchBindings()`` and `empty()` bind none, and passing anything to
    the constructor raises ``TypeError``. A capture binds after its pattern
    matched, so the captures inside that pattern precede it; a capture used
    again keeps its first position and its first-bound expression.

    The bound expressions are the node objects the match saw, so
    ``bindings[x] is node`` holds for the node that matched. ``get``
    returns ``None`` for a capture that is not bound, ``bindings[x]``
    raises ``KeyError`` (``capture `x` is not bound``), and ``has`` and
    ``in`` test whether one is. ``len`` counts the bound captures,
    iteration yields them in binding order, and `items` returns the
    ``(capture, expression)`` pairs.

    Two bindings are equal when they bind the same captures, in any order,
    each to structurally equal expressions; hashing agrees. Bindings are
    immutable and do not pickle.
    """

    __slots__ = ()


FrozenMixin.register(MatchBindings)
MatchBindings._register_public_class()


class Pattern(_rs.Pattern):
    """Base class of the expression-tree patterns.

    A pattern describes a shape an `Expression` may have. `match` tests it
    at the root of an expression only, and returns the `MatchBindings` of
    its captures, or ``None`` when the expression does not match; a
    structural mismatch never raises. Compound patterns match their
    sub-patterns in a fixed order, threading one set of bindings through
    them, and stop at the first that fails.

    The kinds are closed: ``Pattern`` has no constructor, so a Python
    subclass that is none of the kinds below cannot be built
    (``TypeError``); a subclass of a kind is that kind. Matching a pattern
    deeper than the recursion limit raises ``RecursionError``. Only a
    predicate can raise during a match, and its exception propagates
    unchanged.
    """

    __slots__ = ()


FrozenMixin.register(Pattern)


@final
class WildcardPattern(_rs.WildcardPattern, Pattern):
    """Match any expression; capture nothing."""

    __slots__ = ()
    __match_args__ = ()


WildcardPattern._register_public_class()


@final
class CapturePattern(_rs.CapturePattern, Pattern):
    """Match what a sub-pattern matches and bind a capture to it.

    The capture binds after ``sub_pattern`` matched. When it is bound
    already, the match succeeds only if the expression is structurally
    equal to the bound one, which is kept. A capture that is not a
    `Capture`, or a sub-pattern that is not a `Pattern`, raises
    ``TypeError``.

    Attributes:
        capture: The `Capture` bound to the matched expression.
        sub_pattern: The pattern the expression must match first;
            `WildcardPattern()` when omitted.

    """

    __slots__ = ()
    __match_args__ = ("capture", "sub_pattern")


CapturePattern._register_public_class()


@final
class LiteralPattern(_rs.LiteralPattern, Pattern):
    """Match a `LiteralExpression`.

    Attributes:
        value: ``None`` for any literal. Otherwise a value
            `LiteralExpression` accepts, kept as given; the candidate must
            equal ``LiteralExpression(value)``. Literals compare by their
            normalized value and kind, so ``"05"`` matches the literal
            ``5``, a NaN every NaN literal, ``-0.0`` the literal ``0.0``,
            and ``1`` neither ``1.0`` nor ``True``. A value
            `LiteralExpression` refuses raises its error.

    """

    __slots__ = ()
    __match_args__ = ("value",)


LiteralPattern._register_public_class()


@final
class IdentifierPattern(_rs.IdentifierPattern, Pattern):
    """Match an `IdentifierExpression`.

    Attributes:
        identifier: ``None`` for any reference; otherwise the
            `Identifier` the reference must have the id of. Anything else
            raises ``TypeError``.

    """

    __slots__ = ()
    __match_args__ = ("identifier",)


IdentifierPattern._register_public_class()


@final
class UnaryExpressionPattern(_rs.UnaryExpressionPattern, Pattern):
    """Match a `UnaryExpression`.

    Attributes:
        operation: The `UnaryOperation`, or ``None`` for any operation.
        operand: The pattern the operand must match.

    """

    __slots__ = ()
    __match_args__ = ("operation", "operand")


UnaryExpressionPattern._register_public_class()


@final
class BinaryExpressionPattern(_rs.BinaryExpressionPattern, Pattern):
    """Match a `BinaryExpression`: its left operand, then its right.

    Attributes:
        operation: The `BinaryOperation`, or ``None`` for any operation.
        left: The pattern the left operand must match.
        right: The pattern the right operand must match.

    """

    __slots__ = ()
    __match_args__ = ("operation", "left", "right")


BinaryExpressionPattern._register_public_class()


@final
class LogicalExpressionPattern(_rs.LogicalExpressionPattern, Pattern):
    """Match a `LogicalExpression`.

    Attributes:
        operation: The `LogicalOperation`, or ``None`` for either
            connective.
        operands: ``None`` for any operands; otherwise the tuple of
            patterns the operands must match position-wise, with exactly as
            many operands. A logical expression has at least two operands,
            so fewer than two operand patterns build a pattern that matches
            nothing.

    """

    __slots__ = ()
    __match_args__ = ("operation", "operands")


LogicalExpressionPattern._register_public_class()


@final
class PiecewiseExpressionPattern(_rs.PiecewiseExpressionPattern, Pattern):
    """Match a `PiecewiseExpression`: its cases in order, then ``otherwise``.

    Each case's condition is matched, then its value.

    Attributes:
        cases: ``None`` for any cases; otherwise the tuple of
            ``(condition_pattern, value_pattern)`` pairs the cases must
            match position-wise, with exactly as many cases. A piecewise
            expression has at least one case, so ``()`` builds a pattern
            that matches nothing. A case that is not a pair of patterns
            raises ``TypeError``.
        otherwise: The pattern the otherwise branch must match.

    """

    __slots__ = ()
    __match_args__ = ("cases", "otherwise")


PiecewiseExpressionPattern._register_public_class()


@final
class CallExpressionPattern(_rs.CallExpressionPattern, Pattern):
    """Match a `CallExpression`.

    Attributes:
        function_name: ``None`` for any callee; otherwise the callee's
            name, parsed as `CallExpression` parses it: a built-in's name
            is the built-in, and any other non-empty name a user function.
            Calls are compared by callee. An empty name raises
            ``ValueError``.
        arguments: ``None`` for any arguments; otherwise the tuple of
            patterns the arguments must match position-wise, with exactly
            as many arguments, so ``()`` matches only calls without
            arguments.

    """

    __slots__ = ()
    __match_args__ = ("function_name", "arguments")


CallExpressionPattern._register_public_class()


@final
class PredicatePattern(_rs.PredicatePattern, Pattern):
    """Match when a Python predicate over the expression returns a true value.

    Captures nothing. The predicate receives the node object of the
    candidate expression each time the pattern is tried, and its result is
    read by truthiness. Its exceptions propagate unchanged. A value that is
    not callable raises ``TypeError``.

    Attributes:
        predicate: The callable.

    """

    __slots__ = ()
    __match_args__ = ("predicate",)


PredicatePattern._register_public_class()


@final
class AlternativesPattern(_rs.AlternativesPattern, Pattern):
    """Match when any of several sub-patterns matches.

    The alternatives are tried left to right, and the first that matches
    decides. A failed alternative leaves no captures behind, and the choice
    is final: when an enclosing pattern later fails, the remaining
    alternatives are not tried. No alternative matches nothing.

    Attributes:
        alternatives: The tuple of alternatives, in order.

    """

    __slots__ = ()
    __match_args__ = ("alternatives",)


AlternativesPattern._register_public_class()


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
