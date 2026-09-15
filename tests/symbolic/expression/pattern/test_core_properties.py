"""Hypothesis property tests for expression-tree pattern matching.

Covers a pattern built to mirror an expression tree's exact shape
(``build_mirroring_pattern``, defined below): it matches the tree it
mirrors and binds every captured leaf to a structurally equivalent
subtree; ``WildcardPattern`` matches everything; ``does_pattern_match``
agrees with ``match_pattern(...) is not None``; and a mirror pattern
whose root ``BinaryOperation`` has been swapped for a different one no
longer matches.
"""

import pytest

pytest.importorskip("hypothesis")

import dataclasses
import itertools
from collections.abc import Sequence

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    PiecewiseExpression,
    UnaryExpression,
)
from fhy_core.symbolic.expression.pattern import (
    BinaryExpressionPattern,
    CallExpressionPattern,
    CapturePattern,
    LiteralPattern,
    Pattern,
    PiecewiseExpressionPattern,
    UnaryExpressionPattern,
    WildcardPattern,
    does_pattern_match,
    match_pattern,
)

from ....strategies.identifiers import build_identifier_pool
from ....strategies.structural_expressions import build_structural_expression_strategy

pytestmark = pytest.mark.property

_POOL = build_identifier_pool(3)
_STRUCTURAL_MAX_LEAVES = 6

# A decimal-string value longer than `build_decimal_string_value_strategy`'s
# 4-digit-per-part limit, so it can never equal a literal
# `build_structural_expression_strategy` generates; `LiteralPattern` compares
# by exact stored value and type, not by equivalence class, so this is
# genuinely "a literal not in e's alphabet" rather than merely an unlikely
# draw.
_LITERAL_NOT_IN_ALPHABET = LiteralPattern(value="99999.99999")


def build_mirroring_pattern(
    expression: Expression,
) -> tuple[Pattern, dict[str, Expression]]:
    """Build a pattern that mirrors ``expression``'s exact shape.

    Every leaf (a ``LiteralExpression`` or an ``IdentifierExpression``,
    identified here as any node with no visit children) becomes a
    uniquely-named ``CapturePattern(WildcardPattern())``; every
    ``UnaryExpression``, ``BinaryExpression``, ``CallExpression``, and
    ``PiecewiseExpression`` node becomes the matching ``*Pattern`` over
    mirrored children.

    Args:
        expression: Expression tree to mirror.

    Returns:
        The mirroring pattern, and a mapping from each capture name to
        the leaf subtree it was recorded from.

    """
    counter = itertools.count()
    captures: dict[str, Expression] = {}

    def build(node: Expression) -> Pattern:
        if isinstance(node, UnaryExpression):
            return UnaryExpressionPattern(node.operation, build(node.operand))
        if isinstance(node, BinaryExpression):
            return BinaryExpressionPattern(
                node.operation, build(node.left), build(node.right)
            )
        if isinstance(node, CallExpression):
            return CallExpressionPattern(
                node.function_name,
                tuple(build(argument) for argument in node.arguments),
            )
        if isinstance(node, PiecewiseExpression):
            cases = tuple(
                (build(condition), build(value))
                for condition, value in node.get_cases()
            )
            return PiecewiseExpressionPattern(cases, build(node.otherwise))
        name = f"leaf_{next(counter)}"
        captures[name] = node
        return CapturePattern(name, WildcardPattern())

    pattern = build(expression)
    return pattern, captures


# =============================================================================
# The mirror pattern matches and binds every recorded leaf
# =============================================================================


@given(build_structural_expression_strategy(_POOL, _STRUCTURAL_MAX_LEAVES))
def test_mirroring_pattern_matches_and_binds_recorded_leaves(
    expression: Expression,
) -> None:
    """Test a pattern mirroring e's shape matches e and binds every leaf.

    Oracle: ``build_mirroring_pattern`` records the exact leaf subtree
    each capture should bind; the binding must be structurally
    equivalent to it.
    """
    mirror, captures = build_mirroring_pattern(expression)

    bindings = match_pattern(mirror, expression)

    assert bindings is not None
    assert bindings.names() == frozenset(captures.keys())
    for name, recorded_subtree in captures.items():
        assert bindings.get(name).is_structurally_equivalent(recorded_subtree)


# =============================================================================
# WildcardPattern matches every expression
# =============================================================================


@given(build_structural_expression_strategy(_POOL, _STRUCTURAL_MAX_LEAVES))
def test_wildcard_pattern_matches_every_expression(expression: Expression) -> None:
    """Test WildcardPattern() matches every expression."""
    assert match_pattern(WildcardPattern(), expression) is not None


# =============================================================================
# does_pattern_match agrees with match_pattern(...) is not None
# =============================================================================


@given(build_structural_expression_strategy(_POOL, _STRUCTURAL_MAX_LEAVES))
def test_does_pattern_match_agrees_with_match_pattern(expression: Expression) -> None:
    """Test does_pattern_match(p, e) == (match_pattern(p, e) is not None).

    Checked for the mirror pattern, WildcardPattern, and a LiteralPattern
    holding a value outside e's literal alphabet.
    """
    mirror, _captures = build_mirroring_pattern(expression)

    for pattern in (mirror, WildcardPattern(), _LITERAL_NOT_IN_ALPHABET):
        assert does_pattern_match(pattern, expression) == (
            match_pattern(pattern, expression) is not None
        )


# =============================================================================
# A mirror pattern with a different root BinaryOperation does not match
# =============================================================================


@st.composite
def draw_binary_root_with_alternate_operation(
    draw: st.DrawFn,
    identifiers: Sequence[Identifier] = _POOL,
    max_leaves: int = 5,
) -> tuple[BinaryExpression, BinaryOperation]:
    """Draw a BinaryExpression root and a BinaryOperation distinct from its own."""
    operation = draw(st.sampled_from(list(BinaryOperation)))
    left = draw(build_structural_expression_strategy(identifiers, max_leaves))
    right = draw(build_structural_expression_strategy(identifiers, max_leaves))
    alternate_operation = draw(
        st.sampled_from(
            [candidate for candidate in BinaryOperation if candidate != operation]
        )
    )
    return BinaryExpression(operation, left, right), alternate_operation


@given(draw_binary_root_with_alternate_operation())
def test_mirror_pattern_with_different_root_operation_does_not_match(
    pair: tuple[BinaryExpression, BinaryOperation],
) -> None:
    """Test swapping the mirror pattern's root BinaryOperation breaks the match."""
    root, alternate_operation = pair
    mirror, _captures = build_mirroring_pattern(root)
    assert isinstance(mirror, BinaryExpressionPattern)
    altered = dataclasses.replace(mirror, operation=alternate_operation)

    assert match_pattern(altered, root) is None
