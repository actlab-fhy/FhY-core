"""Hypothesis property tests for rule-driven rewriting (P10).

Covers ``apply_rewrite_rules`` with an empty rule set (identity, by
object identity); a semantics-preserving rule set (``x + 0 -> x``,
``x * 1 -> x``, ``-(-x) -> x``) applied to a tree wrapped with those
same no-op forms, checked against the un-wrapped tree's evaluation; and
the documented "identity iff zero rules fired" contract, checked by
wrapping each rule's rewrite callable with a fire counter.
"""

import pytest

pytest.importorskip("hypothesis")

import itertools
from collections.abc import Iterator, Mapping, Sequence
from typing import Final

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
    evaluate_expression_with_numpy,
)
from fhy_core.symbolic.expression.pattern import (
    BinaryExpressionPattern,
    CapturePattern,
    LiteralPattern,
    MatchBindings,
    RewriteRule,
    UnaryExpressionPattern,
    WildcardPattern,
    apply_rewrite_rules,
)

from ....strategies.expressions import (
    build_integer_environment_strategy,
    build_numeric_expression_strategy,
)
from ....strategies.identifiers import build_identifier_pool

pytestmark = pytest.mark.property

np = pytest.importorskip("numpy")

_POOL = build_identifier_pool(3)

_WRAP_KINDS: Final[tuple[str, ...]] = ("add_zero", "multiply_one", "double_negate")
_MAX_WRAPS: Final = 3


# =============================================================================
# Wrapping a numeric gate tree in no-op forms, by construction
# =============================================================================


def _collect_numeric_nodes(expression: Expression) -> list[Expression]:
    """Return every node of a piecewise-free numeric gate tree, pre-order.

    Every node such a tree can hold (a literal or identifier leaf, a
    unary/binary arithmetic node, or a native call argument) is
    numeric-sorted, so every one of them is safe to wrap in `+ 0`,
    `* 1`, or double negation without changing its value or sort.
    """
    nodes = [expression]
    if isinstance(expression, UnaryExpression):
        nodes.extend(_collect_numeric_nodes(expression.operand))
    elif isinstance(expression, BinaryExpression):
        nodes.extend(_collect_numeric_nodes(expression.left))
        nodes.extend(_collect_numeric_nodes(expression.right))
    elif isinstance(expression, CallExpression):
        for argument in expression.arguments:
            nodes.extend(_collect_numeric_nodes(argument))
    return nodes


def _wrap_expression(expression: Expression, wrap_kind: str) -> Expression:
    """Wrap ``expression`` in a value-preserving no-op form."""
    if wrap_kind == "add_zero":
        return BinaryExpression(BinaryOperation.ADD, expression, LiteralExpression(0))
    if wrap_kind == "multiply_one":
        return BinaryExpression(
            BinaryOperation.MULTIPLY, expression, LiteralExpression(1)
        )
    return UnaryExpression(
        UnaryOperation.NEGATE, UnaryExpression(UnaryOperation.NEGATE, expression)
    )


def _rebuild_with_wraps(
    expression: Expression, wraps: Mapping[int, str], counter: Iterator[int]
) -> Expression:
    """Rebuild ``expression``, wrapping the nodes at the chosen pre-order indices.

    ``counter`` yields the same pre-order index sequence
    :func:`_collect_numeric_nodes` used to choose ``wraps``, so a
    wrapped index always lands on the node it was chosen for.
    """
    index = next(counter)
    rebuilt: Expression
    if isinstance(expression, UnaryExpression):
        rebuilt = UnaryExpression(
            expression.operation,
            _rebuild_with_wraps(expression.operand, wraps, counter),
        )
    elif isinstance(expression, BinaryExpression):
        rebuilt = BinaryExpression(
            expression.operation,
            _rebuild_with_wraps(expression.left, wraps, counter),
            _rebuild_with_wraps(expression.right, wraps, counter),
        )
    elif isinstance(expression, CallExpression):
        rebuilt = CallExpression(
            expression.function_name,
            tuple(
                _rebuild_with_wraps(argument, wraps, counter)
                for argument in expression.arguments
            ),
        )
    else:
        rebuilt = expression
    wrap_kind = wraps.get(index)
    if wrap_kind is None:
        return rebuilt
    return _wrap_expression(rebuilt, wrap_kind)


@st.composite
def draw_wrapped_numeric_tree(
    draw: st.DrawFn, identifiers: Sequence[Identifier], max_leaves: int = 8
) -> tuple[Expression, Expression, dict[Identifier, int]]:
    """Draw e, a wrapped variant e', and an environment binding every identifier.

    e' wraps 0 to 3 randomly chosen nodes of e (including, sometimes,
    the root) in `x + 0`, `x * 1`, or `-(-x)`; every wrap preserves e's
    value under every environment. Piecewise is excluded from e so
    every node is numeric-sorted and wrappable without a sort check.
    """
    expression = draw(
        build_numeric_expression_strategy(
            identifiers, max_leaves, include_piecewise=False
        )
    )
    numeric_nodes = _collect_numeric_nodes(expression)
    wrap_count = draw(
        st.integers(min_value=0, max_value=min(_MAX_WRAPS, len(numeric_nodes)))
    )
    chosen_indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=len(numeric_nodes) - 1),
            unique=True,
            min_size=wrap_count,
            max_size=wrap_count,
        )
    )
    chosen_kinds = draw(
        st.lists(st.sampled_from(_WRAP_KINDS), min_size=wrap_count, max_size=wrap_count)
    )
    wraps = dict(zip(chosen_indices, chosen_kinds, strict=True))
    wrapped = _rebuild_with_wraps(expression, wraps, itertools.count())
    environment = draw(build_integer_environment_strategy(identifiers))
    return expression, wrapped, environment


# =============================================================================
# The rule set under test: x + 0 -> x, x * 1 -> x, -(-x) -> x
# =============================================================================


def _build_rewrite_rules() -> tuple[RewriteRule, ...]:
    """Return the semantics-preserving rule set, without fire counting."""
    return (
        RewriteRule(
            pattern=BinaryExpressionPattern(
                BinaryOperation.ADD,
                CapturePattern("x", WildcardPattern()),
                LiteralPattern(value=0),
            ),
            rewrite=lambda bindings: bindings.get("x"),
            name="x + 0 -> x",
        ),
        RewriteRule(
            pattern=BinaryExpressionPattern(
                BinaryOperation.MULTIPLY,
                CapturePattern("x", WildcardPattern()),
                LiteralPattern(value=1),
            ),
            rewrite=lambda bindings: bindings.get("x"),
            name="x * 1 -> x",
        ),
        RewriteRule(
            pattern=UnaryExpressionPattern(
                UnaryOperation.NEGATE,
                UnaryExpressionPattern(
                    UnaryOperation.NEGATE, CapturePattern("x", WildcardPattern())
                ),
            ),
            rewrite=lambda bindings: bindings.get("x"),
            name="-(-x) -> x",
        ),
    )


def _build_counting_rewrite_rules() -> tuple[list[int], tuple[RewriteRule, ...]]:
    """Return a shared fire counter and the same rule set instrumented with it.

    Each rule's ``rewrite`` callable increments the counter before
    returning its capture, so a caller can tell whether any rule fired
    across a whole ``apply_rewrite_rules`` walk.
    """
    fire_count = [0]

    def _count_and_return_x(bindings: MatchBindings) -> Expression:
        fire_count[0] += 1
        return bindings.get("x")

    rules = (
        RewriteRule(
            pattern=BinaryExpressionPattern(
                BinaryOperation.ADD,
                CapturePattern("x", WildcardPattern()),
                LiteralPattern(value=0),
            ),
            rewrite=_count_and_return_x,
            name="x + 0 -> x",
        ),
        RewriteRule(
            pattern=BinaryExpressionPattern(
                BinaryOperation.MULTIPLY,
                CapturePattern("x", WildcardPattern()),
                LiteralPattern(value=1),
            ),
            rewrite=_count_and_return_x,
            name="x * 1 -> x",
        ),
        RewriteRule(
            pattern=UnaryExpressionPattern(
                UnaryOperation.NEGATE,
                UnaryExpressionPattern(
                    UnaryOperation.NEGATE, CapturePattern("x", WildcardPattern())
                ),
            ),
            rewrite=_count_and_return_x,
            name="-(-x) -> x",
        ),
    )
    return fire_count, rules


# =============================================================================
# P10a: an empty rule set is the identity, by object identity
# =============================================================================


@given(build_numeric_expression_strategy(_POOL))
def test_apply_rewrite_rules_with_no_rules_is_identity(expression: Expression) -> None:
    """Test apply_rewrite_rules(e, []) returns e itself, unchanged."""
    assert apply_rewrite_rules(expression, []) is expression


# =============================================================================
# P10b: the rule set preserves evaluation
# =============================================================================


@given(draw_wrapped_numeric_tree(_POOL))
def test_rewrite_rules_preserve_evaluation(
    triple: tuple[Expression, Expression, dict[Identifier, int]],
) -> None:
    """Test the no-op rule set does not change what the wrapped tree evaluates to.

    Oracle: evaluate_expression_with_numpy on the original, un-wrapped
    tree ``e``.
    """
    expression, wrapped, environment = triple
    rules = _build_rewrite_rules()

    rewritten = apply_rewrite_rules(wrapped, rules)

    assert int(evaluate_expression_with_numpy(rewritten, environment)) == int(
        evaluate_expression_with_numpy(expression, environment)
    )


# =============================================================================
# P10c: identity holds iff zero rules fired (the documented contract)
# =============================================================================


@given(draw_wrapped_numeric_tree(_POOL))
def test_rewrite_rules_identity_holds_iff_no_rule_fired(
    triple: tuple[Expression, Expression, dict[Identifier, int]],
) -> None:
    """Test apply_rewrite_rules(e', rules) is e' iff the fire count is zero.

    Oracle: apply_rewrite_rules's documented identity contract.
    """
    _expression, wrapped, _environment = triple
    fire_count, rules = _build_counting_rewrite_rules()

    result = apply_rewrite_rules(wrapped, rules)

    assert (result is wrapped) == (fire_count[0] == 0)
