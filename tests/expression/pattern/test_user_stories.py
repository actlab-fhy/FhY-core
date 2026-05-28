"""End-to-end scenarios for the expression pattern-matching framework.

Each test composes the public API in the same way a real caller would:
build a rule set, run ``apply_rewrite_rules`` (or use ``match_pattern``
directly), and assert on the resulting expression tree.
"""

from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy_core.expression.pattern import (
    BinaryExpressionPattern,
    CapturePattern,
    LiteralPattern,
    MatchBindings,
    Pattern,
    RewriteRule,
    UnaryExpressionPattern,
    WildcardPattern,
    apply_rewrite_rules,
    match_pattern,
)

from ..conftest import mock_identifier

# ===========================================================================
# Story 1: Local algebraic simplifier
# ===========================================================================


def _make_algebraic_rule_set() -> list[RewriteRule]:
    """Build the four canonical algebraic simplifications."""
    capture_x = CapturePattern("x", WildcardPattern())
    rule_left_plus_zero = RewriteRule(
        pattern=BinaryExpressionPattern(
            BinaryOperation.ADD, capture_x, LiteralPattern(value=0)
        ),
        rewrite=lambda bindings: bindings.get("x"),
        name="x + 0 -> x",
    )
    rule_right_plus_zero = RewriteRule(
        pattern=BinaryExpressionPattern(
            BinaryOperation.ADD, LiteralPattern(value=0), capture_x
        ),
        rewrite=lambda bindings: bindings.get("x"),
        name="0 + x -> x",
    )
    rule_times_one = RewriteRule(
        pattern=BinaryExpressionPattern(
            BinaryOperation.MULTIPLY, capture_x, LiteralPattern(value=1)
        ),
        rewrite=lambda bindings: bindings.get("x"),
        name="x * 1 -> x",
    )
    rule_self_subtract = RewriteRule(
        pattern=BinaryExpressionPattern(
            BinaryOperation.SUBTRACT,
            CapturePattern("x", WildcardPattern()),
            CapturePattern("x", WildcardPattern()),
        ),
        rewrite=lambda _: LiteralExpression(0),
        name="x - x -> 0",
    )
    return [
        rule_left_plus_zero,
        rule_right_plus_zero,
        rule_times_one,
        rule_self_subtract,
    ]


def test_algebraic_simplifier_collapses_nested_neutral_operations() -> None:
    """Test the four-rule simplifier reduces ``((a + 0) * 1) - ((a + 0) * 1)``."""
    a = mock_identifier("a", 0)
    subtree = BinaryExpression(
        BinaryOperation.MULTIPLY,
        BinaryExpression(
            BinaryOperation.ADD, IdentifierExpression(a), LiteralExpression(0)
        ),
        LiteralExpression(1),
    )
    expression = BinaryExpression(BinaryOperation.SUBTRACT, subtree, subtree)

    result = apply_rewrite_rules(expression, _make_algebraic_rule_set())

    assert result.is_structurally_equivalent(LiteralExpression(0))


def test_algebraic_simplifier_leaves_non_matching_input_unchanged() -> None:
    """Test the simplifier preserves identity on inputs no rule matches."""
    a = mock_identifier("a", 0)
    expression = BinaryExpression(
        BinaryOperation.MULTIPLY,
        IdentifierExpression(a),
        LiteralExpression(2),
    )

    result = apply_rewrite_rules(expression, _make_algebraic_rule_set())

    assert result is expression


def test_algebraic_simplifier_handles_left_zero_addend() -> None:
    """Test the ``0 + x`` rule fires alongside the ``x + 0`` rule."""
    a = mock_identifier("a", 0)
    expression = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(0), IdentifierExpression(a)
    )

    result = apply_rewrite_rules(expression, _make_algebraic_rule_set())

    assert result.is_structurally_equivalent(IdentifierExpression(a))


# ===========================================================================
# Story 2: Subtraction canonicalizer (a - b -> a + (-b))
# ===========================================================================


def test_subtraction_canonicalizer_rewrites_at_every_depth() -> None:
    """Test ``a - b -> a + (-b)`` rewrites every subtraction in a nested tree."""
    a = mock_identifier("a", 0)
    b = mock_identifier("b", 1)
    c = mock_identifier("c", 2)

    rule = RewriteRule(
        pattern=BinaryExpressionPattern(
            BinaryOperation.SUBTRACT,
            CapturePattern("left", WildcardPattern()),
            CapturePattern("right", WildcardPattern()),
        ),
        rewrite=lambda bindings: BinaryExpression(
            BinaryOperation.ADD,
            bindings.get("left"),
            UnaryExpression(UnaryOperation.NEGATE, bindings.get("right")),
        ),
        name="a - b -> a + (-b)",
    )

    expression = BinaryExpression(
        BinaryOperation.SUBTRACT,
        IdentifierExpression(a),
        BinaryExpression(
            BinaryOperation.SUBTRACT,
            IdentifierExpression(b),
            IdentifierExpression(c),
        ),
    )

    result = apply_rewrite_rules(expression, [rule])

    expected = BinaryExpression(
        BinaryOperation.ADD,
        IdentifierExpression(a),
        UnaryExpression(
            UnaryOperation.NEGATE,
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(b),
                UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(c)),
            ),
        ),
    )
    assert result.is_structurally_equivalent(expected)


# ===========================================================================
# Story 3: Double-negation peephole
# ===========================================================================


def _make_double_negation_rule() -> RewriteRule:
    """Build the peephole rule ``!(!x) -> x``."""
    return RewriteRule(
        pattern=UnaryExpressionPattern(
            UnaryOperation.LOGICAL_NOT,
            UnaryExpressionPattern(
                UnaryOperation.LOGICAL_NOT,
                CapturePattern("x", WildcardPattern()),
            ),
        ),
        rewrite=lambda bindings: bindings.get("x"),
        name="!(!x) -> x",
    )


def test_double_negation_peephole_collapses_inner_double_negation() -> None:
    """Test the peephole simplifies ``!(!a)`` to ``a``."""
    a = mock_identifier("a", 0)
    expression = UnaryExpression(
        UnaryOperation.LOGICAL_NOT,
        UnaryExpression(UnaryOperation.LOGICAL_NOT, IdentifierExpression(a)),
    )

    result = apply_rewrite_rules(expression, [_make_double_negation_rule()])

    assert result.is_structurally_equivalent(IdentifierExpression(a))


def test_double_negation_peephole_requires_two_passes_for_four_negations() -> None:
    """Test four-deep negation requires caller-driven fixpoint iteration."""
    a = mock_identifier("a", 0)
    expression: Expression = IdentifierExpression(a)
    for _ in range(4):
        expression = UnaryExpression(UnaryOperation.LOGICAL_NOT, expression)

    rules = [_make_double_negation_rule()]
    once = apply_rewrite_rules(expression, rules)
    twice = apply_rewrite_rules(once, rules)

    assert twice.is_structurally_equivalent(IdentifierExpression(a))


# ===========================================================================
# Story 4: Sub-tree finder via match_pattern (no rewrite)
# ===========================================================================


def _collect_matching_subexpressions(
    pattern: Pattern, expression: Expression
) -> list[tuple[Expression, MatchBindings]]:
    """Collect every sub-expression matching ``pattern``."""
    collected: list[tuple[Expression, MatchBindings]] = []
    bindings = match_pattern(pattern, expression)
    if bindings is not None:
        collected.append((expression, bindings))
    for child in expression.get_visit_children():
        collected.extend(_collect_matching_subexpressions(pattern, child))
    return collected


def test_match_pattern_powers_a_manual_subtree_finder() -> None:
    """Test ``match_pattern`` is usable for a read-only subtree-collecting walk."""
    a = mock_identifier("a", 0)
    expression = BinaryExpression(
        BinaryOperation.MULTIPLY,
        BinaryExpression(
            BinaryOperation.ADD, IdentifierExpression(a), LiteralExpression(0)
        ),
        BinaryExpression(
            BinaryOperation.ADD, IdentifierExpression(a), LiteralExpression(0)
        ),
    )
    pattern = BinaryExpressionPattern(
        BinaryOperation.ADD,
        CapturePattern("x", WildcardPattern()),
        LiteralPattern(value=0),
    )

    matches = _collect_matching_subexpressions(pattern, expression)

    assert len(matches) == 2
    for matched_expression, bindings in matches:
        assert matched_expression.is_structurally_equivalent(
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(a),
                LiteralExpression(0),
            )
        )
        assert bindings.get("x").is_structurally_equivalent(IdentifierExpression(a))
