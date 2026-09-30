"""Tests for ``fhy_core.symbolic.expression.pattern.rewrite``."""

from collections.abc import Sequence

import pytest

from fhy_core.diagnostic import DiagnosticLevel
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import CompilerPass, PassExecutionError
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy_core.symbolic.expression.pattern import (
    AlternativesPattern,
    BinaryExpressionPattern,
    CallExpressionPattern,
    Capture,
    CapturePattern,
    FiredRule,
    IdentifierPattern,
    LiteralPattern,
    LogicalExpressionPattern,
    MatchBindings,
    Pattern,
    PiecewiseExpressionPattern,
    PredicatePattern,
    RewriteCallbackError,
    RewriteRule,
    RewriteRuleApplier,
    UnaryExpressionPattern,
    WildcardPattern,
    apply_rewrite_rule,
    apply_rewrite_rules,
)
from fhy_core.traits import FrozenMutationError

from ..conftest import mock_identifier


def _make_x_plus_zero(identifier: Identifier) -> BinaryExpression:
    """Build the expression ``x + 0`` for a given identifier ``x``."""
    return BinaryExpression(
        BinaryOperation.ADD,
        IdentifierExpression(identifier),
        LiteralExpression(0),
    )


def _make_x_minus_x(identifier: Identifier) -> BinaryExpression:
    """Build the expression ``x - x`` for a given identifier ``x``."""
    return BinaryExpression(
        BinaryOperation.SUBTRACT,
        IdentifierExpression(identifier),
        IdentifierExpression(identifier),
    )


def _make_x_plus_zero_rule() -> RewriteRule:
    """Build the rule ``x + 0 -> x``."""
    x = Capture("x")
    return RewriteRule(
        pattern=BinaryExpressionPattern(
            BinaryOperation.ADD,
            CapturePattern(x),
            LiteralPattern(value=0),
        ),
        rewrite=lambda bindings: bindings[x],
        name="x + 0 -> x",
    )


def _make_x_minus_x_rule() -> RewriteRule:
    """Build the rule ``x - x -> 0``."""
    x = Capture("x")
    return RewriteRule(
        pattern=BinaryExpressionPattern(
            BinaryOperation.SUBTRACT,
            CapturePattern(x),
            CapturePattern(x),
        ),
        rewrite=lambda _: LiteralExpression(0),
        name="x - x -> 0",
    )


def _make_x_times_one_rule() -> RewriteRule:
    """Build the rule ``x * 1 -> x``."""
    x = Capture("x")
    return RewriteRule(
        pattern=BinaryExpressionPattern(
            BinaryOperation.MULTIPLY,
            CapturePattern(x),
            LiteralPattern(value=1),
        ),
        rewrite=lambda bindings: bindings[x],
        name="x * 1 -> x",
    )


# ===========================================================================
# RewriteRule construction
# ===========================================================================


def test_rewrite_rule_defaults_for_optional_fields() -> None:
    """Test ``RewriteRule`` defaults to no guard and no name.

    The guards are a tuple, empty when none are given.
    """
    rule = RewriteRule(
        pattern=WildcardPattern(),
        rewrite=lambda _: LiteralExpression(0),
    )

    assert rule.guards == ()
    assert rule.name is None


def test_rewrite_rule_compares_and_hashes_by_identity() -> None:
    """Test ``RewriteRule`` instances compare and hash by identity.

    Callables cannot be compared, so two rules of the same content are
    distinct.
    """
    pattern = LiteralPattern(value=5)
    rewrite = lambda _: LiteralExpression(0)  # noqa: E731

    left = RewriteRule(pattern=pattern, rewrite=rewrite, name="r")
    right = RewriteRule(pattern=pattern, rewrite=rewrite, name="r")

    assert left == left  # noqa: PLR0124
    assert left != right
    assert hash(left) == hash(left)
    assert len({left, right}) == 2


def test_rewrite_rule_is_frozen() -> None:
    """Test ``RewriteRule`` rejects post-construction mutation."""
    rule = RewriteRule(
        pattern=WildcardPattern(),
        rewrite=lambda _: LiteralExpression(0),
    )

    with pytest.raises(FrozenMutationError):
        rule.name = "renamed"  # type: ignore[misc]


# ===========================================================================
# apply_rewrite_rule
# ===========================================================================


def test_apply_rewrite_rule_returns_rewritten_expression_on_match() -> None:
    """Test ``apply_rewrite_rule`` returns the rule's output on a matching pattern."""
    x = mock_identifier("x", 0)
    rule = _make_x_plus_zero_rule()
    expression = _make_x_plus_zero(x)

    result = apply_rewrite_rule(rule, expression)

    assert result is not None
    assert result.is_structurally_equivalent(IdentifierExpression(x))


def test_apply_rewrite_rule_returns_none_when_pattern_does_not_match() -> None:
    """Test ``apply_rewrite_rule`` returns ``None`` when the pattern does not match."""
    x = mock_identifier("x", 0)
    rule = _make_x_plus_zero_rule()
    expression = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(x), LiteralExpression(1)
    )

    assert apply_rewrite_rule(rule, expression) is None


def test_apply_rewrite_rule_returns_none_when_guard_returns_false() -> None:
    """Test ``apply_rewrite_rule`` returns ``None`` when the guard rejects bindings."""
    captured = Capture("x")
    x = mock_identifier("x", 0)
    rule = RewriteRule(
        pattern=BinaryExpressionPattern(
            BinaryOperation.ADD,
            CapturePattern(captured),
            LiteralPattern(),
        ),
        guard=lambda _: False,
        rewrite=lambda bindings: bindings[captured],
    )

    assert apply_rewrite_rule(rule, _make_x_plus_zero(x)) is None


def test_apply_rewrite_rule_fires_when_guard_returns_true() -> None:
    """Test ``apply_rewrite_rule`` fires when the guard accepts the bindings."""
    captured = Capture("x")
    x = mock_identifier("x", 0)
    rule = RewriteRule(
        pattern=BinaryExpressionPattern(
            BinaryOperation.ADD,
            CapturePattern(captured),
            LiteralPattern(),
        ),
        guard=lambda _: True,
        rewrite=lambda bindings: bindings[captured],
    )

    result = apply_rewrite_rule(rule, _make_x_plus_zero(x))

    assert result is not None
    assert result.is_structurally_equivalent(IdentifierExpression(x))


def test_apply_rewrite_rule_propagates_guard_exception() -> None:
    """Test ``apply_rewrite_rule`` re-raises a guard exception."""

    def guard(_: MatchBindings) -> bool:
        raise RuntimeError("guard failed")

    rule = RewriteRule(
        pattern=WildcardPattern(),
        guard=guard,
        rewrite=lambda _: LiteralExpression(0),
    )

    with pytest.raises(RuntimeError, match="guard failed"):
        apply_rewrite_rule(rule, LiteralExpression(5))


def test_apply_rewrite_rule_propagates_rewrite_exception() -> None:
    """Test ``apply_rewrite_rule`` re-raises a rewrite exception."""

    def rewrite(_: MatchBindings) -> Expression:
        raise RuntimeError("rewrite failed")

    rule = RewriteRule(pattern=WildcardPattern(), rewrite=rewrite)

    with pytest.raises(RuntimeError, match="rewrite failed"):
        apply_rewrite_rule(rule, LiteralExpression(5))


def test_apply_rewrite_rule_raises_type_error_when_rewrite_returns_non_expression() -> (
    None
):
    """Test ``apply_rewrite_rule`` rejects a rewrite that returns a non-`Expression`."""
    rule = RewriteRule(
        pattern=WildcardPattern(),
        rewrite=lambda _: None,  # type: ignore[arg-type,return-value]
        name="bug",
    )

    with pytest.raises(TypeError, match="bug"):
        apply_rewrite_rule(rule, LiteralExpression(5))


def test_apply_rewrite_rule_type_error_names_unnamed_rule_with_placeholder() -> None:
    """Test the rewrite-return-type check uses ``<unnamed>`` for nameless rules."""
    rule = RewriteRule(
        pattern=WildcardPattern(),
        rewrite=lambda _: 42,  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(TypeError, match="<unnamed>"):
        apply_rewrite_rule(rule, LiteralExpression(5))


def test_apply_rewrite_rule_operates_at_root_only() -> None:
    """Test ``apply_rewrite_rule`` does not recurse into children."""
    x = mock_identifier("x", 0)
    rule = _make_x_plus_zero_rule()
    inner = _make_x_plus_zero(x)
    outer = BinaryExpression(BinaryOperation.MULTIPLY, inner, LiteralExpression(2))

    assert apply_rewrite_rule(rule, outer) is None


# ===========================================================================
# apply_rewrite_rules: empty / identity
# ===========================================================================


def test_apply_rewrite_rules_with_empty_rules_returns_input_unchanged() -> None:
    """Test ``apply_rewrite_rules(expression, [])`` returns the input unchanged."""
    expression = LiteralExpression(5)

    assert apply_rewrite_rules(expression, []) is expression


def test_apply_rewrite_rules_preserves_identity_when_no_rule_matches() -> None:
    """Test ``apply_rewrite_rules`` preserves identity when no rule matches."""
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(x), LiteralExpression(1)
    )

    result = apply_rewrite_rules(expression, [_make_x_plus_zero_rule()])

    assert result is expression


# ===========================================================================
# apply_rewrite_rules: root and leaf
# ===========================================================================


def test_apply_rewrite_rules_rewrites_at_root() -> None:
    """Test ``apply_rewrite_rules`` rewrites a matching root node."""
    x = mock_identifier("x", 0)
    expression = _make_x_plus_zero(x)

    result = apply_rewrite_rules(expression, [_make_x_plus_zero_rule()])

    assert result.is_structurally_equivalent(IdentifierExpression(x))


def test_apply_rewrite_rules_rewrites_in_subtree_and_preserves_spine() -> None:
    """Test ``apply_rewrite_rules`` rewrites a subtree and rebuilds the spine."""
    x = mock_identifier("x", 0)
    subtree = _make_x_plus_zero(x)
    expression = BinaryExpression(
        BinaryOperation.MULTIPLY, subtree, LiteralExpression(2)
    )

    result = apply_rewrite_rules(expression, [_make_x_plus_zero_rule()])

    expected = BinaryExpression(
        BinaryOperation.MULTIPLY, IdentifierExpression(x), LiteralExpression(2)
    )
    assert result.is_structurally_equivalent(expected)


# ===========================================================================
# apply_rewrite_rules: first-match priority semantics
# ===========================================================================


def test_apply_rewrite_rules_uses_first_matching_rule() -> None:
    """Test the first rule that matches fires; subsequent rules are not tried."""
    rule_marker_a = RewriteRule(
        pattern=WildcardPattern(),
        rewrite=lambda _: LiteralExpression(101),
    )
    rule_marker_b = RewriteRule(
        pattern=WildcardPattern(),
        rewrite=lambda _: LiteralExpression(202),
    )

    result = apply_rewrite_rules(LiteralExpression(0), [rule_marker_a, rule_marker_b])

    assert result.is_structurally_equivalent(LiteralExpression(101))


def test_apply_rewrite_rules_skips_guarded_failure_and_tries_next_rule() -> None:
    """Test a guard returning ``False`` causes the next rule to be tried."""
    rule_marker_a = RewriteRule(
        pattern=WildcardPattern(),
        guard=lambda _: False,
        rewrite=lambda _: LiteralExpression(101),
    )
    rule_marker_b = RewriteRule(
        pattern=WildcardPattern(),
        rewrite=lambda _: LiteralExpression(202),
    )

    result = apply_rewrite_rules(LiteralExpression(0), [rule_marker_a, rule_marker_b])

    assert result.is_structurally_equivalent(LiteralExpression(202))


# ===========================================================================
# apply_rewrite_rules: bottom-up traversal
# ===========================================================================


def test_apply_rewrite_rules_walks_bottom_up_single_pass() -> None:
    """Test bottom-up firing collapses nested simplifications in one walk."""
    x = mock_identifier("x", 0)
    inner = _make_x_plus_zero(x)
    expression = BinaryExpression(BinaryOperation.MULTIPLY, inner, LiteralExpression(1))

    result = apply_rewrite_rules(
        expression, [_make_x_plus_zero_rule(), _make_x_times_one_rule()]
    )

    assert result.is_structurally_equivalent(IdentifierExpression(x))


def test_apply_rewrite_rules_does_not_iterate_to_fixpoint() -> None:
    """Test a rule that would re-match its own output does not loop."""
    rule = RewriteRule(
        pattern=LiteralPattern(value=0),
        rewrite=lambda _: BinaryExpression(
            BinaryOperation.ADD, LiteralExpression(0), LiteralExpression(0)
        ),
    )

    result = apply_rewrite_rules(LiteralExpression(0), [rule])

    expected = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(0), LiteralExpression(0)
    )
    assert result.is_structurally_equivalent(expected)


def test_apply_rewrite_rules_threads_rewritten_children_into_parent_match() -> None:
    """Test parents see rewritten children before their own rule is tried."""
    x = mock_identifier("x", 0)
    inner = _make_x_plus_zero(x)
    expression = BinaryExpression(BinaryOperation.ADD, inner, LiteralExpression(0))

    result = apply_rewrite_rules(expression, [_make_x_plus_zero_rule()])

    assert result.is_structurally_equivalent(IdentifierExpression(x))


def test_apply_rewrite_rules_rewrites_inside_piecewise_branch() -> None:
    """Test a rewrite inside one piecewise branch leaves the sibling branches intact."""
    condition_id = mock_identifier("c", 0)
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    condition = IdentifierExpression(condition_id)
    value_branch = _make_x_plus_zero(x)
    otherwise_branch = IdentifierExpression(y)
    expression = PiecewiseExpression((condition,), (value_branch,), otherwise_branch)

    result = apply_rewrite_rules(expression, [_make_x_plus_zero_rule()])

    assert isinstance(result, PiecewiseExpression)
    assert result.conditions[0] is condition
    assert result.values[0].is_structurally_equivalent(IdentifierExpression(x))
    assert result.otherwise is otherwise_branch


# ===========================================================================
# apply_rewrite_rules: capture consistency, advanced
# ===========================================================================


def test_apply_rewrite_rules_x_minus_x_rule() -> None:
    """Test ``x - x -> 0`` matches structurally equivalent operands."""
    x = mock_identifier("x", 0)
    expression = _make_x_minus_x(x)

    result = apply_rewrite_rules(expression, [_make_x_minus_x_rule()])

    assert result.is_structurally_equivalent(LiteralExpression(0))


def test_apply_rewrite_rules_x_minus_x_does_not_match_distinct_operands() -> None:
    """Test ``x - x -> 0`` does not match when the operands differ."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = BinaryExpression(
        BinaryOperation.SUBTRACT,
        IdentifierExpression(x),
        IdentifierExpression(y),
    )

    result = apply_rewrite_rules(expression, [_make_x_minus_x_rule()])

    assert result is expression


# ===========================================================================
# apply_rewrite_rules: exception wrapping
# ===========================================================================


def test_apply_rewrite_rules_wraps_guard_exception_as_rewrite_callback_error() -> None:
    """Test ``apply_rewrite_rules`` raises a guard exception as its cause.

    The walk raises ``RewriteCallbackError`` naming the rule, not the pass
    framework's ``PassExecutionError``.
    """
    failure = RuntimeError("guard failed")

    def guard(_: MatchBindings) -> bool:
        raise failure

    rule = RewriteRule(
        pattern=WildcardPattern(),
        guard=guard,
        rewrite=lambda _: LiteralExpression(0),
    )

    with pytest.raises(RewriteCallbackError, match="rewrite rule 0 failed") as caught:
        apply_rewrite_rules(LiteralExpression(5), [rule])

    assert caught.value.__cause__ is failure
    assert caught.value.rule_index == 0


def test_apply_rewrite_rules_wraps_rewrite_exception_as_rewrite_callback_error() -> (
    None
):
    """Test ``apply_rewrite_rules`` raises a rewrite exception as its cause.

    The walk raises ``RewriteCallbackError`` naming the rule, not the pass
    framework's ``PassExecutionError``.
    """
    failure = RuntimeError("rewrite failed")

    def rewrite(_: MatchBindings) -> Expression:
        raise failure

    rule = RewriteRule(pattern=WildcardPattern(), rewrite=rewrite, name="broken")

    with pytest.raises(
        RewriteCallbackError, match=r"rewrite rule 0 \(broken\) failed"
    ) as caught:
        apply_rewrite_rules(LiteralExpression(5), [rule])

    assert caught.value.__cause__ is failure
    assert caught.value.rule_name == "broken"


# ===========================================================================
# RewriteRuleApplier
# ===========================================================================


def test_rewrite_rule_applier_is_registered_with_pass_registry() -> None:
    """Test ``RewriteRuleApplier`` is registered under its declared pass name."""
    registered = CompilerPass.get_registered_passes()

    assert "fhy_core.symbolic.expression.apply_rewrite_rules" in registered
    assert (
        registered["fhy_core.symbolic.expression.apply_rewrite_rules"].pass_type
        is RewriteRuleApplier
    )


def test_rewrite_rule_applier_execute_output_matches_free_function() -> None:
    """Test the applier's output matches ``apply_rewrite_rules`` for the same input."""
    x = mock_identifier("x", 0)
    expression = _make_x_plus_zero(x)
    rules: Sequence[RewriteRule] = [_make_x_plus_zero_rule()]

    applier = RewriteRuleApplier(rules)
    pass_result = applier.execute(expression)
    function_result = apply_rewrite_rules(expression, rules)

    assert pass_result.output.is_structurally_equivalent(function_result)


def test_rewrite_rule_applier_reports_no_change_when_no_rule_fires() -> None:
    """Test the applier reports ``changed=False`` when no rule fires."""
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(x), LiteralExpression(1)
    )

    applier = RewriteRuleApplier([_make_x_plus_zero_rule()])
    pass_result = applier.execute(expression)

    assert pass_result.changed is False
    assert pass_result.output is expression


def test_rewrite_rule_applier_reports_change_when_a_rule_fires() -> None:
    """Test the applier reports ``changed=True`` when at least one rule fires."""
    x = mock_identifier("x", 0)
    expression = _make_x_plus_zero(x)

    applier = RewriteRuleApplier([_make_x_plus_zero_rule()])
    pass_result = applier.execute(expression)

    assert pass_result.changed is True


def test_rewrite_rule_applier_exposes_rules() -> None:
    """Test ``RewriteRuleApplier.rules`` returns the supplied rule sequence."""
    rules = (_make_x_plus_zero_rule(), _make_x_times_one_rule())

    applier = RewriteRuleApplier(rules)

    assert applier.rules == rules


def test_rewrite_rule_applier_defensively_copies_rule_sequence() -> None:
    """Test the applier is unaffected by mutation of the input rule sequence."""
    rule = _make_x_plus_zero_rule()
    mutable: list[RewriteRule] = [rule]

    applier = RewriteRuleApplier(mutable)
    mutable.clear()

    assert applier.rules == (rule,)


def test_rewrite_rule_applier_wraps_guard_exception_as_pass_execution_error() -> None:
    """Test ``execute`` wraps a guard exception as ``PassExecutionError``.

    The cause is the walk's ``RewriteCallbackError``, whose cause is the
    guard's exception.
    """
    failure = RuntimeError("guard failed")

    def guard(_: MatchBindings) -> bool:
        raise failure

    rule = RewriteRule(
        pattern=WildcardPattern(),
        guard=guard,
        rewrite=lambda _: LiteralExpression(0),
    )
    applier = RewriteRuleApplier([rule])

    with pytest.raises(PassExecutionError) as caught:
        applier.execute(LiteralExpression(5))

    assert isinstance(caught.value.__cause__, RewriteCallbackError)
    assert caught.value.__cause__.__cause__ is failure


def test_rewrite_rule_applier_wraps_rewrite_exception_as_pass_execution_error() -> None:
    """Test ``execute`` wraps a rewrite exception as ``PassExecutionError``.

    The cause is the walk's ``RewriteCallbackError``, whose cause is the
    rewrite's exception.
    """
    failure = RuntimeError("rewrite failed")

    def rewrite(_: MatchBindings) -> Expression:
        raise failure

    rule = RewriteRule(pattern=WildcardPattern(), rewrite=rewrite)
    applier = RewriteRuleApplier([rule])

    with pytest.raises(PassExecutionError) as caught:
        applier.execute(LiteralExpression(5))

    assert isinstance(caught.value.__cause__, RewriteCallbackError)
    assert caught.value.__cause__.__cause__ is failure


def test_rewrite_rule_applier_emits_diagnostic_when_named_rule_fires() -> None:
    """Test a named rule firing emits an ``INFO`` diagnostic naming the rule.

    The text is the core's, ``applied rewrite rule "name"``.
    """
    x = mock_identifier("x", 0)
    rule = _make_x_plus_zero_rule()
    applier = RewriteRuleApplier([rule])

    pass_result = applier.execute(_make_x_plus_zero(x))

    assert [
        (diagnostic.level, diagnostic.message_text)
        for diagnostic in pass_result.diagnostics
    ] == [(DiagnosticLevel.INFO, 'applied rewrite rule "x + 0 -> x"')]


def test_rewrite_rule_applier_does_not_emit_diagnostic_for_unnamed_rule() -> None:
    """Test an unnamed rule firing does not emit a diagnostic."""
    rule = RewriteRule(
        pattern=WildcardPattern(),
        rewrite=lambda _: LiteralExpression(0),
    )
    applier = RewriteRuleApplier([rule])

    pass_result = applier.execute(LiteralExpression(5))

    assert pass_result.diagnostics == ()


# ===========================================================================
# Adversarial cases
# ===========================================================================


def test_apply_rewrite_rules_with_identity_rewrite_reports_unchanged() -> None:
    """Test a rule whose rewrite returns the matched expression declines.

    Returning the matched node itself is no firing, so identity is
    preserved, and the next rule is tried and fires.
    """
    x = Capture("x")
    identity_rule = RewriteRule(
        pattern=CapturePattern(x),
        rewrite=lambda bindings: bindings[x],
    )
    replacement = LiteralExpression(6)
    next_rule = RewriteRule(
        pattern=LiteralPattern(value=5), rewrite=lambda _: replacement
    )
    expression = LiteralExpression(5)

    alone = RewriteRuleApplier([identity_rule])
    alone_result = alone.execute(expression)
    both = RewriteRuleApplier([identity_rule, next_rule])
    both_result = both.execute(expression)

    assert alone_result.output is expression
    assert alone_result.changed is False
    assert alone.fired == ()
    assert both_result.output is replacement
    assert both_result.changed is True
    assert both.fired == (FiredRule(1, None),)


def test_apply_rewrite_rules_in_call_arguments() -> None:
    """Test rules apply to expressions inside call arguments."""
    x = mock_identifier("x", 0)
    inner = _make_x_plus_zero(x)
    expression = CallExpression("f", (inner, LiteralExpression(3)))

    result = apply_rewrite_rules(expression, [_make_x_plus_zero_rule()])

    expected = CallExpression("f", (IdentifierExpression(x), LiteralExpression(3)))
    assert result.is_structurally_equivalent(expected)


def test_apply_rewrite_rules_in_unary_operand() -> None:
    """Test rules apply to expressions inside a unary operand."""
    x = mock_identifier("x", 0)
    inner = _make_x_plus_zero(x)
    expression = UnaryExpression(UnaryOperation.NEGATE, inner)

    result = apply_rewrite_rules(expression, [_make_x_plus_zero_rule()])

    expected = UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(x))
    assert result.is_structurally_equivalent(expected)


def test_apply_rewrite_rules_replaces_a_rule_set_matching_a_long_chain() -> None:
    """Test repeated rule firings collapse a long chain in a single bottom-up pass."""
    x = mock_identifier("x", 0)
    expression: Expression = IdentifierExpression(x)
    for _ in range(8):
        expression = BinaryExpression(
            BinaryOperation.ADD, expression, LiteralExpression(0)
        )

    result = apply_rewrite_rules(expression, [_make_x_plus_zero_rule()])

    assert result.is_structurally_equivalent(IdentifierExpression(x))


# ===========================================================================
# Fixpoint usage pattern (caller-driven)
# ===========================================================================


def test_caller_can_iterate_until_no_change_reported() -> None:
    """Test the caller-driven fixpoint loop terminates when ``changed`` is ``False``."""
    x = mock_identifier("x", 0)
    expression: Expression = _make_x_plus_zero(x)
    expression = BinaryExpression(BinaryOperation.ADD, expression, LiteralExpression(0))
    applier = RewriteRuleApplier([_make_x_plus_zero_rule()])

    current: Expression = expression
    iterations = 0
    while True:
        pass_result = applier.execute(current)
        iterations += 1
        if not pass_result.changed:
            break
        current = pass_result.output
        assert iterations < 10, "fixpoint loop did not terminate"

    assert current.is_structurally_equivalent(IdentifierExpression(x))


# ===========================================================================
# Pattern parameter typing
# ===========================================================================


@pytest.mark.parametrize(
    "rule_pattern",
    [
        WildcardPattern(),
        CapturePattern(Capture("x")),
        LiteralPattern(value=0),
        IdentifierPattern(),
        UnaryExpressionPattern(None, WildcardPattern()),
        BinaryExpressionPattern(None, WildcardPattern(), WildcardPattern()),
        LogicalExpressionPattern(None, None),
        PiecewiseExpressionPattern(None, WildcardPattern()),
        CallExpressionPattern(None, None),
        PredicatePattern(lambda _: True),
        AlternativesPattern((WildcardPattern(),)),
    ],
    ids=lambda pattern: type(pattern).__name__,
)
def test_rewrite_rule_accepts_every_pattern_kind(rule_pattern: Pattern) -> None:
    """Test ``RewriteRule`` accepts a pattern of every kind without coercion.

    The kinds are closed, so these are all the patterns there are.
    """
    rule = RewriteRule(pattern=rule_pattern, rewrite=lambda _: LiteralExpression(1))

    assert rule.pattern is rule_pattern
