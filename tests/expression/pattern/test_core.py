"""Tests for ``fhy_core.expression.pattern.core``."""

from collections.abc import Callable

import pytest
from immutabledict import immutabledict

from fhy_core.expression import (
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
from fhy_core.expression.pattern import (
    AlternativesPattern,
    BinaryExpressionPattern,
    CallExpressionPattern,
    CapturePattern,
    IdentifierPattern,
    LiteralPattern,
    MatchBindings,
    Pattern,
    PiecewiseExpressionPattern,
    PredicatePattern,
    UnaryExpressionPattern,
    WildcardPattern,
    does_pattern_match,
    match_pattern,
)
from fhy_core.traits import FrozenMutationError

from ..conftest import mock_identifier


def _make_simple_binary_expression(operation: BinaryOperation) -> BinaryExpression:
    return BinaryExpression(operation, LiteralExpression(1), LiteralExpression(2))


# ===========================================================================
# MatchBindings
# ===========================================================================


def test_match_bindings_empty_has_no_bound_names() -> None:
    """Test ``MatchBindings.empty()`` reports no bound names."""
    bindings = MatchBindings.empty()

    assert bindings.is_empty()
    assert bindings.names() == frozenset()


def test_match_bindings_try_bind_records_first_binding() -> None:
    """Test the first binding for a fresh name is recorded."""
    bindings = MatchBindings.empty()
    expression = LiteralExpression(5)

    bound = bindings.try_bind("x", expression)

    assert bound is not None
    assert bound.has("x")
    assert bound.get("x") is expression


def test_match_bindings_try_bind_leaves_receiver_untouched() -> None:
    """Test ``try_bind`` does not mutate the receiver."""
    bindings = MatchBindings.empty()

    bindings.try_bind("x", LiteralExpression(1))

    assert not bindings.has("x")
    assert bindings.is_empty()


def test_match_bindings_try_bind_repeated_with_equivalent_returns_self() -> None:
    """Test re-binding to a structurally equivalent expression returns the receiver."""
    bindings = MatchBindings.empty().try_bind("x", LiteralExpression(5))
    assert bindings is not None

    rebound = bindings.try_bind("x", LiteralExpression(5))

    assert rebound is bindings


def test_match_bindings_try_bind_retains_original_expression_on_reconfirm() -> None:
    """Test ``try_bind`` retains the first-bound expression on reconfirm."""
    original = LiteralExpression(5)
    bindings = MatchBindings.empty().try_bind("x", original)
    assert bindings is not None

    rebound = bindings.try_bind("x", LiteralExpression(5))

    assert rebound is not None
    assert rebound.get("x") is original


def test_match_bindings_try_bind_repeated_with_distinct_returns_none() -> None:
    """Test re-binding to a structurally distinct expression returns ``None``."""
    bindings = MatchBindings.empty().try_bind("x", LiteralExpression(5))
    assert bindings is not None

    rebound = bindings.try_bind("x", LiteralExpression(6))

    assert rebound is None


def test_match_bindings_try_bind_repeated_with_equivalent_compound() -> None:
    """Test re-binding to a structurally equivalent compound returns the receiver."""
    expression = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )
    equivalent = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )
    bindings = MatchBindings.empty().try_bind("x", expression)
    assert bindings is not None

    rebound = bindings.try_bind("x", equivalent)

    assert rebound is bindings


def test_match_bindings_get_raises_key_error_for_unbound_name() -> None:
    """Test ``get`` raises ``KeyError`` when the name is unbound."""
    bindings = MatchBindings.empty()

    with pytest.raises(KeyError):
        bindings.get("x")


def test_match_bindings_has_reports_bound_name() -> None:
    """Test ``has`` returns ``True`` after binding and ``False`` before."""
    bindings = MatchBindings.empty()

    assert not bindings.has("x")

    bound = bindings.try_bind("x", LiteralExpression(0))
    assert bound is not None

    assert bound.has("x")


def test_match_bindings_names_after_multiple_binds() -> None:
    """Test ``names`` reports the set of bound names."""
    bindings = MatchBindings.empty()
    step_one = bindings.try_bind("x", LiteralExpression(1))
    assert step_one is not None
    step_two = step_one.try_bind("y", LiteralExpression(2))
    assert step_two is not None

    assert step_two.names() == frozenset({"x", "y"})


def test_match_bindings_equality_by_structure() -> None:
    """Test two ``MatchBindings`` with the same content compare equal."""
    left = MatchBindings.empty().try_bind("x", LiteralExpression(7))
    right = MatchBindings.empty().try_bind("x", LiteralExpression(7))
    assert left is not None and right is not None

    assert left == right
    assert hash(left) == hash(right)


def test_match_bindings_inequality_for_distinct_content() -> None:
    """Test bindings with different content do not compare equal."""
    left = MatchBindings.empty().try_bind("x", LiteralExpression(1))
    right = MatchBindings.empty().try_bind("x", LiteralExpression(2))
    assert left is not None and right is not None

    assert left != right


def test_match_bindings_hash_collides_on_matching_key_sets() -> None:
    """Test ``__hash__`` collapses on the key set; ``__eq__`` distinguishes values."""
    left = MatchBindings.empty().try_bind("x", LiteralExpression(1))
    right = MatchBindings.empty().try_bind("x", LiteralExpression(2))
    assert left is not None and right is not None

    assert hash(left) == hash(right)
    assert left != right


def test_match_bindings_is_frozen() -> None:
    """Test mutation attempts on ``MatchBindings`` raise ``FrozenMutationError``."""
    bindings = MatchBindings.empty()

    with pytest.raises(FrozenMutationError):
        bindings.something_new = 1


def test_match_bindings_equality_against_non_match_bindings_is_false() -> None:
    """Test ``MatchBindings`` compares as unequal to non-``MatchBindings`` values."""
    bindings = MatchBindings.empty()

    assert bindings != "not a match bindings"
    assert bindings != 0


def test_match_bindings_with_different_key_sets_are_unequal() -> None:
    """Test ``MatchBindings`` with different key sets do not compare equal."""
    left = MatchBindings.empty().try_bind("x", LiteralExpression(1))
    right = MatchBindings.empty().try_bind("y", LiteralExpression(1))
    assert left is not None and right is not None

    assert left != right


def test_match_bindings_post_init_rejects_non_string_keys() -> None:
    """Test the constructor rejects keys that are not strings."""
    with pytest.raises(TypeError, match="keys"):
        MatchBindings(immutabledict({0: LiteralExpression(1)}))  # type: ignore[dict-item]


def test_match_bindings_post_init_rejects_non_expression_values() -> None:
    """Test the constructor rejects values that are not `Expression`."""
    with pytest.raises(TypeError, match="Expression"):
        MatchBindings(immutabledict({"x": "not an expression"}))  # type: ignore[dict-item]


# ===========================================================================
# WildcardPattern
# ===========================================================================


def test_wildcard_pattern_matches_any_literal() -> None:
    """Test ``WildcardPattern`` matches a literal expression."""
    pattern = WildcardPattern()

    result = pattern.match(LiteralExpression(5))

    assert result is not None
    assert result.is_empty()


def test_wildcard_pattern_matches_any_compound_expression() -> None:
    """Test ``WildcardPattern`` matches a compound expression."""
    pattern = WildcardPattern()
    expression = _make_simple_binary_expression(BinaryOperation.ADD)

    result = pattern.match(expression)

    assert result is not None
    assert result.is_empty()


def test_wildcard_pattern_match_under_returns_input_bindings() -> None:
    """Test ``WildcardPattern.match_under`` returns the supplied bindings unchanged."""
    starting = MatchBindings.empty().try_bind("a", LiteralExpression(0))
    assert starting is not None
    pattern = WildcardPattern()

    result = pattern.match_under(LiteralExpression(1), starting)

    assert result is starting


# ===========================================================================
# CapturePattern
# ===========================================================================


def test_capture_pattern_with_wildcard_sub_pattern_binds_any_expression() -> None:
    """Test a wildcard-backed capture binds the matched expression."""
    pattern = CapturePattern("x", WildcardPattern())
    expression = LiteralExpression(5)

    result = pattern.match(expression)

    assert result is not None
    assert result.get("x") is expression


def test_capture_pattern_stores_name_as_plain_string() -> None:
    """Test ``CapturePattern.name`` is stored as a plain string."""
    pattern = CapturePattern("x", WildcardPattern())

    assert pattern.name == "x"


def test_capture_pattern_siblings_with_shared_name_share_the_same_capture() -> None:
    """Test two sibling ``CapturePattern("x", ...)`` share one capture key."""
    pattern = BinaryExpressionPattern(
        BinaryOperation.SUBTRACT,
        CapturePattern("x", WildcardPattern()),
        CapturePattern("x", WildcardPattern()),
    )
    expression = BinaryExpression(
        BinaryOperation.SUBTRACT, LiteralExpression(5), LiteralExpression(5)
    )

    result = pattern.match(expression)

    assert result is not None
    assert result.names() == frozenset({"x"})


def test_capture_pattern_runs_sub_pattern_before_capture() -> None:
    """Test the sub-pattern is checked before the capture takes effect."""
    pattern = CapturePattern("x", LiteralPattern(value=5))

    result = pattern.match(LiteralExpression(6))

    assert result is None


def test_capture_pattern_succeeds_when_sub_pattern_matches() -> None:
    """Test capture succeeds when the sub-pattern matches."""
    pattern = CapturePattern("x", LiteralPattern(value=5))
    expression = LiteralExpression(5)

    result = pattern.match(expression)

    assert result is not None
    assert result.get("x") is expression


def test_capture_pattern_repeated_consistent_capture_succeeds() -> None:
    """Test repeated captures with structurally equivalent expressions succeed."""
    pattern = BinaryExpressionPattern(
        BinaryOperation.SUBTRACT,
        CapturePattern("x", WildcardPattern()),
        CapturePattern("x", WildcardPattern()),
    )
    expression = BinaryExpression(
        BinaryOperation.SUBTRACT, LiteralExpression(5), LiteralExpression(5)
    )

    result = pattern.match(expression)

    assert result is not None
    assert result.get("x").is_structurally_equivalent(LiteralExpression(5))


def test_capture_pattern_repeated_inconsistent_capture_fails() -> None:
    """Test repeating a capture name with a structurally distinct expression fails."""
    pattern = BinaryExpressionPattern(
        BinaryOperation.SUBTRACT,
        CapturePattern("x", WildcardPattern()),
        CapturePattern("x", WildcardPattern()),
    )
    expression = BinaryExpression(
        BinaryOperation.SUBTRACT, LiteralExpression(5), LiteralExpression(6)
    )

    result = pattern.match(expression)

    assert result is None


def test_capture_pattern_nested_capture_records_both_bindings() -> None:
    """Test nesting ``CapturePattern`` records both outer and inner captures."""
    pattern = CapturePattern("outer", CapturePattern("inner", WildcardPattern()))
    expression = LiteralExpression(5)

    result = pattern.match(expression)

    assert result is not None
    assert result.get("outer") is expression
    assert result.get("inner") is expression


# ===========================================================================
# LiteralPattern
# ===========================================================================


def test_literal_pattern_with_no_value_matches_any_literal() -> None:
    """Test ``LiteralPattern()`` matches any literal expression."""
    pattern = LiteralPattern()

    assert pattern.match(LiteralExpression(5)) is not None
    assert pattern.match(LiteralExpression(3.14)) is not None
    assert pattern.match(LiteralExpression(True)) is not None


def test_literal_pattern_does_not_match_non_literal_expression() -> None:
    """Test ``LiteralPattern`` does not match a non-literal expression."""
    pattern = LiteralPattern()
    x = mock_identifier("x", 0)

    assert pattern.match(IdentifierExpression(x)) is None


def test_literal_pattern_with_value_requires_value_equality() -> None:
    """Test ``LiteralPattern(value=5)`` matches only ``LiteralExpression(5)``."""
    pattern = LiteralPattern(value=5)

    assert pattern.match(LiteralExpression(5)) is not None
    assert pattern.match(LiteralExpression(6)) is None


def test_literal_pattern_distinguishes_int_from_float() -> None:
    """Test ``LiteralPattern(value=5)`` rejects ``LiteralExpression(5.0)``."""
    pattern = LiteralPattern(value=5)

    assert pattern.match(LiteralExpression(5.0)) is None


def test_literal_pattern_distinguishes_int_from_bool() -> None:
    """Test ``LiteralPattern(value=1)`` rejects ``LiteralExpression(True)``."""
    pattern = LiteralPattern(value=1)

    assert pattern.match(LiteralExpression(True)) is None


# ===========================================================================
# IdentifierPattern
# ===========================================================================


def test_identifier_pattern_with_no_identifier_matches_any_identifier() -> None:
    """Test ``IdentifierPattern()`` matches any identifier expression."""
    pattern = IdentifierPattern()
    x = mock_identifier("x", 0)

    assert pattern.match(IdentifierExpression(x)) is not None


def test_identifier_pattern_does_not_match_non_identifier_expression() -> None:
    """Test ``IdentifierPattern`` does not match a non-identifier expression."""
    pattern = IdentifierPattern()

    assert pattern.match(LiteralExpression(5)) is None


def test_identifier_pattern_with_identifier_requires_identity() -> None:
    """Test ``IdentifierPattern(identifier=x)`` requires by-identity match."""
    x = mock_identifier("x", 0)
    pattern = IdentifierPattern(identifier=x)

    assert pattern.match(IdentifierExpression(x)) is not None


def test_identifier_pattern_rejects_distinct_identifier_with_same_hint() -> None:
    """Test ``IdentifierPattern`` rejects a distinct identifier sharing a hint."""
    x_one = mock_identifier("x", 0)
    x_two = mock_identifier("x", 1)
    pattern = IdentifierPattern(identifier=x_one)

    assert pattern.match(IdentifierExpression(x_two)) is None


# ===========================================================================
# UnaryExpressionPattern
# ===========================================================================


def test_unary_expression_pattern_matches_specific_operation() -> None:
    """Test ``UnaryExpressionPattern`` matches the specified operation."""
    pattern = UnaryExpressionPattern(UnaryOperation.NEGATE, WildcardPattern())
    expression = UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(5))

    assert pattern.match(expression) is not None


def test_unary_expression_pattern_rejects_wrong_operation() -> None:
    """Test ``UnaryExpressionPattern`` rejects the wrong operation."""
    pattern = UnaryExpressionPattern(UnaryOperation.NEGATE, WildcardPattern())
    expression = UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(5))

    assert pattern.match(expression) is None


def test_unary_expression_pattern_with_none_operation_matches_any() -> None:
    """Test ``UnaryExpressionPattern(operation=None, ...)`` matches any unary op."""
    pattern = UnaryExpressionPattern(None, WildcardPattern())

    assert (
        pattern.match(UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(5)))
        is not None
    )
    assert (
        pattern.match(UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(5)))
        is not None
    )


def test_unary_expression_pattern_rejects_non_unary_expression() -> None:
    """Test ``UnaryExpressionPattern`` rejects a non-unary expression."""
    pattern = UnaryExpressionPattern(None, WildcardPattern())

    assert pattern.match(LiteralExpression(5)) is None


def test_unary_expression_pattern_threads_bindings_through_operand() -> None:
    """Test ``UnaryExpressionPattern`` records bindings from the operand sub-pattern."""
    pattern = UnaryExpressionPattern(
        UnaryOperation.NEGATE, CapturePattern("x", WildcardPattern())
    )
    operand = LiteralExpression(5)
    expression = UnaryExpression(UnaryOperation.NEGATE, operand)

    result = pattern.match(expression)

    assert result is not None
    assert result.get("x") is operand


def test_unary_expression_pattern_propagates_operand_failure() -> None:
    """Test ``UnaryExpressionPattern`` fails when the operand sub-pattern fails."""
    pattern = UnaryExpressionPattern(UnaryOperation.NEGATE, LiteralPattern(value=5))
    expression = UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(6))

    assert pattern.match(expression) is None


# ===========================================================================
# BinaryExpressionPattern
# ===========================================================================


def test_binary_expression_pattern_matches_specific_operation() -> None:
    """Test ``BinaryExpressionPattern`` matches the specified operation."""
    pattern = BinaryExpressionPattern(
        BinaryOperation.ADD, WildcardPattern(), WildcardPattern()
    )

    assert (
        pattern.match(_make_simple_binary_expression(BinaryOperation.ADD)) is not None
    )


def test_binary_expression_pattern_rejects_wrong_operation() -> None:
    """Test ``BinaryExpressionPattern`` rejects a different operation."""
    pattern = BinaryExpressionPattern(
        BinaryOperation.ADD, WildcardPattern(), WildcardPattern()
    )

    assert (
        pattern.match(_make_simple_binary_expression(BinaryOperation.SUBTRACT)) is None
    )


def test_binary_expression_pattern_with_none_operation_matches_any() -> None:
    """Test ``BinaryExpressionPattern(operation=None, ...)`` matches any operation."""
    pattern = BinaryExpressionPattern(None, WildcardPattern(), WildcardPattern())

    assert (
        pattern.match(_make_simple_binary_expression(BinaryOperation.ADD)) is not None
    )
    assert (
        pattern.match(_make_simple_binary_expression(BinaryOperation.MULTIPLY))
        is not None
    )


def test_binary_expression_pattern_rejects_non_binary_expression() -> None:
    """Test ``BinaryExpressionPattern`` rejects a non-binary expression."""
    pattern = BinaryExpressionPattern(
        BinaryOperation.ADD, WildcardPattern(), WildcardPattern()
    )

    assert pattern.match(LiteralExpression(5)) is None


def test_binary_expression_pattern_threads_bindings_through_operands() -> None:
    """Test bindings from the left and right operand sub-patterns combine."""
    pattern = BinaryExpressionPattern(
        BinaryOperation.ADD,
        CapturePattern("a", WildcardPattern()),
        CapturePattern("b", WildcardPattern()),
    )
    left = LiteralExpression(1)
    right = LiteralExpression(2)
    expression = BinaryExpression(BinaryOperation.ADD, left, right)

    result = pattern.match(expression)

    assert result is not None
    assert result.get("a") is left
    assert result.get("b") is right


def test_binary_expression_pattern_propagates_left_failure() -> None:
    """Test ``BinaryExpressionPattern`` fails when the left sub-pattern fails."""
    pattern = BinaryExpressionPattern(
        BinaryOperation.ADD,
        LiteralPattern(value=99),
        WildcardPattern(),
    )

    assert pattern.match(_make_simple_binary_expression(BinaryOperation.ADD)) is None


def test_binary_expression_pattern_propagates_right_failure() -> None:
    """Test ``BinaryExpressionPattern`` fails when the right sub-pattern fails."""
    pattern = BinaryExpressionPattern(
        BinaryOperation.ADD,
        WildcardPattern(),
        LiteralPattern(value=99),
    )

    assert pattern.match(_make_simple_binary_expression(BinaryOperation.ADD)) is None


# ===========================================================================
# PiecewiseExpressionPattern
# ===========================================================================


def test_piecewise_expression_pattern_matches_single_case_piecewise() -> None:
    """Test ``PiecewiseExpressionPattern`` matches a one-case piecewise expression."""
    pattern = PiecewiseExpressionPattern(
        ((WildcardPattern(), WildcardPattern()),), WildcardPattern()
    )
    expression = PiecewiseExpression(
        (LiteralExpression(True),), (LiteralExpression(1),), LiteralExpression(2)
    )

    assert pattern.match(expression) is not None


def test_piecewise_expression_pattern_rejects_non_piecewise_expression() -> None:
    """Test ``PiecewiseExpressionPattern`` rejects a non-piecewise expression."""
    pattern = PiecewiseExpressionPattern(None, WildcardPattern())

    assert pattern.match(LiteralExpression(5)) is None


def test_piecewise_expression_pattern_with_none_cases_matches_any_case_count() -> None:
    """Test ``cases=None`` matches piecewise expressions of any case count."""
    pattern = PiecewiseExpressionPattern(None, WildcardPattern())
    single_case = PiecewiseExpression(
        (LiteralExpression(True),), (LiteralExpression(1),), LiteralExpression(2)
    )
    multi_case = PiecewiseExpression(
        (LiteralExpression(True), LiteralExpression(False)),
        (LiteralExpression(1), LiteralExpression(2)),
        LiteralExpression(3),
    )

    assert pattern.match(single_case) is not None
    assert pattern.match(multi_case) is not None


def test_piecewise_expression_pattern_rejects_case_count_mismatch() -> None:
    """Test a non-``None`` ``cases`` tuple requires an exact case-count match."""
    pattern = PiecewiseExpressionPattern(
        ((WildcardPattern(), WildcardPattern()),), WildcardPattern()
    )
    two_case_expression = PiecewiseExpression(
        (LiteralExpression(True), LiteralExpression(False)),
        (LiteralExpression(1), LiteralExpression(2)),
        LiteralExpression(3),
    )

    assert pattern.match(two_case_expression) is None


def test_piecewise_expression_pattern_threads_bindings_through_case_and_otherwise() -> (
    None
):
    """Test bindings from the case condition, case value, and otherwise combine."""
    pattern = PiecewiseExpressionPattern(
        (
            (
                CapturePattern("c", WildcardPattern()),
                CapturePattern("v", WildcardPattern()),
            ),
        ),
        CapturePattern("o", WildcardPattern()),
    )
    condition = LiteralExpression(True)
    value = LiteralExpression(1)
    otherwise = LiteralExpression(2)
    expression = PiecewiseExpression((condition,), (value,), otherwise)

    result = pattern.match(expression)

    assert result is not None
    assert result.get("c") is condition
    assert result.get("v") is value
    assert result.get("o") is otherwise


def test_piecewise_expression_pattern_threads_bindings_across_cases_in_order() -> None:
    """Test bindings from every case thread through in evaluation order."""
    pattern = PiecewiseExpressionPattern(
        (
            (
                CapturePattern("c1", WildcardPattern()),
                CapturePattern("v1", WildcardPattern()),
            ),
            (
                CapturePattern("c2", WildcardPattern()),
                CapturePattern("v2", WildcardPattern()),
            ),
        ),
        WildcardPattern(),
    )
    c1, c2 = LiteralExpression(True), LiteralExpression(False)
    v1, v2 = LiteralExpression(1), LiteralExpression(2)
    expression = PiecewiseExpression((c1, c2), (v1, v2), LiteralExpression(0))

    result = pattern.match(expression)

    assert result is not None
    assert result.get("c1") is c1
    assert result.get("v1") is v1
    assert result.get("c2") is c2
    assert result.get("v2") is v2


def test_piecewise_expression_pattern_propagates_condition_failure_within_a_case() -> (
    None
):
    """Test a failing condition sub-pattern in any case fails the whole match."""
    pattern = PiecewiseExpressionPattern(
        ((LiteralPattern(value=99), WildcardPattern()),), WildcardPattern()
    )
    expression = PiecewiseExpression(
        (LiteralExpression(True),), (LiteralExpression(1),), LiteralExpression(2)
    )

    assert pattern.match(expression) is None


def test_piecewise_expression_pattern_propagates_value_failure_within_a_case() -> None:
    """Test a failing value sub-pattern in any case fails the whole match."""
    pattern = PiecewiseExpressionPattern(
        ((WildcardPattern(), LiteralPattern(value=99)),), WildcardPattern()
    )
    expression = PiecewiseExpression(
        (LiteralExpression(True),), (LiteralExpression(1),), LiteralExpression(2)
    )

    assert pattern.match(expression) is None


def test_piecewise_expression_pattern_propagates_otherwise_failure() -> None:
    """Test a failing ``otherwise`` sub-pattern fails the match even if cases match."""
    pattern = PiecewiseExpressionPattern(
        ((WildcardPattern(), WildcardPattern()),), LiteralPattern(value=99)
    )
    expression = PiecewiseExpression(
        (LiteralExpression(True),), (LiteralExpression(1),), LiteralExpression(2)
    )

    assert pattern.match(expression) is None


# ===========================================================================
# CallExpressionPattern
# ===========================================================================


def test_call_expression_pattern_matches_specific_function_name() -> None:
    """Test ``CallExpressionPattern(function_name="f", ...)`` matches ``f``."""
    pattern = CallExpressionPattern(function_name="f", arguments=(WildcardPattern(),))
    expression = CallExpression("f", (LiteralExpression(1),))

    assert pattern.match(expression) is not None


def test_call_expression_pattern_rejects_wrong_function_name() -> None:
    """Test ``CallExpressionPattern`` rejects a call to a different function name."""
    pattern = CallExpressionPattern(function_name="f", arguments=(WildcardPattern(),))
    expression = CallExpression("g", (LiteralExpression(1),))

    assert pattern.match(expression) is None


def test_call_expression_pattern_with_none_function_name_matches_any() -> None:
    """Test ``CallExpressionPattern(function_name=None, ...)`` matches any name."""
    pattern = CallExpressionPattern(function_name=None, arguments=(WildcardPattern(),))

    assert pattern.match(CallExpression("f", (LiteralExpression(1),))) is not None
    assert pattern.match(CallExpression("g", (LiteralExpression(1),))) is not None


def test_call_expression_pattern_rejects_arity_mismatch() -> None:
    """Test fixed-length ``arguments`` rejects calls with the wrong arity."""
    pattern = CallExpressionPattern(
        function_name="f",
        arguments=(WildcardPattern(), WildcardPattern()),
    )

    assert pattern.match(CallExpression("f", (LiteralExpression(1),))) is None


def test_call_expression_pattern_with_none_arguments_matches_any_arity() -> None:
    """Test ``CallExpressionPattern(arguments=None)`` matches any arity."""
    pattern = CallExpressionPattern(function_name="f", arguments=None)

    assert pattern.match(CallExpression("f", ())) is not None
    assert pattern.match(CallExpression("f", (LiteralExpression(1),))) is not None
    assert (
        pattern.match(CallExpression("f", (LiteralExpression(1), LiteralExpression(2))))
        is not None
    )


def test_call_expression_pattern_threads_bindings_through_arguments() -> None:
    """Test ``CallExpressionPattern`` records bindings from argument sub-patterns."""
    pattern = CallExpressionPattern(
        function_name="f",
        arguments=(
            CapturePattern("a", WildcardPattern()),
            CapturePattern("b", WildcardPattern()),
        ),
    )
    a_argument = LiteralExpression(1)
    b_argument = LiteralExpression(2)
    expression = CallExpression("f", (a_argument, b_argument))

    result = pattern.match(expression)

    assert result is not None
    assert result.get("a") is a_argument
    assert result.get("b") is b_argument


def test_call_expression_pattern_rejects_non_call_expression() -> None:
    """Test ``CallExpressionPattern`` rejects a non-call expression."""
    pattern = CallExpressionPattern(function_name="f", arguments=None)

    assert pattern.match(LiteralExpression(5)) is None


def test_call_expression_pattern_matches_empty_argument_call() -> None:
    """Test ``CallExpressionPattern(arguments=())`` matches a zero-argument call."""
    pattern = CallExpressionPattern(function_name="f", arguments=())

    assert pattern.match(CallExpression("f", ())) is not None
    assert pattern.match(CallExpression("f", (LiteralExpression(1),))) is None


# ===========================================================================
# PredicatePattern
# ===========================================================================


def test_predicate_pattern_matches_when_predicate_returns_true() -> None:
    """Test ``PredicatePattern`` matches when the predicate returns ``True``."""
    pattern = PredicatePattern(lambda _: True)

    assert pattern.match(LiteralExpression(5)) is not None


def test_predicate_pattern_does_not_match_when_predicate_returns_false() -> None:
    """Test ``PredicatePattern`` does not match when the predicate returns ``False``."""
    pattern = PredicatePattern(lambda _: False)

    assert pattern.match(LiteralExpression(5)) is None


def test_predicate_pattern_receives_candidate_expression() -> None:
    """Test the predicate is called with the candidate expression."""
    captured: list[Expression] = []

    def predicate(expression: Expression) -> bool:
        captured.append(expression)
        return True

    pattern = PredicatePattern(predicate)
    expression = LiteralExpression(5)

    pattern.match(expression)

    assert captured == [expression]


def test_predicate_pattern_propagates_exceptions() -> None:
    """Test predicate exceptions propagate from ``match``."""

    def predicate(_: Expression) -> bool:
        raise RuntimeError("predicate failed")

    pattern = PredicatePattern(predicate)

    with pytest.raises(RuntimeError, match="predicate failed"):
        pattern.match(LiteralExpression(5))


def test_predicate_pattern_captures_nothing() -> None:
    """Test ``PredicatePattern`` does not add any bindings on a successful match."""
    starting = MatchBindings.empty().try_bind("x", LiteralExpression(0))
    assert starting is not None
    pattern = PredicatePattern(lambda _: True)

    result = pattern.match_under(LiteralExpression(5), starting)

    assert result == starting


def test_predicate_pattern_using_isinstance_check() -> None:
    """Test a realistic ``PredicatePattern`` filtering for ``LiteralExpression``."""
    pattern = PredicatePattern(lambda e: isinstance(e, LiteralExpression))
    x = mock_identifier("x", 0)

    assert pattern.match(LiteralExpression(5)) is not None
    assert pattern.match(IdentifierExpression(x)) is None


# ===========================================================================
# AlternativesPattern
# ===========================================================================


def test_alternatives_pattern_with_empty_alternatives_raises_value_error() -> None:
    """Test ``AlternativesPattern`` rejects an empty alternatives tuple."""
    with pytest.raises(ValueError, match="alternatives"):
        AlternativesPattern(())


def test_alternatives_pattern_returns_first_match() -> None:
    """Test ``AlternativesPattern`` returns bindings from the first matching child."""
    pattern = AlternativesPattern(
        (
            CapturePattern("x", LiteralPattern()),
            CapturePattern("x", IdentifierPattern()),
        )
    )
    expression = LiteralExpression(5)

    result = pattern.match(expression)

    assert result is not None
    assert result.get("x") is expression


def test_alternatives_pattern_falls_through_to_later_alternative() -> None:
    """Test ``AlternativesPattern`` falls through past failing earlier alternatives."""
    x = mock_identifier("x", 0)
    pattern = AlternativesPattern(
        (
            CapturePattern("x", LiteralPattern()),
            CapturePattern("x", IdentifierPattern()),
        )
    )
    expression = IdentifierExpression(x)

    result = pattern.match(expression)

    assert result is not None
    assert result.get("x") is expression


def test_alternatives_pattern_fails_when_all_alternatives_fail() -> None:
    """Test ``AlternativesPattern`` fails when every alternative fails."""
    pattern = AlternativesPattern((LiteralPattern(value=5), LiteralPattern(value=6)))

    assert pattern.match(LiteralExpression(7)) is None


def test_alternatives_pattern_isolates_failed_attempts() -> None:
    """Test a failing alternative does not taint the bindings for the next attempt."""
    pattern = AlternativesPattern(
        (
            BinaryExpressionPattern(
                BinaryOperation.ADD,
                CapturePattern("x", LiteralPattern(value=99)),
                WildcardPattern(),
            ),
            BinaryExpressionPattern(
                BinaryOperation.ADD,
                CapturePattern("x", WildcardPattern()),
                WildcardPattern(),
            ),
        )
    )
    expression = _make_simple_binary_expression(BinaryOperation.ADD)

    result = pattern.match(expression)

    assert result is not None
    assert result.get("x").is_structurally_equivalent(LiteralExpression(1))


def test_alternatives_pattern_discards_captures_from_failed_alternative() -> None:
    """Test a capture that fired inside a failed alternative does not leak."""
    pattern = AlternativesPattern(
        (
            BinaryExpressionPattern(
                BinaryOperation.ADD,
                CapturePattern("captured_in_failed", LiteralPattern(value=1)),
                LiteralPattern(value=99),
            ),
            CapturePattern("captured_in_successful", WildcardPattern()),
        )
    )
    expression = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )

    result = pattern.match(expression)

    assert result is not None
    assert not result.has("captured_in_failed")
    assert result.has("captured_in_successful")


# ===========================================================================
# Threading: repeated capture across sub-patterns
# ===========================================================================


def test_repeated_capture_across_call_arguments_requires_equivalence() -> None:
    """Test repeated capture across two call arguments requires equivalence."""
    pattern = CallExpressionPattern(
        function_name="f",
        arguments=(
            CapturePattern("x", WildcardPattern()),
            CapturePattern("x", WildcardPattern()),
        ),
    )
    equal_call = CallExpression("f", (LiteralExpression(1), LiteralExpression(1)))
    unequal_call = CallExpression("f", (LiteralExpression(1), LiteralExpression(2)))

    assert pattern.match(equal_call) is not None
    assert pattern.match(unequal_call) is None


# ===========================================================================
# Free functions
# ===========================================================================


def test_match_pattern_delegates_to_pattern_match() -> None:
    """Test ``match_pattern`` returns ``pattern.match(expression)``."""
    pattern = LiteralPattern(value=5)
    expression = LiteralExpression(5)

    assert match_pattern(pattern, expression) == pattern.match(expression)


def test_match_pattern_returns_none_on_mismatch() -> None:
    """Test ``match_pattern`` returns ``None`` for a non-matching pattern."""
    pattern = LiteralPattern(value=5)

    assert match_pattern(pattern, LiteralExpression(6)) is None


def test_does_pattern_match_true_on_match() -> None:
    """Test ``does_pattern_match`` returns ``True`` on a successful match."""
    pattern = LiteralPattern(value=5)

    assert does_pattern_match(pattern, LiteralExpression(5))


def test_does_pattern_match_false_on_mismatch() -> None:
    """Test ``does_pattern_match`` returns ``False`` on a failed match."""
    pattern = LiteralPattern(value=5)

    assert not does_pattern_match(pattern, LiteralExpression(6))


# ===========================================================================
# Edge cases
# ===========================================================================


def _build_left_leaning_addition_chain(depth: int) -> Expression:
    """Build ``((((0 + 0) + 0) + 0) ...)`` with ``depth`` additions."""
    expression: Expression = LiteralExpression(0)
    for _ in range(depth):
        expression = BinaryExpression(
            BinaryOperation.ADD, expression, LiteralExpression(0)
        )
    return expression


def _build_left_leaning_addition_pattern(depth: int) -> Pattern:
    """Build the pattern matching ``_build_left_leaning_addition_chain(depth)``."""
    pattern: Pattern = LiteralPattern(value=0)
    for _ in range(depth):
        pattern = BinaryExpressionPattern(
            BinaryOperation.ADD, pattern, LiteralPattern(value=0)
        )
    return pattern


def test_match_pattern_against_deeply_nested_expression() -> None:
    """Test matching a deeply nested expression succeeds without recursion failure."""
    expression = _build_left_leaning_addition_chain(50)
    pattern = _build_left_leaning_addition_pattern(50)

    assert match_pattern(pattern, expression) is not None


# ===========================================================================
# Determinism
# ===========================================================================


@pytest.mark.parametrize(
    "build_pattern",
    [
        lambda: LiteralPattern(value=5),
        lambda: BinaryExpressionPattern(
            BinaryOperation.ADD,
            CapturePattern("a", WildcardPattern()),
            CapturePattern("b", WildcardPattern()),
        ),
        lambda: AlternativesPattern((LiteralPattern(value=5), IdentifierPattern())),
    ],
    ids=["literal", "binary_capture", "alternatives"],
)
def test_match_pattern_is_deterministic(
    build_pattern: Callable[[], Pattern],
) -> None:
    """Test repeated calls to ``match_pattern`` return equal bindings."""
    pattern = build_pattern()
    expression = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )

    first = match_pattern(pattern, expression)
    second = match_pattern(pattern, expression)

    assert first == second
