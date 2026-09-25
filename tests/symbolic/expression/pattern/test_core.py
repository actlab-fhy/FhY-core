"""Tests for ``fhy_core.symbolic.expression.pattern.core``."""

import math
from collections.abc import Callable
from decimal import Decimal

import pytest

from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    LogicalExpression,
    LogicalOperation,
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
    IdentifierPattern,
    LiteralPattern,
    LogicalExpressionPattern,
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


def _match_capture(capture: Capture, expression: Expression) -> MatchBindings:
    """Return the bindings of ``CapturePattern(capture)`` matching ``expression``."""
    bindings = CapturePattern(capture).match(expression)
    assert bindings is not None
    return bindings


def _match_difference(
    left_capture: Capture, right_capture: Capture, expression: Expression
) -> MatchBindings | None:
    """Return the bindings of ``left - right`` capturing both operands."""
    pattern = BinaryExpressionPattern(
        BinaryOperation.SUBTRACT,
        CapturePattern(left_capture),
        CapturePattern(right_capture),
    )
    return pattern.match(expression)


def test_match_bindings_empty_has_no_bound_names() -> None:
    """Test ``MatchBindings.empty()`` binds no capture."""
    bindings = MatchBindings.empty()

    assert bindings.is_empty()
    assert len(bindings) == 0
    assert list(bindings) == []


def test_match_bindings_of_a_capture_match_record_the_binding() -> None:
    """Test a match records the capture bound to the expression it matched."""
    x = Capture("x")
    expression = LiteralExpression(5)

    bindings = _match_capture(x, expression)

    assert bindings.has(x)
    assert bindings[x] is expression


def test_matching_leaves_earlier_bindings_untouched() -> None:
    """Test a later match of the same pattern does not change earlier bindings."""
    x = Capture("x")
    pattern = CapturePattern(x)
    first = LiteralExpression(1)

    bindings = pattern.match(first)
    pattern.match(LiteralExpression(2))

    assert bindings is not None
    assert bindings[x] is first
    assert len(bindings) == 1


def test_repeated_capture_of_equal_expressions_binds_the_capture_once() -> None:
    """Test a capture matched twice against equal expressions is bound once."""
    x = Capture("x")
    expression = BinaryExpression(
        BinaryOperation.SUBTRACT, LiteralExpression(5), LiteralExpression(5)
    )

    bindings = _match_difference(x, x, expression)

    assert bindings is not None
    assert list(bindings) == [x]


def test_repeated_capture_keeps_the_first_bound_expression() -> None:
    """Test a repeated capture keeps the expression it bound first."""
    x = Capture("x")
    original = LiteralExpression(5)
    expression = BinaryExpression(
        BinaryOperation.SUBTRACT, original, LiteralExpression(5)
    )

    bindings = _match_difference(x, x, expression)

    assert bindings is not None
    assert bindings[x] is original


def test_repeated_capture_of_distinct_expressions_does_not_match() -> None:
    """Test a capture matched against structurally distinct expressions fails."""
    x = Capture("x")
    expression = BinaryExpression(
        BinaryOperation.SUBTRACT, LiteralExpression(5), LiteralExpression(6)
    )

    assert _match_difference(x, x, expression) is None


def test_repeated_capture_of_equal_compound_expressions_matches() -> None:
    """Test a repeated capture accepts structurally equal compound expressions."""
    x = Capture("x")
    expression = BinaryExpression(
        BinaryOperation.SUBTRACT,
        BinaryExpression(
            BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
        ),
        BinaryExpression(
            BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
        ),
    )

    bindings = _match_difference(x, x, expression)

    assert bindings is not None
    assert list(bindings) == [x]


def test_match_bindings_get_returns_none_for_an_unbound_capture() -> None:
    """Test ``get`` returns ``None`` for a capture that is not bound."""
    bindings = _match_capture(Capture("x"), LiteralExpression(1))

    assert bindings.get(Capture("x")) is None
    assert MatchBindings.empty().get(Capture("y")) is None


def test_match_bindings_index_raises_key_error_for_an_unbound_capture() -> None:
    """Test ``bindings[capture]`` raises ``KeyError`` naming an unbound capture."""
    bindings = MatchBindings.empty()

    with pytest.raises(KeyError, match="capture `x` is not bound"):
        bindings[Capture("x")]


def test_match_bindings_has_reports_bound_capture() -> None:
    """Test ``has`` and ``in`` report a bound capture, and only a bound one."""
    x = Capture("x")

    assert not MatchBindings.empty().has(x)
    assert x not in MatchBindings.empty()

    bindings = _match_capture(x, LiteralExpression(0))

    assert bindings.has(x)
    assert x in bindings


def test_match_bindings_iterate_over_bound_captures_in_binding_order() -> None:
    """Test iteration yields the bound captures, and ``len`` counts them."""
    x, y = Capture("x"), Capture("y")
    expression = BinaryExpression(
        BinaryOperation.SUBTRACT, LiteralExpression(1), LiteralExpression(2)
    )

    bindings = _match_difference(x, y, expression)

    assert bindings is not None
    assert list(bindings) == [x, y]
    assert len(bindings) == 2


def test_match_bindings_equality_by_structure() -> None:
    """Test bindings of the same capture to equal expressions compare equal."""
    x = Capture("x")
    left = _match_capture(x, LiteralExpression(7))
    right = _match_capture(x, LiteralExpression(7))

    assert left == right
    assert hash(left) == hash(right)


def test_match_bindings_inequality_for_distinct_content() -> None:
    """Test bindings of a capture to unequal expressions do not compare equal."""
    x = Capture("x")
    left = _match_capture(x, LiteralExpression(1))
    right = _match_capture(x, LiteralExpression(2))

    assert left != right


def test_match_bindings_equality_and_hash_ignore_binding_order() -> None:
    """Test bindings binding the same captures in another order are equal."""
    x, y = Capture("x"), Capture("y")
    expression = BinaryExpression(
        BinaryOperation.SUBTRACT, LiteralExpression(1), LiteralExpression(1)
    )

    left = _match_difference(x, y, expression)
    right = _match_difference(y, x, expression)

    assert left is not None and right is not None
    assert list(left) == [x, y]
    assert list(right) == [y, x]
    assert left == right
    assert hash(left) == hash(right)


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


def test_match_bindings_with_different_captures_are_unequal() -> None:
    """Test bindings of different captures to one expression are unequal."""
    expression = LiteralExpression(1)
    left = _match_capture(Capture("x"), expression)
    right = _match_capture(Capture("x"), expression)

    assert left != right


def test_match_bindings_constructor_refuses_an_argument() -> None:
    """Test no public constructor binds a capture: only a match does."""
    with pytest.raises(TypeError, match="only a match produces bindings"):
        MatchBindings({Capture("x"): LiteralExpression(1)})  # type: ignore[call-arg]


def test_match_bindings_constructed_without_arguments_bind_nothing() -> None:
    """Test ``MatchBindings()`` binds nothing and equals ``empty()``."""
    bindings = MatchBindings()

    assert bindings.is_empty()
    assert bindings == MatchBindings.empty()


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


def test_wildcard_pattern_keeps_the_bindings_threaded_to_it() -> None:
    """Test a wildcard adds no capture to the bindings a compound threads to it.

    D-S5-4 removed ``match_under``; the threading shows through a sibling
    capture, which the wildcard neither drops nor adds to.
    """
    a = Capture("a")
    pattern = BinaryExpressionPattern(
        BinaryOperation.ADD, CapturePattern(a), WildcardPattern()
    )
    left = LiteralExpression(0)

    result = pattern.match(BinaryExpression(BinaryOperation.ADD, left, left + 1))

    assert result is not None
    assert list(result) == [a]
    assert result[a] is left


# ===========================================================================
# CapturePattern
# ===========================================================================


def test_capture_pattern_with_wildcard_sub_pattern_binds_any_expression() -> None:
    """Test a wildcard-backed capture binds the matched expression."""
    x = Capture("x")
    pattern = CapturePattern(x)
    expression = LiteralExpression(5)

    result = pattern.match(expression)

    assert result is not None
    assert result[x] is expression


def test_capture_pattern_stores_its_capture_and_a_wildcard_by_default() -> None:
    """Test ``CapturePattern`` keeps the ``Capture`` object it binds.

    Its sub-pattern defaults to a ``WildcardPattern``.
    """
    x = Capture("x")
    pattern = CapturePattern(x)

    assert pattern.capture is x
    assert pattern.capture.name == "x"
    assert pattern.sub_pattern == WildcardPattern()


def test_capture_pattern_siblings_share_one_capture_object() -> None:
    """Test siblings using one ``Capture`` share it, and same-named ones do not.

    D-S5-2: identity, not name, decides: two ``Capture("x")`` objects are
    independent captures.
    """
    x = Capture("x")
    shared = BinaryExpressionPattern(
        BinaryOperation.SUBTRACT, CapturePattern(x), CapturePattern(x)
    )
    first, second = Capture("x"), Capture("x")
    independent = BinaryExpressionPattern(
        BinaryOperation.SUBTRACT, CapturePattern(first), CapturePattern(second)
    )
    equal = BinaryExpression(
        BinaryOperation.SUBTRACT, LiteralExpression(5), LiteralExpression(5)
    )
    unequal = BinaryExpression(
        BinaryOperation.SUBTRACT, LiteralExpression(5), LiteralExpression(6)
    )

    shared_result = shared.match(equal)
    independent_result = independent.match(unequal)

    assert shared_result is not None
    assert list(shared_result) == [x]
    assert shared.match(unequal) is None
    assert independent_result is not None
    assert list(independent_result) == [first, second]


def test_capture_pattern_runs_sub_pattern_before_capture() -> None:
    """Test the sub-pattern is checked before the capture takes effect."""
    x = Capture("x")
    pattern = CapturePattern(x, LiteralPattern(value=5))

    result = pattern.match(LiteralExpression(6))

    assert result is None


def test_capture_pattern_succeeds_when_sub_pattern_matches() -> None:
    """Test capture succeeds when the sub-pattern matches."""
    x = Capture("x")
    pattern = CapturePattern(x, LiteralPattern(value=5))
    expression = LiteralExpression(5)

    result = pattern.match(expression)

    assert result is not None
    assert result[x] is expression


def test_capture_pattern_repeated_consistent_capture_succeeds() -> None:
    """Test repeated captures with structurally equivalent expressions succeed."""
    x = Capture("x")
    pattern = BinaryExpressionPattern(
        BinaryOperation.SUBTRACT,
        CapturePattern(x),
        CapturePattern(x),
    )
    expression = BinaryExpression(
        BinaryOperation.SUBTRACT, LiteralExpression(5), LiteralExpression(5)
    )

    result = pattern.match(expression)

    assert result is not None
    assert result[x].is_structurally_equivalent(LiteralExpression(5))


def test_capture_pattern_repeated_inconsistent_capture_fails() -> None:
    """Test repeating a capture name with a structurally distinct expression fails."""
    x = Capture("x")
    pattern = BinaryExpressionPattern(
        BinaryOperation.SUBTRACT,
        CapturePattern(x),
        CapturePattern(x),
    )
    expression = BinaryExpression(
        BinaryOperation.SUBTRACT, LiteralExpression(5), LiteralExpression(6)
    )

    result = pattern.match(expression)

    assert result is None


def test_capture_pattern_nested_capture_records_both_bindings() -> None:
    """Test nesting ``CapturePattern`` records both outer and inner captures."""
    outer = Capture("outer")
    inner = Capture("inner")
    pattern = CapturePattern(outer, CapturePattern(inner))
    expression = LiteralExpression(5)

    result = pattern.match(expression)

    assert result is not None
    assert result[outer] is expression
    assert result[inner] is expression


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


def test_literal_pattern_matches_the_normalized_value_of_its_text() -> None:
    """Test a text pattern value matches the literal the text normalizes to.

    A literal keeps no spelling, so ``LiteralPattern(value="05")`` stands
    for the literal ``5`` and ``"1.50"`` for ``Decimal("1.5")``.
    """
    assert LiteralPattern(value="05").match(LiteralExpression(5)) is not None
    assert LiteralPattern(value="1.50").match(LiteralExpression("1.5")) is not None
    assert (
        LiteralPattern(value="1.50").match(LiteralExpression(Decimal("1.5")))
        is not None
    )


def test_literal_pattern_distinguishes_decimal_from_float() -> None:
    """Test a decimal pattern value rejects the binary float with the same digits."""
    pattern = LiteralPattern(value=Decimal("1.5"))

    assert pattern.match(LiteralExpression(1.5)) is None


def test_literal_pattern_with_nan_value_matches_a_nan_literal() -> None:
    """Test ``LiteralPattern(value=nan)`` matches a NaN literal, as ``==`` does."""
    pattern = LiteralPattern(value=math.nan)

    assert pattern.match(LiteralExpression(math.nan)) is not None
    assert pattern.match(LiteralExpression(0.0)) is None


def test_literal_pattern_with_negative_zero_matches_positive_zero() -> None:
    """Test ``LiteralPattern(value=-0.0)`` matches ``LiteralExpression(0.0)``."""
    assert LiteralPattern(value=-0.0).match(LiteralExpression(0.0)) is not None


def test_literal_pattern_rejects_text_outside_the_literal_grammar() -> None:
    """Test a text value ``LiteralExpression`` refuses is refused at construction."""
    with pytest.raises(ValueError, match="invalid literal text"):
        LiteralPattern(value="abc")


def test_literal_pattern_rejects_an_unsupported_value_type() -> None:
    """Test a value of a type ``LiteralExpression`` refuses raises ``TypeError``."""
    with pytest.raises(TypeError):
        LiteralPattern(value=object())  # type: ignore[arg-type]


def test_literal_pattern_keeps_the_value_it_was_given() -> None:
    """Test the pattern's ``value`` field is the given value, not the normalized one."""
    assert LiteralPattern(value="05").value == "05"


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
    x = Capture("x")
    pattern = UnaryExpressionPattern(UnaryOperation.NEGATE, CapturePattern(x))
    operand = LiteralExpression(5)
    expression = UnaryExpression(UnaryOperation.NEGATE, operand)

    result = pattern.match(expression)

    assert result is not None
    assert result[x] is operand


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
    a = Capture("a")
    b = Capture("b")
    pattern = BinaryExpressionPattern(
        BinaryOperation.ADD,
        CapturePattern(a),
        CapturePattern(b),
    )
    left = LiteralExpression(1)
    right = LiteralExpression(2)
    expression = BinaryExpression(BinaryOperation.ADD, left, right)

    result = pattern.match(expression)

    assert result is not None
    assert result[a] is left
    assert result[b] is right


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


def test_piecewise_expression_pattern_with_empty_cases_matches_nothing() -> None:
    """Test an empty (non-``None``) cases tuple builds a pattern matching nothing.

    D-S5-5: a real ``PiecewiseExpression`` always has at least one case, so
    the pattern can never match; it builds, as the core's does, rather than
    raising. ``None`` is the spelling for "match any case count."
    """
    pattern = PiecewiseExpressionPattern((), WildcardPattern())

    assert pattern.cases == ()
    assert (
        pattern.match(
            PiecewiseExpression(
                (LiteralExpression(True),),
                (LiteralExpression(1),),
                LiteralExpression(2),
            )
        )
        is None
    )


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
    c = Capture("c")
    v = Capture("v")
    o = Capture("o")
    pattern = PiecewiseExpressionPattern(
        (
            (
                CapturePattern(c),
                CapturePattern(v),
            ),
        ),
        CapturePattern(o),
    )
    condition = LiteralExpression(True)
    value = LiteralExpression(1)
    otherwise = LiteralExpression(2)
    expression = PiecewiseExpression((condition,), (value,), otherwise)

    result = pattern.match(expression)

    assert result is not None
    assert result[c] is condition
    assert result[v] is value
    assert result[o] is otherwise


def test_piecewise_expression_pattern_threads_bindings_across_cases_in_order() -> None:
    """Test bindings from every case thread through in evaluation order."""
    c1, v1, c2, v2 = Capture("c1"), Capture("v1"), Capture("c2"), Capture("v2")
    pattern = PiecewiseExpressionPattern(
        (
            (CapturePattern(c1), CapturePattern(v1)),
            (CapturePattern(c2), CapturePattern(v2)),
        ),
        WildcardPattern(),
    )
    condition_1, condition_2 = LiteralExpression(True), LiteralExpression(False)
    value_1, value_2 = LiteralExpression(1), LiteralExpression(2)
    expression = PiecewiseExpression(
        (condition_1, condition_2), (value_1, value_2), LiteralExpression(0)
    )

    result = pattern.match(expression)

    assert result is not None
    assert list(result) == [c1, v1, c2, v2]
    assert result[c1] is condition_1
    assert result[v1] is value_1
    assert result[c2] is condition_2
    assert result[v2] is value_2


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
# PiecewiseExpressionPattern: __post_init__ coercion and validation
# ===========================================================================


def test_piecewise_expression_pattern_none_cases_remains_none() -> None:
    """Test constructing with ``cases=None`` leaves the field as ``None``."""
    pattern = PiecewiseExpressionPattern(None, WildcardPattern())

    assert pattern.cases is None


def test_piecewise_expression_pattern_coerces_list_cases_to_tuple_of_tuples() -> None:
    """Test constructing with list ``cases`` (and list pairs) coerces to tuples.

    ``cases`` is declared as ``tuple[tuple[Pattern, Pattern], ...] | None``, but
    nothing at the language level stops a caller from passing a list of lists
    instead; ``__post_init__`` normalizes both levels to real tuples.
    """
    condition_pattern = WildcardPattern()
    value_pattern = LiteralPattern(value=1)

    pattern = PiecewiseExpressionPattern(
        [[condition_pattern, value_pattern]],  # type: ignore[list-item]
        WildcardPattern(),
    )

    assert type(pattern.cases) is tuple
    assert pattern.cases is not None
    assert type(pattern.cases[0]) is tuple
    assert pattern.cases == ((condition_pattern, value_pattern),)


def test_piecewise_expression_pattern_cases_unaffected_by_later_list_mutation() -> None:
    """Test mutating the list passed as ``cases`` after construction has no effect."""
    condition_pattern = WildcardPattern()
    value_pattern = LiteralPattern(value=1)
    cases_list: list[tuple[Pattern, Pattern]] = [(condition_pattern, value_pattern)]

    pattern = PiecewiseExpressionPattern(cases_list, WildcardPattern())
    cases_list.append((LiteralPattern(value=2), LiteralPattern(value=3)))

    assert pattern.cases == ((condition_pattern, value_pattern),)


def test_piecewise_expression_pattern_case_pair_unaffected_by_later_list_mutation() -> (
    None
):
    """Test mutating a case-pair list passed inside ``cases`` has no effect.

    Each case pair may itself be supplied as a list rather than a tuple;
    ``__post_init__`` must coerce that inner sequence too, or a caller
    mutating the pair in place after construction would silently change
    the pattern's matching behavior.
    """
    original_condition = WildcardPattern()
    value_pattern = LiteralPattern(value=1)
    pair = [original_condition, value_pattern]

    pattern = PiecewiseExpressionPattern([pair], WildcardPattern())  # type: ignore[list-item]
    pair[0] = LiteralPattern(value=99)

    assert pattern.cases == ((original_condition, value_pattern),)


def test_piecewise_expression_pattern_rejects_non_pattern_condition_in_case() -> None:
    """Test a non-``Pattern`` condition inside a case raises ``TypeError``."""
    with pytest.raises(
        TypeError,
        match="PiecewiseExpressionPattern case conditions must be a Pattern, got str",
    ):
        PiecewiseExpressionPattern(
            (("not a pattern", WildcardPattern()),),  # type: ignore[arg-type]
            WildcardPattern(),
        )


def test_piecewise_expression_pattern_rejects_non_pattern_value_in_case() -> None:
    """Test a non-``Pattern`` value inside a case raises ``TypeError``."""
    with pytest.raises(
        TypeError,
        match="PiecewiseExpressionPattern case values must be a Pattern, got str",
    ):
        PiecewiseExpressionPattern(
            ((WildcardPattern(), "not a pattern"),),  # type: ignore[arg-type]
            WildcardPattern(),
        )


def test_piecewise_expression_pattern_rejects_non_pattern_otherwise() -> None:
    """Test a non-``Pattern`` ``otherwise`` raises ``TypeError``."""
    with pytest.raises(
        TypeError, match="PiecewiseExpressionPattern otherwise must be a Pattern"
    ):
        PiecewiseExpressionPattern(
            ((WildcardPattern(), WildcardPattern()),),
            "not a pattern",  # type: ignore[arg-type]
        )


def test_piecewise_expression_pattern_rejects_case_with_wrong_pair_length() -> None:
    """Test a case that is not a two-element pair raises ``TypeError``."""
    with pytest.raises(TypeError, match="pairs"):
        PiecewiseExpressionPattern(
            (  # type: ignore[arg-type]
                (WildcardPattern(), WildcardPattern(), WildcardPattern()),
            ),
            WildcardPattern(),
        )


NON_PATTERN_SUB_PATTERN_CASES = [
    (
        "unary_operand",
        lambda: UnaryExpressionPattern(None, "not a pattern"),  # type: ignore[arg-type]
        "UnaryExpressionPattern operand must be a Pattern, got str",
    ),
    (
        "binary_left",
        lambda: BinaryExpressionPattern(
            None,
            "not a pattern",  # type: ignore[arg-type]
            WildcardPattern(),
        ),
        "BinaryExpressionPattern left must be a Pattern, got str",
    ),
    (
        "binary_right",
        lambda: BinaryExpressionPattern(
            None,
            WildcardPattern(),
            "not a pattern",  # type: ignore[arg-type]
        ),
        "BinaryExpressionPattern right must be a Pattern, got str",
    ),
    (
        "capture_sub_pattern",
        lambda: CapturePattern(Capture("x"), "not a pattern"),  # type: ignore[arg-type]
        "CapturePattern sub_pattern must be a Pattern, got str",
    ),
]


@pytest.mark.parametrize(
    "construct, expected_context",
    [(construct, context) for _, construct, context in NON_PATTERN_SUB_PATTERN_CASES],
    ids=[identifier for identifier, _, _ in NON_PATTERN_SUB_PATTERN_CASES],
)
def test_composite_patterns_reject_a_non_pattern_sub_pattern(
    construct: Callable[[], Pattern], expected_context: str
) -> None:
    """Test every composite rejects a non-``Pattern`` child at construction.

    D-S5-5: the binding's own type check raises ``TypeError`` naming the
    field, uniformly across the composites.
    """
    with pytest.raises(TypeError, match=expected_context):
        construct()


def test_capture_pattern_rejects_a_capture_that_is_not_a_capture() -> None:
    """Test a capture that is not a ``Capture``, such as a name, raises.

    D-S5-2: captures are ``Capture`` handles, and a ``str`` name is refused
    at construction rather than bound by spelling.
    """
    with pytest.raises(TypeError, match="CapturePattern capture must be a Capture"):
        CapturePattern("x", WildcardPattern())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Capture name must be a str, got int"):
        Capture(99)  # type: ignore[arg-type]


def test_capture_with_an_empty_name_binds_like_any_other() -> None:
    """Test a capture named ``""`` is valid and binds what it matched.

    D-S5-2: any ``str`` is a capture name; the name serves only ``str``,
    ``repr`` and messages.
    """
    unnamed = Capture("")
    expression = LiteralExpression(5)

    result = CapturePattern(unnamed).match(expression)

    assert result is not None
    assert result[unnamed] is expression
    assert str(unnamed) == ""


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
    a = Capture("a")
    b = Capture("b")
    pattern = CallExpressionPattern(
        function_name="f",
        arguments=(
            CapturePattern(a),
            CapturePattern(b),
        ),
    )
    a_argument = LiteralExpression(1)
    b_argument = LiteralExpression(2)
    expression = CallExpression("f", (a_argument, b_argument))

    result = pattern.match(expression)

    assert result is not None
    assert result[a] is a_argument
    assert result[b] is b_argument


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
# CallExpressionPattern: __post_init__ coercion and validation
# ===========================================================================


def test_call_expression_pattern_none_arguments_remains_none() -> None:
    """Test constructing with ``arguments=None`` leaves the field as ``None``."""
    pattern = CallExpressionPattern(function_name="f", arguments=None)

    assert pattern.arguments is None


def test_call_expression_pattern_empty_tuple_arguments_remains_empty_tuple() -> None:
    """Test constructing with ``arguments=()`` leaves the field as an empty tuple."""
    pattern = CallExpressionPattern(function_name="f", arguments=())

    assert pattern.arguments == ()


def test_call_expression_pattern_coerces_list_arguments_to_tuple() -> None:
    """Test constructing with a list ``arguments`` coerces it to a tuple.

    ``arguments`` is declared as ``tuple[Pattern, ...] | None``, but nothing
    at the language level stops a caller from passing a list instead;
    ``__post_init__`` normalizes it to a real tuple.
    """
    argument_pattern = WildcardPattern()

    pattern = CallExpressionPattern(
        function_name="f",
        arguments=[argument_pattern],
    )

    assert type(pattern.arguments) is tuple
    assert pattern.arguments == (argument_pattern,)


def test_call_expression_pattern_arguments_unaffected_by_later_list_mutation() -> None:
    """Test post-construction mutation of the ``arguments`` list has no effect."""
    argument_pattern = WildcardPattern()
    arguments_list: list[Pattern] = [argument_pattern]

    pattern = CallExpressionPattern(
        function_name="f",
        arguments=arguments_list,
    )
    arguments_list.append(LiteralPattern(value=1))

    assert pattern.arguments == (argument_pattern,)


def test_call_expression_pattern_rejects_non_pattern_argument() -> None:
    """Test a non-``Pattern`` element in ``arguments`` raises ``TypeError``."""
    with pytest.raises(
        TypeError, match="CallExpressionPattern arguments must be a Pattern, got str"
    ):
        CallExpressionPattern(
            function_name="f",
            arguments=(WildcardPattern(), "not a pattern"),  # type: ignore[arg-type]
        )


# ===========================================================================
# LogicalExpressionPattern
# ===========================================================================


def _make_conjunction(*operands: Expression) -> LogicalExpression:
    return LogicalExpression(LogicalOperation.AND, operands)


def test_logical_expression_pattern_matches_specific_operation() -> None:
    """Test ``LogicalExpressionPattern(AND, ...)`` matches a conjunction."""
    pattern = LogicalExpressionPattern(
        LogicalOperation.AND, (WildcardPattern(), WildcardPattern())
    )
    expression = _make_conjunction(LiteralExpression(True), LiteralExpression(False))

    assert pattern.match(expression) is not None


def test_logical_expression_pattern_rejects_wrong_operation() -> None:
    """Test ``LogicalExpressionPattern(AND, ...)`` rejects a disjunction."""
    pattern = LogicalExpressionPattern(
        LogicalOperation.AND, (WildcardPattern(), WildcardPattern())
    )
    expression = LogicalExpression(
        LogicalOperation.OR, (LiteralExpression(True), LiteralExpression(False))
    )

    assert pattern.match(expression) is None


def test_logical_expression_pattern_with_none_operation_matches_any() -> None:
    """Test ``LogicalExpressionPattern(None, ...)`` matches either connective."""
    pattern = LogicalExpressionPattern(None, (WildcardPattern(), WildcardPattern()))
    operands = (LiteralExpression(True), LiteralExpression(False))

    assert pattern.match(LogicalExpression(LogicalOperation.AND, operands)) is not None
    assert pattern.match(LogicalExpression(LogicalOperation.OR, operands)) is not None


def test_logical_expression_pattern_rejects_operand_count_mismatch() -> None:
    """Test a two-operand pattern rejects a three-operand conjunction.

    A logical expression keeps its operands as given, so a pattern of two
    operands does not match the flattened shape of three.
    """
    pattern = LogicalExpressionPattern(
        LogicalOperation.AND, (WildcardPattern(), WildcardPattern())
    )
    expression = _make_conjunction(
        LiteralExpression(True), LiteralExpression(False), LiteralExpression(True)
    )

    assert pattern.match(expression) is None


def test_logical_expression_pattern_with_none_operands_matches_any_count() -> None:
    """Test ``LogicalExpressionPattern(operands=None)`` matches any operand count."""
    pattern = LogicalExpressionPattern(LogicalOperation.AND, None)

    assert (
        pattern.match(
            _make_conjunction(LiteralExpression(True), LiteralExpression(False))
        )
        is not None
    )
    assert (
        pattern.match(
            _make_conjunction(
                LiteralExpression(True),
                LiteralExpression(False),
                LiteralExpression(True),
            )
        )
        is not None
    )


def test_logical_expression_pattern_threads_bindings_through_operands() -> None:
    """Test ``LogicalExpressionPattern`` records bindings from operand sub-patterns."""
    a = Capture("a")
    b = Capture("b")
    pattern = LogicalExpressionPattern(
        LogicalOperation.AND,
        (
            CapturePattern(a),
            CapturePattern(b),
        ),
    )
    a_operand = IdentifierExpression(mock_identifier("a", 0))
    b_operand = IdentifierExpression(mock_identifier("b", 1))

    result = pattern.match(_make_conjunction(a_operand, b_operand))

    assert result is not None
    assert result[a] is a_operand
    assert result[b] is b_operand


def test_logical_expression_pattern_requires_repeated_captures_to_agree() -> None:
    """Test a capture repeated across operands matches only equal operands."""
    x = Capture("x")
    pattern = LogicalExpressionPattern(
        LogicalOperation.OR,
        (
            CapturePattern(x),
            CapturePattern(x),
        ),
    )
    a = IdentifierExpression(mock_identifier("a", 0))
    b = IdentifierExpression(mock_identifier("b", 1))

    assert pattern.match(LogicalExpression(LogicalOperation.OR, (a, a))) is not None
    assert pattern.match(LogicalExpression(LogicalOperation.OR, (a, b))) is None


def test_logical_expression_pattern_does_not_splice_a_nested_operand() -> None:
    """Test a nested conjunction is one operand, matched by a nested pattern."""
    inner = _make_conjunction(LiteralExpression(True), LiteralExpression(False))
    expression = _make_conjunction(LiteralExpression(True), inner)
    pattern = LogicalExpressionPattern(
        LogicalOperation.AND,
        (
            LiteralPattern(value=True),
            LogicalExpressionPattern(LogicalOperation.AND, None),
        ),
    )

    assert pattern.match(expression) is not None


def test_logical_expression_pattern_rejects_non_logical_expression() -> None:
    """Test ``LogicalExpressionPattern`` rejects a binary comparison."""
    pattern = LogicalExpressionPattern(None, None)

    assert pattern.match(_make_simple_binary_expression(BinaryOperation.LESS)) is None


def test_logical_expression_pattern_coerces_list_operands_to_tuple() -> None:
    """Test constructing with a list ``operands`` coerces it to a tuple."""
    operand_patterns: list[Pattern] = [WildcardPattern(), WildcardPattern()]

    pattern = LogicalExpressionPattern(
        LogicalOperation.AND,
        operand_patterns,
    )
    operand_patterns.append(WildcardPattern())

    assert type(pattern.operands) is tuple
    assert pattern.operands == (WildcardPattern(), WildcardPattern())


@pytest.mark.parametrize("count", [0, 1])
def test_logical_expression_pattern_with_fewer_than_two_operands_matches_nothing(
    count: int,
) -> None:
    """Test fewer than two operand patterns build a pattern matching nothing.

    D-S5-5: a logical expression has at least two operands, so the pattern
    can never match; it builds, as the core's does, rather than raising.
    """
    pattern = LogicalExpressionPattern(
        LogicalOperation.AND, tuple(WildcardPattern() for _ in range(count))
    )

    assert pattern.operands is not None
    assert len(pattern.operands) == count
    assert (
        pattern.match(
            _make_conjunction(LiteralExpression(True), LiteralExpression(False))
        )
        is None
    )


def test_logical_expression_pattern_rejects_non_pattern_operand() -> None:
    """Test a non-``Pattern`` element in ``operands`` raises ``TypeError``."""
    with pytest.raises(
        TypeError, match="LogicalExpressionPattern operands must be a Pattern, got str"
    ):
        LogicalExpressionPattern(
            LogicalOperation.AND,
            (WildcardPattern(), "not a pattern"),  # type: ignore[arg-type]
        )


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
    """Test ``PredicatePattern`` adds no binding to those threaded to it.

    D-S5-4 removed ``match_under``; the threading shows through a sibling
    capture, which stays the only binding.
    """
    x = Capture("x")
    pattern = BinaryExpressionPattern(
        BinaryOperation.ADD, CapturePattern(x), PredicatePattern(lambda _: True)
    )
    left = LiteralExpression(0)

    result = pattern.match(BinaryExpression(BinaryOperation.ADD, left, left + 5))

    assert result is not None
    assert list(result) == [x]


def test_predicate_pattern_using_isinstance_check() -> None:
    """Test a realistic ``PredicatePattern`` filtering for ``LiteralExpression``."""
    pattern = PredicatePattern(lambda e: isinstance(e, LiteralExpression))
    x = mock_identifier("x", 0)

    assert pattern.match(LiteralExpression(5)) is not None
    assert pattern.match(IdentifierExpression(x)) is None


# ===========================================================================
# AlternativesPattern
# ===========================================================================


def test_alternatives_pattern_with_empty_alternatives_matches_nothing() -> None:
    """Test ``AlternativesPattern(())`` builds a pattern matching nothing.

    D-S5-5: no alternative can match, so the pattern builds, as the core's
    does, rather than raising.
    """
    pattern = AlternativesPattern(())

    assert pattern.alternatives == ()
    assert pattern.match(LiteralExpression(5)) is None


def test_alternatives_pattern_returns_first_match() -> None:
    """Test ``AlternativesPattern`` returns bindings from the first matching child."""
    x = Capture("x")
    pattern = AlternativesPattern(
        (
            CapturePattern(x, LiteralPattern()),
            CapturePattern(x, IdentifierPattern()),
        )
    )
    expression = LiteralExpression(5)

    result = pattern.match(expression)

    assert result is not None
    assert result[x] is expression


def test_alternatives_pattern_falls_through_to_later_alternative() -> None:
    """Test ``AlternativesPattern`` falls through past failing earlier alternatives."""
    x = Capture("x")
    identifier = mock_identifier("x", 0)
    pattern = AlternativesPattern(
        (
            CapturePattern(x, LiteralPattern()),
            CapturePattern(x, IdentifierPattern()),
        )
    )
    expression = IdentifierExpression(identifier)

    result = pattern.match(expression)

    assert result is not None
    assert result[x] is expression


def test_alternatives_pattern_fails_when_all_alternatives_fail() -> None:
    """Test ``AlternativesPattern`` fails when every alternative fails."""
    pattern = AlternativesPattern((LiteralPattern(value=5), LiteralPattern(value=6)))

    assert pattern.match(LiteralExpression(7)) is None


def test_alternatives_pattern_isolates_failed_attempts() -> None:
    """Test a failing alternative does not taint the bindings for the next attempt."""
    x = Capture("x")
    pattern = AlternativesPattern(
        (
            BinaryExpressionPattern(
                BinaryOperation.ADD,
                CapturePattern(x, LiteralPattern(value=99)),
                WildcardPattern(),
            ),
            BinaryExpressionPattern(
                BinaryOperation.ADD,
                CapturePattern(x),
                WildcardPattern(),
            ),
        )
    )
    expression = _make_simple_binary_expression(BinaryOperation.ADD)

    result = pattern.match(expression)

    assert result is not None
    assert result[x].is_structurally_equivalent(LiteralExpression(1))


def test_alternatives_pattern_discards_captures_from_failed_alternative() -> None:
    """Test a capture that fired inside a failed alternative does not leak."""
    captured_in_failed = Capture("captured_in_failed")
    captured_in_successful = Capture("captured_in_successful")
    pattern = AlternativesPattern(
        (
            BinaryExpressionPattern(
                BinaryOperation.ADD,
                CapturePattern(captured_in_failed, LiteralPattern(value=1)),
                LiteralPattern(value=99),
            ),
            CapturePattern(captured_in_successful),
        )
    )
    expression = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )

    result = pattern.match(expression)

    assert result is not None
    assert not result.has(captured_in_failed)
    assert result.has(captured_in_successful)


# ===========================================================================
# AlternativesPattern: __post_init__ coercion and validation
# ===========================================================================


def test_alternatives_pattern_coerces_list_alternatives_to_tuple() -> None:
    """Test constructing with a list ``alternatives`` coerces it to a tuple.

    ``alternatives`` is declared as ``tuple[Pattern, ...]``, but nothing at
    the language level stops a caller from passing a list instead;
    ``__post_init__`` normalizes it to a real tuple.
    """
    first = LiteralPattern(value=1)
    second = LiteralPattern(value=2)

    pattern = AlternativesPattern([first, second])

    assert type(pattern.alternatives) is tuple
    assert pattern.alternatives == (first, second)


def test_alternatives_pattern_unaffected_by_later_list_mutation() -> None:
    """Test post-construction mutation of the ``alternatives`` list has no effect."""
    first = LiteralPattern(value=1)
    alternatives_list = [first]

    pattern = AlternativesPattern(alternatives_list)
    alternatives_list.append(LiteralPattern(value=2))

    assert pattern.alternatives == (first,)


def test_alternatives_pattern_rejects_non_pattern_alternative() -> None:
    """Test a non-``Pattern`` element in ``alternatives`` raises ``TypeError``."""
    with pytest.raises(
        TypeError, match="AlternativesPattern alternatives must be a Pattern, got str"
    ):
        AlternativesPattern((LiteralPattern(value=1), "not a pattern"))  # type: ignore[arg-type]


# ===========================================================================
# Threading: repeated capture across sub-patterns
# ===========================================================================


def test_repeated_capture_across_call_arguments_requires_equivalence() -> None:
    """Test repeated capture across two call arguments requires equivalence."""
    x = Capture("x")
    pattern = CallExpressionPattern(
        function_name="f",
        arguments=(
            CapturePattern(x),
            CapturePattern(x),
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
            CapturePattern(Capture("a")),
            CapturePattern(Capture("b")),
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
