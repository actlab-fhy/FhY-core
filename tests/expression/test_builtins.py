"""Tests for the built-in pre-registered functions ``max`` and ``min``."""

from fhy_core.expression import (
    BUILTIN_FUNCTIONS,
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    LiteralExpression,
    TernaryExpression,
    get_registered_function,
    inline_functions,
    is_function_registered,
)

# =============================================================================
# BUILTIN_FUNCTIONS mapping
# =============================================================================


def test_builtin_functions_contains_max_and_min() -> None:
    """Test ``BUILTIN_FUNCTIONS`` exposes ``max`` and ``min`` entries."""
    assert set(BUILTIN_FUNCTIONS.keys()) == {"max", "min"}


def test_builtin_functions_entries_carry_canonical_name() -> None:
    """Test each ``BUILTIN_FUNCTIONS`` entry's ``name`` matches its key."""
    assert BUILTIN_FUNCTIONS["max"].name == "max"
    assert BUILTIN_FUNCTIONS["min"].name == "min"


# =============================================================================
# Built-in registrations exist at import time
# =============================================================================


def test_max_is_registered_at_import_time() -> None:
    """Test ``max`` is in the registry as soon as the package is imported."""
    assert is_function_registered("max") is True


def test_min_is_registered_at_import_time() -> None:
    """Test ``min`` is in the registry as soon as the package is imported."""
    assert is_function_registered("min") is True


def test_max_registration_has_two_parameters() -> None:
    """Test the ``max`` registration declares exactly two parameters."""
    registered = get_registered_function("max")
    assert len(registered.parameters) == 2


def test_min_registration_has_two_parameters() -> None:
    """Test the ``min`` registration declares exactly two parameters."""
    registered = get_registered_function("min")
    assert len(registered.parameters) == 2


def test_builtin_functions_entries_match_registry() -> None:
    """Test ``BUILTIN_FUNCTIONS`` values equal the registered functions."""
    assert BUILTIN_FUNCTIONS["max"] is get_registered_function("max")
    assert BUILTIN_FUNCTIONS["min"] is get_registered_function("min")


# =============================================================================
# Inlining built-in calls
# =============================================================================


def _make_expected_max_body(a: Expression, b: Expression) -> Expression:
    return TernaryExpression(
        BinaryExpression(BinaryOperation.GREATER, a, b),
        a,
        b,
    )


def _make_expected_min_body(a: Expression, b: Expression) -> Expression:
    return TernaryExpression(
        BinaryExpression(BinaryOperation.LESS, a, b),
        a,
        b,
    )


def test_inlining_max_call_produces_greater_ternary() -> None:
    """Test inlining ``max(a, b)`` yields ``a > b ? a : b``."""
    a = LiteralExpression(7)
    b = LiteralExpression(2)

    result = inline_functions(CallExpression("max", (a, b)))

    assert result.is_structurally_equivalent(_make_expected_max_body(a, b))


def test_inlining_min_call_produces_less_ternary() -> None:
    """Test inlining ``min(a, b)`` yields ``a < b ? a : b``."""
    a = LiteralExpression(7)
    b = LiteralExpression(2)

    result = inline_functions(CallExpression("min", (a, b)))

    assert result.is_structurally_equivalent(_make_expected_min_body(a, b))


def test_inlining_min_of_max_call_folds_correctly() -> None:
    """Test ``min(max(a, b), c)`` inlines into a nested ternary expression."""
    a = LiteralExpression(1)
    b = LiteralExpression(2)
    c = LiteralExpression(3)

    expression = CallExpression(
        "min",
        (CallExpression("max", (a, b)), c),
    )
    expected_inner = _make_expected_max_body(a, b)
    expected = _make_expected_min_body(expected_inner, c)

    result = inline_functions(expression)

    assert result.is_structurally_equivalent(expected)
