"""Tests for ``PiecewiseExpression``, ``CallExpression``, and their helpers."""

import pytest

from fhy_core.expression import (
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    PiecewiseExpression,
    call,
    piecewise,
)
from fhy_core.traits import FrozenMutationError, HasOperands

from .conftest import mock_identifier

# =============================================================================
# PiecewiseExpression: construction and accessors
# =============================================================================


def test_piecewise_expression_stores_conditions_values_and_otherwise() -> None:
    """Test ``PiecewiseExpression`` exposes the three fields it was built with."""
    condition = LiteralExpression(True)
    value = LiteralExpression(1)
    otherwise = LiteralExpression(2)

    expression = PiecewiseExpression((condition,), (value,), otherwise)

    assert expression.conditions == (condition,)
    assert expression.values == (value,)
    assert expression.otherwise is otherwise


def test_piecewise_expression_supports_multiple_cases_in_declared_order() -> None:
    """Test a multi-case ``PiecewiseExpression`` preserves case order."""
    c1, c2 = LiteralExpression(True), LiteralExpression(False)
    v1, v2 = LiteralExpression(1), LiteralExpression(2)
    otherwise = LiteralExpression(0)

    expression = PiecewiseExpression((c1, c2), (v1, v2), otherwise)

    assert expression.conditions == (c1, c2)
    assert expression.values == (v1, v2)
    assert expression.otherwise is otherwise


def test_piecewise_expression_is_frozen_after_construction() -> None:
    """Test ``PiecewiseExpression`` instances reject attribute mutation."""
    expression = PiecewiseExpression(
        (LiteralExpression(True),), (LiteralExpression(1),), LiteralExpression(2)
    )

    with pytest.raises(FrozenMutationError):
        expression._mutation = "denied"


def test_piecewise_expression_satisfies_has_operands_protocol() -> None:
    """Test ``PiecewiseExpression`` satisfies the ``HasOperands`` runtime protocol."""
    expression = PiecewiseExpression(
        (LiteralExpression(True),), (LiteralExpression(1),), LiteralExpression(2)
    )
    assert isinstance(expression, HasOperands)


# =============================================================================
# PiecewiseExpression.__post_init__: validation errors
# =============================================================================


def test_piecewise_expression_rejects_zero_cases() -> None:
    """Test a ``PiecewiseExpression`` with no cases raises ``ValueError``."""
    with pytest.raises(ValueError, match=r"(?i)case"):
        PiecewiseExpression((), (), LiteralExpression(0))


def test_piecewise_expression_rejects_mismatched_condition_and_value_lengths() -> None:
    """Test unequal ``conditions``/``values`` lengths raise ``ValueError``."""
    with pytest.raises(ValueError, match=r"(?i)(length|condition)"):
        PiecewiseExpression(
            (LiteralExpression(True), LiteralExpression(False)),
            (LiteralExpression(1),),
            LiteralExpression(0),
        )


# =============================================================================
# PiecewiseExpression.get_cases()
# =============================================================================


def test_get_cases_returns_zipped_condition_value_pairs_in_order() -> None:
    """Test ``get_cases()`` returns ``(condition, value)`` pairs in evaluation order."""
    c1, c2 = LiteralExpression(True), LiteralExpression(False)
    v1, v2 = LiteralExpression(1), LiteralExpression(2)
    expression = PiecewiseExpression((c1, c2), (v1, v2), LiteralExpression(0))

    assert expression.get_cases() == ((c1, v1), (c2, v2))


def test_get_cases_on_single_case_expression() -> None:
    """Test ``get_cases()`` returns a single pair for a one-case expression."""
    condition = LiteralExpression(True)
    value = LiteralExpression(1)
    expression = PiecewiseExpression((condition,), (value,), LiteralExpression(0))

    assert expression.get_cases() == ((condition, value),)


# =============================================================================
# PiecewiseExpression.get_operands() / get_visit_children(): interleaved order
# =============================================================================


def test_get_operands_returns_interleaved_cases_then_otherwise() -> None:
    """Test ``get_operands()`` returns ``(c1, v1, c2, v2, ..., otherwise)``."""
    c1, c2 = LiteralExpression(True), LiteralExpression(False)
    v1, v2 = LiteralExpression(1), LiteralExpression(2)
    otherwise = LiteralExpression(0)
    expression = PiecewiseExpression((c1, c2), (v1, v2), otherwise)

    assert expression.get_operands() == (c1, v1, c2, v2, otherwise)


def test_get_visit_children_returns_interleaved_cases_then_otherwise() -> None:
    """Test ``get_visit_children()`` returns ``(c1, v1, c2, v2, ..., otherwise)``."""
    c1, c2 = LiteralExpression(True), LiteralExpression(False)
    v1, v2 = LiteralExpression(1), LiteralExpression(2)
    otherwise = LiteralExpression(0)
    expression = PiecewiseExpression((c1, c2), (v1, v2), otherwise)

    assert expression.get_visit_children() == (c1, v1, c2, v2, otherwise)


def test_get_visit_children_on_single_case_expression() -> None:
    """Test ``get_visit_children()`` returns ``(condition, value, otherwise)``."""
    condition = LiteralExpression(True)
    value = LiteralExpression(1)
    otherwise = LiteralExpression(0)
    expression = PiecewiseExpression((condition,), (value,), otherwise)

    assert expression.get_visit_children() == (condition, value, otherwise)


# =============================================================================
# PiecewiseExpression.rebuild_with_visit_children()
# =============================================================================


def test_rebuild_with_visit_children_round_trips_single_case() -> None:
    """Test rebuilding from ``get_visit_children()`` round-trips a one-case node."""
    condition = LiteralExpression(True)
    value = LiteralExpression(1)
    otherwise = LiteralExpression(0)
    expression = PiecewiseExpression((condition,), (value,), otherwise)

    rebuilt = expression.rebuild_with_visit_children(expression.get_visit_children())

    assert rebuilt.is_structurally_equivalent(expression)


def test_rebuild_with_visit_children_round_trips_multiple_cases() -> None:
    """Test rebuilding from ``get_visit_children()`` round-trips a multi-case node."""
    c1, c2, c3 = (
        LiteralExpression(True),
        LiteralExpression(False),
        LiteralExpression(True),
    )
    v1, v2, v3 = LiteralExpression(1), LiteralExpression(2), LiteralExpression(3)
    otherwise = LiteralExpression(0)
    expression = PiecewiseExpression((c1, c2, c3), (v1, v2, v3), otherwise)

    rebuilt = expression.rebuild_with_visit_children(expression.get_visit_children())

    assert rebuilt.is_structurally_equivalent(expression)
    assert rebuilt.conditions == (c1, c2, c3)
    assert rebuilt.values == (v1, v2, v3)
    assert rebuilt.otherwise is otherwise


@pytest.mark.parametrize(
    "child_count",
    [0, 1, 2, 4, 6],
    ids=["zero", "one", "two", "four", "six"],
)
def test_rebuild_with_visit_children_rejects_even_or_too_short_child_count(
    child_count: int,
) -> None:
    """Test rebuilding from a non-odd or too-short child sequence raises."""
    expression = PiecewiseExpression(
        (LiteralExpression(True),), (LiteralExpression(1),), LiteralExpression(0)
    )
    bad_children = tuple(LiteralExpression(i) for i in range(child_count))

    with pytest.raises(ValueError, match=r"(?i)child"):
        expression.rebuild_with_visit_children(bad_children)


# =============================================================================
# piecewise() constructor helper
# =============================================================================


def test_piecewise_helper_wraps_single_case_expressions_directly() -> None:
    """Test ``piecewise((c, v), otherwise=o)`` returns expression operands unchanged."""
    condition = LiteralExpression(True)
    value = LiteralExpression(1)
    otherwise = LiteralExpression(2)

    expression = piecewise((condition, value), otherwise=otherwise)

    assert isinstance(expression, PiecewiseExpression)
    assert expression.conditions == (condition,)
    assert expression.values == (value,)
    assert expression.otherwise is otherwise


def test_piecewise_helper_builds_multiple_cases_in_declared_order() -> None:
    """Test ``piecewise(...)`` accepts multiple ``(condition, value)`` pairs."""
    c1, c2 = LiteralExpression(True), LiteralExpression(False)
    v1, v2 = LiteralExpression(1), LiteralExpression(2)
    otherwise = LiteralExpression(0)

    expression = piecewise((c1, v1), (c2, v2), otherwise=otherwise)

    assert expression.conditions == (c1, c2)
    assert expression.values == (v1, v2)


def test_piecewise_helper_coerces_identifier_operand_in_condition() -> None:
    """Test ``piecewise(...)`` wraps an ``Identifier`` condition in an expression."""
    identifier = mock_identifier("flag", 0)

    expression = piecewise((identifier, 1), otherwise=0)

    assert isinstance(expression.conditions[0], IdentifierExpression)
    assert expression.conditions[0].identifier is identifier


def test_piecewise_helper_coerces_identifier_operand_in_value() -> None:
    """Test ``piecewise(...)`` wraps an ``Identifier`` value in an expression."""
    identifier = mock_identifier("x", 0)

    expression = piecewise((LiteralExpression(True), identifier), otherwise=0)

    assert isinstance(expression.values[0], IdentifierExpression)
    assert expression.values[0].identifier is identifier


def test_piecewise_helper_coerces_identifier_operand_in_otherwise() -> None:
    """Test ``piecewise(...)`` wraps an ``Identifier`` otherwise in an expression."""
    identifier = mock_identifier("fallback", 0)

    expression = piecewise((LiteralExpression(True), 1), otherwise=identifier)

    assert isinstance(expression.otherwise, IdentifierExpression)
    assert expression.otherwise.identifier is identifier


def test_piecewise_helper_coerces_python_literal_operands() -> None:
    """Test ``piecewise(...)`` wraps Python literal operands in an expression."""
    expression = piecewise((True, 5), otherwise=10)

    assert isinstance(expression.conditions[0], LiteralExpression)
    assert expression.conditions[0].value is True
    assert isinstance(expression.values[0], LiteralExpression)
    assert expression.values[0].value == 5
    assert isinstance(expression.otherwise, LiteralExpression)
    assert expression.otherwise.value == 10


def test_piecewise_helper_rejects_zero_cases() -> None:
    """Test ``piecewise(otherwise=...)`` with no cases raises ``ValueError``."""
    with pytest.raises(ValueError, match=r"(?i)case"):
        piecewise(otherwise=0)


def test_piecewise_helper_rejects_non_two_tuple_case() -> None:
    """Test a non-2-tuple case raises ``ValueError`` naming the offending position."""
    with pytest.raises(ValueError, match=r"1"):
        piecewise(
            (LiteralExpression(True), LiteralExpression(1)),
            (LiteralExpression(False), LiteralExpression(2), LiteralExpression(3)),  # type: ignore[arg-type]
            otherwise=0,
        )


def test_piecewise_helper_rejects_unsupported_operand_type() -> None:
    """Test ``piecewise(...)`` raises ``ValueError`` for an unsupported operand type."""
    with pytest.raises(ValueError):
        piecewise((LiteralExpression(True), object()), otherwise=0)  # type: ignore[arg-type]


def test_piecewise_helper_requires_otherwise_as_keyword() -> None:
    """Test omitting ``otherwise`` raises ``TypeError`` (keyword-only, required)."""
    with pytest.raises(TypeError):
        piecewise((LiteralExpression(True), LiteralExpression(1)))  # type: ignore[call-arg]


def test_piecewise_single_case_reduces_to_a_condition_value_and_fallback() -> None:
    """Test a single-case ``piecewise`` reduces to one condition, value, and fallback.

    ``piecewise((c, v), otherwise=o)`` is the minimal total conditional
    shape: exactly one condition, exactly one value, and a required
    fallback. This pins that shape down structurally so any regression
    in the single-case path is caught.
    """
    condition = LiteralExpression(True)
    true_value = LiteralExpression(1)
    false_value = LiteralExpression(2)

    expression = piecewise((condition, true_value), otherwise=false_value)

    assert expression.conditions == (condition,)
    assert expression.values == (true_value,)
    assert expression.otherwise is false_value


# =============================================================================
# Expression.piecewise: delegation
# =============================================================================


def test_expression_piecewise_delegates_to_module_level_helper() -> None:
    """Test ``Expression.piecewise`` produces the same result as ``piecewise(...)``."""
    condition = LiteralExpression(True)
    value = LiteralExpression(1)
    otherwise = LiteralExpression(2)

    via_static_method = Expression.piecewise((condition, value), otherwise=otherwise)
    via_free_function = piecewise((condition, value), otherwise=otherwise)

    assert isinstance(via_static_method, PiecewiseExpression)
    assert via_static_method.is_structurally_equivalent(via_free_function)


# =============================================================================
# CallExpression: construction and accessors
# =============================================================================


def test_call_expression_stores_name_and_arguments() -> None:
    """Test ``CallExpression`` exposes the name and argument tuple it was built with."""
    arg_a = LiteralExpression(1)
    arg_b = LiteralExpression(2)

    expression = CallExpression("max", (arg_a, arg_b))

    assert expression.function_name == "max"
    assert expression.arguments == (arg_a, arg_b)


def test_call_expression_supports_zero_arguments() -> None:
    """Test ``CallExpression`` accepts an empty arguments tuple structurally."""
    expression = CallExpression("nullary", ())

    assert expression.function_name == "nullary"
    assert expression.arguments == ()


def test_call_expression_rejects_empty_function_name() -> None:
    """Test ``CallExpression`` rejects an empty ``function_name`` at construction.

    This is a value constraint enforced by ``__post_init__``, independent of
    serialization; deserializing such a payload surfaces it through the generic
    engine as a value error.
    """
    with pytest.raises(ValueError, match="non-empty"):
        CallExpression("", (LiteralExpression(1),))


def test_call_expression_is_frozen_after_construction() -> None:
    """Test ``CallExpression`` instances reject attribute mutation."""
    expression = CallExpression("max", (LiteralExpression(1), LiteralExpression(2)))

    with pytest.raises(FrozenMutationError):
        expression._mutation = "denied"


def test_call_expression_satisfies_has_operands_protocol() -> None:
    """Test ``CallExpression`` satisfies the ``HasOperands`` runtime protocol."""
    expression = CallExpression("max", (LiteralExpression(1), LiteralExpression(2)))
    assert isinstance(expression, HasOperands)


def test_call_expression_get_operands_returns_arguments_in_order() -> None:
    """Test ``CallExpression.get_operands()`` returns the argument tuple in order."""
    arg_a = LiteralExpression(1)
    arg_b = LiteralExpression(2)
    arg_c = LiteralExpression(3)
    expression = CallExpression("select3", (arg_a, arg_b, arg_c))

    assert expression.get_operands() == (arg_a, arg_b, arg_c)


def test_call_expression_get_visit_children_returns_arguments_in_order() -> None:
    """Test ``CallExpression.get_visit_children()`` exposes its arguments."""
    arg_a = LiteralExpression(1)
    arg_b = LiteralExpression(2)
    expression = CallExpression("max", (arg_a, arg_b))

    assert expression.get_visit_children() == (arg_a, arg_b)


# =============================================================================
# call() constructor helper
# =============================================================================


def test_call_helper_returns_call_expression_with_name_and_arguments() -> None:
    """Test ``call(...)`` produces a ``CallExpression`` carrying its name and args."""
    arg_a = LiteralExpression(1)
    arg_b = LiteralExpression(2)

    expression = call("max", arg_a, arg_b)

    assert isinstance(expression, CallExpression)
    assert expression.function_name == "max"
    assert expression.arguments == (arg_a, arg_b)


def test_call_helper_supports_zero_arguments() -> None:
    """Test ``call(name)`` produces a ``CallExpression`` with empty arguments."""
    expression = call("nullary")

    assert isinstance(expression, CallExpression)
    assert expression.arguments == ()


def test_call_helper_coerces_identifier_argument_to_identifier_expression() -> None:
    """Test ``call(...)`` wraps an ``Identifier`` argument in identifier expr."""
    identifier = mock_identifier("x", 0)

    expression = call("f", identifier)

    arguments = expression.arguments
    assert len(arguments) == 1
    assert isinstance(arguments[0], IdentifierExpression)
    assert arguments[0].identifier is identifier


def test_call_helper_coerces_python_literal_to_literal_expression() -> None:
    """Test ``call(...)`` wraps a Python literal argument in ``LiteralExpression``."""
    expression = call("max", 1, 2)

    arguments = expression.arguments
    assert len(arguments) == 2
    assert isinstance(arguments[0], LiteralExpression)
    assert arguments[0].value == 1
    assert isinstance(arguments[1], LiteralExpression)
    assert arguments[1].value == 2


def test_call_helper_rejects_unsupported_argument_type() -> None:
    """Test ``call(...)`` raises ``ValueError`` for unsupported argument types."""
    with pytest.raises(ValueError):
        call("f", object())  # type: ignore[arg-type]
