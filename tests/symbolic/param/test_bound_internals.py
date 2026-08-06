"""Tests for the private interval-bound helpers in ``fhy_core.symbolic.param.core``.

These cover the bound-expression parsing used by interval-integer parameter
arithmetic: ``_invert_comparison``, ``_bound_from_literal``, and the
``_iter_interval_bounds`` guards (driven through the public ``+`` operator).
"""

from typing import Any

import pytest

from fhy_core.symbolic.constraint import EquationConstraint, InSetConstraint
from fhy_core.symbolic.expression import (
    BinaryOperation,
    LiteralExpression,
)
from fhy_core.symbolic.param import create_interval_integer_param
from fhy_core.symbolic.param.core import (
    _bound_from_literal,
    _invert_comparison,
    _normalize_natural_bound,
)

from .conftest import mock_identifier

# =============================================================================
# `_invert_comparison`
# =============================================================================


@pytest.mark.parametrize(
    "input_op, expected_op",
    [
        pytest.param(
            BinaryOperation.GREATER, BinaryOperation.LESS, id="greater-becomes-less"
        ),
        pytest.param(
            BinaryOperation.GREATER_EQUAL,
            BinaryOperation.LESS_EQUAL,
            id="greater-equal-becomes-less-equal",
        ),
        pytest.param(
            BinaryOperation.LESS, BinaryOperation.GREATER, id="less-becomes-greater"
        ),
        pytest.param(
            BinaryOperation.LESS_EQUAL,
            BinaryOperation.GREATER_EQUAL,
            id="less-equal-becomes-greater-equal",
        ),
    ],
)
def test_invert_binary_comparison_operation_inverts_each_comparison_operator(
    input_op: BinaryOperation, expected_op: BinaryOperation
) -> None:
    """Test the helper inverts each of the four comparison operators."""
    assert _invert_comparison(input_op) == expected_op


def test_invert_binary_comparison_operation_rejects_non_comparison_operator() -> None:
    """Test the helper raises ``ValueError`` for a non-comparison operator."""
    with pytest.raises(ValueError, match="non-comparison"):
        _invert_comparison(BinaryOperation.ADD)


# =============================================================================
# `_bound_from_literal`
# =============================================================================


@pytest.mark.parametrize(
    "op, expected_is_lower, expected_inclusive",
    [
        pytest.param(BinaryOperation.GREATER, True, False, id="gt"),
        pytest.param(BinaryOperation.GREATER_EQUAL, True, True, id="ge"),
        pytest.param(BinaryOperation.LESS, False, False, id="lt"),
        pytest.param(BinaryOperation.LESS_EQUAL, False, True, id="le"),
    ],
)
def test_get_bound_from_expression_decodes_each_comparison_operator(
    op: BinaryOperation, expected_is_lower: bool, expected_inclusive: bool
) -> None:
    """Test the helper decodes each comparison operator into a bound triple."""
    is_lower, value, inclusive = _bound_from_literal(LiteralExpression(7), op)

    assert is_lower is expected_is_lower
    assert value == 7
    assert inclusive is expected_inclusive


def test_get_bound_from_expression_rejects_non_int_literal() -> None:
    """Test the helper raises ``RuntimeError`` for a non-``int`` literal value."""
    with pytest.raises(RuntimeError):
        _bound_from_literal(LiteralExpression(1.5), BinaryOperation.GREATER)


# =============================================================================
# `_normalize_natural_bound` - equivalent bound spellings collapse together
#
# An exclusive fractional upper bound, an inclusive fractional upper bound,
# and an inclusive integer upper bound at the fractional bound's floor all
# admit exactly the same integers, so they must normalize to the same
# ``(integer bound, inclusive)`` gate value.
# =============================================================================


@pytest.mark.parametrize(
    "bound, is_inclusive",
    [
        pytest.param(2.5, False, id="x-less-than-2.5"),
        pytest.param(2.5, True, id="x-less-equal-2.5"),
        pytest.param(2, True, id="x-less-equal-2"),
    ],
)
def test_normalize_natural_bound_collapses_equivalent_upper_bound_spellings(
    bound: int | float, is_inclusive: bool
) -> None:
    """Test equivalent upper-bound spellings normalize to the same gate value."""
    normalized = _normalize_natural_bound(
        bound, is_lower=False, is_inclusive=is_inclusive
    )

    assert normalized == (2, True)


# =============================================================================
# Defensive guards in ``_iter_interval_bounds``
#
# These guards are unreachable through the public API because
# ``validate_constraint`` rejects every malformed input that would surface
# them. Inject malformed state via ``object.__setattr__`` to drive each branch.
# =============================================================================


def _build_interval_param_with_injected_constraints(
    constraints: tuple[Any, ...],
) -> Any:
    """Build an interval-integer param with an injected tuple of constraints."""
    param = create_interval_integer_param()
    object.__setattr__(param, "constraints", constraints)
    return param


def test_bound_int_param_iter_bounds_rejects_non_equation_constraint_in_state() -> None:
    """Test ``_iter_interval_bounds`` raises for a non-equation constraint in state."""
    param = _build_interval_param_with_injected_constraints(
        (InSetConstraint(mock_identifier("x", 1), {1}),)
    )

    with pytest.raises(RuntimeError):
        param + 0  # test: must raise


def test_bound_int_param_iter_bounds_rejects_non_bound_expression_in_state() -> None:
    """Test ``_iter_interval_bounds`` raises for a non-bound expression in state."""
    var = mock_identifier("x", 1)
    bad = EquationConstraint(var, LiteralExpression(0))
    param = _build_interval_param_with_injected_constraints((bad,))

    with pytest.raises(RuntimeError, match="non-bound"):
        param + 0  # test: must raise
