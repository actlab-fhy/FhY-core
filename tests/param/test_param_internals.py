"""Tests for private helpers in `fhy_core.param.core` and `fhy_core.param.bound`.

The helpers exercised here cover validation paths that the public-API tests
cannot easily reach because the public constructors and validators reject
malformed inputs before they propagate. Each test calls the private helper
directly, mirroring the convention established in
`tests/param/test_nat_param_helpers.py`.
"""

from typing import Any

import pytest

from fhy_core.constraint import EquationConstraint, InSetConstraint
from fhy_core.expression import (
    BinaryOperation,
    LiteralExpression,
)
from fhy_core.identifier import Identifier
from fhy_core.param import BoundIntParam
from fhy_core.param.bound import (
    _get_bound_from_expression,
    _invert_binary_comparison_operation,
)
from fhy_core.param.core import (
    ParamError,
    _serialize_typed_wrapped_leaf_value,
)

# =============================================================================
# `_invert_binary_comparison_operation`
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
    assert _invert_binary_comparison_operation(input_op) == expected_op


def test_invert_binary_comparison_operation_rejects_non_comparison_operator() -> None:
    """Test the helper raises `ValueError` for a non-comparison operator."""
    with pytest.raises(ValueError, match="non-comparison"):
        _invert_binary_comparison_operation(BinaryOperation.ADD)


# =============================================================================
# `_get_bound_from_expression`
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
    is_lower, value, inclusive = _get_bound_from_expression(LiteralExpression(7), op)
    assert is_lower is expected_is_lower
    assert value == 7
    assert inclusive is expected_inclusive


def test_get_bound_from_expression_rejects_non_int_literal() -> None:
    """Test the helper raises `RuntimeError` for a non-`int` literal value."""
    with pytest.raises(RuntimeError, match="integer LiteralExpression"):
        _get_bound_from_expression(LiteralExpression(1.5), BinaryOperation.GREATER)


# =============================================================================
# `_serialize_typed_wrapped_leaf_value`
# =============================================================================


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="bool"),
        pytest.param(1, id="int"),
        pytest.param(1.5, id="float"),
        pytest.param("text", id="str"),
        pytest.param(Identifier("x"), id="serializable"),
    ],
)
def test_serialize_typed_wrapped_leaf_value_accepts_each_supported_type(
    value: Any,
) -> None:
    """Test the helper serializes each supported leaf-value type without raising."""
    _serialize_typed_wrapped_leaf_value(value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([1, 2, 3], id="list"),
        pytest.param({1: 2}, id="dict"),
        pytest.param(object(), id="opaque-object"),
    ],
)
def test_serialize_typed_wrapped_leaf_value_rejects_unsupported_type(
    value: Any,
) -> None:
    """Test the helper raises `ParamError` for a value of an unsupported type."""
    with pytest.raises(ParamError, match="serializable leaf"):
        _serialize_typed_wrapped_leaf_value(value)


# =============================================================================
# Defensive guards in `_iter_bounds`
#
# These guards are unreachable through the public API because `validate_constraint`
# rejects every malformed input that would surface them. Inject malformed state
# via ``object.__setattr__`` to drive each branch.
# =============================================================================


def _build_bound_int_param_with_injected_constraints(
    constraints: tuple[Any, ...],
) -> BoundIntParam:
    """Build a `BoundIntParam` with an arbitrary tuple of injected constraints."""
    param = BoundIntParam()
    object.__setattr__(param, "_constraints", constraints)
    return param


def test_bound_int_param_iter_bounds_rejects_non_equation_constraint_in_state() -> None:
    """Test `_iter_bounds` raises `RuntimeError` for a non-`EquationConstraint`."""
    param = _build_bound_int_param_with_injected_constraints(
        (InSetConstraint(Identifier("x"), {1}),)
    )
    with pytest.raises(RuntimeError, match="non-EquationConstraint"):
        param + 0  # noqa: B015  # test: must raise


def test_bound_int_param_iter_bounds_rejects_non_bound_expression_in_state() -> None:
    """Test `_iter_bounds` raises `RuntimeError` for a non-bound expression."""
    var = Identifier("x")
    bad = EquationConstraint(var, LiteralExpression(0))
    param = _build_bound_int_param_with_injected_constraints((bad,))
    with pytest.raises(RuntimeError, match="non-bound"):
        param + 0  # noqa: B015  # test: must raise
