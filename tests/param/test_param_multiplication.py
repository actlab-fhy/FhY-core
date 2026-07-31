"""Tests for interval-integer parameter multiplication (``__mul__``/``__rmul__``).

Multiplication mirrors ``__add__``/``__sub__``: it is defined only for
interval-integer parameters, reuses the same coercion machinery, and
preserves ``non_negative`` only when both operands are non-negative. The
tricky part is the sign matrix and unbounded-end propagation, which these
tests pin down against hand-computed extended-integer products.
"""

from typing import Any

import pytest
from hypothesis import given  # type: ignore[import-not-found]
from hypothesis import strategies as st

from fhy_core.constraint import EquationConstraint
from fhy_core.param import (
    Param,
    ParamError,
    create_categorical_param,
    create_integer_param,
    create_integer_param_between,
    create_interval_integer_param,
    create_interval_integer_param_between,
    create_interval_integer_param_exactly,
    create_interval_integer_param_with_lower_bound,
    create_interval_integer_param_with_upper_bound,
    create_interval_natural_param,
    create_real_param,
)

from .conftest import (
    assert_all_satisfied,
    assert_none_satisfied,
    assert_param_round_trips_in_all_formats,
    mock_identifier,
)

# =============================================================================
# Sign matrix
# =============================================================================


def test_multiplication_of_positive_intervals_is_positive_product() -> None:
    """Test ``[2,3] * [4,5] -> [8,15]``."""
    x = create_interval_integer_param_between(2, 3)
    y = create_interval_integer_param_between(4, 5)

    z = x * y

    assert_all_satisfied(z, [8, 15])
    assert_none_satisfied(z, [7, 16])


def test_multiplication_of_mixed_sign_interval_uses_min_from_extreme_product() -> None:
    """Test ``[-2,3] * [4,5] -> [-10,15]``.

    The minimum comes from ``-2 * 5``, not from multiplying the two smallest
    magnitudes.
    """
    x = create_interval_integer_param_between(-2, 3)
    y = create_interval_integer_param_between(4, 5)

    z = x * y

    assert_all_satisfied(z, [-10, 15])
    assert_none_satisfied(z, [-11, 16])


def test_multiplication_of_two_negative_intervals_flips_to_positive() -> None:
    """Test ``[-2,-1] * [-3,-1] -> [1,6]``: negative times negative flips sign."""
    x = create_interval_integer_param_between(-2, -1)
    y = create_interval_integer_param_between(-3, -1)

    z = x * y

    assert_all_satisfied(z, [1, 6])
    assert_none_satisfied(z, [0, 7])


# =============================================================================
# Unbounded-end propagation
# =============================================================================


def test_multiplication_with_half_bounded_positive_operand_propagates_lower_bound() -> (
    None
):
    """Test ``[1,+inf) * [2,3] -> [2,+inf)``."""
    x = create_interval_integer_param_with_lower_bound(1)
    y = create_interval_integer_param_between(2, 3)

    z = x * y

    assert_all_satisfied(z, [2, 3, 10**6])
    assert_none_satisfied(z, [1])


def test_multiplication_with_fully_unbounded_operand_stays_fully_unbounded() -> None:
    """Test ``(-inf,+inf) * [2,3] -> (-inf,+inf)``."""
    x = create_interval_integer_param()
    y = create_interval_integer_param_between(2, 3)

    z = x * y

    assert_all_satisfied(z, [-(10**6), 0, 10**6])


def test_multiplication_of_zero_singleton_with_unbounded_operand_is_zero() -> None:
    """Test ``[0,0] * (-inf,+inf) -> [0,0]``: zero absorbs an unbounded operand."""
    x = create_interval_integer_param_exactly(0)
    y = create_interval_integer_param()

    z = x * y

    assert_all_satisfied(z, [0])
    assert_none_satisfied(z, [-1, 1])


def test_multiplication_of_non_negative_interval_with_unbounded_below_operand() -> None:
    """Test ``[0,5] * (-inf,0] -> (-inf,0]``."""
    x = create_interval_integer_param_between(0, 5)
    y = create_interval_integer_param_with_upper_bound(0)

    z = x * y

    assert_all_satisfied(z, [0, -(10**6)])
    assert_none_satisfied(z, [1])


# =============================================================================
# Scalar operands
# =============================================================================


def test_multiplication_with_int_scalar_on_right_scales_interval() -> None:
    """Test multiplication with a plain ``int`` on the right scales the interval."""
    x = create_interval_integer_param_between(3, 5)

    z = x * 2

    assert_all_satisfied(z, [6, 10])
    assert_none_satisfied(z, [5, 11])


def test_multiplication_with_int_scalar_on_left_scales_interval() -> None:
    """Test multiplication with a plain ``int`` on the left scales the interval."""
    x = create_interval_integer_param_between(3, 5)

    z = 2 * x

    assert_all_satisfied(z, [6, 10])
    assert_none_satisfied(z, [5, 11])


def test_multiplication_with_bool_operand_raises_type_error() -> None:
    """Test multiplication rejects a ``bool`` operand (``bool`` is not ``int`` here)."""
    x = create_interval_integer_param_between(0, 1)

    with pytest.raises(TypeError):
        _ = x * True


def test_multiplication_with_bool_on_left_raises_type_error() -> None:
    """Test reflected multiplication rejects a ``bool`` operand on the left."""
    x = create_interval_integer_param_between(0, 1)

    with pytest.raises(TypeError):
        _ = True * x


# =============================================================================
# Unsupported operand kinds
# =============================================================================


def test_multiplication_with_unsupported_type_raises_type_error() -> None:
    """Test multiplication raises ``TypeError`` for an unsupported operand type."""
    x = create_interval_integer_param_between(0, 1)

    with pytest.raises(TypeError):
        _ = x * "nope"


def test_multiplication_with_real_param_raises_type_error() -> None:
    """Test multiplication raises ``TypeError`` against a real-valued parameter."""
    x = create_interval_integer_param_between(0, 1)
    y = create_real_param()

    with pytest.raises(TypeError):
        _ = x * y


def test_multiplication_with_categorical_param_raises_type_error() -> None:
    """Test multiplication raises ``TypeError`` against a categorical parameter."""
    x = create_interval_integer_param_between(0, 1)
    y = create_categorical_param({1, 2})

    with pytest.raises(TypeError):
        _ = x * y


# =============================================================================
# Coercion of a plain integer parameter (non-interval-integer)
# =============================================================================


def test_multiplication_accepts_plain_integer_param_on_right() -> None:
    """Test multiplication of interval-integer param with a plain integer param."""
    x = create_interval_integer_param_between(3, 5)
    y = create_integer_param_between(2, 3)

    z = x * y

    assert_all_satisfied(z, [6, 15])
    assert_none_satisfied(z, [5, 16])


def test_multiplication_accepts_plain_integer_param_on_left() -> None:
    """Test multiplication with a plain integer param on the left (via reflection)."""
    x = create_interval_integer_param_between(3, 5)
    y = create_integer_param_between(2, 3)

    z = y * x

    assert_all_satisfied(z, [6, 15])
    assert_none_satisfied(z, [5, 16])


def test_multiplication_rejects_integer_param_with_non_bound_constraint() -> None:
    """Test multiplication rejects an integer operand with a non-bound constraint."""
    integer = create_integer_param()
    integer = integer.add_constraint(
        EquationConstraint(
            integer.variable, (integer.variable_expression % 5).equals(0)
        )
    )
    bound = create_interval_integer_param_exactly(2)

    with pytest.raises(TypeError):
        _ = bound * integer


# =============================================================================
# Class preservation: `non_negative` and `zero_included`
# =============================================================================


def test_multiplication_keeps_non_negative_when_both_operands_are() -> None:
    """Test ``non_negative`` is preserved when both operands are non-negative."""
    x = create_interval_natural_param(zero_included=False)
    x = x.add_lower_bound_constraint(1).add_upper_bound_constraint(3)
    y = create_interval_natural_param(zero_included=False)
    y = y.add_lower_bound_constraint(1).add_upper_bound_constraint(3)

    z = x * y

    assert isinstance(z.domain, type(x.domain))
    assert z.domain.non_negative  # type: ignore[attr-defined]


def test_multiplication_drops_non_negative_when_only_one_operand_is() -> None:
    """Test ``non_negative`` is dropped when only one operand is non-negative."""
    x = create_interval_natural_param()
    x = x.add_upper_bound_constraint(3)
    y = create_interval_integer_param_between(-2, 2)

    z = x * y

    assert not z.domain.non_negative  # type: ignore[attr-defined]


def test_multiplication_zero_included_is_or_of_operands() -> None:
    """Test the multiplication-specific ``zero_included = left or right`` rule.

    ``x`` is non-negative and excludes zero (``[1,3]``); ``y`` is non-negative
    and includes zero (``[0,3]``). Their product can be zero (``1 * 0``), so
    the result must include zero even though ``x`` alone would not.
    """
    x = create_interval_natural_param(zero_included=False)
    x = x.add_lower_bound_constraint(1).add_upper_bound_constraint(3)
    y = create_interval_natural_param(zero_included=True)
    y = y.add_upper_bound_constraint(3)

    z = x * y

    assert z.domain.non_negative  # type: ignore[attr-defined]
    assert z.domain.zero_included  # type: ignore[attr-defined]
    assert_all_satisfied(z, [0])


def test_multiplication_zero_included_false_when_both_operands_exclude_zero() -> None:
    """Test ``zero_included`` stays ``False`` when neither operand admits zero."""
    x = create_interval_natural_param(zero_included=False)
    x = x.add_lower_bound_constraint(1).add_upper_bound_constraint(3)
    y = create_interval_natural_param(zero_included=False)
    y = y.add_lower_bound_constraint(1).add_upper_bound_constraint(3)

    z = x * y

    assert z.domain.non_negative  # type: ignore[attr-defined]
    assert not z.domain.zero_included  # type: ignore[attr-defined]


# =============================================================================
# `prefer_inclusive` rendering
# =============================================================================


def test_multiplication_rendering_follows_left_operand_prefer_inclusive() -> None:
    """Test the result's ``str`` follows the left operand's `prefer_inclusive`."""
    x_incl = create_interval_integer_param_between(2, 3, prefer_inclusive=True)
    y_incl = create_interval_integer_param_between(4, 5, prefer_inclusive=True)
    x_excl = create_interval_integer_param_between(2, 3, prefer_inclusive=False)
    y_excl = create_interval_integer_param_between(4, 5, prefer_inclusive=False)

    z_incl = x_incl * y_incl
    z_excl = x_excl * y_excl

    for v in range(0, 20):
        assert z_incl.is_constraints_satisfied(v) == z_excl.is_constraints_satisfied(v)
    assert str(z_incl) != str(z_excl)


# =============================================================================
# Empty operand
# =============================================================================


def test_multiplication_with_empty_operand_raises_param_error() -> None:
    """Test multiplication raises ``ParamError`` when an operand is empty."""
    x = create_interval_integer_param()
    x = x.add_lower_bound_constraint(10).add_upper_bound_constraint(5)
    y = create_interval_integer_param_between(1, 2)

    with pytest.raises(ParamError):
        _ = x * y


# =============================================================================
# Result variable reuse
# =============================================================================


def test_multiplication_result_reuses_left_operand_variable() -> None:
    """Test the multiplication result reuses the left operand's variable."""
    x = create_interval_integer_param_between(2, 3, name=mock_identifier("x", 1))
    y = create_interval_integer_param_between(4, 5, name=mock_identifier("y", 2))

    z = x * y

    assert z.variable is x.variable


def test_rmul_result_reuses_coerced_operand_variable() -> None:
    """Test ``int * Param`` reuses the (only) interval operand's variable."""
    x = create_interval_integer_param_between(2, 3, name=mock_identifier("x", 1))

    z = 2 * x

    assert z.variable is x.variable


# =============================================================================
# Brute-force property check
# =============================================================================


@pytest.mark.parametrize(
    "lower, upper",
    [
        pytest.param(0, 0, id="0-0"),
        pytest.param(0, 3, id="0-3"),
        pytest.param(-3, 3, id="neg3-3"),
        pytest.param(-3, -1, id="neg3-neg1"),
        pytest.param(1, 3, id="1-3"),
    ],
)
def test_multiplication_matches_brute_force_corner_products_over_bounded_intervals(
    lower: int, upper: int
) -> None:
    """Test multiplication's satisfied range matches the four-corner hull.

    Interval multiplication is an interval-hull operation, not exact set
    multiplication: for ``[a,b] * [c,d]`` the result is
    ``[min(ac,ad,bc,bd), max(ac,ad,bc,bd)]``. That hull can (and, self-
    multiplied over more than two values, generally does) admit integers
    that are not the product of any actual pair -- e.g. ``[1,3] * [1,3]``
    admits ``5``, which is not ``a * b`` for any ``a, b`` in ``{1,2,3}``.
    So this test checks the satisfied range against the corner-product hull
    directly, not against the brute-force set of actual products.
    """
    corners = [lower * lower, lower * upper, upper * lower, upper * upper]
    expected_min, expected_max = min(corners), max(corners)
    x = create_interval_integer_param_between(lower, upper)
    y = create_interval_integer_param_between(lower, upper)

    z = x * y

    for v in range(expected_min - 2, expected_max + 3):
        assert z.is_constraints_satisfied(v) == (expected_min <= v <= expected_max)


def test_signature_accepts_no_keyword_only_params_for_mul() -> None:
    """Test ``__mul__``/``__rmul__`` take a single positional operand (dunder shape)."""
    x = create_interval_integer_param_between(1, 2)
    y: Any = create_interval_integer_param_between(1, 2)

    assert isinstance(x.__mul__(y), Param)
    assert isinstance(x.__rmul__(2), Param)


# =============================================================================
# Integration: serialization, subset/feasibility/assign interop, chaining
# =============================================================================


def test_multiplication_result_round_trips_through_serialization() -> None:
    """Test a multiplication result round-trips through DICT, JSON, and BINARY."""
    x = create_interval_integer_param_between(2, 3)
    y = create_interval_integer_param_between(4, 5)

    z = x * y

    assert_param_round_trips_in_all_formats(z)


def test_multiplication_result_interoperates_with_is_subset() -> None:
    """Test ``[2,3] * [4,5]``'s result is a subset of a wider hand-built interval."""
    x = create_interval_integer_param_between(2, 3)
    y = create_interval_integer_param_between(4, 5)
    wider = create_interval_integer_param_between(0, 20)

    z = x * y

    assert z.is_subset(wider)
    assert not wider.is_subset(z)


def test_multiplication_result_interoperates_with_is_feasible() -> None:
    """Test a non-empty multiplication result reports feasible."""
    x = create_interval_integer_param_between(2, 3)
    y = create_interval_integer_param_between(4, 5)

    z = x * y

    assert z.is_feasible()
    assert not z.is_empty()


def test_multiplication_result_interoperates_with_assign() -> None:
    """Test a value in the product interval can be assigned to the result."""
    x = create_interval_integer_param_between(2, 3)
    y = create_interval_integer_param_between(4, 5)

    z = x * y
    assignment = z.assign(10)

    assert assignment.value == 10
    with pytest.raises(ParamError):
        z.assign(7)


def test_chained_addition_then_multiplication() -> None:
    """Test ``(a + b) * c`` chains interval addition into interval multiplication."""
    a = create_interval_integer_param_between(1, 2)
    b = create_interval_integer_param_between(3, 4)
    c = create_interval_integer_param_between(2, 2)

    result = (a + b) * c

    # a + b -> [4, 6]; (a + b) * c -> [8, 12].
    assert_all_satisfied(result, [8, 12])
    assert_none_satisfied(result, [7, 13])


# =============================================================================
# Property: soundness of interval multiplication
# =============================================================================


@pytest.mark.property
@given(  # type: ignore[untyped-decorator]
    bound_1=st.integers(min_value=-25, max_value=25),
    bound_2=st.integers(min_value=-25, max_value=25),
    bound_3=st.integers(min_value=-25, max_value=25),
    bound_4=st.integers(min_value=-25, max_value=25),
    data=st.data(),
)
def test_multiplication_is_sound_for_every_concrete_pair_in_range(
    bound_1: int, bound_2: int, bound_3: int, bound_4: int, data: st.DataObject
) -> None:
    """Test that for any concrete ``x in [a,b]``, ``y in [c,d]``, ``x*y`` is valid.

    The interval-product result must admit every actual product of a
    concrete value drawn from each operand's interval -- this is the
    defining soundness property of interval arithmetic.
    """
    lower_1, upper_1 = sorted((bound_1, bound_2))
    lower_2, upper_2 = sorted((bound_3, bound_4))
    x = create_interval_integer_param_between(lower_1, upper_1)
    y = create_interval_integer_param_between(lower_2, upper_2)
    concrete_x = data.draw(st.integers(min_value=lower_1, max_value=upper_1))
    concrete_y = data.draw(st.integers(min_value=lower_2, max_value=upper_2))

    z = x * y

    assert z.is_constraints_satisfied(concrete_x * concrete_y)
