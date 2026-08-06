"""Hypothesis property tests for interval-integer parameter multiplication.

Split out of ``test_param_multiplication.py`` so a test environment without
``hypothesis`` installed (the CI ``tests`` lane syncs only the ``test``
dependency group) can still collect the ordinary unit tests there; this
module is skipped wholesale via the ``importorskip`` below.
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.symbolic.param import create_interval_integer_param_between

pytestmark = pytest.mark.property


# =============================================================================
# Property: soundness of interval multiplication
# =============================================================================


@given(
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
