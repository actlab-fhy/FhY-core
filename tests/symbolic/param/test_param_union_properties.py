"""Hypothesis property tests for `create_union_param`.

Split out of ``test_param_union.py`` so a test environment without
``hypothesis`` installed (the CI ``tests`` lane syncs only the ``test``
dependency group) can still collect the ordinary unit tests there; this
module is skipped wholesale via the ``importorskip`` below.
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings
from hypothesis import strategies as st

from fhy_core.symbolic.param import create_ordinal_param, create_union_param

pytestmark = pytest.mark.property

# The property runs without a hypothesis deadline. Every example takes about
# a millisecond and no draw is filtered, so a deadline here would time only
# the scheduler: on a contended machine one example can be descheduled for
# hundreds of milliseconds, failing a membership law for a reason unrelated
# to it.


# =============================================================================
# Property: finite-set membership law
# =============================================================================


@settings(deadline=None)
@given(
    left_values=st.sets(st.integers(min_value=0, max_value=12), min_size=1, max_size=6),
    right_values=st.sets(
        st.integers(min_value=0, max_value=12), min_size=1, max_size=6
    ),
    candidate=st.integers(min_value=0, max_value=15),
)
def test_union_membership_law_holds_for_random_ordinal_sets(
    left_values: set[int], right_values: set[int], candidate: int
) -> None:
    """Test a value is valid for the union iff valid for either operand.

    Holds for arbitrary (non-empty) ordinal value sets, not just the fixed
    examples above.
    """
    left = create_ordinal_param(sorted(left_values))
    right = create_ordinal_param(sorted(right_values))

    result = create_union_param(left, right)

    expected = candidate in left_values or candidate in right_values
    assert result.is_value_valid(candidate) == expected
