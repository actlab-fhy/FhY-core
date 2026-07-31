"""Hypothesis property tests for `create_intersection_param`.

Split out of ``test_param_intersection.py`` so a test environment without
``hypothesis`` installed (the CI ``tests`` lane syncs only the ``test``
dependency group) can still collect the ordinary unit tests there; this
module is skipped wholesale via the ``importorskip`` below.
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import assume, given  # type: ignore[import-not-found]
from hypothesis import strategies as st

from fhy_core.symbolic.param import create_intersection_param, create_ordinal_param

pytestmark = pytest.mark.property


# =============================================================================
# Property: finite-set membership law
# =============================================================================


@given(  # type: ignore[untyped-decorator]
    left_values=st.sets(st.integers(min_value=0, max_value=12), min_size=1, max_size=6),
    right_values=st.sets(
        st.integers(min_value=0, max_value=12), min_size=1, max_size=6
    ),
    candidate=st.integers(min_value=0, max_value=15),
)
def test_intersection_membership_law_holds_for_random_ordinal_sets(
    left_values: set[int], right_values: set[int], candidate: int
) -> None:
    """Test a value is valid for the intersection iff valid for both operands.

    Holds for arbitrary (non-empty) ordinal value sets; when the sets happen
    to be disjoint the intersection is empty, which `create_intersection_param`
    signals by raising `ParamError` rather than returning a param -- covered
    separately by the disjoint-set unit tests, so this property assumes a
    non-empty intersection here to keep the assertion meaningful.
    """
    assume(left_values & right_values)
    left = create_ordinal_param(sorted(left_values))
    right = create_ordinal_param(sorted(right_values))

    result = create_intersection_param(left, right)

    expected = candidate in left_values and candidate in right_values
    assert result.is_value_valid(candidate) == expected
