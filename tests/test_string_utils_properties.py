"""Hypothesis property tests for `format_comma_separated_list`.

Covers: every item's stringified form appears exactly once, in order; the
number of separators is `len(items) - 1` (no conjunction is ever inserted,
so no adjustment applies); and empty and singleton inputs behave as the
docstring implies.
"""

from collections.abc import Callable

import pytest

pytest.importorskip("hypothesis")

from hypothesis import example, given
from hypothesis import strategies as st

from fhy_core.utils.str_utils import format_comma_separated_list

pytestmark = pytest.mark.property


@example(items=[1, 2, 3], add_space=True)
@example(items=[1, 2, 3], add_space=False)
@example(items=[], add_space=True)
@example(items=[1], add_space=True)
@given(
    items=st.lists(
        st.integers(min_value=-1000, max_value=1000), min_size=0, max_size=8
    ),
    add_space=st.booleans(),
)
def test_every_item_appears_exactly_once_in_order(
    items: list[int], add_space: bool
) -> None:
    """Test each item's str() appears exactly once, in the original order.

    Oracle: splitting the joined result back on the separator reconstructs
    the same sequence of stringified items, since str(int) never contains a
    comma.
    """
    result = format_comma_separated_list(items, str_func=str, add_space=add_space)
    join_char = ", " if add_space else ","
    parts = result.split(join_char) if items else []
    assert parts == [str(item) for item in items]


@given(
    items=st.lists(
        st.integers(min_value=-1000, max_value=1000), min_size=0, max_size=8
    ),
    add_space=st.booleans(),
)
def test_separator_count_is_length_minus_one(items: list[int], add_space: bool) -> None:
    """Test the number of separators equals max(len(items) - 1, 0).

    Oracle: a joined list of n items has exactly n - 1 separators (0 for an
    empty or singleton list); the function inserts no conjunction, so no
    adjustment applies. str(int) never contains a comma, so counting
    top-level separators is exact.
    """
    result = format_comma_separated_list(items, str_func=str, add_space=add_space)
    join_char = ", " if add_space else ","
    expected_separator_count = max(len(items) - 1, 0)
    assert result.count(join_char) == expected_separator_count


@given(str_func=st.sampled_from([str, repr]), add_space=st.booleans())
def test_empty_iterable_yields_empty_string(
    str_func: Callable[[int], str], add_space: bool
) -> None:
    """Test formatting an empty iterable yields the empty string, for any str_func."""
    assert format_comma_separated_list([], str_func=str_func, add_space=add_space) == ""


@given(item=st.integers(min_value=-1000, max_value=1000), add_space=st.booleans())
def test_singleton_iterable_yields_bare_str_func_output(
    item: int, add_space: bool
) -> None:
    """Test a singleton iterable yields exactly str_func(item), with no separator."""
    assert format_comma_separated_list(
        [item], str_func=str, add_space=add_space
    ) == str(item)
