"""Property tests of the expression binding's big integers.

Ints cross into the core as bytes, not decimal text, so an int of any size
is a literal and decodes.
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.symbolic.expression import LiteralExpression

from ...strategies.settings import cap_max_examples

# An int of up to 20,000 digits, built from bounded draws so each example
# stays within the generator's budget: a head of up to 300 digits shifted by
# up to 19,700 digits, plus a tail of up to 300 digits.
_HUGE_INTS = st.builds(
    lambda head, shift, tail: head * 10**shift + tail,
    st.integers(min_value=-(10**300), max_value=10**300),
    st.integers(min_value=0, max_value=19_700),
    st.integers(min_value=0, max_value=10**300),
)


@pytest.mark.property
@given(value=_HUGE_INTS)
@cap_max_examples(60)
def test_any_int_up_to_twenty_thousand_digits_round_trips(value: int) -> None:
    """Test an int of up to 20,000 digits crosses into the core and back."""
    literal = LiteralExpression(value)

    assert literal.value == value
    assert LiteralExpression.from_json(literal.to_json()).value == value
