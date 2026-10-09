"""Hypothesis property tests for `invert_dict` and `invert_frozen_dict`.

Covers two laws: inversion is an involution when the input dict is
injective (unique keys zipped with unique values, by construction), and for
a non-injective dict, inversion keeps exactly one entry per distinct value,
matching the `{v: k for k, v in d.items()}` construction (a later key
overwrites an earlier one that maps to the same value).
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import example, given
from hypothesis import strategies as st
from immutabledict import immutabledict

from fhy_core.utils.dict_utils import invert_dict, invert_frozen_dict

pytestmark = pytest.mark.property

_KEY_ALPHABET = "abcdefgh"


@st.composite
def draw_injective_dict(draw: st.DrawFn) -> dict[str, int]:
    """Draw a dict with unique keys and unique values, so inversion is bijective.

    Keys and values are each drawn as separately-unique lists of the same
    length and zipped pairwise, guaranteeing injectivity by construction
    (no `.filter()` needed).
    """
    size = draw(st.integers(min_value=0, max_value=6))
    keys = draw(
        st.lists(
            st.text(alphabet=_KEY_ALPHABET, min_size=1, max_size=3),
            min_size=size,
            max_size=size,
            unique=True,
        )
    )
    values = draw(
        st.lists(
            st.integers(min_value=-100, max_value=100),
            min_size=size,
            max_size=size,
            unique=True,
        )
    )
    return dict(zip(keys, values, strict=True))


@st.composite
def draw_dict_with_duplicate_values(draw: st.DrawFn) -> dict[str, int]:
    """Draw a dict whose values may repeat, to exercise invert_dict's collision rule.

    Unique keys are zipped against values drawn from a value range much
    smaller than the key count, so duplicate values are frequent without a
    `.filter()`.
    """
    keys = draw(
        st.lists(
            st.text(alphabet=_KEY_ALPHABET, min_size=1, max_size=2),
            min_size=0,
            max_size=6,
            unique=True,
        )
    )
    values = draw(
        st.lists(
            st.integers(min_value=0, max_value=2),
            min_size=len(keys),
            max_size=len(keys),
        )
    )
    return dict(zip(keys, values, strict=True))


@example(d={"a": 1, "b": 2, "c": 3})
@given(d=draw_injective_dict())
def test_invert_dict_is_an_involution_on_injective_dicts(d: dict[str, int]) -> None:
    """Test invert_dict(invert_dict(d)) == d when d is injective.

    Oracle: dict inversion is a bijection when d has no two keys sharing a
    value, so inverting twice must reconstruct the original mapping.
    """
    assert invert_dict(invert_dict(d)) == d


@example(d={"a": 1, "b": 2, "c": 3})
@given(d=draw_injective_dict())
def test_invert_frozen_dict_is_an_involution_on_injective_dicts(
    d: dict[str, int],
) -> None:
    """Test invert_frozen_dict(invert_frozen_dict(d)) == d when d is injective."""
    frozen = immutabledict(d)
    assert invert_frozen_dict(invert_frozen_dict(frozen)) == frozen


@given(d=draw_dict_with_duplicate_values())
def test_invert_dict_keeps_one_entry_per_distinct_value(d: dict[str, int]) -> None:
    """Test invert_dict has exactly one entry per distinct value of d.

    Oracle: the set of distinct values in d, independent of invert_dict's
    own comprehension.
    """
    inverted = invert_dict(d)
    assert set(inverted.keys()) == set(d.values())
    assert len(inverted) == len(set(d.values()))


@given(d=draw_dict_with_duplicate_values())
def test_invert_frozen_dict_keeps_one_entry_per_distinct_value(
    d: dict[str, int],
) -> None:
    """Test invert_frozen_dict has exactly one entry per distinct value of d.

    Mirrors test_invert_dict_keeps_one_entry_per_distinct_value:
    invert_frozen_dict shares invert_dict's underlying comprehension.
    """
    inverted = invert_frozen_dict(immutabledict(d))
    assert set(inverted.keys()) == set(d.values())
    assert len(inverted) == len(set(d.values()))
