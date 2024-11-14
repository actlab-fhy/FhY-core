"""Tests the general array utilities."""

import pytest
from fhy_core.utils.array_utils import get_array_size_in_bits


@pytest.mark.parametrize(
    "shape, element_size_in_bits, expected",
    [
        ([], 8, 8),  # Empty shape is a scalar
        ([1, 2, 3], 8, 48),
        ([4, 5, 6], 16, 1920),
        ([7, 8, 9], 32, 16128),
        ([10, 11, 12], 64, 84480),
    ],
)
def test_get_array_size_in_bits(
    shape: list[int], element_size_in_bits: int, expected: int
):
    """Test calculating the array size in bits is correct."""
    assert get_array_size_in_bits(shape, element_size_in_bits) == expected
