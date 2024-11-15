"""Array utilities."""

__all__ = ["get_array_size_in_bits"]

from collections.abc import Sequence
from math import prod


def get_array_size_in_bits(shape: Sequence[int], element_size_in_bits: int) -> int:
    """Return the size of an array in bits."""
    return prod(shape) * element_size_in_bits
