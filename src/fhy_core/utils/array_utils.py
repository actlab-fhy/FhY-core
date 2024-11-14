"""Array utilities."""

__all__ = ["get_array_size_in_bits"]

from collections.abc import Sequence


def get_array_size_in_bits(shape: Sequence[int], element_size_in_bits: int) -> int:
    """Return the size of an array in bits."""
    size = 1
    for dim in shape:
        size *= dim
    return size * element_size_in_bits
