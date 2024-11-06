"""Dictionary manipulation utilities."""

__all__ = ["invert_dict", "invert_frozen_dict"]

from typing import TypeVar

from frozendict import frozendict

K = TypeVar("K")
V = TypeVar("V")


def invert_dict(d: dict[K, V]) -> dict[V, K]:
    """Return a dictionary with keys and values swapped."""
    return {v: k for k, v in d.items()}


def invert_frozen_dict(d: frozendict[K, V]) -> frozendict[V, K]:
    """Return a frozen dictionary with keys and values swapped."""
    return frozendict({v: k for k, v in d.items()})
