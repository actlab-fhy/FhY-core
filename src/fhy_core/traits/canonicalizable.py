"""`Canonicalizable` trait."""

__all__ = ["Canonicalizable"]

from typing import Protocol, runtime_checkable

from fhy_core.utils import Self


@runtime_checkable
class Canonicalizable(Protocol):
    """Protocol for objects with a canonical local form.

    The return value discriminates by mutability, which is what lets the
    trait compose with both frozen IR nodes and mutable containers:

    - Immutable (``Frozen``) objects cannot self-modify, so they return a
      new canonicalized copy.
    - Mutable objects canonicalize in place and return ``None``.

    The caller-side idiom is uniform: the canonical object is
    ``obj.canonicalize() or obj``, the returned copy when there is one,
    otherwise the now-canonicalized receiver.
    """

    def canonicalize(self) -> "Self | None":
        """Canonicalize this object.

        Returns:
            A new canonicalized copy for frozen objects; ``None`` for
            mutable objects, which canonicalize in place.
        """
