"""Lexical scope utility.

A ``Scope`` is a stack of frames with shadowing lookup: the mechanics that
``SymbolTable`` implements by hand, generalized so other IRs that introduce
lexical scoping do not reinvent them. A fresh scope has a single root frame;
``push``/``pop`` (or the ``scope`` context manager) nest and un-nest frames in
LIFO order, ``define`` writes into the innermost frame, and ``lookup`` walks
frames innermost-to-outermost so an inner binding shadows an outer one.
"""

__all__ = ["Scope"]

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Generic, TypeVar

from .stack import Stack

_K = TypeVar("_K")
_V = TypeVar("_V")


class Scope(Generic[_K, _V]):
    """A stack of lexical frames with shadowing lookup.

    Frames are pushed and popped in LIFO order. A fresh scope starts with one
    root frame that cannot be popped. ``define`` writes into the innermost
    frame; ``lookup`` searches innermost-to-outermost so an inner binding
    shadows an outer one with the same key.
    """

    _frames: Stack[dict[_K, _V]]

    def __init__(self) -> None:
        self._frames = Stack[dict[_K, _V]]()
        self._frames.push({})

    def push(self) -> None:
        """Push a new, empty innermost frame."""
        self._frames.push({})

    def pop(self) -> None:
        """Pop the innermost frame.

        Raises:
            IndexError: If only the root frame remains; the root frame
                cannot be popped.
        """
        if len(self._frames) <= 1:
            raise IndexError("Cannot pop the root scope frame.")
        self._frames.pop()

    def scope(self) -> AbstractContextManager["Scope[_K, _V]"]:
        """Return a context manager that pushes a frame and pops it on exit.

        The frame is popped on exit even if the ``with`` body raises.

        Returns:
            A context manager yielding this same scope with one extra frame
            pushed.
        """
        return self._scoped()

    @contextmanager
    def _scoped(self) -> Iterator["Scope[_K, _V]"]:
        self.push()
        try:
            yield self
        finally:
            self.pop()

    def define(self, key: _K, value: _V) -> None:
        """Bind ``key`` to ``value`` in the innermost frame.

        Overwrites a binding for ``key`` already in the innermost frame and
        shadows any binding for ``key`` in an outer frame.
        """
        self._frames.peek()[key] = value

    def lookup(self, key: _K) -> _V:
        """Return the value bound to ``key``, searching innermost-to-outermost.

        Raises:
            KeyError: If ``key`` is bound in no frame.
        """
        for frame in reversed(list(self._frames)):
            if key in frame:
                return frame[key]
        raise KeyError(key)

    def lookup_local(self, key: _K) -> _V:
        """Return the value bound to ``key`` in the innermost frame only.

        Raises:
            KeyError: If ``key`` is not bound in the innermost frame.
        """
        return self._frames.peek()[key]

    def is_defined(self, key: _K) -> bool:
        """Return whether ``key`` is bound in any frame."""
        return any(key in frame for frame in self._frames)

    def is_defined_local(self, key: _K) -> bool:
        """Return whether ``key`` is bound in the innermost frame."""
        return key in self._frames.peek()

    def get_depth(self) -> int:
        """Return the number of frames, including the root (always >= 1)."""
        return len(self._frames)
