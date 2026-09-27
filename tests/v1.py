"""Helpers for the tests that pin the deprecated V1 wire format.

V1 is written only inside ``wire_version(WireVersion.V1)``, and reading or
writing it warns with ``DeprecationWarning`` (slice S17 of
``docs/design/python-switch.md``, D-S17-16). The tests that pin V1 payloads
use these helpers, which silence that warning, and are deleted with V1.
"""

import contextlib
import warnings
from collections.abc import Iterator

from fhy_core.serialization import _READING_V1, WireVersion, wire_version

__all__ = ["reading_v1", "writing_v1"]


@contextlib.contextmanager
def writing_v1() -> Iterator[None]:
    """Write, and read, V1 inside the block without the deprecation warning.

    Every payload read inside the block is read as V1, as the payloads a V1
    payload nests are.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with wire_version(WireVersion.V1):
            token = _READING_V1.set(True)
            try:
                yield
            finally:
                _READING_V1.reset(token)


@contextlib.contextmanager
def reading_v1() -> Iterator[None]:
    """Read V1 inside the block without the deprecation warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield
