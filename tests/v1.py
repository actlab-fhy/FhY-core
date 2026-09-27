"""Helpers for the tests that pin the deprecated V1 wire format.

V1 is written only inside ``wire_version(WireVersion.V1)``, and reading or
writing it warns with ``DeprecationWarning`` (slice S17 of
``docs/design/python-switch.md``, D-S17-16). The tests that pin V1 payloads
use these helpers, which silence that warning, and are deleted with V1.

The suite turns an unmarked V1 warning into an error (``filterwarnings`` in
``pyproject.toml``), so a test that reads V1 on purpose says so: with
``@reads_v1``, these helpers, the ``v1_wire`` fixture, or
``pytest.warns(DeprecationWarning, match="V1 wire format")`` where the
warning is the point.
"""

import contextlib
import warnings
from collections.abc import Iterator

import pytest

from fhy_core.serialization import _READING_V1, WireVersion, wire_version

__all__ = ["V1_WARNING_FILTER", "reading_v1", "reads_v1", "writing_v1"]

V1_WARNING_FILTER = "ignore:.*V1 wire format.*:DeprecationWarning"
"""The ``filterwarnings`` entry that silences the V1 deprecation warnings."""

reads_v1 = pytest.mark.filterwarnings(V1_WARNING_FILTER)
"""Mark a test that reads V1 payloads on purpose, silencing their warning."""


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
