"""Hypothesis strategies for pools of mock identifiers.

Every identifier a strategy in this package produces is a
``tests.conftest.mock_identifier``, never a real
``fhy_core.identifier.Identifier``. Mock ids drawn here start at
``MOCK_IDENTIFIER_ID_BASE`` (10000), a range no native constant occupies,
so ``mock_identifier`` never raises ``MockIdentifierAliasError`` for a
pool this module builds.
"""

from collections.abc import Sequence
from typing import Final

from hypothesis import strategies as st

from fhy_core.identifier import Identifier

from ..conftest import mock_identifier

__all__ = [
    "MOCK_IDENTIFIER_ID_BASE",
    "build_identifier_pool",
    "build_identifier_strategy",
]

MOCK_IDENTIFIER_ID_BASE: Final = 10_000
"""First id a pool built here assigns; no native constant reaches this high."""


def build_identifier_pool(size: int, name_prefix: str = "v") -> tuple[Identifier, ...]:
    """Return a deterministic pool of mock identifiers.

    Args:
        size: Number of identifiers to build.
        name_prefix: Prefix for each identifier's name hint.

    Returns:
        Mock identifiers with ids ``MOCK_IDENTIFIER_ID_BASE`` through
        ``MOCK_IDENTIFIER_ID_BASE + size - 1`` and name hints
        ``f"{name_prefix}{index}"``, in index order.

    """
    return tuple(
        mock_identifier(f"{name_prefix}{index}", MOCK_IDENTIFIER_ID_BASE + index)
        for index in range(size)
    )


def build_identifier_strategy(
    pool: Sequence[Identifier],
) -> st.SearchStrategy[Identifier]:
    """Return a strategy sampling identifiers from an existing pool.

    Args:
        pool: Non-empty sequence of identifiers to sample from.

    Returns:
        A strategy drawing one identifier from ``pool``.

    """
    return st.sampled_from(pool)
