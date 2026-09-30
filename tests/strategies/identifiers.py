"""Hypothesis strategies for pools of identifiers.

Every identifier a pool here holds is a real
``fhy_core.identifier.Identifier`` with a fixed id, restored through
``Identifier.deserialize_from_dict``, which never issues an id from the
counter (it only advances the counter past the id), so a pool is
deterministic and a hypothesis example replays with the same ids. The ids
start at ``MOCK_IDENTIFIER_ID_BASE`` (10000), a range no native constant
occupies, so no pool identifier is a native constant's canonical
identifier.

An identifier compares and hashes by id alone, so two pools share an
identifier wherever their id ranges overlap. Boolean-sorted identifiers
therefore come from their own pool, whose ids start at
``BOOLEAN_MOCK_IDENTIFIER_ID_BASE`` (15000), clear of every integer pool
of up to 5000 identifiers.
"""

from collections.abc import Sequence
from typing import Final

from hypothesis import strategies as st

from fhy_core.identifier import Identifier

__all__ = [
    "BOOLEAN_MOCK_IDENTIFIER_ID_BASE",
    "MOCK_IDENTIFIER_ID_BASE",
    "build_boolean_identifier_pool",
    "build_identifier_pool",
    "build_identifier_strategy",
]

MOCK_IDENTIFIER_ID_BASE: Final = 10_000
"""First id a pool built here assigns; no native constant reaches this high."""

BOOLEAN_MOCK_IDENTIFIER_ID_BASE: Final = 15_000
"""First id a Boolean pool assigns, clear of the ids an integer pool uses."""


def _restore_identifier(name_hint: str, identifier_id: int) -> Identifier:
    """Return the identifier with the fixed id ``identifier_id``."""
    return Identifier.deserialize_from_dict(
        {"id": identifier_id, "name_hint": name_hint}
    )


def build_identifier_pool(size: int, name_prefix: str = "v") -> tuple[Identifier, ...]:
    """Return a deterministic pool of identifiers.

    Args:
        size: Number of identifiers to build.
        name_prefix: Prefix for each identifier's name hint.

    Returns:
        Identifiers with ids ``MOCK_IDENTIFIER_ID_BASE`` through
        ``MOCK_IDENTIFIER_ID_BASE + size - 1`` and name hints
        ``f"{name_prefix}{index}"``, in index order.

    """
    return tuple(
        _restore_identifier(f"{name_prefix}{index}", MOCK_IDENTIFIER_ID_BASE + index)
        for index in range(size)
    )


def build_boolean_identifier_pool(
    size: int, name_prefix: str = "b"
) -> tuple[Identifier, ...]:
    """Return a deterministic pool of identifiers for Boolean-sorted variables.

    The ids never overlap those of a :func:`build_identifier_pool` pool of
    up to 5000 identifiers, so a tree may draw from both pools and keep
    every Boolean identifier distinct from every integer one.

    Args:
        size: Number of identifiers to build.
        name_prefix: Prefix for each identifier's name hint.

    Returns:
        Identifiers with ids ``BOOLEAN_MOCK_IDENTIFIER_ID_BASE``
        through ``BOOLEAN_MOCK_IDENTIFIER_ID_BASE + size - 1`` and name
        hints ``f"{name_prefix}{index}"``, in index order.

    """
    return tuple(
        _restore_identifier(
            f"{name_prefix}{index}", BOOLEAN_MOCK_IDENTIFIER_ID_BASE + index
        )
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
