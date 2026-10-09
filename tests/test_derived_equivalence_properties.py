"""Property test for ``DerivedEquivalenceMixin`` against a hand-written oracle.

Over four small synthetic dataclasses exercising a value field, a
reference-identifier field, a sequence field, and an optional field, the
mixin's derived ``is_structurally_equivalent`` must agree with a
hand-written comparator written independently of the field-walk under test
(``==`` for value fields, identifier ``id`` for reference fields, elementwise
comparison for sequences, and "``None`` only equals ``None``" for optionals).
"""

from dataclasses import dataclass, field

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.term.derived_equivalence import (
    DerivedEquivalenceMixin,
    compared_as_reference,
    compared_as_value,
)

from .strategies.identifiers import build_identifier_pool, build_identifier_strategy

pytestmark = pytest.mark.property

_POOL = build_identifier_pool(4, name_prefix="r")


# =============================================================================
# Synthetic dataclasses under test
# =============================================================================


@dataclass(frozen=True, eq=False)
class _ValueLeaf(DerivedEquivalenceMixin):
    """A leaf compared purely by value: a short tag and a small count."""

    tag: str = field(metadata=compared_as_value())
    count: int = field(metadata=compared_as_value())


@dataclass(frozen=True, eq=False)
class _ReferenceHolder(DerivedEquivalenceMixin):
    """A node holding a reference identifier alongside a plain value field."""

    identifier: Identifier = field(metadata=compared_as_reference())
    label: str = field(metadata=compared_as_value())


@dataclass(frozen=True, eq=False)
class _SequenceHolder(DerivedEquivalenceMixin):
    """A node holding a homogeneous tuple of value leaves."""

    items: tuple[_ValueLeaf, ...]


@dataclass(frozen=True, eq=False)
class _OptionalHolder(DerivedEquivalenceMixin):
    """A node holding an optional value leaf."""

    maybe: _ValueLeaf | None = None


# =============================================================================
# Strategies
# =============================================================================


@st.composite
def draw_value_leaf(draw: st.DrawFn) -> _ValueLeaf:
    """Draw a ``_ValueLeaf`` with a short tag and a small count."""
    tag = draw(st.text(alphabet="abc", min_size=0, max_size=3))
    count = draw(st.integers(min_value=-10, max_value=10))
    return _ValueLeaf(tag, count)


@st.composite
def draw_reference_holder(draw: st.DrawFn) -> _ReferenceHolder:
    """Draw a ``_ReferenceHolder`` over the shared identifier pool."""
    identifier = draw(build_identifier_strategy(_POOL))
    label = draw(st.text(alphabet="xyz", min_size=0, max_size=3))
    return _ReferenceHolder(identifier, label)


@st.composite
def draw_sequence_holder(draw: st.DrawFn) -> _SequenceHolder:
    """Draw a ``_SequenceHolder`` with zero to three value leaves."""
    items = draw(st.lists(draw_value_leaf(), min_size=0, max_size=3))
    return _SequenceHolder(tuple(items))


@st.composite
def draw_optional_holder(draw: st.DrawFn) -> _OptionalHolder:
    """Draw an ``_OptionalHolder`` that is sometimes ``None``, sometimes a leaf."""
    maybe = draw(st.one_of(st.none(), draw_value_leaf()))
    return _OptionalHolder(maybe)


# =============================================================================
# Hand-written oracle comparators (independent of DerivedEquivalenceMixin)
# =============================================================================


def compute_value_leaf_equivalence(left: _ValueLeaf, right: _ValueLeaf) -> bool:
    """Return the hand-written equivalence of two ``_ValueLeaf`` instances.

    Value fields compare by ``==``.
    """
    return left.tag == right.tag and left.count == right.count


def compute_reference_holder_equivalence(
    left: _ReferenceHolder, right: _ReferenceHolder
) -> bool:
    """Return the hand-written equivalence of two ``_ReferenceHolder`` instances.

    A reference field compares by identifier ``id``, not object identity or
    ``==`` on the mock.
    """
    return left.identifier.id == right.identifier.id and left.label == right.label


def compute_sequence_holder_equivalence(
    left: _SequenceHolder, right: _SequenceHolder
) -> bool:
    """Return the hand-written equivalence of two ``_SequenceHolder`` instances.

    Sequences compare elementwise, and only when the same length.
    """
    if len(left.items) != len(right.items):
        return False
    return all(
        compute_value_leaf_equivalence(one, other)
        for one, other in zip(left.items, right.items, strict=True)
    )


def compute_optional_holder_equivalence(
    left: _OptionalHolder, right: _OptionalHolder
) -> bool:
    """Return the hand-written equivalence of two ``_OptionalHolder`` instances.

    ``None`` compares equivalent only to ``None``: one side ``None`` and the
    other not is never equivalent, matching the mixin's documented
    ``left is right`` shortcut for a ``None`` pair.
    """
    if left.maybe is None or right.maybe is None:
        return left.maybe is right.maybe
    return compute_value_leaf_equivalence(left.maybe, right.maybe)


# =============================================================================
# Properties
# =============================================================================


@given(pair=st.tuples(draw_value_leaf(), draw_value_leaf()))
def test_value_leaf_equivalence_matches_the_hand_written_comparator(
    pair: tuple[_ValueLeaf, _ValueLeaf],
) -> None:
    """Test derived equivalence agrees with the hand-written value comparator."""
    left, right = pair
    assert left.is_structurally_equivalent(right) == compute_value_leaf_equivalence(
        left, right
    )


@given(pair=st.tuples(draw_reference_holder(), draw_reference_holder()))
def test_reference_holder_equivalence_matches_the_hand_written_comparator(
    pair: tuple[_ReferenceHolder, _ReferenceHolder],
) -> None:
    """Test derived equivalence agrees with the hand-written reference comparator."""
    left, right = pair
    assert left.is_structurally_equivalent(
        right
    ) == compute_reference_holder_equivalence(left, right)


@given(pair=st.tuples(draw_sequence_holder(), draw_sequence_holder()))
def test_sequence_holder_equivalence_matches_the_hand_written_comparator(
    pair: tuple[_SequenceHolder, _SequenceHolder],
) -> None:
    """Test derived equivalence agrees with the hand-written elementwise comparator."""
    left, right = pair
    assert left.is_structurally_equivalent(
        right
    ) == compute_sequence_holder_equivalence(left, right)


@given(pair=st.tuples(draw_optional_holder(), draw_optional_holder()))
def test_optional_holder_equivalence_matches_the_hand_written_comparator(
    pair: tuple[_OptionalHolder, _OptionalHolder],
) -> None:
    """Test derived equivalence agrees with the hand-written None-aware comparator."""
    left, right = pair
    assert left.is_structurally_equivalent(
        right
    ) == compute_optional_holder_equivalence(left, right)
