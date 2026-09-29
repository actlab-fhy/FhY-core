"""Tests for strict value matching and its domain-level effects.

Every finite parameter domain matches values type-strictly. It treats
``bool``, ``int``, and ``float`` as mutually disjoint value kinds and ``str``
as distinct, so ``True`` never matches ``1`` and ``1`` never matches ``1.0``
even though Python considers them ``==``. These tests exercise the matching
through a one-value domain's admissibility, cover the index-wise sequence
form through the equivalence of ordered domains, and confirm the domain
builders inherit the strict semantics. The matching runs in the Rust core;
its rules are pinned here and in
``rust/fhy-core/tests/it/param/domain_stories.rs``.
"""

from typing import Any

import pytest

from fhy_core.symbolic.param.domains import (
    PermutationDomain,
    build_categorical_domain,
    build_ordinal_domain,
    build_permutation_domain,
)


def _matches(candidate: object, allowed: Any) -> bool:
    """Return whether a domain holding only `allowed` admits `candidate`."""
    return build_ordinal_domain((allowed,)).is_value_admissible(candidate)


def _match_in_order(own: tuple[Any, ...], other: tuple[Any, ...]) -> bool:
    """Return whether two permutation domains hold the same members in order."""
    return PermutationDomain(own).is_structurally_equivalent(PermutationDomain(other))


# =============================================================================
# Strict matching
# =============================================================================


@pytest.mark.parametrize(
    ("candidate", "allowed"),
    [
        pytest.param(1, 1.0, id="int-float"),
        pytest.param(1.0, 1, id="float-int"),
        pytest.param(True, 1, id="bool-int"),
        pytest.param(1, True, id="int-bool"),
        pytest.param(False, 0, id="bool-int-zero"),
        pytest.param(True, 1.0, id="bool-float"),
    ],
)
def test_matching_rejects_cross_kind_numeric_values(
    candidate: object, allowed: object
) -> None:
    """Test matching reports cross-kind numeric values as non-matching.

    ``bool``, ``int``, and ``float`` are mutually disjoint, so equal-valued
    numbers of different kinds must not match.
    """
    assert not _matches(candidate, allowed)


@pytest.mark.parametrize(
    ("candidate", "allowed"),
    [
        pytest.param(1, 1, id="int-int"),
        pytest.param(1.0, 1.0, id="float-float"),
        pytest.param(True, True, id="bool-bool"),
        pytest.param("a", "a", id="str-str"),
    ],
)
def test_matching_accepts_same_kind_equal_values(
    candidate: object, allowed: object
) -> None:
    """Test matching reports equal same-kind values as matching."""
    assert _matches(candidate, allowed)


def test_matching_rejects_unequal_same_kind_values() -> None:
    """Test matching reports unequal same-kind values as non-matching."""
    assert not _matches(1, 2)
    assert not _matches("a", "b")


def test_matching_treats_str_as_distinct_from_numbers() -> None:
    """Test a ``str`` never matches a numeric value of equal textual form."""
    assert not _matches("1", 1)
    assert not _matches(1, "1")


# =============================================================================
# Ordered matching
# =============================================================================


def test_ordered_matching_accepts_position_wise_equal_sequences() -> None:
    """Test two sequences holding the same values at the same positions match."""
    assert _match_in_order((1, 2, "a"), (1, 2, "a"))


def test_ordered_matching_rejects_sequences_of_different_lengths() -> None:
    """Test a prefix does not match the longer sequence it is a prefix of."""
    assert not _match_in_order((1, 2), (1, 2, 3))
    assert not _match_in_order((1, 2, 3), (1, 2))


def test_ordered_matching_rejects_cross_kind_value_at_a_position() -> None:
    """Test one kind-distinct position makes the whole sequence non-matching."""
    assert not _match_in_order((1, 2), (True, 2))
    assert not _match_in_order((1, 2), (1.0, 2))


def test_ordered_matching_is_order_sensitive() -> None:
    """Test the same values in a different order do not match."""
    assert not _match_in_order((1, 2), (2, 1))


# =============================================================================
# Domain-level acceptance criteria
# =============================================================================


def test_build_ordinal_domain_treats_int_and_float_as_distinct() -> None:
    """Test an ordinal domain accepts ``1`` and ``1.0`` as two distinct values."""
    domain = build_ordinal_domain((1, 1.0))

    assert len(domain.sorted_values) == 2


def test_ordinal_domain_of_ints_does_not_admit_equal_float() -> None:
    """Test an integer ordinal domain rejects the equal-valued float ``1.0``."""
    domain = build_ordinal_domain((1, 2, 3))

    assert not domain.is_value_admissible(1.0)


def test_build_categorical_domain_treats_bool_and_int_as_distinct() -> None:
    """Test a categorical domain accepts ``True`` and ``1`` as two distinct values."""
    domain = build_categorical_domain((True, 1))

    assert len(domain.categories) == 2


def test_categorical_domain_of_ints_does_not_admit_float() -> None:
    """Test an integer categorical domain rejects the float ``1.0``.

    ``float`` is not a categorical value kind, so ``1.0`` is inadmissible
    regardless of numeric-kind strictness.
    """
    domain = build_categorical_domain((1, 2, 3))

    assert not domain.is_value_admissible(1.0)


def test_build_permutation_domain_treats_int_and_float_as_distinct() -> None:
    """Test a permutation domain accepts ``1`` and ``1.0`` as two distinct members."""
    domain = build_permutation_domain((1, 1.0))

    assert len(domain.ordered_members) == 2
