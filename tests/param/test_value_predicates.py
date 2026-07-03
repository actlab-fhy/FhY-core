"""Tests for the strict value-matching predicate and its domain-level effects.

`do_param_values_match` is the central membership predicate for every finite
parameter domain. It treats ``bool``, ``int``, and ``float`` as mutually
disjoint value kinds and ``str`` as distinct, so ``True`` never matches ``1`` and
``1`` never matches ``1.0`` even though Python considers them ``==``. These tests
exercise the predicate directly and confirm the four domain builders inherit the
strict semantics.
"""

import pytest

from fhy_core.param.domains import (
    build_categorical_domain,
    build_ordinal_domain,
    build_permutation_domain,
)
from fhy_core.param.values import do_param_values_match

# =============================================================================
# `do_param_values_match` strictness
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
def test_do_param_values_match_rejects_cross_kind_numeric_values(
    candidate: object, allowed: object
) -> None:
    """Test the predicate reports cross-kind numeric values as non-matching.

    ``bool``, ``int``, and ``float`` are mutually disjoint, so equal-valued
    numbers of different kinds must not match.
    """
    assert not do_param_values_match(candidate, allowed)


@pytest.mark.parametrize(
    ("candidate", "allowed"),
    [
        pytest.param(1, 1, id="int-int"),
        pytest.param(1.0, 1.0, id="float-float"),
        pytest.param(True, True, id="bool-bool"),
        pytest.param("a", "a", id="str-str"),
    ],
)
def test_do_param_values_match_accepts_same_kind_equal_values(
    candidate: object, allowed: object
) -> None:
    """Test the predicate reports equal same-kind values as matching."""
    assert do_param_values_match(candidate, allowed)


def test_do_param_values_match_rejects_unequal_same_kind_values() -> None:
    """Test the predicate reports unequal same-kind values as non-matching."""
    assert not do_param_values_match(1, 2)
    assert not do_param_values_match("a", "b")


def test_do_param_values_match_treats_str_as_distinct_from_numbers() -> None:
    """Test a ``str`` never matches a numeric value of equal textual form."""
    assert not do_param_values_match("1", 1)
    assert not do_param_values_match(1, "1")


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
