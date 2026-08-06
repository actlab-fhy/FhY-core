"""Tests that finite-domain constructors self-enforce their invariants.

The ``build_*_domain`` factories are thin delegators; the non-empty, uniqueness,
type, and canonical-ordering invariants live in each domain's ``__post_init__``,
so directly constructing a domain cannot produce an invalid or non-canonical
instance.
"""

from collections.abc import Callable

import pytest

from fhy_core.symbolic.param import ParamError
from fhy_core.symbolic.param.domains import (
    CategoricalDomain,
    OrdinalDomain,
    ParamDomain,
    PermutationDomain,
)

_FINITE_DOMAINS: list[Callable[..., ParamDomain]] = [
    OrdinalDomain,
    CategoricalDomain,
    PermutationDomain,
]


def test_ordinal_domain_sorts_values_on_construction() -> None:
    """Test a directly-constructed ordinal domain stores its values sorted."""
    assert OrdinalDomain((3, 1, 2)).sorted_values == (1, 2, 3)


def test_categorical_domain_canonicalizes_order_on_construction() -> None:
    """Test direct construction yields an order-independent categorical domain."""
    assert (
        CategoricalDomain(("b", "a")).categories
        == CategoricalDomain(("a", "b")).categories
    )


@pytest.mark.parametrize("constructor", _FINITE_DOMAINS)
def test_finite_domain_rejects_empty_values(
    constructor: Callable[..., ParamDomain],
) -> None:
    """Test each finite domain rejects an empty value set on construction."""
    with pytest.raises(ParamError):
        constructor(())


@pytest.mark.parametrize("constructor", _FINITE_DOMAINS)
def test_finite_domain_rejects_duplicate_values(
    constructor: Callable[..., ParamDomain],
) -> None:
    """Test each finite domain rejects duplicate values on construction."""
    with pytest.raises(ParamError):
        constructor((1, 2, 1))


def test_ordinal_domain_treats_int_and_float_as_distinct_kinds() -> None:
    """Test ``1`` and ``1.0`` are distinct members but two ``1``s are duplicates."""
    assert OrdinalDomain((1, 1.0)).sorted_values == (1, 1.0)
    with pytest.raises(ParamError):
        OrdinalDomain((1, 1))
