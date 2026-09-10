"""Tests that finite-domain constructors self-enforce their invariants.

The ``build_*_domain`` factories are thin delegators; the non-empty, uniqueness,
type, and canonical-ordering invariants live in each domain's ``__post_init__``,
so directly constructing a domain cannot produce an invalid or non-canonical
instance.

Also pins the idempotence of ``normalize_value`` across every built-in
domain kind.
"""

from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.symbolic.param import ParamError
from fhy_core.symbolic.param.domains import (
    CategoricalDomain,
    IntegerDomain,
    IntervalIntegerDomain,
    OrdinalDomain,
    ParamDomain,
    PermutationDomain,
    RealDomain,
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


@pytest.mark.parametrize(
    "values",
    [(1, True), (1, 1.0)],
    ids=["int-and-bool", "int-and-float"],
)
def test_ordinal_domain_canonical_order_ignores_construction_order(
    values: tuple[Any, ...],
) -> None:
    """Test values that compare equal get an order independent of construction.

    The ascending sort cannot separate ``1`` from ``True`` or from ``1.0``, so
    without a tiebreak the stored tuple would simply keep whichever order the
    caller passed and the same value set would have two canonical forms. Compares
    the member types, because the tuples themselves compare equal either way.
    """
    forward = OrdinalDomain(values).sorted_values
    reverse = OrdinalDomain(tuple(reversed(values))).sorted_values

    assert [type(value) for value in forward] == [type(value) for value in reverse]


def test_ordinal_domain_treats_int_and_float_as_distinct_kinds() -> None:
    """Test ``1`` and ``1.0`` are distinct members but two ``1``s are duplicates."""
    assert OrdinalDomain((1, 1.0)).sorted_values == (1, 1.0)
    with pytest.raises(ParamError):
        OrdinalDomain((1, 1))


_BUILT_IN_DOMAIN_VALUES: list[tuple[ParamDomain, Any]] = [
    (IntegerDomain(), 3),
    (RealDomain(), 2.5),
    (IntervalIntegerDomain(), 4),
    (OrdinalDomain((1, 2, 3)), 2),
    (CategoricalDomain(("a", "b")), "a"),
    (PermutationDomain(("a", "b", "c")), ["c", "a", "b"]),
]


@pytest.mark.parametrize(
    ("domain", "value"),
    _BUILT_IN_DOMAIN_VALUES,
    ids=[
        "integer",
        "real",
        "interval-integer",
        "ordinal",
        "categorical",
        "permutation",
    ],
)
def test_normalize_value_is_idempotent(domain: ParamDomain, value: Any) -> None:
    """Test normalizing an already canonical value yields an equal value."""
    canonical = domain.normalize_value(value)

    assert domain.normalize_value(canonical) == canonical
