"""Tests for round-trip serialization of the parameter value domains.

These exercise the wrapped-family serialization envelope at the domain level,
independent of any composing `Param`, so a failure localizes to the domain
family rather than the parameter container.
"""

from typing import Any

import pytest
from immutabledict import immutabledict

from fhy_core.symbolic.param.domains import (
    CategoricalDomain,
    IntegerDomain,
    IntervalIntegerDomain,
    OrdinalDomain,
    ParamDomain,
    PermutationDomain,
    RealDomain,
    build_categorical_domain,
    build_ordinal_domain,
    build_permutation_domain,
)


@pytest.mark.parametrize(
    "domain",
    [
        IntegerDomain(),
        IntegerDomain(non_negative=True, zero_included=True),
        IntegerDomain(non_negative=True, zero_included=False),
        RealDomain(),
        IntervalIntegerDomain(),
        IntervalIntegerDomain(
            prefer_inclusive=False, non_negative=True, zero_included=False
        ),
        build_ordinal_domain((1, 2, 3)),
        build_ordinal_domain((1, 1.0)),
        build_categorical_domain((True, 1, "a")),
        build_permutation_domain((1, 2, 3)),
    ],
)
def test_domain_round_trips_through_family_serialization(domain: ParamDomain) -> None:
    """Test each domain round-trips via the wrapped-family serializer.

    Deserialization dispatches on the wrapped envelope's type tag, so the
    restored value must be the same concrete leaf and structurally equivalent.
    """
    restored = ParamDomain.deserialize_from_dict(domain.serialize_to_dict())

    assert type(restored) is type(domain)
    assert domain.is_structurally_equivalent(restored)
    assert restored.is_structurally_equivalent(domain)


@pytest.mark.parametrize(
    "values",
    [(1, True), (1, 1.0)],
    ids=["int-and-bool", "int-and-float"],
)
def test_ordinal_domain_serializes_the_same_value_set_identically(
    values: tuple[Any, ...],
) -> None:
    """Test an ordinal domain's serialized form does not depend on construction order.

    ``1`` and ``True`` (or ``1.0``) compare equal, so the ascending sort alone
    leaves their positions to the caller. The wrapped payload records each value's
    kind, so without a canonical-order tiebreak one value set would have two
    serialized forms.
    """
    forward = build_ordinal_domain(values).serialize_to_dict()
    reverse = build_ordinal_domain(tuple(reversed(values))).serialize_to_dict()

    assert forward == reverse


def test_categorical_domain_round_trip_preserves_bool_int_distinction() -> None:
    """Test a mixed bool/int categorical domain round-trips both kinds distinctly."""
    domain = build_categorical_domain((True, 1))

    restored = ParamDomain.deserialize_from_dict(domain.serialize_to_dict())

    assert restored.is_value_admissible(True)
    assert restored.is_value_admissible(1)


# =============================================================================
# construct_from_fields accepts any Mapping, not only dict
# =============================================================================


def test_ordinal_domain_construct_from_fields_accepts_an_immutabledict() -> None:
    """Test `OrdinalDomain`'s reconstruction hook accepts an `immutabledict`."""
    domain = build_ordinal_domain((1, 2, 3))
    fields = immutabledict({"sorted_values": domain.sorted_values})

    rebuilt = OrdinalDomain.construct_from_fields(fields)

    assert rebuilt.is_structurally_equivalent(domain)


def test_categorical_domain_construct_from_fields_accepts_an_immutabledict() -> None:
    """Test `CategoricalDomain`'s reconstruction hook accepts an `immutabledict`."""
    domain = build_categorical_domain((True, 1, "a"))
    fields = immutabledict({"categories": domain.categories})

    rebuilt = CategoricalDomain.construct_from_fields(fields)

    assert rebuilt.is_structurally_equivalent(domain)


def test_permutation_domain_construct_from_fields_accepts_an_immutabledict() -> None:
    """Test `PermutationDomain`'s reconstruction hook accepts an `immutabledict`."""
    domain = build_permutation_domain((1, 2, 3))
    fields = immutabledict({"ordered_members": domain.ordered_members})

    rebuilt = PermutationDomain.construct_from_fields(fields)

    assert rebuilt.is_structurally_equivalent(domain)
