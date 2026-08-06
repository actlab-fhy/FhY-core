"""Tests for round-trip serialization of the parameter value domains.

These exercise the wrapped-family serialization envelope at the domain level,
independent of any composing `Param`, so a failure localizes to the domain
family rather than the parameter container.
"""

import pytest

from fhy_core.symbolic.param.domains import (
    IntegerDomain,
    IntervalIntegerDomain,
    ParamDomain,
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


def test_categorical_domain_round_trip_preserves_bool_int_distinction() -> None:
    """Test a mixed bool/int categorical domain round-trips both kinds distinctly."""
    domain = build_categorical_domain((True, 1))

    restored = ParamDomain.deserialize_from_dict(domain.serialize_to_dict())

    assert restored.is_value_admissible(True)
    assert restored.is_value_admissible(1)
