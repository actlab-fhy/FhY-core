"""Tests for interval-natural parameters."""

import pytest

from fhy_core.serialization import (
    DeserializationDictStructureError,
)
from fhy_core.symbolic.param import (
    IntervalIntegerDomain,
    Param,
    create_interval_integer_param_between,
    create_interval_natural_param,
)

from .conftest import assert_all_satisfied, assert_none_satisfied, mock_identifier


def _create_interval_natural_param_between(
    lower: int,
    upper: int,
    *,
    prefer_inclusive: bool = True,
    zero_included: bool = True,
) -> Param[int]:
    """Create an interval-natural param bounded to ``[lower, upper]``."""
    p = create_interval_natural_param(
        prefer_inclusive=prefer_inclusive, zero_included=zero_included
    )
    p = p.add_lower_bound_constraint(lower, is_inclusive=True)
    return p.add_upper_bound_constraint(upper, is_inclusive=True)


# =============================================================================
# Keyword-only signatures
# =============================================================================


def test_bound_nat_param_init_accepts_post_marker_args_as_keywords() -> None:
    """Test ``create_interval_natural_param`` accepts keyword args."""
    create_interval_natural_param(
        name=mock_identifier("x", 1), zero_included=False, prefer_inclusive=False
    )


def test_bound_nat_param_init_rejects_name_passed_positionally() -> None:
    """Test ``create_interval_natural_param`` rejects ``name`` passed positionally.

    The factory signature is ``create_interval_natural_param(*, name=None, ...)``,
    all keyword-only. Passing a positional argument raises ``TypeError``.
    """
    with pytest.raises(TypeError):
        create_interval_natural_param(mock_identifier("x", 1))  # type: ignore[misc]  # test: keyword-only


# =============================================================================
# Default-flag invariants
# =============================================================================


def test_bound_nat_param_init_defaults_to_zero_included_true() -> None:
    """Test ``create_interval_natural_param()`` defaults to ``zero_included=True``."""
    assert create_interval_natural_param().is_value_valid(0)


def test_bound_nat_param_init_defaults_to_prefer_inclusive_true() -> None:
    """Test ``create_interval_natural_param()`` defaults to ``prefer_inclusive=True``.

    Constructs the param via the bare factory (no bound-specific helper) so
    the default is the *only* default in play. Verifies via the constraint
    form produced by an arithmetic operation.
    """
    bounded = create_interval_natural_param().add_lower_bound_constraint(3)

    assert ">=" in str(bounded + 1)


# =============================================================================
# Serialization
# =============================================================================


def test_bound_nat_param_serialization_round_trip_preserves_constraints() -> None:
    """Test interval-natural param round-trips through dict serialization."""
    p = create_interval_natural_param().add_lower_bound_constraint(2, is_inclusive=True)

    dictionary = p.serialize_to_dict()
    restored: Param[int] = Param.deserialize_from_dict(dictionary)
    redictionary = restored.serialize_to_dict()

    assert len(dictionary["constraint_system"]["__data__"]["constraints"]) == 2  # type: ignore[index]  # test: dict shape known
    assert_all_satisfied(restored, [2, 5, 100])
    assert_none_satisfied(restored, [0, 1])
    assert len(redictionary["constraint_system"]["__data__"]["constraints"]) == 2  # type: ignore[index]  # test: dict shape known


def test_bound_nat_param_deserialize_round_trip_preserves_zero_excluded_flag() -> None:
    """Test ``create_interval_natural_param(zero_included=False)`` round-trips.

    The ``zero_included`` flag is carried explicitly in the domain envelope, so
    a round-trip reproduces a structurally-equivalent param.
    """
    original = create_interval_natural_param(zero_included=False)

    restored: Param[int] = Param.deserialize_from_dict(original.serialize_to_dict())

    assert original.is_structurally_equivalent(restored)


def test_bound_nat_param_deserialize_rejects_payload_with_malformed_domain_data() -> (
    None
):
    """Test ``Param.deserialize_from_dict`` rejects malformed bound_nat domain data.

    A domain envelope whose ``__data__`` omits the interval-integer flags fails
    the derived structure check for ``IntervalIntegerDomain``.
    """
    payload = create_interval_natural_param().serialize_to_dict()
    payload["domain"]["__data__"] = {}  # type: ignore[index]  # test: modify serialized

    with pytest.raises(DeserializationDictStructureError):
        Param.deserialize_from_dict(payload)


def test_bound_nat_param_deserialize_recovers_zero_exclusion_without_constraint() -> (
    None
):
    """Test the zero-exclusion flag survives even when constraints are stripped.

    ``zero_included`` lives in the domain envelope, so emptying the stored
    ``constraints`` list still round-trips to a param that rejects ``0`` (the
    implied ``> 0`` constraint is re-derived on construction).
    """
    payload = create_interval_natural_param(zero_included=False).serialize_to_dict()
    payload["constraint_system"]["__data__"]["constraints"] = []  # type: ignore[index]  # test: modify serialized

    restored: Param[int] = Param.deserialize_from_dict(payload)

    assert not restored.is_value_valid(0)
    assert restored.is_value_valid(1)


# =============================================================================
# Arithmetic type preservation vs. widening
# =============================================================================


def test_bound_nat_param_plus_bound_nat_param_preserves_class() -> None:
    """Test adding two interval-natural params returns an interval-natural param."""
    left = _create_interval_natural_param_between(0, 5)
    right = _create_interval_natural_param_between(0, 3)

    result = left + right

    assert isinstance(result.domain, IntervalIntegerDomain)
    assert result.domain.non_negative


def test_bound_nat_param_plus_int_literal_widens_to_bound_int_param() -> None:
    """Test ``nat_param + int`` widens to a non-natural interval-integer param."""
    left = _create_interval_natural_param_between(1, 5)

    result = left + 2

    assert isinstance(result.domain, IntervalIntegerDomain)
    assert not result.domain.non_negative


def test_bound_nat_param_minus_bound_nat_param_widens_to_bound_int_param() -> None:
    """Test ``nat_param - nat_param`` widens to a non-natural interval-integer param."""
    left = _create_interval_natural_param_between(0, 3)
    right = _create_interval_natural_param_between(0, 5)

    result = left - right

    assert isinstance(result.domain, IntervalIntegerDomain)
    assert not result.domain.non_negative


def test_negation_of_bound_nat_param_widens_to_bound_int_param() -> None:
    """Test negating an interval-natural param widens to a non-natural interval."""
    base = _create_interval_natural_param_between(0, 5)

    result = -base

    assert isinstance(result.domain, IntervalIntegerDomain)
    assert not result.domain.non_negative


def test_bound_int_param_plus_bound_int_param_returns_bound_int_param() -> None:
    """Test ``interval_int_param + interval_int_param`` returns non-natural interval."""
    left = create_interval_integer_param_between(-3, 3)
    right = create_interval_integer_param_between(-1, 1)

    result = left + right

    assert isinstance(result.domain, IntervalIntegerDomain)
    assert not result.domain.non_negative


def test_bound_nat_param_addition_preserves_zero_excluded_flag() -> None:
    """Test ``nat_param + nat_param`` preserves the non-negative (natural) domain."""
    zero_excluded_left = _create_interval_natural_param_between(
        1, 5, prefer_inclusive=True
    )
    zero_excluded_right = _create_interval_natural_param_between(
        1, 3, prefer_inclusive=True
    )

    result = zero_excluded_left + zero_excluded_right

    assert isinstance(result.domain, IntervalIntegerDomain)
    assert result.domain.non_negative
