"""Tests for `BoundNatParam`."""

import pytest

from fhy_core.identifier import Identifier
from fhy_core.param import BoundNatParam
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
)

from .conftest import assert_all_satisfied, assert_none_satisfied

# =============================================================================
# Keyword-only signatures
# =============================================================================


def test_bound_nat_param_init_accepts_post_marker_args_as_keywords() -> None:
    """Test `BoundNatParam.__init__` accepts post-``*`` args as keywords."""
    BoundNatParam(name=Identifier("x"), is_zero_included=False, prefer_inclusive=False)


def test_bound_nat_param_init_rejects_name_passed_positionally() -> None:
    """Test `BoundNatParam.__init__` rejects ``name`` passed positionally."""
    with pytest.raises(TypeError):
        BoundNatParam(Identifier("x"))  # type: ignore[misc]  # test: keyword-only


# =============================================================================
# Default-flag invariants
# =============================================================================


def test_bound_nat_param_init_defaults_to_zero_included_true() -> None:
    """Test `BoundNatParam()` defaults to ``is_zero_included=True``."""
    assert BoundNatParam().is_value_valid(0)


def test_bound_nat_param_init_defaults_to_prefer_inclusive_true() -> None:
    """Test `BoundNatParam()` defaults to ``prefer_inclusive=True``.

    Constructs the param via the bare ``BoundNatParam()`` constructor (no
    classmethod) so the `__init__` default is the *only* default in play —
    inherited classmethods like `with_lower_bound` carry their own
    ``prefer_inclusive`` default and pass it explicitly to ``cls(...)``,
    which masks the `__init__` default. Verifies via the constraint form
    produced by an arithmetic operation.
    """
    p = BoundNatParam().add_lower_bound_constraint(3) + 1
    assert ">=" in str(p)


# =============================================================================
# Serialization
# =============================================================================


def test_bound_nat_param_serialization_round_trip_preserves_constraints() -> None:
    """Test `BoundNatParam` round-trips through dict serialization with constraints."""
    p = BoundNatParam.with_lower_bound(2, is_inclusive=True)
    dictionary = p.serialize_to_dict()
    assert len(dictionary["__data__"]["constraints"]) == 2  # type: ignore[index,arg-type,call-overload]  # test: dict shape known
    restored = BoundNatParam.deserialize_from_dict(dictionary)
    assert_all_satisfied(restored, [2, 5, 100])
    assert_none_satisfied(restored, [0, 1])
    redictionary = restored.serialize_to_dict()
    assert len(redictionary["__data__"]["constraints"]) == 2  # type: ignore[index,arg-type,call-overload]  # test: dict shape known


def test_bound_nat_param_deserialize_round_trip_preserves_zero_excluded_flag() -> None:
    """Test `BoundNatParam(is_zero_included=False)` round-trips through serialization.

    Pins down the ``is_zero_included = False`` assignment in
    `BoundNatParam.deserialize_data_from_dict`'s
    `is_zero_not_included_constraint_exists` branch. A ``False → True`` flip
    leaves admissibility unchanged on every integer (the basic ``var > 0``
    constraint from the payload survives the filter alongside the
    constructor's added ``var >= 0``), so ``is_value_valid`` cannot
    discriminate the two. Asserts structural equivalence instead — the
    mutant produces a param with two constraints and ``_is_zero_included =
    True``, which is not structurally equivalent to the original.
    """
    original = BoundNatParam(is_zero_included=False)
    restored = BoundNatParam.deserialize_from_dict(original.serialize_to_dict())
    assert original.is_structurally_equivalent(restored)


def test_bound_nat_param_deserialize_rejects_payload_with_malformed_data() -> None:
    """Test `BoundNatParam.deserialize_from_dict` rejects malformed payload data.

    A wrapped envelope whose ``__data__`` is missing required fields fails
    the inner ``_is_valid_bound_param_data`` check.
    """
    payload = {"__type__": "bound_nat_param", "__data__": {}}
    with pytest.raises(DeserializationDictStructureError):
        BoundNatParam.deserialize_from_dict(payload)  # type: ignore[arg-type]  # test: dict shape


def test_bound_nat_param_deserialize_rejects_payload_without_basic_constraint() -> None:
    """Test `BoundNatParam.deserialize_from_dict` rejects without basic constraint.

    Falls through to the ``DeserializationValueError`` raise when neither
    ``is_zero_included_constraint_exists`` nor
    ``is_zero_not_included_constraint_exists`` finds a matching constraint.
    """
    payload = BoundNatParam(is_zero_included=False).serialize_to_dict()
    payload["__data__"]["constraints"] = []  # type: ignore[index]  # test: modify serialized
    with pytest.raises(DeserializationValueError):
        BoundNatParam.deserialize_from_dict(payload)
