"""Tests for `NatParam`."""

from functools import partial
from typing import Any

import pytest

from fhy_core.constraint import EquationConstraint
from fhy_core.identifier import Identifier
from fhy_core.param import IntParam, NatParam
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
)

from .conftest import assert_all_satisfied, assert_none_satisfied

# =============================================================================
# Construction & assignment
# =============================================================================


def test_nat_param_with_zero_included_admits_zero_and_positive_integers() -> None:
    """Test a `NatParam` with zero included admits zero and positive integers."""
    param = NatParam()
    assert param.assign(0).is_value_set()
    assert param.assign(1).is_value_set()
    with pytest.raises(ValueError):
        param.assign(-1)


def test_nat_param_with_zero_excluded_rejects_zero() -> None:
    """Test a `NatParam` with zero excluded rejects zero."""
    param = NatParam(is_zero_included=False)
    assert param.assign(1).is_value_set()
    with pytest.raises(ValueError):
        param.assign(0)
    with pytest.raises(ValueError):
        param.assign(-1)


def test_nat_param_with_zero_excluded_preserves_zero_exclusion_after_add() -> None:
    """Test adding constraints preserves the zero-excluded `NatParam` semantics."""
    param = NatParam(is_zero_included=False)
    updated = param.add_lower_bound_constraint(2, is_inclusive=True)
    with pytest.raises(ValueError):
        updated.add_lower_bound_constraint(0, is_inclusive=True)


def test_nat_param_with_lower_bound_zero_inclusive_admits_zero() -> None:
    """Test `NatParam.with_lower_bound(0, is_inclusive=True)` admits zero.

    Pins down the constructor argument that includes zero in the natural number
    constraints added before the lower bound is applied.
    """
    param = NatParam.with_lower_bound(0, is_inclusive=True)
    assert_all_satisfied(param, [0, 1, 2, 100])


# =============================================================================
# Lower / upper bound and `between` rejection paths
# =============================================================================


@pytest.mark.parametrize(
    "factory, ops",
    [
        pytest.param(
            partial(NatParam, is_zero_included=False),
            [("add_lower_bound_constraint", (0,))],
            id="lower-mutating-zero-excluded",
        ),
        pytest.param(
            partial(NatParam, is_zero_included=False),
            [("add_lower_bound_constraint", (-1,))],
            id="lower-mutating-negative-bound",
        ),
        pytest.param(
            partial(NatParam, is_zero_included=True),
            [("add_lower_bound_constraint", (-1,))],
            id="lower-mutating-zero-included-exclusive",
        ),
        pytest.param(
            partial(NatParam, is_zero_included=False),
            [("add_upper_bound_constraint", (0,))],
            id="upper-mutating-zero-included-exclusive",
        ),
        pytest.param(
            partial(NatParam, is_zero_included=False),
            [("add_upper_bound_constraint", (-1,))],
            id="upper-mutating-zero-excluded",
        ),
        pytest.param(
            partial(NatParam, is_zero_included=True),
            [("add_upper_bound_constraint", (-1,))],
            id="upper-mutating-negative-bound",
        ),
        pytest.param(
            partial(
                NatParam.between,
                -1,
                1,
                is_lower_inclusive=False,
                is_upper_inclusive=False,
            ),
            [],
            id="between-constructor-negative-bound",
        ),
        pytest.param(
            partial(
                NatParam.between,
                3,
                2,
                is_lower_inclusive=False,
                is_upper_inclusive=False,
            ),
            [],
            id="between-constructor-reversed-bound",
        ),
    ],
)
def test_nat_param_bounded_construction_with_invalid_inputs_raises(
    factory: Any,
    ops: list[tuple[str, tuple[Any, ...]]],
) -> None:
    """Test bounded `NatParam` constructions reject invalid bounds with `ValueError`."""
    with pytest.raises(ValueError):
        param = factory()
        for op in ops:
            name, args = op
            param = getattr(param, name)(*args)


# =============================================================================
# Bound-threshold boundary matrix
#
# Each row pins down the exact threshold of one comparison operator in
# `add_lower_bound_constraint` / `add_upper_bound_constraint`. The four-tuple
# ``(is_zero_included, is_inclusive, bound, kind)`` selects the validation
# branch and the boundary value; ``kind`` is ``"raises"`` or ``"succeeds"``.
# Together the rows discriminate every comparison-operator and
# `NumberReplacer` mutant on lines 150, 156, 160, 175, 179, 185, and 189 of
# `fhy_core.param.fundamental`.
# =============================================================================


_LOWER_BOUND_BOUNDARY_CASES = [
    pytest.param(True, False, 0, "raises", id="zero-incl-excl-0-raises"),
    pytest.param(True, False, 1, "succeeds", id="zero-incl-excl-1-succeeds"),
    pytest.param(True, False, 2, "succeeds", id="zero-incl-excl-2-succeeds"),
    pytest.param(False, True, 0, "raises", id="zero-excl-incl-0-raises"),
    pytest.param(False, True, 1, "succeeds", id="zero-excl-incl-1-succeeds"),
    pytest.param(False, True, 2, "succeeds", id="zero-excl-incl-2-succeeds"),
    pytest.param(False, False, -1, "raises", id="zero-excl-excl-neg1-raises"),
    pytest.param(False, False, 0, "succeeds", id="zero-excl-excl-0-succeeds"),
    pytest.param(False, False, 1, "succeeds", id="zero-excl-excl-1-succeeds"),
]


@pytest.mark.parametrize(
    "is_zero_included, is_inclusive, lower_bound, kind", _LOWER_BOUND_BOUNDARY_CASES
)
def test_nat_param_add_lower_bound_constraint_threshold_matrix(
    is_zero_included: bool, is_inclusive: bool, lower_bound: int, kind: str
) -> None:
    """Test `NatParam.add_lower_bound_constraint` thresholds raise or succeed."""
    param = NatParam(is_zero_included=is_zero_included)
    if kind == "raises":
        with pytest.raises(ValueError):
            param.add_lower_bound_constraint(lower_bound, is_inclusive=is_inclusive)
    else:
        param.add_lower_bound_constraint(lower_bound, is_inclusive=is_inclusive)


_UPPER_BOUND_BOUNDARY_CASES = [
    pytest.param(True, True, -1, "raises", id="zero-incl-incl-neg1-raises"),
    pytest.param(True, True, 0, "succeeds", id="zero-incl-incl-0-succeeds"),
    pytest.param(True, False, 0, "raises", id="zero-incl-excl-0-raises"),
    pytest.param(True, False, 1, "succeeds", id="zero-incl-excl-1-succeeds"),
    pytest.param(True, False, 2, "succeeds", id="zero-incl-excl-2-succeeds"),
    pytest.param(False, True, 0, "raises", id="zero-excl-incl-0-raises"),
    pytest.param(False, True, 1, "succeeds", id="zero-excl-incl-1-succeeds"),
    pytest.param(False, True, 2, "succeeds", id="zero-excl-incl-2-succeeds"),
    pytest.param(False, False, 1, "raises", id="zero-excl-excl-1-raises"),
    pytest.param(False, False, 2, "succeeds", id="zero-excl-excl-2-succeeds"),
    pytest.param(False, False, 3, "succeeds", id="zero-excl-excl-3-succeeds"),
]


@pytest.mark.parametrize(
    "is_zero_included, is_inclusive, upper_bound, kind", _UPPER_BOUND_BOUNDARY_CASES
)
def test_nat_param_add_upper_bound_constraint_threshold_matrix(
    is_zero_included: bool, is_inclusive: bool, upper_bound: int, kind: str
) -> None:
    """Test `NatParam.add_upper_bound_constraint` thresholds raise or succeed."""
    param = NatParam(is_zero_included=is_zero_included)
    if kind == "raises":
        with pytest.raises(ValueError):
            param.add_upper_bound_constraint(upper_bound, is_inclusive=is_inclusive)
    else:
        param.add_upper_bound_constraint(upper_bound, is_inclusive=is_inclusive)


# =============================================================================
# Default-inclusivity invariant
# =============================================================================


def test_nat_param_add_upper_bound_constraint_defaults_to_inclusive() -> None:
    """Test `NatParam.add_upper_bound_constraint` defaults to an inclusive bound."""
    param = NatParam().add_upper_bound_constraint(5)
    assert param.is_value_valid(5)


# =============================================================================
# Keyword-only signatures
# =============================================================================


def test_nat_param_init_accepts_name_and_is_zero_included_as_keywords() -> None:
    """Test `NatParam.__init__` accepts post-``*`` args as keywords."""
    NatParam(name=Identifier("x"), is_zero_included=False)


def test_nat_param_init_rejects_name_passed_positionally() -> None:
    """Test `NatParam.__init__` rejects ``name`` passed positionally."""
    with pytest.raises(TypeError):
        NatParam(Identifier("x"))  # type: ignore[misc]  # test: keyword-only


def test_nat_param_add_lower_bound_constraint_rejects_is_inclusive_positional() -> None:
    """Test `add_lower_bound_constraint` rejects positional ``is_inclusive``."""
    param = NatParam()
    with pytest.raises(TypeError):
        param.add_lower_bound_constraint(1, True)  # type: ignore[misc]  # test: keyword-only


def test_nat_param_add_upper_bound_constraint_rejects_is_inclusive_positional() -> None:
    """Test `add_upper_bound_constraint` rejects positional ``is_inclusive``."""
    param = NatParam()
    with pytest.raises(TypeError):
        param.add_upper_bound_constraint(1, True)  # type: ignore[misc]  # test: keyword-only


# =============================================================================
# Structural equivalence
# =============================================================================


def test_nat_param_is_structurally_equivalent_to_self() -> None:
    """Test `NatParam.is_structurally_equivalent` is reflexive."""
    param = NatParam(is_zero_included=True)
    assert param.is_structurally_equivalent(param)


def test_nat_param_is_not_structurally_equivalent_when_zero_inclusion_differs() -> None:
    """Test `NatParam`s with mismatched ``is_zero_included`` are not equivalent.

    Asserts non-equivalence in *both* directions: with `bool` operands,
    ``False <= True`` and ``True >= False`` would each pass a one-directional
    assertion. Asserting both directions pins down ``==`` against ordered
    comparisons.
    """
    shared_name = Identifier("x")
    shared_name_copy = Identifier.deserialize_from_dict(shared_name.serialize_to_dict())
    included = NatParam(name=shared_name, is_zero_included=True)
    excluded = NatParam(name=shared_name_copy, is_zero_included=False)
    assert not included.is_structurally_equivalent(excluded)
    assert not excluded.is_structurally_equivalent(included)


def test_nat_param_is_not_structurally_equivalent_when_super_constraints_differ() -> (
    None
):
    """Test same-flag `NatParam`s with different non-basic constraints differ.

    The base-class `is_structurally_equivalent` returns ``False`` for the
    constraint mismatch even though the `isinstance` check and the
    ``is_zero_included`` check both pass. Pins down the ``and`` between the
    two NatParam-level checks against an ``or`` weakening.
    """
    shared_name = Identifier("x")
    shared_name_copy = Identifier.deserialize_from_dict(shared_name.serialize_to_dict())
    smaller_upper = NatParam(name=shared_name).add_upper_bound_constraint(5)
    larger_upper = NatParam(name=shared_name_copy).add_upper_bound_constraint(10)
    assert not smaller_upper.is_structurally_equivalent(larger_upper)


def test_nat_param_is_not_structurally_equivalent_to_int_param() -> None:
    """Test a `NatParam` is not structurally equivalent to a plain `IntParam`.

    Pins down the `isinstance(other, NatParam)` short-circuit against an
    ``or`` weakening that would short-circuit on ``self._is_zero_included ==
    other._is_zero_included`` and raise ``AttributeError`` when ``other`` is
    not a `NatParam`.
    """
    shared_name = Identifier("x")
    shared_name_copy = Identifier.deserialize_from_dict(shared_name.serialize_to_dict())
    nat = NatParam(name=shared_name)
    integer = IntParam(name=shared_name_copy)
    assert not nat.is_structurally_equivalent(integer)


# =============================================================================
# Serialization
# =============================================================================


def test_nat_param_serialization_round_trip_preserves_constraints() -> None:
    """Test `NatParam` round-trips through dict serialization with its constraints."""
    param = NatParam(is_zero_included=False)
    param = param.add_constraint(
        EquationConstraint(param.variable, param.variable_expression >= 1)
    )
    param = param.add_constraint(
        EquationConstraint(param.variable, param.variable_expression <= 10)
    )
    dictionary = param.serialize_to_dict()
    assert len(dictionary["__data__"]["constraints"]) == 3  # type: ignore[index,arg-type,call-overload]  # test: dict shape known
    restored = NatParam.deserialize_from_dict(dictionary)
    assert_all_satisfied(restored, [1, 5, 10])
    assert_none_satisfied(restored, [0, 11])
    redictionary = restored.serialize_to_dict()
    assert len(redictionary["__data__"]["constraints"]) == 3  # type: ignore[index,arg-type,call-overload]  # test: dict shape known


def test_nat_param_deserialize_round_trip_preserves_zero_inclusion_flag() -> None:
    """Test `NatParam` round-trips ``is_zero_included=True`` through serialization.

    Pins down the ``is_zero_included = True`` assignment in
    `deserialize_data_from_dict`: a ``True → False`` flip would yield a
    deserialized param that rejects ``0``.
    """
    original = NatParam(is_zero_included=True)
    restored = NatParam.deserialize_from_dict(original.serialize_to_dict())
    assert restored.is_value_valid(0)


def test_nat_param_deserialize_rejects_payload_without_basic_constraint() -> None:
    """Test `NatParam.deserialize_from_dict` rejects payloads with no basic constraint.

    The deserializer falls through to a ``DeserializationValueError`` when
    neither ``is_zero_included_constraint_exists`` nor
    ``is_zero_not_included_constraint_exists`` finds a matching constraint.
    Pins down the ``return False`` fallback in
    `is_zero_not_included_constraint_exists` against a ``True`` flip.
    """
    payload = NatParam(is_zero_included=False).serialize_to_dict()
    payload["__data__"]["constraints"] = []  # type: ignore[index]  # test: modify serialized
    with pytest.raises(DeserializationValueError):
        NatParam.deserialize_from_dict(payload)


def test_nat_param_deserialize_rejects_payload_with_malformed_data() -> None:
    """Test `NatParam.deserialize_from_dict` rejects payloads with malformed data.

    A wrapped envelope whose ``__data__`` lacks the ``variable`` and
    ``constraints`` fields fails the inner ``is_valid_param_data`` check.
    """
    payload = {"__type__": "nat_param", "__data__": {}}
    with pytest.raises(DeserializationDictStructureError):
        NatParam.deserialize_from_dict(payload)  # type: ignore[arg-type]  # test: dict shape
