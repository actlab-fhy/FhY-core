"""Tests for natural-number parameters."""

from functools import partial
from typing import Any

import pytest

from fhy_core.serialization import (
    DeserializationDictStructureError,
)
from fhy_core.symbolic.constraint import EquationConstraint
from fhy_core.symbolic.param import (
    IntegerDomain,
    Param,
    ParamError,
    create_integer_param,
    create_natural_param,
)

from .conftest import assert_all_satisfied, assert_none_satisfied, mock_identifier

# =============================================================================
# Construction & assignment
# =============================================================================


def test_nat_param_with_zero_included_admits_zero_and_positive_integers() -> None:
    """Test a natural param with zero included admits zero and positive integers."""
    param = create_natural_param()

    assert param.assign(0).is_value_set()
    assert param.assign(1).is_value_set()
    with pytest.raises(ParamError):
        param.assign(-1)


def test_nat_param_with_zero_excluded_rejects_zero() -> None:
    """Test a natural param with zero excluded rejects zero."""
    param = create_natural_param(zero_included=False)

    assert param.assign(1).is_value_set()
    with pytest.raises(ParamError):
        param.assign(0)
    with pytest.raises(ParamError):
        param.assign(-1)


def test_nat_param_with_zero_excluded_preserves_zero_exclusion_after_add() -> None:
    """Test adding constraints preserves the zero-excluded natural param semantics."""
    param = create_natural_param(zero_included=False)
    updated = param.add_lower_bound_constraint(2, is_inclusive=True)

    with pytest.raises(ParamError):
        updated.add_lower_bound_constraint(0, is_inclusive=True)


def test_nat_param_with_lower_bound_zero_inclusive_admits_zero() -> None:
    """Test natural param with lower_bound=0 inclusive admits zero and positives."""
    param = create_natural_param().add_lower_bound_constraint(0, is_inclusive=True)

    assert_all_satisfied(param, [0, 1, 2, 100])


# =============================================================================
# Lower / upper bound and `between` rejection paths
# =============================================================================


@pytest.mark.parametrize(
    "factory, ops",
    [
        pytest.param(
            partial(create_natural_param, zero_included=False),
            [("add_lower_bound_constraint", (0,))],
            id="lower-mutating-zero-excluded",
        ),
        pytest.param(
            partial(create_natural_param, zero_included=False),
            [("add_lower_bound_constraint", (-1,))],
            id="lower-mutating-negative-bound",
        ),
        pytest.param(
            partial(create_natural_param, zero_included=True),
            [("add_lower_bound_constraint", (-1,))],
            id="lower-mutating-zero-included-exclusive",
        ),
        pytest.param(
            partial(create_natural_param, zero_included=False),
            [("add_upper_bound_constraint", (0,))],
            id="upper-mutating-zero-included-exclusive",
        ),
        pytest.param(
            partial(create_natural_param, zero_included=False),
            [("add_upper_bound_constraint", (-1,))],
            id="upper-mutating-zero-excluded",
        ),
        pytest.param(
            partial(create_natural_param, zero_included=True),
            [("add_upper_bound_constraint", (-1,))],
            id="upper-mutating-negative-bound",
        ),
        pytest.param(
            # -1 is an invalid lower bound for a natural param.
            lambda: create_natural_param().add_lower_bound_constraint(
                -1, is_inclusive=False
            ),
            [],
            id="between-constructor-negative-bound",
        ),
        pytest.param(
            # There is no create_natural_param_between factory; individual
            # add_lower_bound_constraint + add_upper_bound_constraint do not
            # validate lower <= upper together, so a reversed bound is accepted.
            # Marked xfail until a between-factory for natural params is added.
            lambda: (
                create_natural_param()
                .add_lower_bound_constraint(3, is_inclusive=False)
                .add_upper_bound_constraint(2, is_inclusive=False)
            ),
            [],
            id="between-constructor-reversed-bound",
            marks=pytest.mark.xfail(
                reason=(
                    "No create_natural_param_between factory: individual "
                    "bound-adding does not validate lower <= upper together."
                ),
                strict=False,
            ),
        ),
    ],
)
def test_nat_param_bounded_construction_with_invalid_inputs_raises(
    factory: Any,
    ops: list[tuple[str, tuple[Any, ...]]],
) -> None:
    """Test bounded natural param constructions reject invalid bounds (`ParamError`)."""
    with pytest.raises(ParamError):
        param = factory()
        for op in ops:
            name, args = op
            param = getattr(param, name)(*args)


# =============================================================================
# Bound-threshold boundary matrix
#
# Each row pins down the exact threshold of one comparison operator in
# `add_lower_bound_constraint` / `add_upper_bound_constraint`. The four-tuple
# ``(zero_included, is_inclusive, bound, kind)`` selects the validation
# branch and the boundary value; ``kind`` is ``"raises"`` or ``"succeeds"``.
# Together the rows discriminate every comparison-operator and
# `NumberReplacer` mutant on the natural-bound validation logic.
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
    "zero_included, is_inclusive, lower_bound, kind", _LOWER_BOUND_BOUNDARY_CASES
)
def test_nat_param_add_lower_bound_constraint_threshold_matrix(
    zero_included: bool, is_inclusive: bool, lower_bound: int, kind: str
) -> None:
    """Test natural param `add_lower_bound_constraint` thresholds raise or succeed."""
    param = create_natural_param(zero_included=zero_included)
    if kind == "raises":
        with pytest.raises(ParamError):
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
    "zero_included, is_inclusive, upper_bound, kind", _UPPER_BOUND_BOUNDARY_CASES
)
def test_nat_param_add_upper_bound_constraint_threshold_matrix(
    zero_included: bool, is_inclusive: bool, upper_bound: int, kind: str
) -> None:
    """Test natural param `add_upper_bound_constraint` thresholds raise or succeed."""
    param = create_natural_param(zero_included=zero_included)
    if kind == "raises":
        with pytest.raises(ParamError):
            param.add_upper_bound_constraint(upper_bound, is_inclusive=is_inclusive)
    else:
        param.add_upper_bound_constraint(upper_bound, is_inclusive=is_inclusive)


# =============================================================================
# Default-inclusivity invariant
# =============================================================================


def test_nat_param_add_upper_bound_constraint_defaults_to_inclusive() -> None:
    """Test natural param `add_upper_bound_constraint` defaults to inclusive bound."""
    param = create_natural_param().add_upper_bound_constraint(5)

    assert param.is_value_valid(5)


# =============================================================================
# Keyword-only signatures
# =============================================================================


def test_nat_param_factory_accepts_name_and_zero_included_as_keywords() -> None:
    """Test `create_natural_param` accepts ``name`` and ``zero_included`` as kwargs."""
    create_natural_param(name=mock_identifier("x", 1), zero_included=False)


def test_nat_param_factory_rejects_positional_args() -> None:
    """Test `create_natural_param` rejects positional arguments (keyword-only API)."""
    with pytest.raises(TypeError):
        create_natural_param(mock_identifier("x", 1))  # type: ignore[misc]  # test: keyword-only


def test_nat_param_add_lower_bound_constraint_rejects_is_inclusive_positional() -> None:
    """Test `add_lower_bound_constraint` rejects positional ``is_inclusive``."""
    param = create_natural_param()

    with pytest.raises(TypeError):
        param.add_lower_bound_constraint(1, True)  # type: ignore[misc]  # test: keyword-only


def test_nat_param_add_upper_bound_constraint_rejects_is_inclusive_positional() -> None:
    """Test `add_upper_bound_constraint` rejects positional ``is_inclusive``."""
    param = create_natural_param()

    with pytest.raises(TypeError):
        param.add_upper_bound_constraint(1, True)  # type: ignore[misc]  # test: keyword-only


# =============================================================================
# Structural equivalence
# =============================================================================


def test_nat_param_is_structurally_equivalent_to_self() -> None:
    """Test natural param `is_structurally_equivalent` is reflexive."""
    param = create_natural_param(zero_included=True)

    assert param.is_structurally_equivalent(param)


def test_nat_param_is_not_structurally_equivalent_when_zero_inclusion_differs() -> None:
    """Test natural params with mismatched ``zero_included`` are not equivalent.

    Non-equivalence is asserted in both directions because equivalence is a
    symmetric relation.
    """
    shared_name = mock_identifier("x", 1)
    shared_name_copy = mock_identifier("x", 1)
    included = create_natural_param(name=shared_name, zero_included=True)
    excluded = create_natural_param(name=shared_name_copy, zero_included=False)

    assert not included.is_structurally_equivalent(excluded)
    assert not excluded.is_structurally_equivalent(included)


def test_nat_param_is_not_structurally_equivalent_when_super_constraints_differ() -> (
    None
):
    """Test same-flag natural params with different non-basic constraints differ.

    The two params share a domain and ``zero_included`` flag, so the differing
    upper-bound constraints are the only discriminator.
    """
    shared_name = mock_identifier("x", 1)
    shared_name_copy = mock_identifier("x", 1)
    smaller_upper = create_natural_param(name=shared_name).add_upper_bound_constraint(5)
    larger_upper = create_natural_param(
        name=shared_name_copy
    ).add_upper_bound_constraint(10)

    assert not smaller_upper.is_structurally_equivalent(larger_upper)


def test_nat_param_is_not_structurally_equivalent_to_int_param() -> None:
    """Test a natural param is not structurally equivalent to a plain integer param.

    Both are `Param`, but their domains differ: `IntegerDomain(non_negative=True)`
    vs `IntegerDomain(non_negative=False)`.
    """
    shared_name = mock_identifier("x", 1)
    shared_name_copy = mock_identifier("x", 1)
    nat = create_natural_param(name=shared_name)
    integer = create_integer_param(name=shared_name_copy)

    assert not nat.is_structurally_equivalent(integer)


def test_integer_domain_canonicalizes_zero_included_when_not_non_negative() -> None:
    """Test ``zero_included`` is inert (canonicalized) when ``non_negative`` is False.

    ``zero_included`` only carries meaning for a natural-number domain, so two
    non-restricted integer domains that differ solely in ``zero_included`` are
    the same domain and compare structurally equivalent.
    """
    with_zero = IntegerDomain(non_negative=False, zero_included=True)
    without_zero = IntegerDomain(non_negative=False, zero_included=False)

    assert without_zero.zero_included is True
    assert with_zero.is_structurally_equivalent(without_zero)


# =============================================================================
# Serialization
# =============================================================================


def test_nat_param_serialization_round_trip_preserves_constraints() -> None:
    """Test natural param round-trips through dict serialization with constraints."""
    param = create_natural_param(zero_included=False)
    param = param.add_constraint(
        EquationConstraint(param.variable, param.variable_expression >= 1)
    )
    param = param.add_constraint(
        EquationConstraint(param.variable, param.variable_expression <= 10)
    )

    dictionary = param.serialize_to_dict()
    restored: Param[int] = Param.deserialize_from_dict(dictionary)
    redictionary = restored.serialize_to_dict()

    assert len(dictionary["constraints"]) == 3  # type: ignore[arg-type]  # test: dict shape known
    assert_all_satisfied(restored, [1, 5, 10])
    assert_none_satisfied(restored, [0, 11])
    assert len(redictionary["constraints"]) == 3  # type: ignore[arg-type]  # test: dict shape known


def test_nat_param_deserialize_round_trip_preserves_zero_inclusion_flag() -> None:
    """Test natural param round-trips ``zero_included=True`` through serialization.

    A deserialized ``zero_included=True`` natural param still admits ``0``.
    """
    original = create_natural_param(zero_included=True)

    restored: Param[int] = Param.deserialize_from_dict(original.serialize_to_dict())

    assert restored.is_value_valid(0)


def test_nat_param_deserialize_recovers_zero_exclusion_without_stored_constraint() -> (
    None
):
    """Test the zero-exclusion flag survives even when constraints are stripped.

    Under the derived format the domain carries ``zero_included`` explicitly, so
    the implied ``> 0`` constraint is re-derived on construction rather than
    inferred from a stored bound. A payload whose ``constraints`` list is emptied
    still round-trips to a param that rejects ``0``.
    """
    payload = create_natural_param(zero_included=False).serialize_to_dict()
    payload["constraints"] = []  # test: modify serialized

    restored: Param[int] = Param.deserialize_from_dict(payload)

    assert not restored.is_value_valid(0)
    assert restored.is_value_valid(1)


def test_nat_param_deserialize_rejects_payload_with_malformed_domain_data() -> None:
    """Test `Param.deserialize_from_dict` rejects payloads with malformed domain data.

    A domain envelope whose ``__data__`` omits the ``non_negative`` /
    ``zero_included`` fields fails the derived structure check for
    ``IntegerDomain``.
    """
    payload = create_natural_param().serialize_to_dict()
    payload["domain"]["__data__"] = {}  # type: ignore[index]  # test: modify serialized

    with pytest.raises(DeserializationDictStructureError):
        Param.deserialize_from_dict(payload)


# =============================================================================
# Domain kind checks
# =============================================================================


def test_nat_param_domain_is_integer_domain_with_non_negative() -> None:
    """Test a natural param's domain is `IntegerDomain` with ``non_negative=True``."""
    param = create_natural_param()

    assert isinstance(param.domain, IntegerDomain)
    assert param.domain.non_negative


def test_nat_param_is_value_admissible_does_not_gate_on_sign() -> None:
    """Test `is_value_admissible` is True for negative integers on a natural param.

    Natural params express non-negativity via constraints, not admissibility.
    ``is_value_admissible(-5)`` returns ``True``; ``is_value_valid(-5)`` is ``False``.
    """
    param = create_natural_param()

    assert param.is_value_admissible(-5)
    assert not param.is_value_valid(-5)
