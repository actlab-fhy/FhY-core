"""Tests for `RealParam`."""

from functools import partial
from typing import Any

import pytest

from fhy_core.constraint import EquationConstraint
from fhy_core.identifier import Identifier
from fhy_core.param import IntParam, RealParam

from .conftest import assert_all_satisfied, assert_none_satisfied

# =============================================================================
# Admissibility
# =============================================================================


def test_real_param_assign_rejects_non_numeric_value(
    default_real_param: RealParam,
) -> None:
    """Test `RealParam.assign` rejects a non-numeric value."""
    with pytest.raises(ValueError):
        default_real_param.assign([])  # type: ignore[arg-type]  # test: invalid input


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param(1.5, True, id="float-admitted"),
        pytest.param("1.5", True, id="numeric-string-admitted"),
        pytest.param(True, False, id="bool-true-rejected"),
        pytest.param(False, False, id="bool-false-rejected"),
        pytest.param("not a number", False, id="non-numeric-string-rejected"),
        pytest.param([], False, id="list-rejected"),
        pytest.param(None, False, id="none-rejected"),
    ],
)
def test_real_param_admissibility_matrix(value: Any, expected: bool) -> None:
    """Test `RealParam.is_value_admissible` admits floats / numeric strings only.

    ``bool`` is a subtype of ``int`` but real-valued semantics treat booleans
    as non-numeric to avoid silent ``True``/``False`` admission.
    """
    assert RealParam().is_value_admissible(value) is expected


def test_real_param_str_uses_R_for_param_set() -> None:
    """Test `str(RealParam())` denotes the param set with ``R``."""
    assert "R" in str(RealParam())


# =============================================================================
# Constraint addition
# =============================================================================


def test_real_param_add_constraint_combines_with_existing_constraints(
    default_real_param: RealParam,
) -> None:
    """Test sequential `add_constraint` calls produce a combined feasibility set."""
    param = default_real_param.add_constraint(
        EquationConstraint(
            default_real_param.variable,
            default_real_param.variable_expression * 3.14 < 20.0,
        )
    )
    param = param.add_constraint(
        EquationConstraint(param.variable, param.variable_expression >= 1.0)
    )
    assert_all_satisfied(param, [2.0])
    assert_none_satisfied(param, [0.5, 7.0])


# =============================================================================
# Lower / upper bound and `between` constructors
# =============================================================================


@pytest.mark.parametrize(
    "factory, ops, pass_values, fail_values",
    [
        pytest.param(
            partial(RealParam),
            [("add_lower_bound_constraint", (1.0,), {"is_inclusive": True})],
            [1.0, 2.0],
            [0.5],
            id="lower-mutating-inclusive",
        ),
        pytest.param(
            partial(RealParam),
            [("add_lower_bound_constraint", (1.0,), {"is_inclusive": False})],
            [1.5, 2.0],
            [1.0, 0.5],
            id="lower-mutating-exclusive",
        ),
        pytest.param(
            partial(RealParam.with_lower_bound, 1.0, is_inclusive=True),
            [],
            [1.0, 2.0],
            [0.5],
            id="lower-constructor-inclusive",
        ),
        pytest.param(
            partial(RealParam.with_lower_bound, 1.0, is_inclusive=False),
            [],
            [1.5, 2.0],
            [1.0, 0.5],
            id="lower-constructor-exclusive",
        ),
        pytest.param(
            partial(RealParam),
            [("add_upper_bound_constraint", (2.0,), {"is_inclusive": True})],
            [2.0, 1.0],
            [2.5],
            id="upper-mutating-inclusive",
        ),
        pytest.param(
            partial(RealParam),
            [("add_upper_bound_constraint", (2.0,), {"is_inclusive": False})],
            [1.0, 1.5],
            [2.0, 2.5],
            id="upper-mutating-exclusive",
        ),
        pytest.param(
            partial(RealParam.with_upper_bound, 2.0, is_inclusive=True),
            [],
            [2.0, 1.0],
            [2.5],
            id="upper-constructor-inclusive",
        ),
        pytest.param(
            partial(RealParam.with_upper_bound, 2.0, is_inclusive=False),
            [],
            [1.0, 1.5],
            [2.0, 2.5],
            id="upper-constructor-exclusive",
        ),
        pytest.param(
            partial(RealParam),
            [
                ("add_lower_bound_constraint", (1.0,), {"is_inclusive": True}),
                ("add_upper_bound_constraint", (2.0,), {"is_inclusive": True}),
            ],
            [1.0, 1.5, 2.0],
            [0.5, 2.5],
            id="between-mutating-inclusive",
        ),
        pytest.param(
            partial(RealParam),
            [
                ("add_lower_bound_constraint", (1.0,), {"is_inclusive": False}),
                ("add_upper_bound_constraint", (2.0,), {"is_inclusive": False}),
            ],
            [1.5],
            [1.0, 2.0, 0.5, 2.5],
            id="between-mutating-exclusive",
        ),
        pytest.param(
            partial(
                RealParam.between,
                1.0,
                2.0,
                is_lower_inclusive=True,
                is_upper_inclusive=True,
            ),
            [],
            [1.0, 1.5, 2.0],
            [0.5, 2.5],
            id="between-constructor-inclusive",
        ),
        pytest.param(
            partial(
                RealParam.between,
                1.0,
                2.0,
                is_lower_inclusive=False,
                is_upper_inclusive=False,
            ),
            [],
            [1.5],
            [1.0, 2.0, 0.5, 2.5],
            id="between-constructor-exclusive",
        ),
    ],
)
def test_real_param_bounded_construction_admits_expected_values(
    factory: Any,
    ops: list[tuple[str, tuple[Any, ...], dict[str, Any]]],
    pass_values: list[Any],
    fail_values: list[Any],
) -> None:
    """Test bounded `RealParam` constructions admit and reject the expected values."""
    param = factory()
    for name, args, kwargs in ops:
        param = getattr(param, name)(*args, **kwargs)
    assert_all_satisfied(param, pass_values)
    assert_none_satisfied(param, fail_values)


@pytest.mark.parametrize(
    "factory, ops",
    [
        pytest.param(
            partial(RealParam),
            [("add_lower_bound_constraint", ("invalid",))],
            id="lower-mutating-invalid",
        ),
        pytest.param(
            partial(RealParam),
            [("add_upper_bound_constraint", ("invalid",))],
            id="upper-mutating-invalid",
        ),
        pytest.param(
            partial(RealParam.with_upper_bound, "invalid"),
            [],
            id="upper-constructor-invalid",
        ),
        pytest.param(
            partial(RealParam.with_lower_bound, "invalid"),
            [],
            id="lower-constructor-invalid",
        ),
        pytest.param(
            partial(RealParam.between, 2.0, 1.0),
            [],
            id="between-constructor-reversed",
        ),
    ],
)
def test_real_param_bounded_construction_with_invalid_inputs_raises(
    factory: Any,
    ops: list[tuple[str, tuple[Any, ...]]],
) -> None:
    """Test bounded `RealParam` constructions reject invalid bounds via `ValueError`."""
    with pytest.raises(ValueError):
        param = factory()
        for op in ops:
            name, args = op
            param = getattr(param, name)(*args)


# =============================================================================
# Default-inclusivity invariant (kills `True -> False` flips on default args)
# =============================================================================


@pytest.mark.parametrize(
    "factory, boundary_value",
    [
        pytest.param(
            partial(RealParam.with_lower_bound, 0.0), 0.0, id="with-lower-bound"
        ),
        pytest.param(
            partial(RealParam.with_upper_bound, 1.0), 1.0, id="with-upper-bound"
        ),
        pytest.param(
            lambda: RealParam().add_lower_bound_constraint(0.0),
            0.0,
            id="add-lower-bound-constraint",
        ),
        pytest.param(
            lambda: RealParam().add_upper_bound_constraint(1.0),
            1.0,
            id="add-upper-bound-constraint",
        ),
        pytest.param(
            partial(RealParam.between, 0.0, 1.0), 0.0, id="between-lower-endpoint"
        ),
        pytest.param(
            partial(RealParam.between, 0.0, 1.0), 1.0, id="between-upper-endpoint"
        ),
    ],
)
def test_real_param_default_bound_inclusivity_admits_endpoint(
    factory: Any, boundary_value: float
) -> None:
    """Test each `RealParam` bound builder defaults to inclusive (admits endpoint)."""
    assert factory().is_value_valid(boundary_value)


# =============================================================================
# Equal-bounds and reversed-bounds invariants for `between`
# =============================================================================


def test_real_param_between_equal_bounds_with_both_inclusive_is_singleton() -> None:
    """Test `RealParam.between(x, x)` with both bounds inclusive admits only ``x``."""
    param = RealParam.between(5.0, 5.0)
    assert param.is_value_valid(5.0)
    assert not param.is_value_valid(4.999)
    assert not param.is_value_valid(5.001)


@pytest.mark.parametrize(
    "is_lower_inclusive, is_upper_inclusive",
    [
        pytest.param(False, True, id="exclusive-inclusive"),
        pytest.param(True, False, id="inclusive-exclusive"),
        pytest.param(False, False, id="exclusive-exclusive"),
    ],
)
def test_real_param_between_equal_bounds_with_any_exclusive_raises(
    is_lower_inclusive: bool, is_upper_inclusive: bool
) -> None:
    """Test `RealParam.between(x, x)` raises when either bound is exclusive.

    Constructs the upper bound as a runtime ``float("5.0")`` so the two bounds
    are equal but not identity-equal; pins down value-equality (``==``) rather
    than identity (``is``) on the bounds-equal check.
    """
    with pytest.raises(ValueError):
        RealParam.between(
            5.0,
            float("5.0"),
            is_lower_inclusive=is_lower_inclusive,
            is_upper_inclusive=is_upper_inclusive,
        )


# =============================================================================
# Structural equivalence vs `IntParam`
# =============================================================================


def test_real_param_is_not_structurally_equivalent_to_int_param() -> None:
    """Test `RealParam` is not equivalent to an otherwise matching `IntParam`."""
    shared_name = Identifier("x")
    shared_name_copy = Identifier.deserialize_from_dict(shared_name.serialize_to_dict())
    real = RealParam(name=shared_name)
    integer = IntParam(name=shared_name_copy)
    assert not real.is_structurally_equivalent(integer)


# =============================================================================
# Serialization
# =============================================================================


def test_real_param_serialization_round_trip_preserves_constraints() -> None:
    """Test `RealParam` round-trips through dict serialization with its constraints."""
    param = RealParam()
    param = param.add_constraint(
        EquationConstraint(param.variable, param.variable_expression > 0)
    )
    param = param.add_constraint(
        EquationConstraint(param.variable, param.variable_expression < 10)
    )
    dictionary = param.serialize_to_dict()
    restored = RealParam.deserialize_from_dict(dictionary)
    assert_all_satisfied(restored, [1.0, 5.0, 9.0])
    assert_none_satisfied(restored, [0.0, 10.0])
