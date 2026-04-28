"""Tests for `IntParam`."""

from functools import partial
from typing import Any

import pytest

from fhy_core.constraint import EquationConstraint
from fhy_core.param import IntParam

from .conftest import assert_all_satisfied, assert_none_satisfied

# =============================================================================
# Admissibility
# =============================================================================


def test_int_param_assign_rejects_float_value(default_int_param: IntParam) -> None:
    """Test `IntParam.assign` rejects a `float` value."""
    with pytest.raises(ValueError):
        default_int_param.assign(1.0)  # type: ignore[arg-type]  # test: invalid input


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param(3, True, id="int-admitted"),
        pytest.param(True, False, id="bool-true-rejected"),
        pytest.param(False, False, id="bool-false-rejected"),
        pytest.param(1.5, False, id="float-rejected"),
        pytest.param("3", False, id="string-rejected"),
    ],
)
def test_int_param_admissibility_matrix(value: Any, expected: bool) -> None:
    """Test `IntParam.is_value_admissible` admits non-bool ints only.

    ``bool`` is a subtype of ``int`` but integer-valued semantics treat booleans
    as non-numeric to avoid silent ``True``/``False`` admission.
    """
    assert IntParam().is_value_admissible(value) is expected


# =============================================================================
# Constraint addition
# =============================================================================


def test_int_param_add_constraint_combines_with_existing_constraints(
    default_int_param: IntParam,
) -> None:
    """Test sequential `add_constraint` calls produce a combined feasibility set."""
    param = default_int_param.add_constraint(
        EquationConstraint(
            default_int_param.variable,
            (default_int_param.variable_expression % 5).equals(0),
        )
    )
    param = param.add_constraint(
        EquationConstraint(param.variable, param.variable_expression > 10)
    )
    assert param.is_constraints_satisfied(15)
    with pytest.raises(ValueError):
        param.assign(12)


# =============================================================================
# Lower / upper bound and `between` constructors
# =============================================================================


@pytest.mark.parametrize(
    "factory, ops, pass_values, fail_values",
    [
        pytest.param(
            partial(IntParam),
            [("add_lower_bound_constraint", (1,), {"is_inclusive": True})],
            [1, 2],
            [0],
            id="lower-mutating-inclusive",
        ),
        pytest.param(
            partial(IntParam),
            [("add_lower_bound_constraint", (1,), {"is_inclusive": False})],
            [2, 5],
            [-1, 0],
            id="lower-mutating-exclusive",
        ),
        pytest.param(
            partial(IntParam.with_lower_bound, 1, is_inclusive=True),
            [],
            [1, 2],
            [0],
            id="lower-constructor-inclusive",
        ),
        pytest.param(
            partial(IntParam.with_lower_bound, 1, is_inclusive=False),
            [],
            [2, 5],
            [1, 0],
            id="lower-constructor-exclusive",
        ),
        pytest.param(
            partial(IntParam),
            [("add_upper_bound_constraint", (2,), {"is_inclusive": True})],
            [2, 1],
            [3],
            id="upper-mutating-inclusive",
        ),
        pytest.param(
            partial(IntParam),
            [("add_upper_bound_constraint", (2,), {"is_inclusive": False})],
            [0, 1],
            [2, 3],
            id="upper-mutating-exclusive",
        ),
        pytest.param(
            partial(IntParam.with_upper_bound, 2, is_inclusive=True),
            [],
            [2, 1],
            [3],
            id="upper-constructor-inclusive",
        ),
        pytest.param(
            partial(IntParam.with_upper_bound, 2, is_inclusive=False),
            [],
            [1, 1],
            [2, 2],
            id="upper-constructor-exclusive",
        ),
        pytest.param(
            partial(IntParam),
            [
                ("add_lower_bound_constraint", (1,), {"is_inclusive": True}),
                ("add_upper_bound_constraint", (2,), {"is_inclusive": True}),
            ],
            [1, 2],
            [-1, 3],
            id="between-mutating-inclusive",
        ),
        pytest.param(
            partial(IntParam),
            [
                ("add_lower_bound_constraint", (1,), {"is_inclusive": False}),
                ("add_upper_bound_constraint", (3,), {"is_inclusive": False}),
            ],
            [2],
            [1, 3, 5, -5],
            id="between-mutating-exclusive",
        ),
        pytest.param(
            partial(
                IntParam.between,
                1,
                2,
                is_lower_inclusive=True,
                is_upper_inclusive=True,
            ),
            [],
            [1, 2],
            [0, 3],
            id="between-constructor-inclusive",
        ),
        pytest.param(
            partial(
                IntParam.between,
                1,
                3,
                is_lower_inclusive=False,
                is_upper_inclusive=False,
            ),
            [],
            [2],
            [1, 0, 3],
            id="between-constructor-exclusive",
        ),
    ],
)
def test_int_param_bounded_construction_admits_expected_values(
    factory: Any,
    ops: list[tuple[str, tuple[Any, ...], dict[str, Any]]],
    pass_values: list[Any],
    fail_values: list[Any],
) -> None:
    """Test bounded `IntParam` constructions admit and reject the expected values."""
    param = factory()
    for name, args, kwargs in ops:
        param = getattr(param, name)(*args, **kwargs)
    assert_all_satisfied(param, pass_values)
    assert_none_satisfied(param, fail_values)


def test_int_param_between_with_reversed_bounds_raises() -> None:
    """Test `IntParam.between` raises when ``lower > upper``."""
    with pytest.raises(ValueError):
        IntParam.between(2, 1)


# =============================================================================
# Default-inclusivity invariant (kills `True -> False` flips on default args)
# =============================================================================


@pytest.mark.parametrize(
    "factory, boundary_value",
    [
        pytest.param(partial(IntParam.with_lower_bound, 0), 0, id="with-lower-bound"),
        pytest.param(partial(IntParam.with_upper_bound, 5), 5, id="with-upper-bound"),
        pytest.param(
            lambda: IntParam().add_lower_bound_constraint(0),
            0,
            id="add-lower-bound-constraint",
        ),
        pytest.param(
            lambda: IntParam().add_upper_bound_constraint(5),
            5,
            id="add-upper-bound-constraint",
        ),
        pytest.param(partial(IntParam.between, 0, 5), 0, id="between-lower-endpoint"),
        pytest.param(partial(IntParam.between, 0, 5), 5, id="between-upper-endpoint"),
    ],
)
def test_int_param_default_bound_inclusivity_admits_endpoint(
    factory: Any, boundary_value: int
) -> None:
    """Test each `IntParam` bound builder defaults to inclusive (admits endpoint)."""
    assert factory().is_value_valid(boundary_value)


# =============================================================================
# Equal-bounds invariants for `between`
# =============================================================================


def test_int_param_between_equal_bounds_with_both_inclusive_is_singleton() -> None:
    """Test `IntParam.between(x, x)` with both bounds inclusive admits only ``x``."""
    param = IntParam.between(5, 5)
    assert param.is_value_valid(5)
    assert not param.is_value_valid(4)
    assert not param.is_value_valid(6)


@pytest.mark.parametrize(
    "is_lower_inclusive, is_upper_inclusive",
    [
        pytest.param(False, True, id="exclusive-inclusive"),
        pytest.param(True, False, id="inclusive-exclusive"),
        pytest.param(False, False, id="exclusive-exclusive"),
    ],
)
def test_int_param_between_equal_bounds_with_any_exclusive_raises(
    is_lower_inclusive: bool, is_upper_inclusive: bool
) -> None:
    """Test `IntParam.between(x, x)` raises when either bound is exclusive.

    Uses an integer above CPython's small-int cache (``257``) and constructs
    the upper bound via ``int("257")`` so the two bounds are equal but not
    identity-equal; pins down value-equality (``==``) rather than identity
    (``is``) on the bounds-equal check.
    """
    with pytest.raises(ValueError):
        IntParam.between(
            257,
            int("257"),
            is_lower_inclusive=is_lower_inclusive,
            is_upper_inclusive=is_upper_inclusive,
        )


# =============================================================================
# Serialization
# =============================================================================


def test_int_param_serialization_round_trip_preserves_constraints() -> None:
    """Test `IntParam` round-trips through dict serialization with its constraints."""
    param = IntParam()
    param = param.add_constraint(
        EquationConstraint(param.variable, param.variable_expression > 0)
    )
    param = param.add_constraint(
        EquationConstraint(param.variable, param.variable_expression < 10)
    )
    dictionary = param.serialize_to_dict()
    restored = IntParam.deserialize_from_dict(dictionary)
    assert_all_satisfied(restored, [1, 5, 9])
    assert_none_satisfied(restored, [0, 10])
