"""Tests for real-valued parameters."""

from functools import partial
from typing import Any

import pytest

from fhy_core.symbolic.constraint import EquationConstraint
from fhy_core.symbolic.param import (
    Param,
    ParamError,
    create_integer_param,
    create_real_param,
    create_real_param_between,
    create_real_param_with_lower_bound,
    create_real_param_with_upper_bound,
)

from .conftest import assert_all_satisfied, assert_none_satisfied, mock_identifier

# =============================================================================
# Admissibility
# =============================================================================


def test_real_param_assign_rejects_non_numeric_value(
    default_real_param: Param[str | float],
) -> None:
    """Test real param `assign` raises `ParamError` for a non-numeric value."""
    with pytest.raises(ParamError):
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
    """Test real param `is_value_admissible` admits floats / numeric strings only.

    ``bool`` is a subtype of ``int`` but real-valued semantics treat booleans
    as non-numeric to avoid silent ``True``/``False`` admission.
    """
    param = create_real_param()

    result = param.is_value_admissible(value)

    assert result is expected


def test_real_param_str_uses_R_for_param_set() -> None:
    """Test `str` of a real param denotes the param set with ``R``."""
    assert "R" in str(create_real_param())


# =============================================================================
# Constraint addition
# =============================================================================


def test_real_param_add_constraint_combines_with_existing_constraints(
    default_real_param: Param[str | float],
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
            partial(create_real_param),
            [("add_lower_bound_constraint", (1.0,), {"is_inclusive": True})],
            [1.0, 2.0],
            [0.5],
            id="lower-mutating-inclusive",
        ),
        pytest.param(
            partial(create_real_param),
            [("add_lower_bound_constraint", (1.0,), {"is_inclusive": False})],
            [1.5, 2.0],
            [1.0, 0.5],
            id="lower-mutating-exclusive",
        ),
        pytest.param(
            partial(create_real_param_with_lower_bound, 1.0, is_inclusive=True),
            [],
            [1.0, 2.0],
            [0.5],
            id="lower-constructor-inclusive",
        ),
        pytest.param(
            partial(create_real_param_with_lower_bound, 1.0, is_inclusive=False),
            [],
            [1.5, 2.0],
            [1.0, 0.5],
            id="lower-constructor-exclusive",
        ),
        pytest.param(
            partial(create_real_param),
            [("add_upper_bound_constraint", (2.0,), {"is_inclusive": True})],
            [2.0, 1.0],
            [2.5],
            id="upper-mutating-inclusive",
        ),
        pytest.param(
            partial(create_real_param),
            [("add_upper_bound_constraint", (2.0,), {"is_inclusive": False})],
            [1.0, 1.5],
            [2.0, 2.5],
            id="upper-mutating-exclusive",
        ),
        pytest.param(
            partial(create_real_param_with_upper_bound, 2.0, is_inclusive=True),
            [],
            [2.0, 1.0],
            [2.5],
            id="upper-constructor-inclusive",
        ),
        pytest.param(
            partial(create_real_param_with_upper_bound, 2.0, is_inclusive=False),
            [],
            [1.0, 1.5],
            [2.0, 2.5],
            id="upper-constructor-exclusive",
        ),
        pytest.param(
            partial(create_real_param),
            [
                ("add_lower_bound_constraint", (1.0,), {"is_inclusive": True}),
                ("add_upper_bound_constraint", (2.0,), {"is_inclusive": True}),
            ],
            [1.0, 1.5, 2.0],
            [0.5, 2.5],
            id="between-mutating-inclusive",
        ),
        pytest.param(
            partial(create_real_param),
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
                create_real_param_between,
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
                create_real_param_between,
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
        pytest.param(
            partial(create_real_param),
            [
                ("add_lower_bound_constraint", ("1.0",), {"is_inclusive": True}),
                ("add_upper_bound_constraint", ("2.0",), {"is_inclusive": True}),
            ],
            [1.0, 1.5, 2.0],
            [0.5, 2.5],
            id="between-mutating-string-bounds",
        ),
    ],
)
def test_real_param_bounded_construction_admits_expected_values(
    factory: Any,
    ops: list[tuple[str, tuple[Any, ...], dict[str, Any]]],
    pass_values: list[Any],
    fail_values: list[Any],
) -> None:
    """Test bounded real param constructions admit and reject the expected values."""
    param = factory()
    for name, args, kwargs in ops:
        param = getattr(param, name)(*args, **kwargs)

    assert_all_satisfied(param, pass_values)
    assert_none_satisfied(param, fail_values)


@pytest.mark.parametrize(
    "factory, ops",
    [
        pytest.param(
            partial(create_real_param),
            [("add_lower_bound_constraint", ("invalid",))],
            id="lower-mutating-invalid",
        ),
        pytest.param(
            partial(create_real_param),
            [("add_upper_bound_constraint", ("invalid",))],
            id="upper-mutating-invalid",
        ),
        pytest.param(
            partial(create_real_param_with_upper_bound, "invalid"),
            [],
            id="upper-constructor-invalid",
        ),
        pytest.param(
            partial(create_real_param_with_lower_bound, "invalid"),
            [],
            id="lower-constructor-invalid",
        ),
    ],
)
def test_real_param_bounded_construction_with_invalid_string_bounds_raises(
    factory: Any,
    ops: list[tuple[str, tuple[Any, ...]]],
) -> None:
    """Test bounded real param constructions reject unparseable string bounds.

    The string ``"invalid"`` reaches the expression layer's
    ``LiteralExpression`` validator and raises a plain ``ValueError`` rather
    than ``ParamError`` -- the failure is in expression construction, not in
    param-domain validation.
    """
    with pytest.raises(ValueError, match="Invalid string-form literal expression"):
        param = factory()
        for name, args in ops:
            param = getattr(param, name)(*args)


def test_real_param_between_with_reversed_bounds_raises() -> None:
    """Test `create_real_param_between` raises `ParamError` when ``lower > upper``."""
    with pytest.raises(ParamError):
        create_real_param_between(2.0, 1.0)


# =============================================================================
# Default-inclusivity invariant (kills `True -> False` flips on default args)
# =============================================================================


@pytest.mark.parametrize(
    "factory, boundary_value",
    [
        pytest.param(
            partial(create_real_param_with_lower_bound, 0.0), 0.0, id="with-lower-bound"
        ),
        pytest.param(
            partial(create_real_param_with_upper_bound, 1.0), 1.0, id="with-upper-bound"
        ),
        pytest.param(
            lambda: create_real_param().add_lower_bound_constraint(0.0),
            0.0,
            id="add-lower-bound-constraint",
        ),
        pytest.param(
            lambda: create_real_param().add_upper_bound_constraint(1.0),
            1.0,
            id="add-upper-bound-constraint",
        ),
        pytest.param(
            partial(create_real_param_between, 0.0, 1.0),
            0.0,
            id="between-lower-endpoint",
        ),
        pytest.param(
            partial(create_real_param_between, 0.0, 1.0),
            1.0,
            id="between-upper-endpoint",
        ),
    ],
)
def test_real_param_default_bound_inclusivity_admits_endpoint(
    factory: Any, boundary_value: float
) -> None:
    """Test each real param bound builder defaults to inclusive (admits endpoint)."""
    assert factory().is_value_valid(boundary_value)


# =============================================================================
# Equal-bounds and reversed-bounds invariants for `between`
# =============================================================================


def test_real_param_between_equal_bounds_with_both_inclusive_is_singleton() -> None:
    """Test `create_real_param_between(x, x)` admits only ``x`` (both inclusive)."""
    param = create_real_param_between(5.0, 5.0)

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
    """Test `create_real_param_between(x, x)` raises when either bound is exclusive.

    The upper bound is a runtime ``float("5.0")`` so the two bounds are equal
    but not identity-equal, exercising value-equality (``==``) rather than
    identity (``is``) on the bounds-equal check.
    """
    with pytest.raises(ParamError):
        create_real_param_between(
            5.0,
            float("5.0"),
            is_lower_inclusive=is_lower_inclusive,
            is_upper_inclusive=is_upper_inclusive,
        )


# =============================================================================
# Structural equivalence vs integer param
# =============================================================================


def test_real_param_is_not_structurally_equivalent_to_int_param() -> None:
    """Test a real param is not equivalent to an otherwise matching integer param."""
    shared_name = mock_identifier("x", 1)
    shared_name_copy = mock_identifier("x", 1)
    real = create_real_param(name=shared_name)
    integer = create_integer_param(name=shared_name_copy)

    assert not real.is_structurally_equivalent(integer)


# =============================================================================
# Serialization
# =============================================================================


def test_real_param_serialization_round_trip_preserves_constraints() -> None:
    """Test real param round-trips through dict serialization with its constraints."""
    param = create_real_param()
    param = param.add_constraint(
        EquationConstraint(param.variable, param.variable_expression > 0)
    )
    param = param.add_constraint(
        EquationConstraint(param.variable, param.variable_expression < 10)
    )

    dictionary = param.serialize_to_dict()
    restored: Param[float] = Param.deserialize_from_dict(dictionary)

    assert_all_satisfied(restored, [1.0, 5.0, 9.0])
    assert_none_satisfied(restored, [0.0, 10.0])
