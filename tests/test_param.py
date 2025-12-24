"""Tests the parameter utility."""

from functools import partial
from typing import TypeVar

import pytest
from fhy_core.constraint import EquationConstraint, InSetConstraint
from fhy_core.expression import SymbolType
from fhy_core.param import (
    BoundIntParam,
    CategoricalParam,
    IntParam,
    NatParam,
    OrdinalParam,
    Param,
    PermParam,
    RealParam,
)

T = TypeVar("T")


def _assert_all_satisfied(param: Param[T], values: list[T]) -> None:
    for v in values:
        assert param.is_constraints_satisfied(
            v
        ), f"Value {v} should satisfy constraints of parameter {param}"


def _assert_none_satisfied(param: Param[T], values: list[T]) -> None:
    for v in values:
        assert not param.is_constraints_satisfied(
            v
        ), f"Value {v} should not satisfy constraints of parameter {param}"


@pytest.fixture
def default_real_param() -> RealParam:
    return RealParam()


@pytest.fixture
def default_int_param() -> IntParam:
    return IntParam()


def test_param_is_not_set_after_initialization(default_real_param):
    """Test that the value of a parameter is not set after initialization."""
    assert not default_real_param.is_value_set()


def test_param_is_set_after_setting_value(default_real_param):
    """Test that the value of a parameter is set after setting a value."""
    default_real_param.set_value(1.0)
    assert default_real_param.is_value_set()


def test_param_get_value_fails_when_not_set(default_real_param):
    """Test that getting the value of a parameter fails when the value is not set."""
    with pytest.raises(ValueError):
        default_real_param.get_value()


def test_get_param_value_after_setting_value(default_real_param):
    """Test that the value of a parameter can be retrieved after setting a value."""
    default_real_param.set_value(1.0)
    assert default_real_param.get_value() == 1.0


def test_real_param_get_symbol_type(default_real_param):
    """Test that the symbol type of a real parameter is correct."""
    assert default_real_param.get_symbol_type() == SymbolType.REAL


def test_int_param_get_symbol_type(default_int_param):
    """Test that the symbol type of an integer parameter is correct."""
    assert default_int_param.get_symbol_type() == SymbolType.INT


def test_real_param_value_set_fails_with_invalid_value(default_real_param):
    """Test that setting a real parameter value fails with an invalid value."""
    with pytest.raises(ValueError):
        default_real_param.set_value([])


def test_int_param_value_set_fails_with_invalid_value(default_int_param):
    """Test that setting an integer parameter value fails with an invalid value."""
    with pytest.raises(ValueError):
        default_int_param.set_value(1.0)


def test_create_real_param_with_value():
    """Test that a real parameter can be created with a value."""
    param = RealParam.with_value(1.0)
    assert param.is_value_set()
    assert param.get_value() == 1.0


def test_create_real_param_with_value_fails_with_invalid_value():
    """Test that creating a real parameter with an invalid value fails."""
    with pytest.raises(ValueError):
        RealParam.with_value("invalid")


def test_create_int_param_with_value():
    """Test that an integer parameter can be created with a value."""
    param = IntParam.with_value(1)
    assert param.is_value_set()
    assert param.get_value() == 1


def test_create_int_param_with_value_fails_with_invalid_value():
    """Test that creating an integer parameter with an invalid value fails."""
    with pytest.raises(ValueError):
        IntParam.with_value(1.2)


def test_add_and_check_real_param_constraints(default_real_param):
    """Test that a real constraint can be added and checked."""
    default_real_param.add_constraint(
        EquationConstraint(
            default_real_param.variable,
            default_real_param.variable_expression * 3.14 < 20.0,
        )
    )
    default_real_param.add_constraint(
        EquationConstraint(
            default_real_param.variable, default_real_param.variable_expression >= 1.0
        )
    )
    _assert_all_satisfied(default_real_param, [2.0])
    _assert_none_satisfied(default_real_param, [0.5, 7.0])


def test_add_and_check_int_param_constraints(default_int_param):
    """Test that an integer constraint can be added and checked."""
    default_int_param.add_constraint(
        EquationConstraint(
            default_int_param.variable,
            (default_int_param.variable_expression % 5).equals(0),
        )
    )
    default_int_param.add_constraint(
        EquationConstraint(
            default_int_param.variable, default_int_param.variable_expression > 10
        )
    )
    assert default_int_param.is_constraints_satisfied(15)
    with pytest.raises(ValueError):
        default_int_param.set_value(12)


def test_set_real_param_and_real_param_is_subset():
    """Test subset relationship between real parameters where they could be set."""
    param1 = RealParam()
    param1.set_value(1.0)
    param2 = RealParam()
    assert param1.is_subset(param2)
    assert not param2.is_subset(param1)
    param2.set_value(1.0)
    assert param1.is_subset(param2)
    assert param2.is_subset(param1)


def test_real_param_is_subset_of_real_param():
    """Test that a real parameter is a subset of another real parameter."""
    param1 = RealParam()
    param2 = RealParam()
    assert param1.is_subset(param2)
    assert param2.is_subset(param1)


def test_positive_real_param_and_real_param_is_subset():
    """Test subset relationship between positive real parameter and real parameter."""
    param1 = RealParam()
    param1.add_constraint(
        EquationConstraint(param1.variable, param1.variable_expression > 0)
    )
    param2 = RealParam()
    assert param1.is_subset(param2)
    assert not param2.is_subset(param1)


def test_interval_real_param_and_interval_real_param_is_subset():
    """Test subset relationship between interval real parameters."""
    param1 = RealParam()
    param1.add_constraint(
        EquationConstraint(param1.variable, param1.variable_expression >= 0)
    )
    param1.add_constraint(
        EquationConstraint(param1.variable, param1.variable_expression <= 3)
    )
    param2 = RealParam()
    param2.add_constraint(
        EquationConstraint(param2.variable, param2.variable_expression >= 0)
    )
    param2.add_constraint(
        EquationConstraint(param2.variable, param2.variable_expression <= 2)
    )
    assert not param1.is_subset(param2)
    assert param2.is_subset(param1)


def test_nat_param_zero_included():
    """Test that a natural number parameter with zero included can be set."""
    param = NatParam()
    param.set_value(0)
    assert param.is_value_set()
    param.set_value(1)
    assert param.is_value_set()
    with pytest.raises(ValueError):
        param.set_value(-1)


def test_nat_param_zero_excluded():
    """Test that a natural number parameter with zero excluded can be set."""
    param = NatParam(is_zero_included=False)
    param.set_value(1)
    assert param.is_value_set()
    with pytest.raises(ValueError):
        param.set_value(0)
    with pytest.raises(ValueError):
        param.set_value(-1)


def test_ordinal_param_initialization():
    """Test that an ordinal parameter can be initialized."""
    param = OrdinalParam([5, 6, 7])
    assert not param.is_value_set()


def test_ordinal_param_initialization_fails_with_non_unique_values():
    """Test that an ordinal parameter initialization fails with non-unique values."""
    with pytest.raises(ValueError):
        OrdinalParam([1, 2, 1])


@pytest.fixture()
def ordinal_param_123() -> OrdinalParam:
    return OrdinalParam([1, 2, 3])


def test_set_ordinal_param_value(ordinal_param_123: OrdinalParam):
    """Test that an ordinal parameter value can be set."""
    ordinal_param_123.set_value(1)
    assert ordinal_param_123.is_value_set()
    ordinal_param_123.set_value(3)
    assert ordinal_param_123.is_value_set()


def test_set_ordinal_param_value_fails_with_invalid_value(
    ordinal_param_123: OrdinalParam,
):
    """Test that setting an ordinal parameter value fails with an invalid value."""
    with pytest.raises(ValueError):
        ordinal_param_123.set_value(4)


def test_add_and_check_ordinal_param_constraints(ordinal_param_123: OrdinalParam):
    """Test that ordinal parameter constraints can be added and checked."""
    ordinal_param_123.add_constraint(
        InSetConstraint({ordinal_param_123.variable}, {1, 2})
    )
    _assert_all_satisfied(ordinal_param_123, [1, 2])
    _assert_none_satisfied(ordinal_param_123, [3])


def test_adding_invalid_constraint_to_ordinal_param_fails(
    ordinal_param_123: OrdinalParam,
):
    """Test that adding an invalid constraint to an ordinal parameter fails."""
    with pytest.raises(ValueError):
        ordinal_param_123.add_constraint(
            EquationConstraint(
                ordinal_param_123.variable, ordinal_param_123.variable_expression > 1
            )
        )


def test_categorical_param_initialization():
    """Test that a categorical parameter can be initialized."""
    param = CategoricalParam({"a", "b", "c"})
    assert not param.is_value_set()


@pytest.fixture()
def categorical_param_abc() -> CategoricalParam:
    return CategoricalParam({"a", "b", "c"})


def test_set_categorical_param_value(categorical_param_abc: CategoricalParam):
    """Test that a categorical parameter value can be set."""
    categorical_param_abc.set_value("a")
    assert categorical_param_abc.is_value_set()
    categorical_param_abc.set_value("c")
    assert categorical_param_abc.is_value_set()


def test_set_categorical_param_value_fails_with_invalid_value(
    categorical_param_abc: CategoricalParam,
):
    """Test that setting a categorical parameter value fails with an invalid value."""
    with pytest.raises(ValueError):
        categorical_param_abc.set_value("d")


def test_add_and_check_categorical_param_constraints(
    categorical_param_abc: CategoricalParam,
):
    """Test that categorical parameter constraints can be added and checked."""
    categorical_param_abc.add_constraint(
        InSetConstraint({categorical_param_abc.variable}, {"a", "b"})
    )
    _assert_all_satisfied(categorical_param_abc, ["a", "b"])
    _assert_none_satisfied(categorical_param_abc, ["c"])


def test_adding_invalid_constraint_to_categorical_param_fails(
    categorical_param_abc: CategoricalParam,
):
    """Test that adding an invalid constraint to a categorical parameter fails."""
    with pytest.raises(ValueError):
        categorical_param_abc.add_constraint(
            EquationConstraint(
                categorical_param_abc.variable,
                categorical_param_abc.variable_expression > "a",
            )
        )


def test_perm_param_initialization():
    """Test that a permutation parameter can be initialized."""
    param = PermParam(["n", "c", "h", "w"])
    assert not param.is_value_set()


def test_perm_param_initialization_fails_with_non_unique_values():
    """Test that a permutation parameter initialization fails with non-unique values."""
    with pytest.raises(ValueError):
        PermParam(["n", "c", "h", "n"])


@pytest.fixture()
def perm_param_nchw() -> PermParam:
    return PermParam(["n", "c", "h", "w"])


def test_set_perm_param_value(perm_param_nchw: PermParam):
    """Test that a permutation parameter value can be set."""
    perm_param_nchw.set_value(["c", "n", "w", "h"])
    assert perm_param_nchw.is_value_set()


def test_set_perm_param_value_fails_with_invalid_value(perm_param_nchw: PermParam):
    """Test that setting a permutation parameter value fails with an invalid value."""
    with pytest.raises(ValueError):
        perm_param_nchw.set_value(["n", "c", "h", "n"])


def test_add_and_check_perm_param_constraints(perm_param_nchw: PermParam):
    """Test that permutation parameter constraints can be added and checked."""
    perm_param_nchw.add_constraint(
        InSetConstraint(
            {perm_param_nchw.variable}, {("n", "c", "h", "w"), ("c", "n", "w", "h")}
        )
    )
    _assert_all_satisfied(perm_param_nchw, [["n", "c", "h", "w"], ["c", "n", "w", "h"]])
    _assert_none_satisfied(perm_param_nchw, [["n", "c", "w", "h"]])


def test_adding_invalid_constraint_to_perm_param_fails(perm_param_nchw: PermParam):
    """Test that adding an invalid constraint to a permutation parameter fails."""
    with pytest.raises(ValueError):
        perm_param_nchw.add_constraint(
            EquationConstraint(
                perm_param_nchw.variable, perm_param_nchw.variable_expression > 1
            )
        )


def test_copy_real_param():
    """Test that a real parameter can be copied."""
    real_param = RealParam()
    real_param_copy = real_param.copy()
    assert real_param_copy.variable is real_param.variable
    assert real_param_copy is not real_param


def test_copy_int_param():
    """Test that an integer parameter can be copied."""
    int_param = IntParam()
    int_param_copy = int_param.copy()
    assert int_param_copy.variable is int_param.variable
    assert int_param_copy is not int_param


def test_copy_ordinal_param(ordinal_param_123: OrdinalParam):
    """Test that an ordinal parameter can be copied."""
    ordinal_param_copy = ordinal_param_123.copy()
    assert ordinal_param_copy.variable is ordinal_param_123.variable
    assert ordinal_param_copy is not ordinal_param_123


def test_copy_categorical_param(categorical_param_abc: CategoricalParam):
    """Test that a categorical parameter can be copied."""
    categorical_param_copy = categorical_param_abc.copy()
    assert categorical_param_copy.variable is categorical_param_abc.variable
    assert categorical_param_copy is not categorical_param_abc


def test_copy_perm_param(perm_param_nchw: PermParam):
    """Test that a permutation parameter can be copied."""
    perm_param_copy = perm_param_nchw.copy()
    assert perm_param_copy.variable is perm_param_nchw.variable
    assert perm_param_copy is not perm_param_nchw


def test_copied_param_keeps_constraints(ordinal_param_123: OrdinalParam):
    """Test that a copied parameter keeps its constraints."""
    ordinal_param_123.add_constraint(
        InSetConstraint({ordinal_param_123.variable}, {1, 2})
    )
    ordinal_param_copy = ordinal_param_123.copy()
    _assert_all_satisfied(ordinal_param_copy, [1, 2])
    _assert_none_satisfied(ordinal_param_copy, [3])


@pytest.mark.parametrize(
    "factory, ops, pass_values, fail_values",
    [
        # --- lower bound (mutating) ---
        pytest.param(
            partial(RealParam),
            [
                ("add_lower_bound_constraint", (1.0,), {"is_inclusive": True}),
            ],
            [1.0, 2.0],
            [0.5],
            id="real-lower-mutating-inclusive-1.0",
        ),
        pytest.param(
            partial(RealParam),
            [("add_lower_bound_constraint", (1.0,), {"is_inclusive": False})],
            [1.5, 2.0],
            [1.0, 0.5],
            id="real-lower-mutating-exclusive-1.0",
        ),
        # --- lower bound (constructor) ---
        pytest.param(
            partial(RealParam.with_lower_bound, 1.0, is_inclusive=True),
            [],
            [1.0, 2.0],
            [0.5],
            id="real-lower-constructor-inclusive-1.0",
        ),
        pytest.param(
            partial(RealParam.with_lower_bound, 1.0, is_inclusive=False),
            [],
            [1.5, 2.0],
            [1.0, 0.5],
            id="real-lower-constructor-exclusive-1.0",
        ),
        # --- upper bound (mutating) ---
        pytest.param(
            partial(RealParam),
            [
                ("add_upper_bound_constraint", (2.0,), {"is_inclusive": True}),
            ],
            [2.0, 1.0],
            [2.5],
            id="real-upper-mutating-inclusive-2.0",
        ),
        pytest.param(
            partial(RealParam),
            [("add_upper_bound_constraint", (2.0,), {"is_inclusive": False})],
            [1.0, 1.5],
            [2.0, 2.5],
            id="real-upper-mutating-exclusive-2.0",
        ),
        # --- upper bound (constructor) ---
        pytest.param(
            partial(RealParam.with_upper_bound, 2.0, is_inclusive=True),
            [],
            [2.0, 1.0],
            [2.5],
            id="real-upper-constructor-inclusive-2.0",
        ),
        pytest.param(
            partial(RealParam.with_upper_bound, 2.0, is_inclusive=False),
            [],
            [1.0, 1.5],
            [2.0, 2.5],
            id="real-upper-constructor-exclusive-2.0",
        ),
        # --- between (mutating) ---
        pytest.param(
            partial(RealParam),
            [
                ("add_lower_bound_constraint", (1.0,), {"is_inclusive": True}),
                ("add_upper_bound_constraint", (2.0,), {"is_inclusive": True}),
            ],
            [1.0, 1.5, 2.0],
            [0.5, 2.5],
            id="real-between-mutating-[1,2]",
        ),
        pytest.param(
            partial(RealParam),
            [
                ("add_lower_bound_constraint", (1.0,), {"is_inclusive": False}),
                ("add_upper_bound_constraint", (2.0,), {"is_inclusive": False}),
            ],
            [1.5],
            [1.0, 2.0, 0.5, 2.5],
            id="real-between-mutating-(1,2)",
        ),
        # --- between (constructor) ---
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
            id="real-between-constructor-[1,2]",
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
            id="real-between-constructor-(1,2)",
        ),
        # --- lower bound (mutating) ---
        pytest.param(
            partial(IntParam),
            [
                ("add_lower_bound_constraint", (1,), {"is_inclusive": True}),
            ],
            [1, 2],
            [0],
            id="int-lower-mutating-inclusive-1",
        ),
        pytest.param(
            partial(IntParam),
            [("add_lower_bound_constraint", (1,), {"is_inclusive": False})],
            [2, 5],
            [-1, 0],
            id="int-lower-mutating-exclusive-1",
        ),
        # --- lower bound (constructor) ---
        pytest.param(
            partial(IntParam.with_lower_bound, 1, is_inclusive=True),
            [],
            [1, 2],
            [0],
            id="int-lower-constructor-inclusive-1",
        ),
        pytest.param(
            partial(IntParam.with_lower_bound, 1, is_inclusive=False),
            [],
            [2, 5],
            [1, 0],
            id="int-lower-constructor-exclusive-1",
        ),
        # --- upper bound (mutating) ---
        pytest.param(
            partial(IntParam),
            [
                ("add_upper_bound_constraint", (2,), {"is_inclusive": True}),
            ],
            [2, 1],
            [3],
            id="int-upper-mutating-inclusive-2",
        ),
        pytest.param(
            partial(IntParam),
            [("add_upper_bound_constraint", (2,), {"is_inclusive": False})],
            [0, 1],
            [2, 3],
            id="int-upper-mutating-exclusive-2",
        ),
        # --- upper bound (constructor) ---
        pytest.param(
            partial(IntParam.with_upper_bound, 2, is_inclusive=True),
            [],
            [2, 1],
            [3],
            id="int-upper-constructor-inclusive-2",
        ),
        pytest.param(
            partial(IntParam.with_upper_bound, 2, is_inclusive=False),
            [],
            [1, 1],
            [2, 2],
            id="int-upper-constructor-exclusive-2",
        ),
        # --- between (mutating) ---
        pytest.param(
            partial(IntParam),
            [
                ("add_lower_bound_constraint", (1,), {"is_inclusive": True}),
                ("add_upper_bound_constraint", (2,), {"is_inclusive": True}),
            ],
            [1, 2],
            [-1, 3],
            id="int-between-mutating-[1,2]",
        ),
        pytest.param(
            partial(IntParam),
            [
                ("add_lower_bound_constraint", (1,), {"is_inclusive": False}),
                ("add_upper_bound_constraint", (3,), {"is_inclusive": False}),
            ],
            [2],
            [1, 3, 5, -5],
            id="int-between-mutating-(1,3)",
        ),
        # --- between (constructor) ---
        pytest.param(
            partial(
                IntParam.between, 1, 2, is_lower_inclusive=True, is_upper_inclusive=True
            ),
            [],
            [1, 2],
            [0, 3],
            id="int-between-constructor-[1,2]",
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
            id="int-between-constructor-(1,3)",
        ),
    ],
)
def test_numeric_params_adding_lower_and_upper_bounds(
    factory, ops, pass_values, fail_values
):
    """Test adding lower and upper bound constraints to int and real parameters."""
    param = factory()
    for name, args, kwargs in ops:
        getattr(param, name)(*args, **kwargs)
    _assert_all_satisfied(param, pass_values)
    _assert_none_satisfied(param, fail_values)


@pytest.mark.parametrize(
    "factory, ops",
    [
        pytest.param(
            partial(RealParam),
            [("add_lower_bound_constraint", ("invalid",))],
            id="real-lower-mutating-invalid",
        ),
        pytest.param(
            partial(RealParam),
            [("add_upper_bound_constraint", ("invalid",))],
            id="real-upper-mutating-invalid",
        ),
        pytest.param(
            partial(RealParam.with_upper_bound, "invalid"),
            [],
            id="real-upper-constructor-invalid",
        ),
        pytest.param(
            partial(RealParam.with_lower_bound, "invalid"),
            [],
            id="real-lower-constructor-invalid",
        ),
        pytest.param(
            partial(RealParam.between, 2.0, 1.0),
            [],
            id="real-between-constructor-invalid",
        ),
        pytest.param(
            partial(IntParam.between, 2, 1),
            [],
            id="int-between-constructor-invalid",
        ),
        pytest.param(
            partial(NatParam, is_zero_included=False),
            [("add_lower_bound_constraint", (0,))],
            id="nat-lower-mutating-invalid-zero-excluded",
        ),
        pytest.param(
            partial(NatParam, is_zero_included=False),
            [("add_lower_bound_constraint", (-1,))],
            id="nat-lower-mutating-invalid-negative-bound",
        ),
        pytest.param(
            partial(NatParam, is_zero_included=True),
            [("add_lower_bound_constraint", (-1,))],
            id="nat-lower-mutating-invalid-zero-included-exclusive",
        ),
        pytest.param(
            partial(NatParam, is_zero_included=False),
            [("add_upper_bound_constraint", (0,))],
            id="nat-upper-mutating-invalid-zero-included-exclusive",
        ),
        pytest.param(
            partial(NatParam, is_zero_included=False),
            [("add_upper_bound_constraint", (-1,))],
            id="nat-upper-mutating-invalid-zero-excluded",
        ),
        pytest.param(
            partial(NatParam, is_zero_included=True),
            [("add_upper_bound_constraint", (-1,))],
            id="nat-upper-mutating-invalid-negative-bound",
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
            id="nat-between-constructor-invalid-negative-bound",
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
            id="nat-between-constructor-invalid-reversed-bound",
        ),
    ],
)
def test_numeric_params_adding_invalid_lower_and_upper_bounds_fails(factory, ops):
    with pytest.raises(ValueError):
        param = factory()
        for name, args, kwargs in ops:
            getattr(param, name)(*args, **kwargs)


def test_zero_included_in_nat_param_with_lower_bound_inclusive_at_zero():
    """Test that creating a NatParam with lower bound 0 inclusive works.

    This is specifically to test the constructor argument for including zero
    in the natural number constraints added before the lower bound constraint
    is added.
    """
    param = NatParam.with_lower_bound(0, is_inclusive=True)
    _assert_all_satisfied(param, [0, 1, 2, 100])


# TODO: Test the repr method.


def test_bound_int_param_between_inclusive_inclusive_constraints_satisfied():
    """Test ``between'' with inclusive bounds.

    Integer semantics: [3,5] => {3,4,5}
    """
    p = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    _assert_all_satisfied(p, [3, 4, 5])
    _assert_none_satisfied(p, [2, 6])


def test_bound_int_param_between_exclusive_exclusive_constraints_satisfied():
    """Test ``between'' with exclusive bounds.

    Integer semantics: (3,5) => only {4}
    """
    p = BoundIntParam.between(3, 5, is_lower_inclusive=False, is_upper_inclusive=False)
    _assert_all_satisfied(p, [4])
    _assert_none_satisfied(p, [3, 5, 2, 6])


def test_bound_int_param_between_exclusive_inclusive_constraints_satisfied():
    """Test ``between'' with exclusive lower bound and inclusive upper bound.

    Integer semantics: (3,5] => {4,5}
    """
    p = BoundIntParam.between(3, 5, is_lower_inclusive=False, is_upper_inclusive=True)
    _assert_all_satisfied(p, [4, 5])
    _assert_none_satisfied(p, [3, 2, 6])


def test_bound_int_param_between_inclusive_exclusive_constraints_satisfied():
    """Test ``between'' with inclusive lower bound and exclusive upper bound.

    Integer semantics: [3,5) => {3,4}
    """
    p = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=False)
    _assert_all_satisfied(p, [3, 4])
    _assert_none_satisfied(p, [5, 2, 6])


def test_bound_int_param_between_invalid_interval_raises_for_strict_equal_bounds():
    """Test that ``between'' raises for intervals with strict equal bounds."""
    with pytest.raises(ValueError):
        BoundIntParam.between(3, 3, is_lower_inclusive=False, is_upper_inclusive=False)


def test_bound_int_param_between_equal_bounds_inclusive_is_singleton():
    """Test that ``between'' with equal inclusive bounds is a singleton."""
    p = BoundIntParam.between(3, 3, is_lower_inclusive=True, is_upper_inclusive=True)
    _assert_all_satisfied(p, [3])
    _assert_none_satisfied(p, [2, 4])


def test_bound_int_param_between_invalid_order_raises_error():
    """Test that ``between'' raises for intervals with reversed bounds."""
    with pytest.raises(ValueError):
        BoundIntParam.between(5, 3)


def test_bound_int_param_with_lower_bound_inclusive():
    """Test ``with_lower_bound'' with inclusive bound."""
    p = BoundIntParam.with_lower_bound(3, is_inclusive=True)
    _assert_all_satisfied(p, [3, 4, 100])
    _assert_none_satisfied(p, [2, -1])


def test_bound_int_param_with_lower_bound_exclusive_integer_semantics():
    """Test ``with_lower_bound'' with exclusive bound.

    Integer semantics: x > 3 => ints {4,5,...}
    """
    p = BoundIntParam.with_lower_bound(3, is_inclusive=False)
    _assert_all_satisfied(p, [4, 5, 100])
    _assert_none_satisfied(p, [3, 2, -10])


def test_bound_int_param_with_upper_bound_inclusive():
    p = BoundIntParam.with_upper_bound(5, is_inclusive=True)
    _assert_all_satisfied(p, [5, 4, -100])
    _assert_none_satisfied(p, [6, 7])


def test_bound_int_param_with_upper_bound_exclusive_integer_semantics():
    """Test ``with_upper_bound'' with exclusive bound.

    Integer semantics: x < 5 => ints {...,3,4}
    """
    p = BoundIntParam.with_upper_bound(5, is_inclusive=False)
    _assert_all_satisfied(p, [4, 3, -100])
    _assert_none_satisfied(p, [5, 6])


def test_bound_int_param_exactly_satisfies_only_that_value():
    """Test ``exactly'' creates a singleton parameter."""
    p = BoundIntParam.exactly(7)
    _assert_all_satisfied(p, [7])
    _assert_none_satisfied(p, [6, 8, 0])


def test_bound_int_param_prefer_inclusive_flag_does_not_change_satisfiable_set():
    """Test that prefer_inclusive flag does not change satisfiable set."""
    p1 = BoundIntParam.between(
        3, 5, is_lower_inclusive=False, is_upper_inclusive=False, prefer_inclusive=True
    )
    p2 = BoundIntParam.between(
        3, 5, is_lower_inclusive=False, is_upper_inclusive=False, prefer_inclusive=False
    )
    for v in range(0, 10):
        assert p1.is_constraints_satisfied(v) == p2.is_constraints_satisfied(v)


def test_bound_int_param_set_value_accepts_int_only():
    """Test that BoundIntParam.set_value only accepts integer values."""
    p = BoundIntParam.with_lower_bound(0)
    p.set_value(1)
    assert p.get_value() == 1
    with pytest.raises(ValueError):
        p.set_value(1.0)
    with pytest.raises(ValueError):
        p.set_value("1")


def test_bound_int_param_set_value_rejects_value_outside_constraints():
    """Test that ``set_value'' rejects values outside constraints."""
    p = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    with pytest.raises(ValueError):
        p.set_value(2)
    with pytest.raises(ValueError):
        p.set_value(6)
    p.set_value(4)
    assert p.get_value() == 4


def test_bound_int_param_addition_of_singletons_is_singleton():
    """Test addition of two singleton BoundIntParams results in a singleton."""
    x = BoundIntParam.exactly(4)
    y = BoundIntParam.exactly(6)
    z = x + y
    _assert_all_satisfied(z, [10])
    _assert_none_satisfied(z, [9, 11])


def test_bound_int_param_addition_with_int_rhs():
    """Test addition of BoundIntParam with an integer on the right-hand side."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    z = x + 2
    _assert_all_satisfied(z, [5, 6, 7])
    _assert_none_satisfied(z, [4, 8])


def test_bound_int_param_addition_with_int_lhs():
    """Test addition of BoundIntParam with an integer on the left-hand side."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    z = 2 + x
    _assert_all_satisfied(z, [5, 6, 7])
    _assert_none_satisfied(z, [4, 8])


def test_bound_int_param_addition_bounds_propagate_semantics_from_strict_inputs():
    """Test addition bounds propagation with strict intervals.

    Integer semantics:
    x: (3,5) => {4}
    y: (5,10) => {6,7,8,9}
    x+y => {10,11,12,13}
    """
    x = BoundIntParam.between(3, 5, is_lower_inclusive=False, is_upper_inclusive=False)
    y = BoundIntParam.between(5, 10, is_lower_inclusive=False, is_upper_inclusive=False)
    z = x + y
    _assert_all_satisfied(z, [10, 11, 12, 13])
    _assert_none_satisfied(z, [9, 14])


def test_bound_int_param_addition_unbounded_lower_or_upper_propagates_unboundedness():
    """Test addition bounds propagation with unbounded inputs.

    Integer semantics:
    x >= 3, y unbounded => z >= 3 + (-inf) = -inf
    """
    x = BoundIntParam.with_lower_bound(3, is_inclusive=True)
    y = BoundIntParam()
    z = x + y
    _assert_all_satisfied(z, [-(10**6), 0, 10**6])


def test_bound_int_param_subtraction_of_singletons_is_singleton():
    """Test subtraction of two singleton BoundIntParams results in a singleton."""
    x = BoundIntParam.exactly(10)
    y = BoundIntParam.exactly(6)
    z = x - y
    _assert_all_satisfied(z, [4])
    _assert_none_satisfied(z, [3, 5])


def test_bound_int_param_subtraction_with_int_rhs():
    """Test subtraction of BoundIntParam with an integer on the right-hand side."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    z = x - 2
    _assert_all_satisfied(z, [1, 2, 3])
    _assert_none_satisfied(z, [0, 4])


def test_bound_int_param_subtraction_with_int_lhs():
    """Test subtraction of BoundIntParam with an integer on the left-hand side."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    z = 10 - x
    _assert_all_satisfied(z, [5, 6, 7])
    _assert_none_satisfied(z, [4, 8])


def test_bound_int_param_subtraction_bounds_propagate_semantics_from_strict_inputs():
    """Test subtraction bounds propagation with strict intervals.

    Integer semantics:
    x: (3,5) => {4}
    y: (5,10) => {6,7,8,9}
    x-y => {4-9 .. 4-6} = {-5,-4,-3,-2}
    """
    x = BoundIntParam.between(3, 5, is_lower_inclusive=False, is_upper_inclusive=False)
    y = BoundIntParam.between(5, 10, is_lower_inclusive=False, is_upper_inclusive=False)
    z = x - y
    _assert_all_satisfied(z, [-5, -4, -3, -2])
    _assert_none_satisfied(z, [-6, -1, 0])


def test_bound_int_param_negation_of_singleton():
    """Test negation of a singleton BoundIntParam results in a singleton."""
    x = BoundIntParam.exactly(4)
    z = -x
    _assert_all_satisfied(z, [-4])
    _assert_none_satisfied(z, [-3, -5])


def test_bound_int_param_negation_of_interval():
    """Test negation of an interval BoundIntParam."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    z = -x
    _assert_all_satisfied(z, [-5, -4, -3])
    _assert_none_satisfied(z, [-6, -2])


def test_bound_int_param_negation_of_strict_interval_integer_semantics():
    """Test negation of a strict interval BoundIntParam."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=False, is_upper_inclusive=False)
    z = -x
    _assert_all_satisfied(z, [-4])
    _assert_none_satisfied(z, [-5, -3])


def test_bound_int_param_prefer_inclusive_changes_str_not_membership_for_addition():
    """Test that prefer_inclusive changes string but not membership for addition."""
    x_incl = BoundIntParam.between(
        3, 5, is_lower_inclusive=False, is_upper_inclusive=False, prefer_inclusive=True
    )
    y_incl = BoundIntParam.between(
        5, 10, is_lower_inclusive=False, is_upper_inclusive=False, prefer_inclusive=True
    )

    x_excl = BoundIntParam.between(
        3, 5, is_lower_inclusive=False, is_upper_inclusive=False, prefer_inclusive=False
    )
    y_excl = BoundIntParam.between(
        5,
        10,
        is_lower_inclusive=False,
        is_upper_inclusive=False,
        prefer_inclusive=False,
    )

    z_incl = x_incl + y_incl
    z_excl = x_excl + y_excl

    for v in range(-20, 40):
        assert z_incl.is_constraints_satisfied(v) == z_excl.is_constraints_satisfied(v)
    assert str(z_incl) != str(z_excl)


def test_bound_int_param_addition_accepts_int_param():
    """Test addition of BoundIntParam with IntParam on the right-hand side."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    y = IntParam.between(5, 10, is_lower_inclusive=True, is_upper_inclusive=True)
    z = x + y
    _assert_all_satisfied(z, [8, 9, 10, 11, 12, 13, 14, 15])
    _assert_none_satisfied(z, [7, 16])


def test_bound_int_param_subtraction_accepts_int_param():
    """Test subtraction of BoundIntParam with IntParam on the right-hand side."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    y = IntParam.between(5, 10, is_lower_inclusive=True, is_upper_inclusive=True)
    z = x - y
    _assert_all_satisfied(z, [-7, -3, 0])
    _assert_none_satisfied(z, [-8, 1])


def test_bound_int_param_rsub_accepts_int_param_on_left():
    """Test subtraction of BoundIntParam with IntParam on the left-hand side."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    y = IntParam.between(5, 10, is_lower_inclusive=True, is_upper_inclusive=True)
    z = y - x
    _assert_all_satisfied(z, [0, 7])
    _assert_none_satisfied(z, [-1, 8])


def test_bound_int_param_addition_with_unsupported_type_raises_error():
    """Test that addition of BoundIntParam with unsupported type raises TypeError."""
    x = BoundIntParam.between(0, 1)
    with pytest.raises(TypeError):
        _ = x + "nope"


@pytest.mark.parametrize(
    "lower,upper,lin,u_in",
    [
        (0, 0, True, True),
        (0, 1, True, True),
        (0, 1, False, True),
        (0, 1, True, False),
        (0, 2, False, False),
        (-3, 3, True, True),
        (-3, 3, False, False),
    ],
)
def test_bound_int_param_addition_matches_brute_force(lower, upper, lin, u_in):
    """Test that addition matches brute-force set addition."""
    x = BoundIntParam.between(
        lower, upper, is_lower_inclusive=lin, is_upper_inclusive=u_in
    )
    y = BoundIntParam.between(
        lower, upper, is_lower_inclusive=lin, is_upper_inclusive=u_in
    )
    z = x + y
    allowed_x = [
        v for v in range(lower - 2, upper + 3) if x.is_constraints_satisfied(v)
    ]
    allowed_y = [
        v for v in range(lower - 2, upper + 3) if y.is_constraints_satisfied(v)
    ]
    allowed_z = {a + b for a in allowed_x for b in allowed_y}
    for v in range(2 * (lower - 2), 2 * (upper + 2) + 1):
        assert z.is_constraints_satisfied(v) == (v in allowed_z)


@pytest.mark.parametrize(
    "lower,upper,lin,u_in",
    [
        (0, 0, True, True),
        (0, 1, True, True),
        (0, 2, False, False),
        (-2, 2, True, True),
        (-2, 2, False, False),
    ],
)
def test_bound_int_param_subtraction_matches_brute_force(lower, upper, lin, u_in):
    """Test that subtraction matches brute-force set subtraction."""
    x = BoundIntParam.between(
        lower, upper, is_lower_inclusive=lin, is_upper_inclusive=u_in
    )
    y = BoundIntParam.between(
        lower, upper, is_lower_inclusive=lin, is_upper_inclusive=u_in
    )
    z = x - y
    allowed_x = [
        v for v in range(lower - 2, upper + 3) if x.is_constraints_satisfied(v)
    ]
    allowed_y = [
        v for v in range(lower - 2, upper + 3) if y.is_constraints_satisfied(v)
    ]
    allowed_z = {a - b for a in allowed_x for b in allowed_y}
    for v in range((lower - 2) - (upper + 2), (upper + 2) - (lower - 2) + 1):
        assert z.is_constraints_satisfied(v) == (v in allowed_z)


# TODO: Update tests that check exceptions to match exception messages.
