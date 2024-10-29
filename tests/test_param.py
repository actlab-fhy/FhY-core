"""Tests the parameter utility."""

import pytest
from fhy_core.constraint import EquationConstraint, InSetConstraint
from fhy_core.param import (
    CategoricalParam,
    IntParam,
    NatParam,
    OrdinalParam,
    PermParam,
    RealParam,
)


def test_param_is_not_set_after_initialization():
    """Test that the value of a parameter is not set after initialization."""
    param = RealParam()
    assert not param.is_set()


def test_param_is_set_after_setting_value():
    """Test that the value of a parameter is set after setting a value."""
    param = RealParam()
    param.set_value(1.0)
    assert param.is_set()


def test_param_get_value_fails_when_not_set():
    """Test that getting the value of a parameter fails when the value is not set."""
    param = RealParam()
    with pytest.raises(ValueError):
        param.get_value()


def test_get_param_value_after_setting_value():
    """Test that the value of a parameter can be retrieved after setting a value."""
    param = RealParam()
    param.set_value(1.0)
    assert param.get_value() == 1.0


def test_real_param_value_set_fails_with_invalid_value():
    """Test that setting a real parameter value fails with an invalid value."""
    param = RealParam()
    with pytest.raises(ValueError):
        param.set_value([])


def test_int_param_value_set_fails_with_invalid_value():
    """Test that setting an integer parameter value fails with an invalid value."""
    param = IntParam()
    with pytest.raises(ValueError):
        param.set_value(1.0)


def test_add_and_check_real_param_constraints():
    """Test that a real constraint can be added and checked."""
    param = RealParam()
    param.add_constraint(EquationConstraint(param.variable_expression * 3.14 < 20.0))
    param.add_constraint(EquationConstraint(param.variable_expression >= 1.0))
    assert param.is_constraints_satisfied(2.0)
    assert not param.is_constraints_satisfied(0.5)
    assert not param.is_constraints_satisfied(7.0)


def test_add_and_check_int_param_constraints():
    """Test that an integer constraint can be added and checked."""
    param = IntParam()
    param.add_constraint(EquationConstraint((param.variable_expression % 5).equals(0)))
    param.add_constraint(EquationConstraint(param.variable_expression > 10))
    assert param.is_constraints_satisfied(15)
    with pytest.raises(ValueError):
        param.set_value(12)


def test_nat_param_zero_included():
    """Test that a natural number parameter with zero included can be set."""
    param = NatParam()
    param.set_value(0)
    assert param.is_set()
    param.set_value(1)
    assert param.is_set()
    with pytest.raises(ValueError):
        param.set_value(-1)


def test_nat_param_zero_excluded():
    """Test that a natural number parameter with zero excluded can be set."""
    param = NatParam(is_zero_included=False)
    param.set_value(1)
    assert param.is_set()
    with pytest.raises(ValueError):
        param.set_value(0)
    with pytest.raises(ValueError):
        param.set_value(-1)


def test_ordinal_param_initialization():
    """Test that an ordinal parameter can be initialized."""
    param = OrdinalParam([5, 6, 7])
    assert not param.is_set()


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
    assert ordinal_param_123.is_set()
    ordinal_param_123.set_value(3)
    assert ordinal_param_123.is_set()


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
    assert ordinal_param_123.is_constraints_satisfied(1)
    assert ordinal_param_123.is_constraints_satisfied(2)
    assert not ordinal_param_123.is_constraints_satisfied(3)


def test_adding_invalid_constraint_to_ordinal_param_fails(
    ordinal_param_123: OrdinalParam,
):
    """Test that adding an invalid constraint to an ordinal parameter fails."""
    with pytest.raises(ValueError):
        ordinal_param_123.add_constraint(
            EquationConstraint(ordinal_param_123.variable_expression > 1)
        )


def test_categorical_param_initialization():
    """Test that a categorical parameter can be initialized."""
    param = CategoricalParam({"a", "b", "c"})
    assert not param.is_set()


@pytest.fixture()
def categorical_param_abc() -> CategoricalParam:
    return CategoricalParam({"a", "b", "c"})


def test_set_categorical_param_value(categorical_param_abc: CategoricalParam):
    """Test that a categorical parameter value can be set."""
    categorical_param_abc.set_value("a")
    assert categorical_param_abc.is_set()
    categorical_param_abc.set_value("c")
    assert categorical_param_abc.is_set()


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
    assert categorical_param_abc.is_constraints_satisfied("a")
    assert categorical_param_abc.is_constraints_satisfied("b")
    assert not categorical_param_abc.is_constraints_satisfied("c")


def test_adding_invalid_constraint_to_categorical_param_fails(
    categorical_param_abc: CategoricalParam,
):
    """Test that adding an invalid constraint to a categorical parameter fails."""
    with pytest.raises(ValueError):
        categorical_param_abc.add_constraint(
            EquationConstraint(categorical_param_abc.variable_expression > "a")
        )


def test_perm_param_initialization():
    """Test that a permutation parameter can be initialized."""
    param = PermParam(["n", "c", "h", "w"])
    assert not param.is_set()


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
    assert perm_param_nchw.is_set()


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
    assert perm_param_nchw.is_constraints_satisfied(["n", "c", "h", "w"])
    assert perm_param_nchw.is_constraints_satisfied(["c", "n", "w", "h"])
    assert not perm_param_nchw.is_constraints_satisfied(["n", "c", "w", "h"])


def test_adding_invalid_constraint_to_perm_param_fails(perm_param_nchw: PermParam):
    """Test that adding an invalid constraint to a permutation parameter fails."""
    with pytest.raises(ValueError):
        perm_param_nchw.add_constraint(
            EquationConstraint(perm_param_nchw.variable_expression > 1)
        )
