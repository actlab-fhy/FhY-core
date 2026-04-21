"""Tests the parameter utility."""

from functools import partial
from typing import Any, TypeVar

import pytest
from fhy_core.constraint import EquationConstraint, InSetConstraint
from fhy_core.expression import SymbolType
from fhy_core.identifier import Identifier
from fhy_core.param import (
    BoundIntParam,
    BoundNatParam,
    CategoricalParam,
    IntParam,
    NatParam,
    OrdinalParam,
    Param,
    ParamAssignment,
    PermParam,
    RealParam,
    create_single_valid_value_param,
)
from fhy_core.serialization import (
    DeserializationValueError,
    Serializable,
    SerializationFormat,
    register_serializable,
    serialize_registry_wrapped_value,
)
from fhy_core.trait import StructuralEquivalence

T = TypeVar("T")


@register_serializable(type_id="tests.param.serializable_hash_only")
class _SerializableHashOnly(Serializable):
    """Serializable value that is hashable but has identity equality semantics."""

    _value: int

    def __init__(self, value: int) -> None:
        self._value = value

    def __hash__(self) -> int:
        return hash(self._value)

    def serialize_to_dict(self) -> dict[str, Any]:
        return {"value": self._value}

    @classmethod
    def deserialize_from_dict(cls, data: dict[str, Any]) -> "_SerializableHashOnly":
        return cls(value=int(data["value"]))


@register_serializable(type_id="tests.param.serializable_equal_no_order")
class _SerializableEqualNoOrder(Serializable):
    """Serializable value with equality semantics but no ordering semantics."""

    _value: int

    def __init__(self, value: int) -> None:
        self._value = value

    def __hash__(self) -> int:
        return hash(self._value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _SerializableEqualNoOrder) and (
            self._value == other._value
        )

    def serialize_to_dict(self) -> dict[str, Any]:
        return {"value": self._value}

    @classmethod
    def deserialize_from_dict(cls, data: dict[str, Any]) -> "_SerializableEqualNoOrder":
        return cls(value=int(data["value"]))


def _assert_all_satisfied(param: Param[Any], values: list[Any]) -> None:
    for v in values:
        assert param.is_constraints_satisfied(
            v
        ), f"Value {v} should satisfy constraints of parameter {param}"


def _assert_none_satisfied(param: Param[Any], values: list[Any]) -> None:
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


def test_param_structural_equivalence_runtime_protocol() -> None:
    """Test `Param` implementations satisfy `StructuralEquivalence` protocol."""
    param = IntParam()
    assert isinstance(param, StructuralEquivalence)


def test_param_structural_equivalence_true_for_constraint_reordering() -> None:
    """Test params compare structurally when equivalent constraints are reordered."""
    shared_name = Identifier("x")
    shared_name_copy = Identifier.deserialize_from_dict(shared_name.serialize_to_dict())

    left_base = IntParam(name=shared_name)
    right_base = IntParam(name=shared_name_copy)

    left = left_base.add_constraints(
        [
            EquationConstraint(left_base.variable, left_base.variable_expression >= 0),
            EquationConstraint(left_base.variable, left_base.variable_expression <= 10),
        ]
    )
    right = right_base.add_constraints(
        [
            EquationConstraint(
                right_base.variable,
                right_base.variable_expression <= 10,
            ),
            EquationConstraint(
                right_base.variable,
                right_base.variable_expression >= 0,
            ),
        ]
    )

    assert left.is_structurally_equivalent(right)


def test_param_structural_equivalence_false_for_different_constraints() -> None:
    """Test params are not structurally equivalent with different constraints."""
    shared_name = Identifier("x")
    shared_name_copy = Identifier.deserialize_from_dict(shared_name.serialize_to_dict())

    left_base = IntParam(name=shared_name)
    right_base = IntParam(name=shared_name_copy)

    left = left_base.add_constraints(
        [
            EquationConstraint(left_base.variable, left_base.variable_expression >= 0),
            EquationConstraint(left_base.variable, left_base.variable_expression <= 10),
        ]
    )
    right = right_base.add_constraints(
        [
            EquationConstraint(
                right_base.variable, right_base.variable_expression >= 1
            ),
            EquationConstraint(
                right_base.variable, right_base.variable_expression <= 10
            ),
        ]
    )

    assert not left.is_structurally_equivalent(right)


def test_param_is_not_set_after_initialization(default_real_param: RealParam) -> None:
    """Test that the value of a parameter is not set after initialization."""
    assignment = default_real_param.assign(1.0)
    assert isinstance(default_real_param, RealParam)
    assert isinstance(assignment, ParamAssignment)
    assert assignment.param is default_real_param
    assert not hasattr(default_real_param, "is_value_set")


def test_param_is_set_after_setting_value(default_real_param: RealParam) -> None:
    """Test that the value of a parameter is set after setting a value."""
    assignment = default_real_param.assign(1.0)
    assert assignment.is_value_set()


def test_param_get_value_fails_when_not_set(default_real_param: RealParam) -> None:
    """Test that parameters no longer expose direct value access."""
    with pytest.raises(AttributeError):
        default_real_param.get_value()  # type: ignore[attr-defined]  # test: method removed


def test_get_param_value_after_setting_value(default_real_param: RealParam) -> None:
    """Test that the value of a parameter can be retrieved after setting a value."""
    assignment = default_real_param.assign(1.0)
    assert assignment.value == 1.0


def test_real_param_get_symbol_type(default_real_param: RealParam) -> None:
    """Test that the symbol type of a real parameter is correct."""
    assert default_real_param.get_symbol_type() == SymbolType.REAL


def test_int_param_get_symbol_type(default_int_param: IntParam) -> None:
    """Test that the symbol type of an integer parameter is correct."""
    assert default_int_param.get_symbol_type() == SymbolType.INT


def test_real_param_value_set_fails_with_invalid_value(
    default_real_param: RealParam,
) -> None:
    """Test that setting a real parameter value fails with an invalid value."""
    with pytest.raises(ValueError):
        default_real_param.assign([])  # type: ignore[arg-type]  # test: invalid input


def test_int_param_value_set_fails_with_invalid_value(
    default_int_param: IntParam,
) -> None:
    """Test that setting an integer parameter value fails with an invalid value."""
    with pytest.raises(ValueError):
        default_int_param.assign(1.0)  # type: ignore[arg-type]  # test: invalid input


def test_create_real_param_with_value() -> None:
    """Test that a real parameter can be created with a value."""
    param = RealParam.with_value(1.0)
    assert param.is_value_set()
    assert param.value == 1.0


def test_create_real_param_with_value_fails_with_invalid_value() -> None:
    """Test that creating a real parameter with an invalid value fails."""
    with pytest.raises(ValueError):
        RealParam.with_value("invalid")


def test_create_int_param_with_value() -> None:
    """Test that an integer parameter can be created with a value."""
    param = IntParam.with_value(1)
    assert param.is_value_set()
    assert param.value == 1


def test_create_int_param_with_value_fails_with_invalid_value() -> None:
    """Test that creating an integer parameter with an invalid value fails."""
    with pytest.raises(ValueError):
        IntParam.with_value(1.2)  # type: ignore[arg-type]  # test: invalid input


def test_param_assign_creates_assignment() -> None:
    """Test that assigning a parameter creates an immutable assignment."""
    param = IntParam.with_lower_bound(0)
    assignment = param.assign(3)
    assert isinstance(assignment, ParamAssignment)
    assert assignment.value == 3
    assert assignment.param is param


@pytest.mark.parametrize(
    ("param", "valid_value", "invalid_type_or_shape_value", "constraint_fail_value"),
    [
        (RealParam.with_lower_bound(0.0), 1.5, [], -1.0),
        (IntParam.with_lower_bound(0), 3, 1.2, -1),
        (NatParam(is_zero_included=False), 2, 1.5, 0),
        (OrdinalParam([1, 2, 3]), 2, "2", 4),
        (CategoricalParam({"a", "b"}), "a", 3, "z"),
        (PermParam(["n", "c", "h", "w"]), ("n", "c", "h", "w"), 7, ("n", "c")),
    ],
)
def test_param_is_value_valid_checks_type_and_constraints(
    param: Param[object],
    valid_value: object,
    invalid_type_or_shape_value: object,
    constraint_fail_value: object,
) -> None:
    """Test `is_value_valid` checks subclass admissibility and constraints."""
    assert param.is_value_valid(valid_value)
    assert not param.is_value_valid(invalid_type_or_shape_value)
    assert not param.is_value_valid(constraint_fail_value)


def test_param_assignment_materialize_returns_bound_param() -> None:
    """Test that materialize returns the parameter definition."""
    assignment = RealParam.with_upper_bound(2.0).assign(1.5)
    bound_param = assignment.materialize()
    assert bound_param is assignment.param


def test_add_and_check_real_param_constraints(default_real_param: RealParam) -> None:
    """Test that a real constraint can be added and checked."""
    default_real_param = default_real_param.add_constraint(
        EquationConstraint(
            default_real_param.variable,
            default_real_param.variable_expression * 3.14 < 20.0,
        )
    )
    default_real_param = default_real_param.add_constraint(
        EquationConstraint(
            default_real_param.variable, default_real_param.variable_expression >= 1.0
        )
    )
    _assert_all_satisfied(default_real_param, [2.0])
    _assert_none_satisfied(default_real_param, [0.5, 7.0])


def test_add_multiple_constraints_at_once() -> None:
    """Test that multiple constraints can be added in one call."""
    param = RealParam()
    constraints = [
        EquationConstraint(param.variable, param.variable_expression >= 1.0),
        EquationConstraint(param.variable, param.variable_expression <= 3.0),
    ]
    param = param.add_constraints(constraints)
    _assert_all_satisfied(param, [1.0, 2.0, 3.0])
    _assert_none_satisfied(param, [0.5, 3.5])


def test_add_multiple_constraints_validates_subclass_constraint_rules() -> None:
    """Test that add_constraints enforces subclass-specific constraint checks."""
    param = OrdinalParam([1, 2, 3])
    with pytest.raises(ValueError):
        param.add_constraints(
            [
                InSetConstraint(param.variable, {1, 2}),
                EquationConstraint(param.variable, param.variable_expression > 1),
            ]
        )


def test_add_and_check_int_param_constraints(default_int_param: IntParam) -> None:
    """Test that an integer constraint can be added and checked."""
    default_int_param = default_int_param.add_constraint(
        EquationConstraint(
            default_int_param.variable,
            (default_int_param.variable_expression % 5).equals(0),
        )
    )
    default_int_param = default_int_param.add_constraint(
        EquationConstraint(
            default_int_param.variable, default_int_param.variable_expression > 10
        )
    )
    assert default_int_param.is_constraints_satisfied(15)
    with pytest.raises(ValueError):
        default_int_param.assign(12)


def test_set_real_param_and_real_param_is_subset() -> None:
    """Test assignment values can be set independently from the same parameter."""
    param1 = RealParam()
    assignment_1 = param1.assign(1.0)
    assignment_2 = param1.assign(1.0)
    assert assignment_1.value == 1.0
    assert assignment_2.value == 1.0
    assert assignment_1.param is param1
    assert assignment_2.param is param1


def test_real_param_is_subset_of_real_param() -> None:
    """Test that a real parameter is a subset of another real parameter."""
    param1 = RealParam()
    param2 = RealParam()
    assert param1.is_subset(param2)
    assert param2.is_subset(param1)


def test_positive_real_param_and_real_param_is_subset() -> None:
    """Test subset relationship between positive real parameter and real parameter."""
    param1 = RealParam()
    param1 = param1.add_constraint(
        EquationConstraint(param1.variable, param1.variable_expression > 0)
    )
    param2 = RealParam()
    assert param1.is_subset(param2)
    assert not param2.is_subset(param1)


def test_interval_real_param_and_interval_real_param_is_subset() -> None:
    """Test subset relationship between interval real parameters."""
    param1 = RealParam()
    param1 = param1.add_constraint(
        EquationConstraint(param1.variable, param1.variable_expression >= 0)
    )
    param1 = param1.add_constraint(
        EquationConstraint(param1.variable, param1.variable_expression <= 3)
    )
    param2 = RealParam()
    param2 = param2.add_constraint(
        EquationConstraint(param2.variable, param2.variable_expression >= 0)
    )
    param2 = param2.add_constraint(
        EquationConstraint(param2.variable, param2.variable_expression <= 2)
    )
    assert not param1.is_subset(param2)
    assert param2.is_subset(param1)


def test_nat_param_zero_included() -> None:
    """Test that a natural number parameter with zero included can be set."""
    param = NatParam()
    assignment_0 = param.assign(0)
    assert assignment_0.is_value_set()
    assignment_1 = param.assign(1)
    assert assignment_1.is_value_set()
    with pytest.raises(ValueError):
        param.assign(-1)


def test_nat_param_zero_excluded() -> None:
    """Test that a natural number parameter with zero excluded can be set."""
    param = NatParam(is_zero_included=False)
    assignment_1 = param.assign(1)
    assert assignment_1.is_value_set()
    with pytest.raises(ValueError):
        param.assign(0)
    with pytest.raises(ValueError):
        param.assign(-1)


def test_nat_param_add_constraint_preserves_zero_excluded_state() -> None:
    """Test that adding constraints preserves zero-excluded NatParam semantics."""
    param = NatParam(is_zero_included=False)
    updated = param.add_lower_bound_constraint(2, is_inclusive=True)
    with pytest.raises(ValueError):
        updated.add_lower_bound_constraint(0, is_inclusive=True)


def test_ordinal_param_initialization() -> None:
    """Test that an ordinal parameter can be initialized."""
    param = OrdinalParam([5, 6, 7])
    assert isinstance(param, OrdinalParam)


def test_ordinal_param_initialization_fails_with_non_unique_values() -> None:
    """Test that an ordinal parameter initialization fails with non-unique values."""
    with pytest.raises(ValueError):
        OrdinalParam([1, 2, 1])


def test_ordinal_param_initialization_requires_orderable_values() -> None:
    """Test ordinal params reject wrapped-leaf values without ordering semantics."""
    with pytest.raises(TypeError):
        OrdinalParam([_SerializableEqualNoOrder(1), _SerializableEqualNoOrder(2)])  # type: ignore[type-var]  # test: invalid input


@pytest.fixture()
def ordinal_param_123() -> OrdinalParam[int]:
    return OrdinalParam([1, 2, 3])


def test_set_ordinal_param_value(ordinal_param_123: OrdinalParam[int]) -> None:
    """Test that an ordinal parameter value can be set."""
    assignment_1 = ordinal_param_123.assign(1)
    assert assignment_1.is_value_set()
    assignment_3 = ordinal_param_123.assign(3)
    assert assignment_3.is_value_set()


def test_set_ordinal_param_value_fails_with_invalid_value(
    ordinal_param_123: OrdinalParam[int],
) -> None:
    """Test that setting an ordinal parameter value fails with an invalid value."""
    with pytest.raises(ValueError):
        ordinal_param_123.assign(4)


def test_ordinal_param_distinguishes_bool_from_numeric_values() -> None:
    """Test ordinal params do not treat bools as interchangeable with numerics."""
    param = OrdinalParam([1, 2, 3])
    assert not param.is_value_admissible(True)


def test_add_and_check_ordinal_param_constraints(
    ordinal_param_123: OrdinalParam[int],
) -> None:
    """Test that ordinal parameter constraints can be added and checked."""
    ordinal_param_123 = ordinal_param_123.add_constraint(
        InSetConstraint(ordinal_param_123.variable, {1, 2})
    )
    _assert_all_satisfied(ordinal_param_123, [1, 2])
    _assert_none_satisfied(ordinal_param_123, [3])


def test_adding_invalid_constraint_to_ordinal_param_fails(
    ordinal_param_123: OrdinalParam[int],
) -> None:
    """Test that adding an invalid constraint to an ordinal parameter fails."""
    with pytest.raises(ValueError):
        ordinal_param_123.add_constraint(
            EquationConstraint(
                ordinal_param_123.variable, ordinal_param_123.variable_expression > 1
            )
        )


def test_categorical_param_initialization() -> None:
    """Test that a categorical parameter can be initialized."""
    param = CategoricalParam({"a", "b", "c"})
    assert isinstance(param, CategoricalParam)


def test_categorical_param_initialization_requires_equal_values() -> None:
    """Test categorical params reject wrapped-leaf values without equal semantics."""
    with pytest.raises(TypeError):
        CategoricalParam({_SerializableHashOnly(1), _SerializableHashOnly(2)})  # type: ignore[type-var]  # test: invalid input


@pytest.fixture()
def categorical_param_abc() -> CategoricalParam[str]:
    return CategoricalParam({"a", "b", "c"})


def test_create_single_valid_value_param() -> None:
    """Test helper creates a categorical param constrained to one value."""
    param = create_single_valid_value_param("only")
    assert isinstance(param, CategoricalParam)
    assert param.get_possible_values() == {"only"}

    assignment = param.assign("only")
    assert assignment.is_value_set()
    assert assignment.value == "only"

    with pytest.raises(ValueError):
        param.assign("different")


def test_set_categorical_param_value(
    categorical_param_abc: CategoricalParam[str],
) -> None:
    """Test that a categorical parameter value can be set."""
    assignment_a = categorical_param_abc.assign("a")
    assert assignment_a.is_value_set()
    assignment_c = categorical_param_abc.assign("c")
    assert assignment_c.is_value_set()


def test_set_categorical_param_value_fails_with_invalid_value(
    categorical_param_abc: CategoricalParam[str],
) -> None:
    """Test that setting a categorical parameter value fails with an invalid value."""
    with pytest.raises(ValueError):
        categorical_param_abc.assign("d")


def test_categorical_param_distinguishes_bool_from_int_values() -> None:
    """Test categorical params do not treat bools as interchangeable with ints."""
    param = CategoricalParam([1, 2, 3])
    assert not param.is_value_admissible(True)


def test_add_and_check_categorical_param_constraints(
    categorical_param_abc: CategoricalParam[str],
) -> None:
    """Test that categorical parameter constraints can be added and checked."""
    categorical_param_abc = categorical_param_abc.add_constraint(
        InSetConstraint(categorical_param_abc.variable, {"a", "b"})
    )
    _assert_all_satisfied(categorical_param_abc, ["a", "b"])
    _assert_none_satisfied(categorical_param_abc, ["c"])


def test_adding_invalid_constraint_to_categorical_param_fails(
    categorical_param_abc: CategoricalParam[str],
) -> None:
    """Test that adding an invalid constraint to a categorical parameter fails."""
    with pytest.raises(ValueError):
        categorical_param_abc.add_constraint(
            EquationConstraint(
                categorical_param_abc.variable,
                categorical_param_abc.variable_expression > "a",
            )
        )


def test_perm_param_initialization() -> None:
    """Test that a permutation parameter can be initialized."""
    param = PermParam(["n", "c", "h", "w"])
    assert isinstance(param, PermParam)


def test_perm_param_initialization_fails_with_non_unique_values() -> None:
    """Test that a permutation parameter initialization fails with non-unique values."""
    with pytest.raises(ValueError):
        PermParam(["n", "c", "h", "n"])


@pytest.fixture()
def perm_param_nchw() -> PermParam[str]:
    return PermParam(["n", "c", "h", "w"])


def test_set_perm_param_value(perm_param_nchw: PermParam[str]) -> None:
    """Test that a permutation parameter value can be set."""
    assignment = perm_param_nchw.assign(["c", "n", "w", "h"])
    assert assignment.is_value_set()


def test_set_perm_param_value_fails_with_invalid_value(
    perm_param_nchw: PermParam[str],
) -> None:
    """Test that setting a permutation parameter value fails with an invalid value."""
    with pytest.raises(ValueError):
        perm_param_nchw.assign(["n", "c", "h", "n"])


def test_perm_param_rejects_string_like_sequences() -> None:
    """Test permutation params reject plain strings as permutation values."""
    param = PermParam(["n", "c", "h", "w"])
    assert not param.is_value_admissible("nchw")


def test_add_and_check_perm_param_constraints(perm_param_nchw: PermParam[str]) -> None:
    """Test that permutation parameter constraints can be added and checked."""
    perm_param_nchw = perm_param_nchw.add_constraint(
        InSetConstraint(
            perm_param_nchw.variable, {("n", "c", "h", "w"), ("c", "n", "w", "h")}
        )
    )
    _assert_all_satisfied(perm_param_nchw, [["n", "c", "h", "w"], ["c", "n", "w", "h"]])
    _assert_none_satisfied(perm_param_nchw, [["n", "c", "w", "h"]])


def test_adding_invalid_constraint_to_perm_param_fails(
    perm_param_nchw: PermParam[str],
) -> None:
    """Test that adding an invalid constraint to a permutation parameter fails."""
    with pytest.raises(ValueError):
        perm_param_nchw = perm_param_nchw.add_constraint(
            EquationConstraint(
                perm_param_nchw.variable, perm_param_nchw.variable_expression > 1
            )
        )


def test_add_constraint_returns_new_param_without_mutating_original() -> None:
    """Test that adding constraints returns a new parameter definition."""
    param = OrdinalParam([1, 2, 3])
    updated = param.add_constraint(InSetConstraint(param.variable, {1, 2}))
    _assert_all_satisfied(updated, [1, 2])
    _assert_none_satisfied(updated, [3])
    _assert_all_satisfied(param, [1, 2, 3])


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
    factory: Any, ops: Any, pass_values: Any, fail_values: Any
) -> None:
    """Test adding lower and upper bound constraints to int and real parameters."""
    param = factory()
    for name, args, kwargs in ops:
        param = getattr(param, name)(*args, **kwargs)
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
def test_numeric_params_adding_invalid_lower_and_upper_bounds_fails(
    factory: Any, ops: Any
) -> None:
    with pytest.raises(ValueError):
        param = factory()
        for name, args, kwargs in ops:
            param = getattr(param, name)(*args, **kwargs)


def test_zero_included_in_nat_param_with_lower_bound_inclusive_at_zero() -> None:
    """Test that creating a NatParam with lower bound 0 inclusive works.

    This is specifically to test the constructor argument for including zero
    in the natural number constraints added before the lower bound constraint
    is added.
    """
    param = NatParam.with_lower_bound(0, is_inclusive=True)
    _assert_all_satisfied(param, [0, 1, 2, 100])


def test_real_param_serialization() -> None:
    """Test a real parameter can be serialized/deserialized via a dictionary."""
    param = RealParam()
    param = param.add_constraint(
        EquationConstraint(param.variable, param.variable_expression > 0)
    )
    param = param.add_constraint(
        EquationConstraint(param.variable, param.variable_expression < 10)
    )
    dictionary = param.serialize_to_dict()
    param2 = RealParam.deserialize_from_dict(dictionary)
    _assert_all_satisfied(param2, [1.0, 5.0, 9.0])
    _assert_none_satisfied(param2, [0.0, 10.0])


def test_int_param_serialization() -> None:
    """Test an int parameter can be serialized/deserialized via a dictionary."""
    param = IntParam()
    param = param.add_constraint(
        EquationConstraint(param.variable, param.variable_expression > 0)
    )
    param = param.add_constraint(
        EquationConstraint(param.variable, param.variable_expression < 10)
    )
    dictionary = param.serialize_to_dict()
    param2 = IntParam.deserialize_from_dict(dictionary)
    _assert_all_satisfied(param2, [1, 5, 9])
    _assert_none_satisfied(param2, [0, 10])


def test_ordinal_param_serialization(ordinal_param_123: OrdinalParam[int]) -> None:
    """Test an ordinal parameter can be serialized/deserialized via a dictionary."""
    ordinal_param_123 = ordinal_param_123.add_constraint(
        InSetConstraint(ordinal_param_123.variable, {1, 2})
    )
    dictionary = ordinal_param_123.serialize_to_dict()
    param2: OrdinalParam[int] = OrdinalParam.deserialize_from_dict(dictionary)
    _assert_all_satisfied(param2, [1, 2])
    _assert_none_satisfied(param2, [3])


def test_ordinal_param_deserialization_rejects_unwrapped_possible_values() -> None:
    """Test ordinal deserialization rejects raw unwrapped possible values."""
    payload = OrdinalParam([1, 2, 3]).serialize_to_dict()
    payload["__data__"]["possible_values"] = [1, 2, 3]  # type: ignore[index]  # test: modify serialized
    with pytest.raises(DeserializationValueError):
        OrdinalParam.deserialize_from_dict(payload)


def test_categorical_param_serialization(
    categorical_param_abc: CategoricalParam[str],
) -> None:
    """Test a categorical parameter can be serialized/deserialized via a dictionary."""
    categorical_param_abc = categorical_param_abc.add_constraint(
        InSetConstraint(categorical_param_abc.variable, {"a", "b"})
    )
    dictionary = categorical_param_abc.serialize_to_dict()
    param2: CategoricalParam[str] = CategoricalParam.deserialize_from_dict(dictionary)
    _assert_all_satisfied(param2, ["a", "b"])
    _assert_none_satisfied(param2, ["c"])


def test_categorical_param_deserialization_rejects_wrapped_non_leaf_values() -> None:
    """Test categorical deserialization rejects wrapped container values."""
    payload = CategoricalParam({"a", "b"}).serialize_to_dict()
    payload["__data__"]["possible_values"] = [  # type: ignore[index]  # test: modify serialized
        serialize_registry_wrapped_value(("a", "b"))
    ]
    with pytest.raises(DeserializationValueError):
        CategoricalParam.deserialize_from_dict(payload)


def test_categorical_param_round_trip_serialization() -> None:
    """Test a categorical parameter can be serialized/deserialized via a dictionary."""
    param = CategoricalParam([Identifier("a"), Identifier("b")])
    data = param.serialize_to_dict()
    param2: CategoricalParam[Identifier] = CategoricalParam.deserialize_from_dict(data)
    assert param.is_structurally_equivalent(param2)


def test_perm_param_serialization(perm_param_nchw: PermParam[str]) -> None:
    """Test a permutation parameter can be serialized/deserialized via a dictionary."""
    perm_param_nchw = perm_param_nchw.add_constraint(
        InSetConstraint(
            perm_param_nchw.variable, {("n", "c", "h", "w"), ("c", "n", "w", "h")}
        )
    )
    dictionary = perm_param_nchw.serialize_to_dict()
    param2: PermParam[str] = PermParam.deserialize_from_dict(dictionary)
    _assert_all_satisfied(param2, [["n", "c", "h", "w"], ["c", "n", "w", "h"]])
    _assert_none_satisfied(param2, [["n", "c", "w", "h"]])


def test_nat_param_serialization() -> None:
    """Test a nat parameter can be serialized/deserialized via a dictionary."""
    param = NatParam(is_zero_included=False)
    param = param.add_constraint(
        EquationConstraint(param.variable, param.variable_expression >= 1)
    )
    param = param.add_constraint(
        EquationConstraint(param.variable, param.variable_expression <= 10)
    )
    dictionary = param.serialize_to_dict()
    assert len(dictionary["__data__"]["constraints"]) == 3  # type: ignore[index,arg-type,call-overload]  # test: dict shape known
    param2 = NatParam.deserialize_from_dict(dictionary)
    _assert_all_satisfied(param2, [1, 5, 10])
    _assert_none_satisfied(param2, [0, 11])
    dictionary = param2.serialize_to_dict()
    assert len(dictionary["__data__"]["constraints"]) == 3  # type: ignore[index,arg-type,call-overload]  # test: dict shape known


def test_param_assignment_serialization_dict_roundtrip() -> None:
    """Test parameter assignments round-trip via dictionary serialization."""
    assignment = IntParam.with_lower_bound(0).assign(3)
    dictionary = assignment.serialize_to_dict()

    assignment2 = ParamAssignment.deserialize_from_dict(dictionary)
    assert assignment2.value == assignment.value
    assert assignment2.param.is_structurally_equivalent(assignment.param)
    assert assignment2.serialize_to_dict() == dictionary


def test_param_assignment_serialization_json_and_binary_roundtrip() -> None:
    """Test parameter assignments round-trip via JSON and binary serialization."""
    assignment = PermParam(["n", "c", "h", "w"]).assign(["n", "c", "h", "w"])

    json_payload = assignment.serialize(SerializationFormat.JSON)
    assignment_from_json: ParamAssignment[Any] = ParamAssignment.deserialize(
        json_payload, SerializationFormat.JSON
    )
    assert isinstance(assignment_from_json.param, PermParam)
    assert assignment_from_json.value == ("n", "c", "h", "w")

    binary_payload = assignment.serialize(SerializationFormat.BINARY)
    assignment_from_binary: ParamAssignment[Any] = ParamAssignment.deserialize(
        binary_payload, SerializationFormat.BINARY
    )
    assert isinstance(assignment_from_binary.param, PermParam)
    assert assignment_from_binary.value == ("n", "c", "h", "w")


def test_param_assignment_deserialization_rejects_value_invalid_for_param() -> None:
    """Test assignment deserialization fails if payload value violates constraints."""
    param = RealParam.with_lower_bound(0.0)
    payload = {
        "param": param.serialize_to_dict(),
        "value": serialize_registry_wrapped_value(-1.0),
    }
    with pytest.raises(DeserializationValueError):
        ParamAssignment.deserialize_from_dict(payload)  # type: ignore[arg-type]  # test: dict shape


# TODO: Check serialization structure errors and value errors for all types.


# TODO: Test the repr method.


def test_bound_int_param_between_inclusive_inclusive_constraints_satisfied() -> None:
    """Test ``between'' with inclusive bounds.

    Integer semantics: [3,5] => {3,4,5}
    """
    p = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    _assert_all_satisfied(p, [3, 4, 5])
    _assert_none_satisfied(p, [2, 6])


def test_bound_int_param_between_exclusive_exclusive_constraints_satisfied() -> None:
    """Test ``between'' with exclusive bounds.

    Integer semantics: (3,5) => only {4}
    """
    p = BoundIntParam.between(3, 5, is_lower_inclusive=False, is_upper_inclusive=False)
    _assert_all_satisfied(p, [4])
    _assert_none_satisfied(p, [3, 5, 2, 6])


def test_bound_int_param_between_exclusive_inclusive_constraints_satisfied() -> None:
    """Test ``between'' with exclusive lower bound and inclusive upper bound.

    Integer semantics: (3,5] => {4,5}
    """
    p = BoundIntParam.between(3, 5, is_lower_inclusive=False, is_upper_inclusive=True)
    _assert_all_satisfied(p, [4, 5])
    _assert_none_satisfied(p, [3, 2, 6])


def test_bound_int_param_between_inclusive_exclusive_constraints_satisfied() -> None:
    """Test ``between'' with inclusive lower bound and exclusive upper bound.

    Integer semantics: [3,5) => {3,4}
    """
    p = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=False)
    _assert_all_satisfied(p, [3, 4])
    _assert_none_satisfied(p, [5, 2, 6])


def test_bound_int_param_between_invalid_interval_raises_for_strict_equal_bounds() -> (
    None
):
    """Test that ``between'' raises for intervals with strict equal bounds."""
    with pytest.raises(ValueError):
        BoundIntParam.between(3, 3, is_lower_inclusive=False, is_upper_inclusive=False)


def test_bound_int_param_between_equal_bounds_inclusive_is_singleton() -> None:
    """Test that ``between'' with equal inclusive bounds is a singleton."""
    p = BoundIntParam.between(3, 3, is_lower_inclusive=True, is_upper_inclusive=True)
    _assert_all_satisfied(p, [3])
    _assert_none_satisfied(p, [2, 4])


def test_bound_int_param_between_invalid_order_raises_error() -> None:
    """Test that ``between'' raises for intervals with reversed bounds."""
    with pytest.raises(ValueError):
        BoundIntParam.between(5, 3)


def test_bound_int_param_with_lower_bound_inclusive() -> None:
    """Test ``with_lower_bound'' with inclusive bound."""
    p = BoundIntParam.with_lower_bound(3, is_inclusive=True)
    _assert_all_satisfied(p, [3, 4, 100])
    _assert_none_satisfied(p, [2, -1])


def test_bound_int_param_with_lower_bound_exclusive_integer_semantics() -> None:
    """Test ``with_lower_bound'' with exclusive bound.

    Integer semantics: x > 3 => ints {4,5,...}
    """
    p = BoundIntParam.with_lower_bound(3, is_inclusive=False)
    _assert_all_satisfied(p, [4, 5, 100])
    _assert_none_satisfied(p, [3, 2, -10])


def test_bound_int_param_with_upper_bound_inclusive() -> None:
    p = BoundIntParam.with_upper_bound(5, is_inclusive=True)
    _assert_all_satisfied(p, [5, 4, -100])
    _assert_none_satisfied(p, [6, 7])


def test_bound_int_param_with_upper_bound_exclusive_integer_semantics() -> None:
    """Test ``with_upper_bound'' with exclusive bound.

    Integer semantics: x < 5 => ints {...,3,4}
    """
    p = BoundIntParam.with_upper_bound(5, is_inclusive=False)
    _assert_all_satisfied(p, [4, 3, -100])
    _assert_none_satisfied(p, [5, 6])


def test_bound_int_param_exactly_satisfies_only_that_value() -> None:
    """Test ``exactly'' creates a singleton parameter."""
    p = BoundIntParam.exactly(7)
    _assert_all_satisfied(p, [7])
    _assert_none_satisfied(p, [6, 8, 0])


def test_bound_int_param_prefer_inclusive_flag_does_not_change_satisfiable_set() -> (
    None
):
    """Test that prefer_inclusive flag does not change satisfiable set."""
    p1 = BoundIntParam.between(
        3, 5, is_lower_inclusive=False, is_upper_inclusive=False, prefer_inclusive=True
    )
    p2 = BoundIntParam.between(
        3, 5, is_lower_inclusive=False, is_upper_inclusive=False, prefer_inclusive=False
    )
    for v in range(0, 10):
        assert p1.is_constraints_satisfied(v) == p2.is_constraints_satisfied(v)


def test_bound_int_param_assign_accepts_int_only() -> None:
    """Test that BoundIntParam.assign only accepts integer values."""
    p = BoundIntParam.with_lower_bound(0)
    assignment = p.assign(1)
    assert assignment.value == 1
    with pytest.raises(ValueError):
        p.assign(1.0)  # type: ignore[arg-type]  # test: invalid input
    with pytest.raises(ValueError):
        p.assign("1")  # type: ignore[arg-type]  # test: invalid input


def test_bound_int_param_assign_rejects_value_outside_constraints() -> None:
    """Test that ``assign'' rejects values outside constraints."""
    p = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    with pytest.raises(ValueError):
        p.assign(2)
    with pytest.raises(ValueError):
        p.assign(6)
    assignment = p.assign(4)
    assert assignment.value == 4


def test_bound_int_param_addition_of_singletons_is_singleton() -> None:
    """Test addition of two singleton BoundIntParams results in a singleton."""
    x = BoundIntParam.exactly(4)
    y = BoundIntParam.exactly(6)
    z = x + y
    _assert_all_satisfied(z, [10])
    _assert_none_satisfied(z, [9, 11])


def test_bound_int_param_addition_with_int_rhs() -> None:
    """Test addition of BoundIntParam with an integer on the right-hand side."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    z = x + 2
    _assert_all_satisfied(z, [5, 6, 7])
    _assert_none_satisfied(z, [4, 8])


def test_bound_int_param_addition_with_int_lhs() -> None:
    """Test addition of BoundIntParam with an integer on the left-hand side."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    z = 2 + x
    _assert_all_satisfied(z, [5, 6, 7])
    _assert_none_satisfied(z, [4, 8])


def test_bound_int_param_addition_bounds_propagate_semantics_from_strict_inputs() -> (
    None
):
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


def test_bound_int_param_addition_unbounded_lower_upper_propagates_unboundedness() -> (
    None
):
    """Test addition bounds propagation with unbounded inputs.

    Integer semantics:
    x >= 3, y unbounded => z >= 3 + (-inf) = -inf
    """
    x = BoundIntParam.with_lower_bound(3, is_inclusive=True)
    y = BoundIntParam()
    z = x + y
    _assert_all_satisfied(z, [-(10**6), 0, 10**6])


def test_bound_int_param_subtraction_of_singletons_is_singleton() -> None:
    """Test subtraction of two singleton BoundIntParams results in a singleton."""
    x = BoundIntParam.exactly(10)
    y = BoundIntParam.exactly(6)
    z = x - y
    _assert_all_satisfied(z, [4])
    _assert_none_satisfied(z, [3, 5])


def test_bound_int_param_subtraction_with_int_rhs() -> None:
    """Test subtraction of BoundIntParam with an integer on the right-hand side."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    z = x - 2
    _assert_all_satisfied(z, [1, 2, 3])
    _assert_none_satisfied(z, [0, 4])


def test_bound_int_param_subtraction_with_int_lhs() -> None:
    """Test subtraction of BoundIntParam with an integer on the left-hand side."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    z = 10 - x
    _assert_all_satisfied(z, [5, 6, 7])
    _assert_none_satisfied(z, [4, 8])


def test_bound_int_param_subtraction_bounds_propagate_semantics_from_inputs() -> None:
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


def test_bound_int_param_negation_of_singleton() -> None:
    """Test negation of a singleton BoundIntParam results in a singleton."""
    x = BoundIntParam.exactly(4)
    z = -x
    _assert_all_satisfied(z, [-4])
    _assert_none_satisfied(z, [-3, -5])


def test_bound_int_param_negation_of_interval() -> None:
    """Test negation of an interval BoundIntParam."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    z = -x
    _assert_all_satisfied(z, [-5, -4, -3])
    _assert_none_satisfied(z, [-6, -2])


def test_bound_int_param_negation_of_strict_interval_integer_semantics() -> None:
    """Test negation of a strict interval BoundIntParam."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=False, is_upper_inclusive=False)
    z = -x
    _assert_all_satisfied(z, [-4])
    _assert_none_satisfied(z, [-5, -3])


def test_bound_int_param_prefer_inclusive_changes_str_not_membership_for_addition() -> (
    None
):
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


def test_bound_int_param_addition_accepts_int_param() -> None:
    """Test addition of BoundIntParam with IntParam on the right-hand side."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    y = IntParam.between(5, 10, is_lower_inclusive=True, is_upper_inclusive=True)
    z = x + y
    _assert_all_satisfied(z, [8, 9, 10, 11, 12, 13, 14, 15])
    _assert_none_satisfied(z, [7, 16])


def test_bound_int_param_subtraction_accepts_int_param() -> None:
    """Test subtraction of BoundIntParam with IntParam on the right-hand side."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    y = IntParam.between(5, 10, is_lower_inclusive=True, is_upper_inclusive=True)
    z = x - y
    _assert_all_satisfied(z, [-7, -3, 0])
    _assert_none_satisfied(z, [-8, 1])


def test_bound_int_param_rsub_accepts_int_param_on_left() -> None:
    """Test subtraction of BoundIntParam with IntParam on the left-hand side."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    y = IntParam.between(5, 10, is_lower_inclusive=True, is_upper_inclusive=True)
    z = y - x
    _assert_all_satisfied(z, [0, 7])
    _assert_none_satisfied(z, [-1, 8])


def test_bound_int_param_addition_with_unsupported_type_raises_error() -> None:
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
def test_bound_int_param_addition_matches_brute_force(
    lower: int, upper: int, lin: bool, u_in: bool
) -> None:
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
def test_bound_int_param_subtraction_matches_brute_force(
    lower: int, upper: int, lin: bool, u_in: bool
) -> None:
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


def test_bound_int_param_serialization() -> None:
    """Test bound int parameter can be serialized/deserialized via a dictionary."""
    p = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=False)
    dictionary = p.serialize_to_dict()
    p2 = BoundIntParam.deserialize_from_dict(dictionary)
    _assert_all_satisfied(p2, [3, 4])
    _assert_none_satisfied(p2, [5])


def test_bound_nat_param_serialization() -> None:
    """Test bound nat parameter can be serialized/deserialized via a dictionary."""
    p = BoundNatParam.with_lower_bound(2, is_inclusive=True)
    dictionary = p.serialize_to_dict()
    assert len(dictionary["__data__"]["constraints"]) == 2  # type: ignore[index,arg-type,call-overload]  # test: dict shape known
    p2 = BoundNatParam.deserialize_from_dict(dictionary)
    _assert_all_satisfied(p2, [2, 5, 100])
    _assert_none_satisfied(p2, [0, 1])
    dictionary = p2.serialize_to_dict()
    assert len(dictionary["__data__"]["constraints"]) == 2  # type: ignore[index,arg-type,call-overload]  # test: dict shape known


# TODO: Update tests that check exceptions to match exception messages.
