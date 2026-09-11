"""Hypothesis property tests for the `tests.strategies` package itself.

Covers six properties: every public strategy draws without raising;
numeric and boolean gate trees respect their leaf budget and pass
`validate_logical_operands`; the plain-Python oracle agrees with the
NumPy evaluator on gate trees; every `SerializableCase` round-trips
through DICT; and `build_identifier_pool` is deterministic with
distinct ids.

`draw_serializable_case` does not cover mock identifiers, even though
`tests.conftest.mock_identifier` is itself `Serializable`: its
`serialize_to_dict`/`deserialize_from_dict` hooks are instance
attributes on the `Mock`, not classmethods, so
`type(instance).deserialize_from_dict(...)` -- the one round-trip
mechanism every `SerializableCase` shares -- raises `AttributeError`
for a mock identifier. `test_build_identifier_pool_...` below checks
identifier determinism directly instead.
"""

import pytest

pytest.importorskip("hypothesis")

from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.symbolic.expression import (
    evaluate_expression_with_numpy,
    validate_logical_operands,
)

from .strategies.constraints import (
    build_integer_bindings_strategy,
    build_set_member_strategy,
    draw_bound_equation_constraint,
    draw_constraint_system,
    draw_in_set_constraint,
    draw_integer_set_constraint,
    draw_not_in_set_constraint,
)
from .strategies.expressions import (
    build_any_sort_expression_strategy,
    build_boolean_expression_strategy,
    build_integer_environment_strategy,
    build_numeric_expression_strategy,
    build_structural_expression_strategy,
    build_sympy_stable_expression_strategy,
    count_expression_leaves,
    draw_boolean_tree_with_environment,
    draw_numeric_tree_with_environment,
    evaluate_with_python,
)
from .strategies.identifiers import build_identifier_pool, build_identifier_strategy
from .strategies.literals import (
    build_any_literal_strategy,
    build_boolean_literal_strategy,
    build_boolean_value_strategy,
    build_decimal_string_literal_strategy,
    build_decimal_string_value_strategy,
    build_finite_float_literal_strategy,
    build_finite_float_value_strategy,
    build_integer_literal_strategy,
    build_integer_value_strategy,
)
from .strategies.params import (
    build_categorical_value_set_strategy,
    build_optional_bound_strategy,
    build_ordinal_value_set_strategy,
    build_permutation_member_set_strategy,
    draw_bounded_integer_param,
    draw_bounded_real_param,
    draw_categorical_param,
    draw_interval_integer_param,
    draw_natural_param,
    draw_ordered_optional_bounds,
    draw_ordinal_param,
    draw_param_over_any_domain,
    draw_param_with_candidate,
    draw_permutation_param,
    draw_same_kind_param_pair_with_candidate,
    draw_single_valid_value_param,
)
from .strategies.serializables import SerializableCase, draw_serializable_case
from .strategies.types import (
    build_core_data_type_strategy,
    build_primitive_data_type_strategy,
    build_type_qualifier_strategy,
    draw_index_type,
    draw_numerical_type,
    draw_shape,
    draw_template_data_type,
    draw_template_free_type,
)

pytestmark = pytest.mark.property

np = pytest.importorskip("numpy")

_POOL = build_identifier_pool(4)
_VARIABLE = _POOL[0]

_Case = tuple[str, st.SearchStrategy[Any]]

_IDENTIFIERS_CASES: list[_Case] = [
    ("build_identifier_strategy", build_identifier_strategy(_POOL)),
]

_LITERALS_CASES: list[_Case] = [
    ("build_integer_value_strategy", build_integer_value_strategy()),
    ("build_boolean_value_strategy", build_boolean_value_strategy()),
    ("build_finite_float_value_strategy", build_finite_float_value_strategy()),
    ("build_decimal_string_value_strategy", build_decimal_string_value_strategy()),
    ("build_integer_literal_strategy", build_integer_literal_strategy()),
    ("build_boolean_literal_strategy", build_boolean_literal_strategy()),
    ("build_finite_float_literal_strategy", build_finite_float_literal_strategy()),
    ("build_decimal_string_literal_strategy", build_decimal_string_literal_strategy()),
    ("build_any_literal_strategy", build_any_literal_strategy()),
]

_EXPRESSIONS_CASES: list[_Case] = [
    ("build_numeric_expression_strategy", build_numeric_expression_strategy(_POOL)),
    ("build_boolean_expression_strategy", build_boolean_expression_strategy(_POOL)),
    ("build_any_sort_expression_strategy", build_any_sort_expression_strategy(_POOL)),
    (
        "build_structural_expression_strategy",
        build_structural_expression_strategy(_POOL),
    ),
    (
        "build_sympy_stable_expression_strategy",
        build_sympy_stable_expression_strategy(_POOL),
    ),
    ("build_integer_environment_strategy", build_integer_environment_strategy(_POOL)),
    ("draw_numeric_tree_with_environment", draw_numeric_tree_with_environment(_POOL)),
    ("draw_boolean_tree_with_environment", draw_boolean_tree_with_environment(_POOL)),
]

_PARAMS_CASES: list[_Case] = [
    ("build_optional_bound_strategy", build_optional_bound_strategy()),
    ("draw_ordered_optional_bounds", draw_ordered_optional_bounds()),
    ("draw_interval_integer_param", draw_interval_integer_param()),
    ("build_ordinal_value_set_strategy", build_ordinal_value_set_strategy()),
    ("build_categorical_value_set_strategy", build_categorical_value_set_strategy()),
    ("build_permutation_member_set_strategy", build_permutation_member_set_strategy()),
    ("draw_ordinal_param", draw_ordinal_param()),
    ("draw_categorical_param", draw_categorical_param()),
    ("draw_permutation_param", draw_permutation_param()),
    ("draw_bounded_integer_param", draw_bounded_integer_param()),
    ("draw_natural_param", draw_natural_param()),
    ("draw_bounded_real_param", draw_bounded_real_param()),
    ("draw_single_valid_value_param", draw_single_valid_value_param()),
    ("draw_param_over_any_domain", draw_param_over_any_domain()),
    ("draw_param_with_candidate", draw_param_with_candidate()),
    (
        "draw_same_kind_param_pair_with_candidate",
        draw_same_kind_param_pair_with_candidate(),
    ),
]

_CONSTRAINTS_CASES: list[_Case] = [
    ("build_set_member_strategy", build_set_member_strategy()),
    ("draw_in_set_constraint", draw_in_set_constraint(_VARIABLE)),
    ("draw_not_in_set_constraint", draw_not_in_set_constraint(_VARIABLE)),
    ("draw_bound_equation_constraint", draw_bound_equation_constraint(_VARIABLE)),
    ("draw_integer_set_constraint", draw_integer_set_constraint(_VARIABLE)),
    ("draw_constraint_system", draw_constraint_system(_POOL)),
    ("build_integer_bindings_strategy", build_integer_bindings_strategy(_POOL)),
]

_TYPES_CASES: list[_Case] = [
    ("build_core_data_type_strategy", build_core_data_type_strategy()),
    ("build_type_qualifier_strategy", build_type_qualifier_strategy()),
    ("build_primitive_data_type_strategy", build_primitive_data_type_strategy()),
    ("draw_shape", draw_shape(_POOL)),
    ("draw_numerical_type", draw_numerical_type(_POOL)),
    ("draw_index_type", draw_index_type(_POOL)),
    ("draw_template_free_type", draw_template_free_type(_POOL)),
    ("draw_template_data_type", draw_template_data_type(_POOL)),
]

_SERIALIZABLES_CASES: list[_Case] = [
    ("draw_serializable_case", draw_serializable_case()),
]


@pytest.mark.parametrize(
    "case", _IDENTIFIERS_CASES, ids=[name for name, _ in _IDENTIFIERS_CASES]
)
@given(data=st.data())
def test_identifiers_strategy_draws_without_raising(
    case: _Case, data: st.DataObject
) -> None:
    """Test every public strategies.identifiers strategy draws without raising."""
    _name, strategy = case
    data.draw(strategy)


@pytest.mark.parametrize(
    "case", _LITERALS_CASES, ids=[name for name, _ in _LITERALS_CASES]
)
@given(data=st.data())
def test_literals_strategy_draws_without_raising(
    case: _Case, data: st.DataObject
) -> None:
    """Test every public strategies.literals strategy draws without raising."""
    _name, strategy = case
    data.draw(strategy)


@pytest.mark.parametrize(
    "case", _EXPRESSIONS_CASES, ids=[name for name, _ in _EXPRESSIONS_CASES]
)
@given(data=st.data())
def test_expressions_strategy_draws_without_raising(
    case: _Case, data: st.DataObject
) -> None:
    """Test every public strategies.expressions strategy draws without raising."""
    _name, strategy = case
    data.draw(strategy)


@pytest.mark.parametrize("case", _PARAMS_CASES, ids=[name for name, _ in _PARAMS_CASES])
@given(data=st.data())
def test_params_strategy_draws_without_raising(
    case: _Case, data: st.DataObject
) -> None:
    """Test every public strategies.params strategy draws without raising."""
    _name, strategy = case
    data.draw(strategy)


@pytest.mark.parametrize(
    "case", _CONSTRAINTS_CASES, ids=[name for name, _ in _CONSTRAINTS_CASES]
)
@given(data=st.data())
def test_constraints_strategy_draws_without_raising(
    case: _Case, data: st.DataObject
) -> None:
    """Test every public strategies.constraints strategy draws without raising."""
    _name, strategy = case
    data.draw(strategy)


@pytest.mark.parametrize("case", _TYPES_CASES, ids=[name for name, _ in _TYPES_CASES])
@given(data=st.data())
def test_types_strategy_draws_without_raising(case: _Case, data: st.DataObject) -> None:
    """Test every public strategies.types strategy draws without raising."""
    _name, strategy = case
    data.draw(strategy)


@pytest.mark.parametrize(
    "case", _SERIALIZABLES_CASES, ids=[name for name, _ in _SERIALIZABLES_CASES]
)
@given(data=st.data())
def test_serializables_strategy_draws_without_raising(
    case: _Case, data: st.DataObject
) -> None:
    """Test every public strategies.serializables strategy draws without raising."""
    _name, strategy = case
    data.draw(strategy)


@given(draw_numeric_tree_with_environment(_POOL))
def test_numeric_gate_tree_respects_its_leaf_budget(
    pair: tuple[Any, dict[Any, int]],
) -> None:
    """Test a numeric gate tree never exceeds the leaf budget it was drawn with."""
    expression, _environment = pair
    assert count_expression_leaves(expression) <= 8


@given(draw_boolean_tree_with_environment(_POOL))
def test_boolean_gate_tree_respects_its_leaf_budget(
    pair: tuple[Any, dict[Any, int]],
) -> None:
    """Test a boolean gate tree never exceeds the leaf budget it was drawn with."""
    expression, _environment = pair
    assert count_expression_leaves(expression) <= 8


@given(draw_numeric_tree_with_environment(_POOL))
def test_numeric_gate_tree_passes_validate_logical_operands(
    pair: tuple[Any, dict[Any, int]],
) -> None:
    """Test a numeric gate tree never trips validate_logical_operands."""
    expression, _environment = pair
    validate_logical_operands(expression)


@given(draw_boolean_tree_with_environment(_POOL))
def test_boolean_gate_tree_passes_validate_logical_operands(
    pair: tuple[Any, dict[Any, int]],
) -> None:
    """Test a boolean gate tree never trips validate_logical_operands."""
    expression, _environment = pair
    validate_logical_operands(expression)


@given(draw_numeric_tree_with_environment(_POOL))
def test_python_oracle_agrees_with_numpy_on_numeric_gate_trees(
    pair: tuple[Any, dict[Any, int]],
) -> None:
    """Test the Python oracle agrees with the NumPy evaluator on numeric trees."""
    expression, environment = pair
    python_result = evaluate_with_python(expression, environment)
    numpy_result = evaluate_expression_with_numpy(expression, environment)
    assert int(python_result) == int(numpy_result)


@given(draw_boolean_tree_with_environment(_POOL))
def test_python_oracle_agrees_with_numpy_on_boolean_gate_trees(
    pair: tuple[Any, dict[Any, int]],
) -> None:
    """Test the Python oracle agrees with the NumPy evaluator on boolean trees."""
    expression, environment = pair
    python_result = evaluate_with_python(expression, environment)
    numpy_result = evaluate_expression_with_numpy(expression, environment)
    assert bool(python_result) == bool(numpy_result)


@given(draw_serializable_case())
def test_serializable_case_round_trips_through_dict(case: SerializableCase) -> None:
    """Test every SerializableCase round-trips through DICT under its equivalence."""
    restored = type(case.instance).deserialize_from_dict(
        case.instance.serialize_to_dict()
    )
    assert case.are_equivalent(restored, case.instance)


@given(st.integers(min_value=1, max_value=10))
def test_build_identifier_pool_is_deterministic_with_distinct_ids(size: int) -> None:
    """Test build_identifier_pool yields distinct ids and is deterministic."""
    pool_a = build_identifier_pool(size)
    pool_b = build_identifier_pool(size)
    assert len(pool_a) == size
    assert len({identifier.id for identifier in pool_a}) == size
    assert all(a == b for a, b in zip(pool_a, pool_b, strict=True))
