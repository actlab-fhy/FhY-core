"""Hypothesis property tests for the `tests.strategies` package itself.

Covers six properties: every public strategy draws without raising;
numeric and boolean gate trees respect their leaf budget and pass
`validate_logical_operands`; the plain-Python oracle agrees with the
NumPy evaluator on gate trees; every `SerializableCase` round-trips
through DICT; and `build_identifier_pool` is deterministic with
distinct ids. Also pins the numeric/boolean gate strategies'
`include_calls`, `native_functions`, and `include_division` options: a
tree drawn with the option off (or restricted to a subset) never holds
the excluded shape anywhere in it, including inside subtrees reached
through a comparison operand or a piecewise condition/value.

Gate trees drawn with a Boolean identifier pool are checked to stay
well-typed (the type checker types a Boolean tree as a Boolean and a
numeric tree as a number, and no integer identifier reaches a Boolean
position), and to reach the shapes the SymPy bridge rewrites: a Boolean
piecewise with a free condition under `&&`/`||`, `!`, and `==`/`!=`,
nested in another Boolean piecewise, over Boolean identifier conditions
and values, and supplied by a substitution; an equality between two
sort-ambiguous operands (bare Boolean identifiers, or piecewise over
them) that only a substitution shows is Boolean; and a numeric piecewise
with a Boolean identifier condition.

The param strategies are also checked for reach: each bounded kind draws
an exclusive bound, each strategy taking `include_empty` draws an empty
param only when it is set, and integer and real bounds reach past the
int64 range; and the endpoints `draw_interval_integer_param_with_bounds`
reports are the param's own least and greatest members.

`cap_max_examples` is checked against the loaded profile: it never raises
the profile's example count, it lowers the `thorough` count, and the
settings it builds while a module imports see the profile
`tests/conftest.py` loads.

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

from collections.abc import Callable, Iterator
from typing import Any, Final

from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
    evaluate_expression_with_numpy,
    validate_logical_operands,
)
from fhy_core.symbolic.param import Param
from fhy_core.symbolic.symbol_type import SymbolType
from fhy_core.types import (
    CoreDataType,
    NumericalType,
    PrimitiveDataType,
    Type,
    TypeQualifier,
)
from fhy_core.types.checking.type_checker import synthesize_expression_type

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
    BOOLEAN_EQUALITY_OPERATIONS,
    COMPARISON_OPERATIONS,
    LOGICAL_BINARY_OPERATIONS,
    NUMERIC_DIVISION_OPERATIONS,
    SYMPY_STABLE_CALL_FUNCTIONS,
    build_any_sort_expression_strategy,
    build_boolean_environment_strategy,
    build_boolean_expression_strategy,
    build_gate_environment_strategy,
    build_integer_environment_strategy,
    build_numeric_expression_strategy,
    build_sympy_stable_expression_strategy,
    count_expression_leaves,
    draw_boolean_tree_with_environment,
    draw_numeric_tree_with_environment,
    draw_simultaneous_substitution_case,
    evaluate_with_python,
)
from .strategies.identifiers import (
    build_boolean_identifier_pool,
    build_identifier_pool,
    build_identifier_strategy,
)
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
    WIDE_REAL_SCALES,
    build_categorical_value_set_strategy,
    build_optional_bound_strategy,
    build_ordinal_value_set_strategy,
    build_permutation_member_set_strategy,
    draw_bounded_integer_param,
    draw_bounded_real_param,
    draw_categorical_param,
    draw_integer_bound,
    draw_interval_integer_param,
    draw_interval_integer_param_with_bounds,
    draw_natural_param,
    draw_ordered_optional_bounds,
    draw_ordinal_param,
    draw_overlapping_intersection_eligible_group,
    draw_overlapping_union_eligible_group,
    draw_param_over_any_domain,
    draw_param_with_candidate,
    draw_permutation_param,
    draw_real_scale,
    draw_same_kind_param_pair_with_candidate,
    draw_single_valid_value_param,
)
from .strategies.serializables import SerializableCase, draw_serializable_case
from .strategies.settings import cap_max_examples
from .strategies.structural_expressions import build_structural_expression_strategy
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
_BOOLEAN_POOL = build_boolean_identifier_pool(2)
_VARIABLE = _POOL[0]
_SYMBOL_TYPES: Final[dict[Identifier, SymbolType]] = {
    **dict.fromkeys(_POOL, SymbolType.INT),
    **dict.fromkeys(_BOOLEAN_POOL, SymbolType.BOOL),
}

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
    (
        "build_numeric_expression_strategy[boolean_identifiers]",
        build_numeric_expression_strategy(_POOL, boolean_identifiers=_BOOLEAN_POOL),
    ),
    (
        "build_boolean_expression_strategy[boolean_identifiers]",
        build_boolean_expression_strategy(_POOL, boolean_identifiers=_BOOLEAN_POOL),
    ),
    (
        "build_any_sort_expression_strategy[boolean_identifiers]",
        build_any_sort_expression_strategy(_POOL, boolean_identifiers=_BOOLEAN_POOL),
    ),
    (
        "build_boolean_environment_strategy",
        build_boolean_environment_strategy(_BOOLEAN_POOL),
    ),
    (
        "build_gate_environment_strategy",
        build_gate_environment_strategy(_POOL, _BOOLEAN_POOL),
    ),
    (
        "draw_simultaneous_substitution_case",
        draw_simultaneous_substitution_case(_POOL, _BOOLEAN_POOL),
    ),
]

_PARAMS_CASES: list[_Case] = [
    ("draw_integer_bound", draw_integer_bound()),
    ("draw_real_scale", draw_real_scale()),
    ("build_optional_bound_strategy", build_optional_bound_strategy()),
    ("draw_ordered_optional_bounds", draw_ordered_optional_bounds()),
    (
        "draw_interval_integer_param_with_bounds",
        draw_interval_integer_param_with_bounds(),
    ),
    (
        "draw_interval_integer_param_with_bounds[include_empty]",
        draw_interval_integer_param_with_bounds(include_empty=True),
    ),
    ("draw_interval_integer_param", draw_interval_integer_param()),
    ("build_ordinal_value_set_strategy", build_ordinal_value_set_strategy()),
    ("build_categorical_value_set_strategy", build_categorical_value_set_strategy()),
    ("build_permutation_member_set_strategy", build_permutation_member_set_strategy()),
    ("draw_ordinal_param", draw_ordinal_param()),
    ("draw_categorical_param", draw_categorical_param()),
    ("draw_permutation_param", draw_permutation_param()),
    ("draw_bounded_integer_param", draw_bounded_integer_param()),
    (
        "draw_bounded_integer_param[include_empty]",
        draw_bounded_integer_param(include_empty=True),
    ),
    ("draw_natural_param", draw_natural_param()),
    ("draw_bounded_real_param", draw_bounded_real_param()),
    ("draw_single_valid_value_param", draw_single_valid_value_param()),
    ("draw_param_over_any_domain", draw_param_over_any_domain()),
    (
        "draw_param_over_any_domain[include_empty]",
        draw_param_over_any_domain(include_empty=True),
    ),
    ("draw_param_with_candidate", draw_param_with_candidate()),
    (
        "draw_param_with_candidate[include_empty]",
        draw_param_with_candidate(include_empty=True),
    ),
    (
        "draw_same_kind_param_pair_with_candidate",
        draw_same_kind_param_pair_with_candidate(),
    ),
    (
        "draw_overlapping_intersection_eligible_group",
        draw_overlapping_intersection_eligible_group(size=3),
    ),
    (
        "draw_overlapping_union_eligible_group",
        draw_overlapping_union_eligible_group(size=3),
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


# Rare shapes need more than a profile's example budget to turn up, and a
# reachability search must not write to or replay from the example database.
_FIND_SETTINGS = settings(max_examples=2000, database=None)

_INT64_MAX = 2**63 - 1
_INT64_MIN = -(2**63)


def _iter_bound_expressions(param: Param[Any]) -> Iterator[BinaryExpression]:
    """Yield the comparison each of ``param``'s bound constraints holds."""
    for constraint in param.constraints:
        expression = constraint.convert_to_expression()
        if isinstance(expression, BinaryExpression):
            yield expression


def _has_exclusive_bound(param: Param[Any]) -> bool:
    """Return whether any of ``param``'s bounds is a strict comparison."""
    return any(
        expression.operation in (BinaryOperation.GREATER, BinaryOperation.LESS)
        for expression in _iter_bound_expressions(param)
    )


def _get_widest_bound_magnitude(param: Param[Any]) -> float:
    """Return the largest magnitude among ``param``'s literal bounds, or ``0``."""
    return max(
        (
            abs(float(expression.right.value))
            for expression in _iter_bound_expressions(param)
            if isinstance(expression.right, LiteralExpression)
        ),
        default=0.0,
    )


_BOUNDED_PARAM_CASES: list[_Case] = [
    ("draw_interval_integer_param", draw_interval_integer_param()),
    ("draw_bounded_integer_param", draw_bounded_integer_param()),
    ("draw_bounded_real_param", draw_bounded_real_param()),
]

_EMPTY_CAPABLE_PARAM_CASES: list[_Case] = [
    (
        "draw_interval_integer_param",
        draw_interval_integer_param(include_empty=True),
    ),
    ("draw_bounded_integer_param", draw_bounded_integer_param(include_empty=True)),
    ("draw_param_over_any_domain", draw_param_over_any_domain(include_empty=True)),
    (
        "draw_param_with_candidate",
        draw_param_with_candidate(include_empty=True).map(lambda case: case[0]),
    ),
]


@pytest.mark.parametrize(
    "case", _BOUNDED_PARAM_CASES, ids=[name for name, _ in _BOUNDED_PARAM_CASES]
)
def test_params_strategy_reaches_an_exclusive_bound(case: _Case) -> None:
    """Test each bounded param strategy draws a param with an exclusive bound."""
    _name, strategy = case

    found = find(strategy, _has_exclusive_bound, settings=_FIND_SETTINGS)

    assert _has_exclusive_bound(found)


@pytest.mark.parametrize(
    "case",
    _EMPTY_CAPABLE_PARAM_CASES,
    ids=[name for name, _ in _EMPTY_CAPABLE_PARAM_CASES],
)
def test_params_strategy_reaches_an_empty_param_when_empties_are_included(
    case: _Case,
) -> None:
    """Test each strategy taking include_empty draws an empty param when it is set."""
    _name, strategy = case

    found = find(strategy, lambda param: param.is_empty(), settings=_FIND_SETTINGS)

    assert found.is_empty()


@pytest.mark.parametrize(
    "is_beyond",
    [
        pytest.param(lambda bound: bound > _INT64_MAX, id="above-int64"),
        pytest.param(lambda bound: bound < _INT64_MIN, id="below-int64"),
        pytest.param(lambda bound: abs(bound) >= 2**100, id="beyond-2-pow-100"),
    ],
)
def test_interval_integer_param_strategy_reaches_an_endpoint_past_int64(
    is_beyond: Any,
) -> None:
    """Test an interval-integer param draws an endpoint past the int64 range."""
    found = find(
        draw_interval_integer_param_with_bounds(),
        lambda case: any(bound is not None and is_beyond(bound) for bound in case[1:]),
        settings=_FIND_SETTINGS,
    )

    assert any(bound is not None and is_beyond(bound) for bound in found[1:])


def test_real_param_strategy_reaches_a_bound_at_the_widest_scale() -> None:
    """Test a bounded real param draws a bound scaled by the widest real scale."""
    widest_scale = max(WIDE_REAL_SCALES)

    found = find(
        draw_bounded_real_param(),
        lambda param: _get_widest_bound_magnitude(param) >= widest_scale,
        settings=_FIND_SETTINGS,
    )

    assert _get_widest_bound_magnitude(found) >= widest_scale


@given(case=draw_interval_integer_param_with_bounds())
def test_interval_integer_param_with_bounds_reports_its_extreme_members(
    case: tuple[Param[int], int | None, int | None],
) -> None:
    """Test the reported endpoints are the least and greatest integers admitted.

    Consumers compute expected results from these endpoints, so each must
    be admitted with its outward neighbor refused, whichever inclusivity
    the strategy drew for that end.
    """
    param, lower, upper = case

    if lower is not None:
        assert param.is_value_valid(lower)
        assert not param.is_value_valid(lower - 1)
    if upper is not None:
        assert param.is_value_valid(upper)
        assert not param.is_value_valid(upper + 1)


@given(case=draw_interval_integer_param_with_bounds(include_empty=True))
def test_interval_integer_param_with_bounds_crosses_endpoints_exactly_when_empty(
    case: tuple[Param[int], int | None, int | None],
) -> None:
    """Test the reported lower endpoint exceeds the upper exactly for an empty param."""
    param, lower, upper = case

    is_crossed = lower is not None and upper is not None and lower > upper

    assert param.is_empty() == is_crossed


@given(param=st.one_of(draw_interval_integer_param(), draw_bounded_integer_param()))
def test_params_strategy_draws_no_empty_param_by_default(param: Param[int]) -> None:
    """Test an empty-capable strategy draws no empty param without include_empty."""
    assert not param.is_empty()


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


@given(draw_numeric_tree_with_environment(_POOL, boolean_identifiers=_BOOLEAN_POOL))
def test_numeric_gate_tree_respects_its_leaf_budget(
    pair: tuple[Expression, dict[Identifier, int | bool]],
) -> None:
    """Test a numeric gate tree never exceeds the leaf budget it was drawn with."""
    expression, _environment = pair
    assert count_expression_leaves(expression) <= 8


@given(draw_boolean_tree_with_environment(_POOL, boolean_identifiers=_BOOLEAN_POOL))
def test_boolean_gate_tree_respects_its_leaf_budget(
    pair: tuple[Expression, dict[Identifier, int | bool]],
) -> None:
    """Test a boolean gate tree never exceeds the leaf budget it was drawn with."""
    expression, _environment = pair
    assert count_expression_leaves(expression) <= 8


@given(draw_numeric_tree_with_environment(_POOL, boolean_identifiers=_BOOLEAN_POOL))
def test_numeric_gate_tree_passes_validate_logical_operands(
    pair: tuple[Expression, dict[Identifier, int | bool]],
) -> None:
    """Test a numeric gate tree never trips validate_logical_operands.

    Declaring the integer pool ``INT`` makes an integer identifier in a
    Boolean position a refusal, so this also checks one never lands there.
    """
    expression, _environment = pair
    validate_logical_operands(expression, symbol_types=_SYMBOL_TYPES)


@given(draw_boolean_tree_with_environment(_POOL, boolean_identifiers=_BOOLEAN_POOL))
def test_boolean_gate_tree_passes_validate_logical_operands(
    pair: tuple[Expression, dict[Identifier, int | bool]],
) -> None:
    """Test a boolean gate tree never trips validate_logical_operands.

    Declaring the integer pool ``INT`` makes an integer identifier in a
    Boolean position a refusal, so this also checks one never lands there.
    """
    expression, _environment = pair
    validate_logical_operands(expression, symbol_types=_SYMBOL_TYPES)


@given(draw_numeric_tree_with_environment(_POOL, boolean_identifiers=_BOOLEAN_POOL))
def test_python_oracle_agrees_with_numpy_on_numeric_gate_trees(
    pair: tuple[Expression, dict[Identifier, int | bool]],
) -> None:
    """Test the Python oracle agrees with the NumPy evaluator on numeric trees."""
    expression, environment = pair
    python_result = evaluate_with_python(expression, environment)
    numpy_result = evaluate_expression_with_numpy(expression, environment)
    assert int(python_result) == int(numpy_result)


@given(draw_boolean_tree_with_environment(_POOL, boolean_identifiers=_BOOLEAN_POOL))
def test_python_oracle_agrees_with_numpy_on_boolean_gate_trees(
    pair: tuple[Expression, dict[Identifier, int | bool]],
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


def _iter_expression_nodes(expression: Expression) -> Iterator[Expression]:
    """Yield expression and every descendant, walked via get_visit_children."""
    yield expression
    for child in expression.get_visit_children():
        yield from _iter_expression_nodes(child)


@given(draw_numeric_tree_with_environment(_POOL, include_calls=False))
def test_numeric_gate_tree_excludes_calls_when_include_calls_is_false(
    pair: tuple[Expression, dict[Identifier, int]],
) -> None:
    """Test include_calls=False draws a numeric tree with no CallExpression."""
    expression, _environment = pair
    assert not any(
        isinstance(node, CallExpression) for node in _iter_expression_nodes(expression)
    )


# A comparison operand is a numeric subtree, so this also checks that
# include_calls reaches through that subtree, not only the boolean root.
@given(draw_boolean_tree_with_environment(_POOL, include_calls=False))
def test_boolean_gate_tree_excludes_calls_when_include_calls_is_false(
    pair: tuple[Expression, dict[Identifier, int]],
) -> None:
    """Test include_calls=False draws a boolean tree with no CallExpression."""
    expression, _environment = pair
    assert not any(
        isinstance(node, CallExpression) for node in _iter_expression_nodes(expression)
    )


@given(draw_numeric_tree_with_environment(_POOL, native_functions=("floor",)))
def test_numeric_gate_tree_calls_only_the_native_functions_option(
    pair: tuple[Expression, dict[Identifier, int]],
) -> None:
    """Test native_functions=("floor",) draws a numeric tree that calls only floor."""
    expression, _environment = pair
    assert all(
        node.function_name == "floor"
        for node in _iter_expression_nodes(expression)
        if isinstance(node, CallExpression)
    )


@given(draw_boolean_tree_with_environment(_POOL, native_functions=("floor",)))
def test_boolean_gate_tree_calls_only_the_native_functions_option(
    pair: tuple[Expression, dict[Identifier, int]],
) -> None:
    """Test native_functions=("floor",) draws a boolean tree that calls only floor."""
    expression, _environment = pair
    assert all(
        node.function_name == "floor"
        for node in _iter_expression_nodes(expression)
        if isinstance(node, CallExpression)
    )


# build_boolean_expression_strategy forwards include_division to every
# numeric subtree it draws for a comparison, so this walks the whole tree
# with get_visit_children rather than checking only the root.
@given(draw_boolean_tree_with_environment(_POOL, include_division=False))
def test_boolean_gate_tree_excludes_division_when_include_division_is_false(
    pair: tuple[Expression, dict[Identifier, int]],
) -> None:
    """Test include_division=False draws a boolean tree with no FLOOR_DIVIDE/MODULO."""
    expression, _environment = pair
    assert not any(
        isinstance(node, BinaryExpression)
        and node.operation in NUMERIC_DIVISION_OPERATIONS
        for node in _iter_expression_nodes(expression)
    )


# =============================================================================
# Expressions: Boolean identifiers and Boolean-position piecewise
# =============================================================================


_INT32_SCALAR: Final[Type] = NumericalType(PrimitiveDataType(CoreDataType.INT32), [])
_BOOL_SCALAR: Final[Type] = NumericalType(PrimitiveDataType(CoreDataType.BOOL), [])


def _look_up_pool_type(identifier: Identifier) -> tuple[Type, TypeQualifier]:
    """Return a pool identifier's type: ``int32`` for an integer, else ``bool``."""
    if identifier in _BOOLEAN_POOL:
        return _BOOL_SCALAR, TypeQualifier.PARAM
    if identifier in _POOL:
        return _INT32_SCALAR, TypeQualifier.PARAM
    raise KeyError(identifier.name_hint)


def _is_typed_boolean(expression: Expression) -> bool:
    """Return whether the type checker synthesizes the Boolean type for ``expression``.

    Raises ``FhYCoreTypeError`` for an ill-typed tree, such as an integer
    identifier in a Boolean position or a Boolean compared with a number.
    """
    synthesized_type, _ = synthesize_expression_type(expression, _look_up_pool_type)
    return (
        isinstance(synthesized_type, NumericalType)
        and isinstance(synthesized_type.data_type, PrimitiveDataType)
        and synthesized_type.data_type.core_data_type is CoreDataType.BOOL
    )


def _is_boolean_sorted(expression: Expression) -> bool:
    """Return whether a gate tree over the two pools denotes a Boolean.

    Reads the sort off the root alone, which is exact for a well-typed
    gate tree: a gate-grammar call always returns an integer, and a
    piecewise has the sort of its ``otherwise``.
    """
    if isinstance(expression, LiteralExpression):
        return isinstance(expression.value, bool)
    if isinstance(expression, IdentifierExpression):
        return expression.identifier in _BOOLEAN_POOL
    if isinstance(expression, UnaryExpression):
        return expression.operation is UnaryOperation.LOGICAL_NOT
    if isinstance(expression, BinaryExpression):
        return expression.operation in (
            *COMPARISON_OPERATIONS,
            *LOGICAL_BINARY_OPERATIONS,
        )
    if isinstance(expression, PiecewiseExpression):
        return _is_boolean_sorted(expression.otherwise)
    return False


def _is_boolean_identifier(expression: Expression) -> bool:
    """Return whether ``expression`` is a bare identifier from the Boolean pool."""
    return (
        isinstance(expression, IdentifierExpression)
        and expression.identifier in _BOOLEAN_POOL
    )


def _is_open_boolean_piecewise(expression: Expression) -> bool:
    """Return whether ``expression`` is a Boolean piecewise with a free condition."""
    return (
        isinstance(expression, PiecewiseExpression)
        and _is_boolean_sorted(expression)
        and any(condition.get_free_identifiers() for condition in expression.conditions)
    )


def _holds_open_boolean_piecewise_under_binary(
    expression: Expression, operations: tuple[BinaryOperation, ...]
) -> bool:
    """Return whether an ``operations`` node has an open Boolean piecewise operand."""
    return any(
        isinstance(node, BinaryExpression)
        and node.operation in operations
        and any(
            _is_open_boolean_piecewise(operand) for operand in (node.left, node.right)
        )
        for node in _iter_expression_nodes(expression)
    )


def _holds_open_boolean_piecewise_under_and_or(expression: Expression) -> bool:
    """Return whether an ``&&``/``||`` node has an open Boolean piecewise operand."""
    return _holds_open_boolean_piecewise_under_binary(
        expression, LOGICAL_BINARY_OPERATIONS
    )


def _holds_open_boolean_piecewise_under_equality(expression: Expression) -> bool:
    """Return whether an ``==``/``!=`` node has an open Boolean piecewise operand."""
    return _holds_open_boolean_piecewise_under_binary(
        expression, BOOLEAN_EQUALITY_OPERATIONS
    )


def _holds_open_boolean_piecewise_under_not(expression: Expression) -> bool:
    """Return whether a ``!`` node negates an open Boolean piecewise."""
    return any(
        isinstance(node, UnaryExpression)
        and node.operation is UnaryOperation.LOGICAL_NOT
        and _is_open_boolean_piecewise(node.operand)
        for node in _iter_expression_nodes(expression)
    )


def _holds_boolean_piecewise_over_boolean_identifiers(expression: Expression) -> bool:
    """Return whether a Boolean piecewise has Boolean identifier conditions and values.

    At least one condition and at least one branch value must each be a
    bare Boolean identifier.
    """
    return any(
        isinstance(node, PiecewiseExpression)
        and _is_boolean_sorted(node)
        and any(_is_boolean_identifier(condition) for condition in node.conditions)
        and any(
            _is_boolean_identifier(value) for value in (*node.values, node.otherwise)
        )
        for node in _iter_expression_nodes(expression)
    )


def _holds_nested_open_boolean_piecewise(expression: Expression) -> bool:
    """Return whether an open Boolean piecewise holds another as a part."""
    return any(
        _is_open_boolean_piecewise(node)
        and isinstance(node, PiecewiseExpression)
        and any(
            _is_open_boolean_piecewise(part)
            for part in (*node.conditions, *node.values, node.otherwise)
        )
        for node in _iter_expression_nodes(expression)
    )


def _holds_numeric_piecewise_with_a_boolean_identifier_condition(
    expression: Expression,
) -> bool:
    """Return whether a numeric piecewise has a bare Boolean identifier condition."""
    return any(
        isinstance(node, PiecewiseExpression)
        and not _is_boolean_sorted(node)
        and any(_is_boolean_identifier(condition) for condition in node.conditions)
        for node in _iter_expression_nodes(expression)
    )


_SubstitutionCase = tuple[
    Expression, dict[Identifier, Expression], dict[Identifier, int | bool]
]


def _supplies_an_open_boolean_piecewise(case: _SubstitutionCase) -> bool:
    """Return whether a free Boolean identifier is replaced by an open piecewise."""
    expression, replacements, _environment = case
    return any(
        identifier in expression.get_free_identifiers()
        and _is_open_boolean_piecewise(replacement)
        for identifier, replacement in replacements.items()
    )


def _is_sort_ambiguous(expression: Expression) -> bool:
    """Return whether only Boolean identifiers show ``expression`` is Boolean.

    True for a bare Boolean identifier, and for a piecewise whose every
    branch is sort-ambiguous in turn.
    """
    if isinstance(expression, PiecewiseExpression):
        return all(
            _is_sort_ambiguous(value)
            for value in (*expression.values, expression.otherwise)
        )
    return _is_boolean_identifier(expression)


def _makes_a_sort_ambiguous_equality_boolean(case: _SubstitutionCase) -> bool:
    """Return whether a replacement types an equality of sort-ambiguous operands.

    The tree must hold an ``==``/``!=`` whose operands are both
    sort-ambiguous, at least one a piecewise, and one of whose Boolean
    identifiers the case replaces with a tree that is not an identifier,
    so only the substituted tree shows the equality compares Booleans.
    """
    expression, replacements, _environment = case
    replaced = {
        identifier
        for identifier, replacement in replacements.items()
        if not isinstance(replacement, IdentifierExpression)
    }
    return any(
        isinstance(node, BinaryExpression)
        and node.operation in BOOLEAN_EQUALITY_OPERATIONS
        and _is_sort_ambiguous(node.left)
        and _is_sort_ambiguous(node.right)
        and PiecewiseExpression in (type(node.left), type(node.right))
        and bool(node.get_free_identifiers() & replaced)
        for node in _iter_expression_nodes(expression)
    )


# A shape search stops at the first draw that holds the shape: shrinking a
# whole tree to its smallest instance costs far more than finding it.
_SHAPE_SEARCH_SETTINGS = settings(
    max_examples=2000, database=None, phases=[Phase.generate]
)

# Drawn with the options the SymPy-backed properties use.
_SYMPY_PROPERTY_TREES: Final = build_any_sort_expression_strategy(
    _POOL,
    6,
    boolean_identifiers=_BOOLEAN_POOL,
    include_division=False,
    include_calls=True,
    native_functions=SYMPY_STABLE_CALL_FUNCTIONS,
)
_SYMPY_PROPERTY_SUBSTITUTION_CASES: Final = draw_simultaneous_substitution_case(
    _POOL,
    _BOOLEAN_POOL,
    6,
    include_division=False,
    include_calls=True,
    native_functions=SYMPY_STABLE_CALL_FUNCTIONS,
)


@given(draw_numeric_tree_with_environment(_POOL, boolean_identifiers=_BOOLEAN_POOL))
def test_numeric_gate_tree_with_boolean_identifiers_is_well_typed(
    pair: tuple[Expression, dict[Identifier, int | bool]],
) -> None:
    """Test the type checker accepts a numeric gate tree and types it as a number."""
    expression, _environment = pair

    assert not _is_typed_boolean(expression)


@given(draw_boolean_tree_with_environment(_POOL, boolean_identifiers=_BOOLEAN_POOL))
def test_boolean_gate_tree_with_boolean_identifiers_is_well_typed(
    pair: tuple[Expression, dict[Identifier, int | bool]],
) -> None:
    """Test the type checker accepts a Boolean gate tree and types it as a Boolean."""
    expression, _environment = pair

    assert _is_typed_boolean(expression)


@given(build_gate_environment_strategy(_POOL, _BOOLEAN_POOL))
def test_gate_environment_binds_each_pool_to_its_own_value_type(
    environment: dict[Identifier, int | bool],
) -> None:
    """Test integer identifiers are bound to ints and Boolean identifiers to bools."""
    assert set(environment) == {*_POOL, *_BOOLEAN_POOL}
    for identifier, value in environment.items():
        assert isinstance(value, bool) == (identifier in _BOOLEAN_POOL)


@given(draw_simultaneous_substitution_case(_POOL, _BOOLEAN_POOL))
def test_substitution_case_replaces_each_identifier_with_a_tree_of_its_sort(
    case: _SubstitutionCase,
) -> None:
    """Test each replacement is well-typed and of its identifier's sort."""
    expression, replacements, _environment = case

    assert _is_typed_boolean(expression) == _is_boolean_sorted(expression)
    assert set(replacements) == {_POOL[0], _POOL[1], _BOOLEAN_POOL[0]}
    for identifier, replacement in replacements.items():
        assert _is_typed_boolean(replacement) == (identifier in _BOOLEAN_POOL)


@given(
    draw_boolean_tree_with_environment(
        _POOL, boolean_identifiers=_BOOLEAN_POOL, include_boolean_comparisons=False
    )
)
def test_boolean_gate_tree_excludes_boolean_equalities_when_the_option_is_false(
    pair: tuple[Expression, dict[Identifier, int | bool]],
) -> None:
    """Test include_boolean_comparisons=False draws no ``==``/``!=`` over Booleans."""
    expression, _environment = pair

    assert not any(
        isinstance(node, BinaryExpression)
        and node.operation in BOOLEAN_EQUALITY_OPERATIONS
        and _is_boolean_sorted(node.left)
        for node in _iter_expression_nodes(expression)
    )


@pytest.mark.parametrize(
    "holds_shape",
    [
        pytest.param(
            _holds_open_boolean_piecewise_under_and_or,
            id="boolean-piecewise-under-and-or",
        ),
        pytest.param(
            _holds_open_boolean_piecewise_under_equality,
            id="boolean-piecewise-under-eq-ne",
        ),
        pytest.param(
            _holds_open_boolean_piecewise_under_not, id="boolean-piecewise-under-not"
        ),
        pytest.param(
            _holds_boolean_piecewise_over_boolean_identifiers,
            id="boolean-piecewise-over-boolean-identifiers",
        ),
        pytest.param(
            _holds_nested_open_boolean_piecewise, id="nested-boolean-piecewise"
        ),
        pytest.param(
            _holds_numeric_piecewise_with_a_boolean_identifier_condition,
            id="numeric-piecewise-with-boolean-identifier-condition",
        ),
    ],
)
def test_sympy_property_trees_reach_a_boolean_bridge_shape(
    holds_shape: Callable[[Expression], bool],
) -> None:
    """Test the trees the SymPy-backed properties draw reach each rewritten shape."""
    found = find(_SYMPY_PROPERTY_TREES, holds_shape, settings=_SHAPE_SEARCH_SETTINGS)

    assert holds_shape(found)


def test_sympy_property_substitution_cases_reach_a_substituted_boolean_piecewise() -> (
    None
):
    """Test a substitution case replaces a free Boolean identifier with a piecewise."""
    found = find(
        _SYMPY_PROPERTY_SUBSTITUTION_CASES,
        _supplies_an_open_boolean_piecewise,
        settings=_SHAPE_SEARCH_SETTINGS,
    )

    assert _supplies_an_open_boolean_piecewise(found)


def test_sympy_property_substitution_cases_reach_a_sort_ambiguous_equality() -> None:
    """Test a substitution case types an equality nothing else shows is Boolean."""
    found = find(
        _SYMPY_PROPERTY_SUBSTITUTION_CASES,
        _makes_a_sort_ambiguous_equality_boolean,
        settings=_SHAPE_SEARCH_SETTINGS,
    )

    assert _makes_a_sort_ambiguous_equality_boolean(found)


# =============================================================================
# Settings: cap_max_examples
# =============================================================================


_IMPORT_TIME_CEILING: Final = 50
_SETTINGS_CAPPED_AT_IMPORT = cap_max_examples(_IMPORT_TIME_CEILING)


def _get_loaded_profile() -> settings:
    """Return the Hypothesis settings profile currently loaded."""
    return settings.get_profile(settings.get_current_profile_name())


@pytest.fixture()
def load_thorough_profile() -> Iterator[settings]:
    """Load the ``thorough`` profile for one test, then reload the previous one."""
    previous_profile_name = settings.get_current_profile_name()
    settings.load_profile("thorough")
    try:
        yield settings.get_profile("thorough")
    finally:
        settings.load_profile(previous_profile_name)


@pytest.mark.parametrize("excess", [0, 1, 1000])
def test_cap_max_examples_never_exceeds_the_loaded_profile_count(excess: int) -> None:
    """Test a ceiling at or above the loaded profile's count keeps that count."""
    profile = _get_loaded_profile()

    capped = cap_max_examples(profile.max_examples + excess)

    assert capped.max_examples == profile.max_examples


def test_cap_max_examples_lowers_the_thorough_profile_count(
    load_thorough_profile: settings,
) -> None:
    """Test a ceiling below the thorough profile's count becomes the count."""
    ceiling = load_thorough_profile.max_examples - 1

    assert cap_max_examples(ceiling).max_examples == ceiling


def test_cap_max_examples_at_import_reads_the_profile_conftest_loads() -> None:
    """Test settings capped while this module was imported saw the loaded profile.

    Under ``dev`` or ``mutation`` (25 examples) this fails if the profile
    were loaded only after test modules import, because the ceiling would
    then be measured against Hypothesis's built-in count of 100.
    """
    expected = min(_IMPORT_TIME_CEILING, _get_loaded_profile().max_examples)

    assert _SETTINGS_CAPPED_AT_IMPORT.max_examples == expected


def test_cap_max_examples_keeps_every_other_setting_of_the_profile(
    load_thorough_profile: settings,
) -> None:
    """Test the capped settings keep the profile's other settings, such as deadline."""
    capped = cap_max_examples(1)

    assert (capped.derandomize, capped.deadline, capped.print_blob) == (
        load_thorough_profile.derandomize,
        load_thorough_profile.deadline,
        load_thorough_profile.print_blob,
    )
