"""Tests for interval-integer parameters."""

from functools import partial
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.serialization import DeserializationDictStructureError
from fhy_core.symbolic.constraint import EquationConstraint, InSetConstraint
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
    make_binary_expression,
)
from fhy_core.symbolic.param import (
    Param,
    ParamError,
    create_integer_param,
    create_integer_param_between,
    create_interval_integer_param,
    create_interval_integer_param_between,
    create_interval_integer_param_exactly,
    create_interval_integer_param_with_lower_bound,
    create_interval_integer_param_with_upper_bound,
)

from .conftest import assert_all_satisfied, assert_none_satisfied, mock_identifier

# =============================================================================
# `between` constructor
# =============================================================================


def test_bound_int_param_between_with_inclusive_bounds_satisfies_endpoints() -> None:
    """Test ``create_interval_integer_param_between(3, 5)`` admits inclusive endpoints.

    Integer semantics: ``[3, 5]`` => ``{3, 4, 5}``.
    """
    p = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=True, is_upper_inclusive=True
    )

    assert_all_satisfied(p, [3, 4, 5])
    assert_none_satisfied(p, [2, 6])


def test_bound_int_param_between_with_exclusive_bounds_excludes_endpoints() -> None:
    """Test ``create_interval_integer_param_between(3, 5)`` excludes both endpoints.

    Exclusive on both sides. Integer semantics: ``(3, 5)`` => ``{4}``.
    """
    p = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=False, is_upper_inclusive=False
    )

    assert_all_satisfied(p, [4])
    assert_none_satisfied(p, [3, 5, 2, 6])


def test_bound_int_param_between_with_exclusive_lower_inclusive_upper() -> None:
    """Test ``create_interval_integer_param_between(3, 5)`` excl lower, incl upper.

    Integer semantics: ``(3, 5]`` => ``{4, 5}``.
    """
    p = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=False, is_upper_inclusive=True
    )

    assert_all_satisfied(p, [4, 5])
    assert_none_satisfied(p, [3, 2, 6])


def test_bound_int_param_between_with_inclusive_lower_exclusive_upper() -> None:
    """Test ``create_interval_integer_param_between(3, 5)`` incl lower, excl upper.

    Integer semantics: ``[3, 5)`` => ``{3, 4}``.
    """
    p = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=True, is_upper_inclusive=False
    )

    assert_all_satisfied(p, [3, 4])
    assert_none_satisfied(p, [5, 2, 6])


def test_bound_int_param_between_with_strict_equal_bounds_raises() -> None:
    """Test ``create_interval_integer_param_between(x, x)`` raises (both exclusive)."""
    with pytest.raises(ParamError):
        create_interval_integer_param_between(
            3, 3, is_lower_inclusive=False, is_upper_inclusive=False
        )


def test_bound_int_param_between_with_inclusive_equal_bounds_is_singleton() -> None:
    """Test ``create_interval_integer_param_between(x, x)`` admits only ``x`` (incl)."""
    p = create_interval_integer_param_between(
        3, 3, is_lower_inclusive=True, is_upper_inclusive=True
    )

    assert_all_satisfied(p, [3])
    assert_none_satisfied(p, [2, 4])


def test_bound_int_param_between_with_reversed_bounds_raises() -> None:
    """Test ``create_interval_integer_param_between`` raises when ``lower > upper``."""
    with pytest.raises(ParamError):
        create_interval_integer_param_between(5, 3)


# =============================================================================
# `with_lower_bound` / `with_upper_bound`
# =============================================================================


@pytest.mark.parametrize(
    "factory, bound, is_inclusive, pass_values, fail_values",
    [
        pytest.param(
            create_interval_integer_param_with_lower_bound,
            3,
            True,
            [3, 4, 100],
            [2, -1],
            id="lower-inclusive",
        ),
        pytest.param(
            create_interval_integer_param_with_lower_bound,
            3,
            False,
            [4, 5, 100],
            [3, 2, -10],
            id="lower-exclusive",
        ),
        pytest.param(
            create_interval_integer_param_with_upper_bound,
            5,
            True,
            [5, 4, -100],
            [6, 7],
            id="upper-inclusive",
        ),
        pytest.param(
            create_interval_integer_param_with_upper_bound,
            5,
            False,
            [4, 3, -100],
            [5, 6],
            id="upper-exclusive",
        ),
    ],
)
def test_bound_int_param_with_bound_admits_or_excludes_endpoint_per_inclusivity(
    factory: Any,
    bound: int,
    is_inclusive: bool,
    pass_values: list[int],
    fail_values: list[int],
) -> None:
    """Test ``with_lower_bound`` / ``with_upper_bound`` honor ``is_inclusive``.

    Integer semantics: ``x > k`` => ``{k+1, k+2, ...}`` and ``x < k`` =>
    ``{..., k-2, k-1}``.
    """
    p = factory(bound, is_inclusive=is_inclusive)

    assert_all_satisfied(p, pass_values)
    assert_none_satisfied(p, fail_values)


# =============================================================================
# `exactly`
# =============================================================================


def test_bound_int_param_exactly_admits_only_the_given_value() -> None:
    """Test ``create_interval_integer_param_exactly(7)`` admits only ``7``."""
    p = create_interval_integer_param_exactly(7)

    assert_all_satisfied(p, [7])
    assert_none_satisfied(p, [6, 8, 0])


# =============================================================================
# `prefer_inclusive`
# =============================================================================


def test_bound_int_param_prefer_inclusive_does_not_change_satisfiable_set() -> None:
    """Test `prefer_inclusive` does not change the satisfiable set."""
    p1 = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=False, is_upper_inclusive=False, prefer_inclusive=True
    )
    p2 = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=False, is_upper_inclusive=False, prefer_inclusive=False
    )

    for v in range(0, 10):
        assert p1.is_constraints_satisfied(v) == p2.is_constraints_satisfied(v)


# =============================================================================
# `assign`
# =============================================================================


def test_bound_int_param_assign_accepts_int_values_only() -> None:
    """Test interval-integer param ``assign`` only accepts integer values."""
    p = create_interval_integer_param_with_lower_bound(0)

    assignment = p.assign(1)

    assert assignment.value == 1
    with pytest.raises(ParamError):
        p.assign(1.0)  # type: ignore[arg-type]  # test: invalid input
    with pytest.raises(ParamError):
        p.assign("1")  # type: ignore[arg-type]  # test: invalid input


def test_bound_int_param_assign_rejects_value_outside_constraints() -> None:
    """Test interval-integer param ``assign`` rejects values outside the bounds."""
    p = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=True, is_upper_inclusive=True
    )

    with pytest.raises(ParamError):
        p.assign(2)
    with pytest.raises(ParamError):
        p.assign(6)

    assignment = p.assign(4)
    assert assignment.value == 4


# =============================================================================
# Arithmetic - addition
# =============================================================================


def test_bound_int_param_addition_of_singletons_is_singleton() -> None:
    """Test addition of two singleton interval-integer params is a singleton."""
    x = create_interval_integer_param_exactly(4)
    y = create_interval_integer_param_exactly(6)

    z = x + y

    assert_all_satisfied(z, [10])
    assert_none_satisfied(z, [9, 11])


def test_bound_int_param_addition_with_int_on_right_shifts_interval() -> None:
    """Test addition with an ``int`` on the right shifts the interval."""
    x = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=True, is_upper_inclusive=True
    )

    z = x + 2

    assert_all_satisfied(z, [5, 6, 7])
    assert_none_satisfied(z, [4, 8])


def test_bound_int_param_addition_with_int_on_left_shifts_interval() -> None:
    """Test addition with an ``int`` on the left shifts the interval."""
    x = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=True, is_upper_inclusive=True
    )

    z = 2 + x

    assert_all_satisfied(z, [5, 6, 7])
    assert_none_satisfied(z, [4, 8])


def test_bound_int_param_addition_propagates_strict_interval_semantics() -> None:
    """Test addition propagates strict-interval semantics from inputs.

    Integer semantics:
    ``x: (3, 5)`` => ``{4}``
    ``y: (5, 10)`` => ``{6, 7, 8, 9}``
    ``x + y`` => ``{10, 11, 12, 13}``.
    """
    x = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=False, is_upper_inclusive=False
    )
    y = create_interval_integer_param_between(
        5, 10, is_lower_inclusive=False, is_upper_inclusive=False
    )

    z = x + y

    assert_all_satisfied(z, [10, 11, 12, 13])
    assert_none_satisfied(z, [9, 14])


def test_bound_int_param_addition_with_unbounded_propagates_unboundedness() -> None:
    """Test addition with an unbounded operand yields an unbounded result.

    Integer semantics: ``x >= 3``, ``y`` unbounded => ``z >= 3 + (-inf) = -inf``.
    """
    x = create_interval_integer_param_with_lower_bound(3, is_inclusive=True)
    y = create_interval_integer_param()

    z = x + y

    assert_all_satisfied(z, [-(10**6), 0, 10**6])


def test_bound_int_param_addition_accepts_int_param_on_right() -> None:
    """Test addition of interval-integer param with a plain integer param (right)."""
    x = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=True, is_upper_inclusive=True
    )
    y = create_integer_param_between(
        5, 10, is_lower_inclusive=True, is_upper_inclusive=True
    )

    z = x + y

    assert_all_satisfied(z, [8, 9, 10, 11, 12, 13, 14, 15])
    assert_none_satisfied(z, [7, 16])


def test_bound_int_param_addition_with_unsupported_type_raises() -> None:
    """Test interval-integer param ``__add__`` raises ``TypeError`` for unsupported."""
    x = create_interval_integer_param_between(0, 1)

    with pytest.raises(TypeError):
        _ = x + "nope"


def test_bound_int_param_prefer_inclusive_changes_str_not_membership_addition() -> None:
    """Test `prefer_inclusive` changes string form but not membership for addition."""
    x_incl = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=False, is_upper_inclusive=False, prefer_inclusive=True
    )
    y_incl = create_interval_integer_param_between(
        5, 10, is_lower_inclusive=False, is_upper_inclusive=False, prefer_inclusive=True
    )

    x_excl = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=False, is_upper_inclusive=False, prefer_inclusive=False
    )
    y_excl = create_interval_integer_param_between(
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


# =============================================================================
# Arithmetic - subtraction
# =============================================================================


def test_bound_int_param_subtraction_of_singletons_is_singleton() -> None:
    """Test subtraction of two singleton interval-integer params is a singleton."""
    x = create_interval_integer_param_exactly(10)
    y = create_interval_integer_param_exactly(6)

    z = x - y

    assert_all_satisfied(z, [4])
    assert_none_satisfied(z, [3, 5])


def test_bound_int_param_subtraction_with_int_on_right_shifts_interval() -> None:
    """Test subtraction with an ``int`` on the right shifts the interval."""
    x = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=True, is_upper_inclusive=True
    )

    z = x - 2

    assert_all_satisfied(z, [1, 2, 3])
    assert_none_satisfied(z, [0, 4])


def test_bound_int_param_subtraction_with_int_on_left_shifts_interval() -> None:
    """Test subtraction with an ``int`` on the left shifts the interval."""
    x = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=True, is_upper_inclusive=True
    )

    z = 10 - x

    assert_all_satisfied(z, [5, 6, 7])
    assert_none_satisfied(z, [4, 8])


def test_bound_int_param_subtraction_propagates_strict_interval_semantics() -> None:
    """Test subtraction propagates strict-interval semantics from inputs.

    Integer semantics:
    ``x: (3, 5)`` => ``{4}``
    ``y: (5, 10)`` => ``{6, 7, 8, 9}``
    ``x - y`` => ``{-5, -4, -3, -2}``.
    """
    x = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=False, is_upper_inclusive=False
    )
    y = create_interval_integer_param_between(
        5, 10, is_lower_inclusive=False, is_upper_inclusive=False
    )

    z = x - y

    assert_all_satisfied(z, [-5, -4, -3, -2])
    assert_none_satisfied(z, [-6, -1, 0])


def test_bound_int_param_subtraction_accepts_int_param_on_right() -> None:
    """Test subtraction of interval-integer param with a plain integer param (right)."""
    x = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=True, is_upper_inclusive=True
    )
    y = create_integer_param_between(
        5, 10, is_lower_inclusive=True, is_upper_inclusive=True
    )

    z = x - y

    assert_all_satisfied(z, [-7, -3, 0])
    assert_none_satisfied(z, [-8, 1])


def test_bound_int_param_rsub_accepts_int_param_on_left() -> None:
    """Test reflected subtraction with plain integer param on the left."""
    x = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=True, is_upper_inclusive=True
    )
    y = create_integer_param_between(
        5, 10, is_lower_inclusive=True, is_upper_inclusive=True
    )

    z = y - x

    assert_all_satisfied(z, [0, 7])
    assert_none_satisfied(z, [-1, 8])


# =============================================================================
# Arithmetic - negation
# =============================================================================


def test_bound_int_param_negation_of_singleton_is_negated_singleton() -> None:
    """Test negation of a singleton interval-integer param is a negated singleton."""
    x = create_interval_integer_param_exactly(4)

    z = -x

    assert_all_satisfied(z, [-4])
    assert_none_satisfied(z, [-3, -5])


def test_bound_int_param_negation_of_inclusive_interval_reflects_endpoints() -> None:
    """Test negation of an inclusive interval-integer param reflects its endpoints."""
    x = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=True, is_upper_inclusive=True
    )

    z = -x

    assert_all_satisfied(z, [-5, -4, -3])
    assert_none_satisfied(z, [-6, -2])


def test_bound_int_param_negation_of_strict_interval_uses_integer_semantics() -> None:
    """Test negation of a strict-interval param uses integer semantics."""
    x = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=False, is_upper_inclusive=False
    )

    z = -x

    assert_all_satisfied(z, [-4])
    assert_none_satisfied(z, [-5, -3])


# =============================================================================
# Brute-force interval property tests
# =============================================================================


@pytest.mark.parametrize(
    "lower, upper, is_lower_inclusive, is_upper_inclusive",
    [
        pytest.param(0, 0, True, True, id="0-0-incl-incl"),
        pytest.param(0, 1, True, True, id="0-1-incl-incl"),
        pytest.param(0, 1, False, True, id="0-1-excl-incl"),
        pytest.param(0, 1, True, False, id="0-1-incl-excl"),
        pytest.param(0, 2, False, False, id="0-2-excl-excl"),
        pytest.param(-3, 3, True, True, id="neg3-3-incl-incl"),
        pytest.param(-3, 3, False, False, id="neg3-3-excl-excl"),
    ],
)
def test_bound_int_param_addition_matches_brute_force(
    lower: int, upper: int, is_lower_inclusive: bool, is_upper_inclusive: bool
) -> None:
    """Test addition matches brute-force set addition over the input interval."""
    x = create_interval_integer_param_between(
        lower,
        upper,
        is_lower_inclusive=is_lower_inclusive,
        is_upper_inclusive=is_upper_inclusive,
    )
    y = create_interval_integer_param_between(
        lower,
        upper,
        is_lower_inclusive=is_lower_inclusive,
        is_upper_inclusive=is_upper_inclusive,
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
    "lower, upper, is_lower_inclusive, is_upper_inclusive",
    [
        pytest.param(0, 0, True, True, id="0-0-incl-incl"),
        pytest.param(0, 1, True, True, id="0-1-incl-incl"),
        pytest.param(0, 2, False, False, id="0-2-excl-excl"),
        pytest.param(-2, 2, True, True, id="neg2-2-incl-incl"),
        pytest.param(-2, 2, False, False, id="neg2-2-excl-excl"),
    ],
)
def test_bound_int_param_subtraction_matches_brute_force(
    lower: int, upper: int, is_lower_inclusive: bool, is_upper_inclusive: bool
) -> None:
    """Test subtraction matches brute-force set subtraction over the input interval."""
    x = create_interval_integer_param_between(
        lower,
        upper,
        is_lower_inclusive=is_lower_inclusive,
        is_upper_inclusive=is_upper_inclusive,
    )
    y = create_interval_integer_param_between(
        lower,
        upper,
        is_lower_inclusive=is_lower_inclusive,
        is_upper_inclusive=is_upper_inclusive,
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


# =============================================================================
# Serialization
# =============================================================================


def test_bound_int_param_serialization_round_trip_preserves_constraints() -> None:
    """Test interval-integer param round-trips through dict serialization."""
    p = create_interval_integer_param_between(
        3, 5, is_lower_inclusive=True, is_upper_inclusive=False
    )

    dictionary = p.serialize_to_dict()
    restored: Param[int] = Param.deserialize_from_dict(dictionary)

    assert_all_satisfied(restored, [3, 4])
    assert_none_satisfied(restored, [5])


# =============================================================================
# Keyword-only signatures
# =============================================================================


def test_bound_int_param_init_accepts_post_marker_args_as_keywords() -> None:
    """Test ``create_interval_integer_param`` accepts keyword args."""
    create_interval_integer_param(name=mock_identifier("x", 1), prefer_inclusive=False)


def test_bound_int_param_init_rejects_name_passed_positionally() -> None:
    """Test ``create_interval_integer_param`` rejects ``name`` passed positionally.

    The factory signature is ``create_interval_integer_param(*, name=None, ...)``,
    all keyword-only. Passing a positional argument raises ``TypeError``.
    """
    with pytest.raises(TypeError):
        create_interval_integer_param(mock_identifier("x", 1))  # type: ignore[misc]  # test: keyword-only


@pytest.mark.parametrize(
    "callable_, positional_args",
    [
        pytest.param(create_interval_integer_param_between, (1, 2), id="between"),
        pytest.param(
            create_interval_integer_param_with_lower_bound, (1,), id="with-lower-bound"
        ),
        pytest.param(
            create_interval_integer_param_with_upper_bound, (2,), id="with-upper-bound"
        ),
        pytest.param(create_interval_integer_param_exactly, (1,), id="exactly"),
    ],
)
def test_bound_int_param_classmethod_rejects_name_passed_positionally(
    callable_: object, positional_args: tuple[int, ...]
) -> None:
    """Test each interval-integer factory rejects ``name`` passed positionally."""
    with pytest.raises(TypeError):
        callable_(*positional_args, mock_identifier("x", 1))  # type: ignore[operator]  # test: keyword-only


# =============================================================================
# Default-flag invariants
# =============================================================================


@pytest.mark.parametrize(
    "factory, expected_substring",
    [
        # Bare ``create_interval_integer_param()`` exercises the default directly.
        pytest.param(
            lambda: create_interval_integer_param().add_lower_bound_constraint(3) + 1,
            ">=",
            id="init",
        ),
        pytest.param(
            lambda: create_interval_integer_param_between(3, 5) + 1, ">=", id="between"
        ),
        pytest.param(
            lambda: create_interval_integer_param_with_lower_bound(3) + 1,
            ">=",
            id="with-lower-bound",
        ),
        pytest.param(
            lambda: create_interval_integer_param_with_upper_bound(5) + 1,
            "<=",
            id="with-upper-bound",
        ),
        pytest.param(
            lambda: create_interval_integer_param_exactly(3) + 1, ">=", id="exactly"
        ),
    ],
)
def test_bound_int_param_default_prefer_inclusive_emits_inclusive_form(
    factory: Any, expected_substring: str
) -> None:
    """Test each public factory defaults to ``prefer_inclusive=True``.

    Verified through the constraint form produced by an arithmetic operation:
    with the inclusive-form preference, the resulting param's repr embeds
    ``>=`` / ``<=`` rather than ``>`` / ``<``.
    """
    assert expected_substring in str(factory())


@pytest.mark.parametrize(
    "factory, boundary_value",
    [
        pytest.param(
            partial(create_interval_integer_param_between, 3, 5), 3, id="between-lower"
        ),
        pytest.param(
            partial(create_interval_integer_param_between, 3, 5), 5, id="between-upper"
        ),
        pytest.param(
            partial(create_interval_integer_param_with_lower_bound, 3),
            3,
            id="with-lower-bound",
        ),
        pytest.param(
            partial(create_interval_integer_param_with_upper_bound, 5),
            5,
            id="with-upper-bound",
        ),
    ],
)
def test_bound_int_param_default_inclusivity_admits_endpoint(
    factory: Any, boundary_value: int
) -> None:
    """Test each public factory defaults to inclusive bounds (admits endpoint)."""
    assert factory().is_value_valid(boundary_value)


# =============================================================================
# Literal-on-left bound expressions and `_invert_binary_comparison_operation`
# =============================================================================


def _build_literal_left_constraint(
    variable: Identifier, op: BinaryOperation, literal_value: int
) -> EquationConstraint:
    """Build a ``literal op variable`` `EquationConstraint`."""
    return EquationConstraint(
        variable, make_binary_expression(op, literal_value, variable)
    )


@pytest.mark.parametrize(
    "literal_value, op, expected_min, expected_max",
    [
        # `1 <= x` is equivalent to `x >= 1` -> lower bound 1.
        pytest.param(1, BinaryOperation.LESS_EQUAL, 1, None, id="le-becomes-ge"),
        # `1 < x` is equivalent to `x > 1` -> lower bound 2 after inclusive shift.
        pytest.param(1, BinaryOperation.LESS, 2, None, id="lt-becomes-gt"),
        # `5 >= x` is equivalent to `x <= 5` -> upper bound 5.
        pytest.param(5, BinaryOperation.GREATER_EQUAL, None, 5, id="ge-becomes-le"),
        # `5 > x` is equivalent to `x < 5` -> upper bound 4 after inclusive shift.
        pytest.param(5, BinaryOperation.GREATER, None, 4, id="gt-becomes-lt"),
    ],
)
def test_bound_int_param_handles_literal_on_left_bound_expressions(
    literal_value: int,
    op: BinaryOperation,
    expected_min: int | None,
    expected_max: int | None,
) -> None:
    """Test interval-integer param handles ``literal op variable`` constraints.

    The four cases together drive ``_invert_comparison``'s four branches via
    the ``literal op var`` arm of ``_iter_interval_bounds``. Asserting the
    resulting effective interval pins down each branch's return value.
    """
    var = mock_identifier("x", 1)
    p = create_interval_integer_param(name=var).add_constraint(
        _build_literal_left_constraint(var, op, literal_value)
    )

    # Force ``_iter_interval_bounds`` -> ``_invert_comparison`` via arithmetic.
    shifted = p + 0

    if expected_min is not None:
        assert shifted.is_value_valid(expected_min)
        assert not shifted.is_value_valid(expected_min - 1)
    if expected_max is not None:
        assert shifted.is_value_valid(expected_max)
        assert not shifted.is_value_valid(expected_max + 1)


def test_bound_int_param_iter_bounds_accepts_literal_on_left_constraint() -> None:
    """Test ``_iter_interval_bounds`` accepts a ``literal op var`` constraint.

    Triggered through arithmetic, which is the public entry point that calls
    ``_iter_interval_bounds``.
    """
    var = mock_identifier("x", 1)
    p = create_interval_integer_param(name=var).add_constraint(
        _build_literal_left_constraint(var, BinaryOperation.LESS_EQUAL, 1)
    )

    p + 0  # test: must not raise


# =============================================================================
# `_create_param_from_min_max` exact constraint form
# =============================================================================


@pytest.mark.parametrize(
    "prefer_inclusive, must_contain, must_not_contain",
    [
        pytest.param(True, ">=", None, id="inclusive-lower"),
        pytest.param(True, "<=", None, id="inclusive-upper"),
        pytest.param(False, " > ", None, id="exclusive-lower"),
        pytest.param(False, " < ", "<=", id="exclusive-upper"),
    ],
)
def test_bound_int_param_addition_emits_form_per_prefer_inclusive(
    prefer_inclusive: bool, must_contain: str, must_not_contain: str | None
) -> None:
    """Test addition emits inclusive or exclusive form per ``prefer_inclusive``.

    Pins down the two branches in ``_apply_interval_bounds`` against a
    branch flip on either side.
    """
    p = (
        create_interval_integer_param_between(3, 5, prefer_inclusive=prefer_inclusive)
        + 1
    )
    text = str(p)
    assert must_contain in text
    if must_not_contain is not None:
        assert must_not_contain not in text


# =============================================================================
# Structural equivalence - `prefer_inclusive` flag
# =============================================================================


def test_bound_int_param_is_structurally_equivalent_to_self() -> None:
    """Test ``is_structurally_equivalent`` is reflexive for interval-integer params."""
    p = create_interval_integer_param_between(3, 5)

    assert p.is_structurally_equivalent(p)


def test_bound_int_param_is_not_equivalent_when_prefer_inclusive_differs() -> None:
    """Test interval-integer params with mismatched ``prefer_inclusive`` differ."""
    inclusive = create_interval_integer_param_between(
        3,
        5,
        name=mock_identifier("x", 1),
        is_lower_inclusive=True,
        is_upper_inclusive=True,
        prefer_inclusive=True,
    )
    exclusive = create_interval_integer_param_between(
        3,
        5,
        name=mock_identifier("x", 1),
        is_lower_inclusive=True,
        is_upper_inclusive=True,
        prefer_inclusive=False,
    )

    assert not inclusive.is_structurally_equivalent(exclusive)
    assert not exclusive.is_structurally_equivalent(inclusive)


def test_bound_int_param_is_not_equivalent_when_super_constraints_differ() -> None:
    """Test same-flag interval-integer params with different bounds differ."""
    smaller = create_interval_integer_param_between(3, 5, name=mock_identifier("x", 1))
    larger = create_interval_integer_param_between(3, 10, name=mock_identifier("x", 1))

    assert not smaller.is_structurally_equivalent(larger)


def test_bound_int_param_is_not_structurally_equivalent_to_int_param() -> None:
    """Test interval-integer param is not structurally equivalent to integer param."""
    bound = create_interval_integer_param(name=mock_identifier("x", 1))
    integer = create_integer_param(name=mock_identifier("x", 1))

    assert not bound.is_structurally_equivalent(integer)


# =============================================================================
# `validate_constraint` / `_is_valid_bound_expression`
# =============================================================================


def test_bound_int_param_add_constraint_rejects_non_equation_constraint() -> None:
    """Test interval-integer ``add_constraint`` rejects non-equation constraints."""
    p = create_interval_integer_param()

    with pytest.raises(TypeError):
        p.add_constraint(InSetConstraint(p.variable, {1, 2}))


def _build_bound_constraint_with_expression(
    variable: Identifier, expression: Any
) -> EquationConstraint:
    """Wrap an expression in an `EquationConstraint` for the given variable."""
    return EquationConstraint(variable, expression)


@pytest.mark.parametrize(
    "build_expression",
    [
        # Non-`BinaryExpression` falls through ``is_bound_expression`` early.
        pytest.param(lambda var: LiteralExpression(0), id="non-binary"),
        # Non-comparison binary operation (e.g. ``ADD``).
        pytest.param(lambda var: IdentifierExpression(var) + 0, id="non-comparison-op"),
        # Identifier on both sides (no literal operand).
        pytest.param(
            lambda var: IdentifierExpression(var) >= mock_identifier("y", 2),
            id="no-literal-operand",
        ),
        # Literal on both sides (no identifier operand).
        pytest.param(
            lambda var: BinaryExpression(
                BinaryOperation.GREATER_EQUAL,
                LiteralExpression(1),
                LiteralExpression(0),
            ),
            id="no-identifier-operand",
        ),
        # Non-`int` literal.
        pytest.param(
            lambda var: IdentifierExpression(var) >= 1.5, id="non-int-literal"
        ),
    ],
)
def test_bound_int_param_add_constraint_rejects_each_invalid_bound_expression(
    build_expression: Any,
) -> None:
    """Test interval-integer ``add_constraint`` rejects each invalid bound expr."""
    p = create_interval_integer_param()

    with pytest.raises(ParamError):
        p.add_constraint(
            _build_bound_constraint_with_expression(
                p.variable, build_expression(p.variable)
            )
        )


# =============================================================================
# Deserialization - structural rejection
# =============================================================================


def _valid_bound_int_payload() -> dict[str, Any]:
    """Return a well-formed derived-format interval-integer-param payload."""
    param = create_interval_integer_param(name=mock_identifier("x", 1))
    return param.serialize_to_dict()


@pytest.mark.parametrize(
    "mutate",
    [
        # Domain envelope missing its ``prefer_inclusive`` data field.
        pytest.param(
            lambda payload: payload["domain"]["__data__"].__delitem__(
                "prefer_inclusive"
            ),
            id="domain-missing-prefer-inclusive",
        ),
        # ``prefer_inclusive`` of the wrong type inside the domain data.
        pytest.param(
            lambda payload: payload["domain"]["__data__"].__setitem__(
                "prefer_inclusive", "not-a-bool"
            ),
            id="prefer-inclusive-not-bool",
        ),
        # Top-level ``constraints`` field missing.
        pytest.param(
            lambda payload: payload.__delitem__("constraints"),
            id="missing-constraints",
        ),
    ],
)
def test_bound_int_param_deserialize_rejects_each_malformed_payload(
    mutate: Any,
) -> None:
    """Test ``Param.deserialize_from_dict`` rejects each malformed bound_int payload."""
    payload = _valid_bound_int_payload()
    mutate(payload)

    with pytest.raises(DeserializationDictStructureError):
        Param.deserialize_from_dict(payload)


# =============================================================================
# Arithmetic with unbounded operands
# =============================================================================


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(
            lambda: (
                create_interval_integer_param_with_upper_bound(5)
                + create_interval_integer_param()
            ),
            id="addition-half-bounded-upper",
        ),
        pytest.param(
            lambda: (
                create_interval_integer_param_with_lower_bound(3)
                - create_interval_integer_param()
            ),
            id="subtraction-half-bounded-lower",
        ),
        pytest.param(
            lambda: (
                create_interval_integer_param_with_upper_bound(5)
                - create_interval_integer_param()
            ),
            id="subtraction-half-bounded-upper",
        ),
    ],
)
def test_bound_int_param_arithmetic_with_unbounded_operand_does_not_raise(
    operation: Any,
) -> None:
    """Test each arithmetic op with a half-bounded operand does not raise."""
    operation()


# =============================================================================
# `_coerce_other` validation of integer-param operands
# =============================================================================


def test_bound_int_param_addition_rejects_int_param_with_non_bound_constraint() -> None:
    """Test addition rejects integer param operand carrying a non-bound constraint."""
    integer = create_integer_param()
    integer = integer.add_constraint(
        EquationConstraint(
            integer.variable, (integer.variable_expression % 5).equals(0)
        )
    )
    bound = create_interval_integer_param_exactly(1)

    with pytest.raises(TypeError):
        bound + integer
