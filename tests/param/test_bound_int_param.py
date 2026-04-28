"""Tests for `BoundIntParam`."""

from functools import partial
from typing import Any

import pytest

from fhy_core.constraint import EquationConstraint, InSetConstraint
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.identifier import Identifier
from fhy_core.param import BoundIntParam, IntParam, ParamError
from fhy_core.serialization import DeserializationDictStructureError

from .conftest import assert_all_satisfied, assert_none_satisfied

# =============================================================================
# `between` constructor
# =============================================================================


def test_bound_int_param_between_with_inclusive_bounds_satisfies_endpoints() -> None:
    """Test `BoundIntParam.between(3, 5)` admits the inclusive endpoints.

    Integer semantics: ``[3, 5]`` => ``{3, 4, 5}``.
    """
    p = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    assert_all_satisfied(p, [3, 4, 5])
    assert_none_satisfied(p, [2, 6])


def test_bound_int_param_between_with_exclusive_bounds_excludes_endpoints() -> None:
    """Test `BoundIntParam.between(3, 5)` excludes both endpoints when exclusive.

    Integer semantics: ``(3, 5)`` => ``{4}``.
    """
    p = BoundIntParam.between(3, 5, is_lower_inclusive=False, is_upper_inclusive=False)
    assert_all_satisfied(p, [4])
    assert_none_satisfied(p, [3, 5, 2, 6])


def test_bound_int_param_between_with_exclusive_lower_inclusive_upper() -> None:
    """Test `BoundIntParam.between(3, 5)` with exclusive lower, inclusive upper.

    Integer semantics: ``(3, 5]`` => ``{4, 5}``.
    """
    p = BoundIntParam.between(3, 5, is_lower_inclusive=False, is_upper_inclusive=True)
    assert_all_satisfied(p, [4, 5])
    assert_none_satisfied(p, [3, 2, 6])


def test_bound_int_param_between_with_inclusive_lower_exclusive_upper() -> None:
    """Test `BoundIntParam.between(3, 5)` with inclusive lower, exclusive upper.

    Integer semantics: ``[3, 5)`` => ``{3, 4}``.
    """
    p = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=False)
    assert_all_satisfied(p, [3, 4])
    assert_none_satisfied(p, [5, 2, 6])


def test_bound_int_param_between_with_strict_equal_bounds_raises() -> None:
    """Test `BoundIntParam.between(x, x)` raises when both bounds are exclusive."""
    with pytest.raises(ValueError):
        BoundIntParam.between(3, 3, is_lower_inclusive=False, is_upper_inclusive=False)


def test_bound_int_param_between_with_inclusive_equal_bounds_is_singleton() -> None:
    """Test `BoundIntParam.between(x, x)` with inclusive bounds admits only ``x``."""
    p = BoundIntParam.between(3, 3, is_lower_inclusive=True, is_upper_inclusive=True)
    assert_all_satisfied(p, [3])
    assert_none_satisfied(p, [2, 4])


def test_bound_int_param_between_with_reversed_bounds_raises() -> None:
    """Test `BoundIntParam.between` raises when ``lower > upper``."""
    with pytest.raises(ValueError):
        BoundIntParam.between(5, 3)


# =============================================================================
# `with_lower_bound` / `with_upper_bound`
# =============================================================================


@pytest.mark.parametrize(
    "factory, bound, is_inclusive, pass_values, fail_values",
    [
        pytest.param(
            BoundIntParam.with_lower_bound,
            3,
            True,
            [3, 4, 100],
            [2, -1],
            id="lower-inclusive",
        ),
        pytest.param(
            BoundIntParam.with_lower_bound,
            3,
            False,
            [4, 5, 100],
            [3, 2, -10],
            id="lower-exclusive",
        ),
        pytest.param(
            BoundIntParam.with_upper_bound,
            5,
            True,
            [5, 4, -100],
            [6, 7],
            id="upper-inclusive",
        ),
        pytest.param(
            BoundIntParam.with_upper_bound,
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
    """Test `BoundIntParam.exactly(7)` admits only ``7``."""
    p = BoundIntParam.exactly(7)
    assert_all_satisfied(p, [7])
    assert_none_satisfied(p, [6, 8, 0])


# =============================================================================
# `prefer_inclusive`
# =============================================================================


def test_bound_int_param_prefer_inclusive_does_not_change_satisfiable_set() -> None:
    """Test `prefer_inclusive` does not change the satisfiable set."""
    p1 = BoundIntParam.between(
        3, 5, is_lower_inclusive=False, is_upper_inclusive=False, prefer_inclusive=True
    )
    p2 = BoundIntParam.between(
        3, 5, is_lower_inclusive=False, is_upper_inclusive=False, prefer_inclusive=False
    )
    for v in range(0, 10):
        assert p1.is_constraints_satisfied(v) == p2.is_constraints_satisfied(v)


# =============================================================================
# `assign`
# =============================================================================


def test_bound_int_param_assign_accepts_int_values_only() -> None:
    """Test `BoundIntParam.assign` only accepts integer values."""
    p = BoundIntParam.with_lower_bound(0)
    assignment = p.assign(1)
    assert assignment.value == 1
    with pytest.raises(ValueError):
        p.assign(1.0)  # type: ignore[arg-type]  # test: invalid input
    with pytest.raises(ValueError):
        p.assign("1")  # type: ignore[arg-type]  # test: invalid input


def test_bound_int_param_assign_rejects_value_outside_constraints() -> None:
    """Test `BoundIntParam.assign` rejects values outside the bounds."""
    p = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    with pytest.raises(ValueError):
        p.assign(2)
    with pytest.raises(ValueError):
        p.assign(6)
    assignment = p.assign(4)
    assert assignment.value == 4


# =============================================================================
# Arithmetic — addition
# =============================================================================


def test_bound_int_param_addition_of_singletons_is_singleton() -> None:
    """Test addition of two singleton `BoundIntParam`s is a singleton."""
    x = BoundIntParam.exactly(4)
    y = BoundIntParam.exactly(6)
    z = x + y
    assert_all_satisfied(z, [10])
    assert_none_satisfied(z, [9, 11])


def test_bound_int_param_addition_with_int_on_right_shifts_interval() -> None:
    """Test addition with an `int` on the right shifts the interval."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    z = x + 2
    assert_all_satisfied(z, [5, 6, 7])
    assert_none_satisfied(z, [4, 8])


def test_bound_int_param_addition_with_int_on_left_shifts_interval() -> None:
    """Test addition with an `int` on the left shifts the interval."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
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
    x = BoundIntParam.between(3, 5, is_lower_inclusive=False, is_upper_inclusive=False)
    y = BoundIntParam.between(5, 10, is_lower_inclusive=False, is_upper_inclusive=False)
    z = x + y
    assert_all_satisfied(z, [10, 11, 12, 13])
    assert_none_satisfied(z, [9, 14])


def test_bound_int_param_addition_with_unbounded_propagates_unboundedness() -> None:
    """Test addition with an unbounded operand yields an unbounded result.

    Integer semantics: ``x >= 3``, ``y`` unbounded => ``z >= 3 + (-inf) = -inf``.
    """
    x = BoundIntParam.with_lower_bound(3, is_inclusive=True)
    y = BoundIntParam()
    z = x + y
    assert_all_satisfied(z, [-(10**6), 0, 10**6])


def test_bound_int_param_addition_accepts_int_param_on_right() -> None:
    """Test addition of `BoundIntParam` with an `IntParam` on the right."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    y = IntParam.between(5, 10, is_lower_inclusive=True, is_upper_inclusive=True)
    z = x + y
    assert_all_satisfied(z, [8, 9, 10, 11, 12, 13, 14, 15])
    assert_none_satisfied(z, [7, 16])


def test_bound_int_param_addition_with_unsupported_type_raises() -> None:
    """Test addition of `BoundIntParam` with an unsupported type raises `TypeError`."""
    x = BoundIntParam.between(0, 1)
    with pytest.raises(TypeError):
        _ = x + "nope"


def test_bound_int_param_prefer_inclusive_changes_str_not_membership_addition() -> None:
    """Test `prefer_inclusive` changes string form but not membership for addition."""
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


# =============================================================================
# Arithmetic — subtraction
# =============================================================================


def test_bound_int_param_subtraction_of_singletons_is_singleton() -> None:
    """Test subtraction of two singleton `BoundIntParam`s is a singleton."""
    x = BoundIntParam.exactly(10)
    y = BoundIntParam.exactly(6)
    z = x - y
    assert_all_satisfied(z, [4])
    assert_none_satisfied(z, [3, 5])


def test_bound_int_param_subtraction_with_int_on_right_shifts_interval() -> None:
    """Test subtraction with an `int` on the right shifts the interval."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    z = x - 2
    assert_all_satisfied(z, [1, 2, 3])
    assert_none_satisfied(z, [0, 4])


def test_bound_int_param_subtraction_with_int_on_left_shifts_interval() -> None:
    """Test subtraction with an `int` on the left shifts the interval."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
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
    x = BoundIntParam.between(3, 5, is_lower_inclusive=False, is_upper_inclusive=False)
    y = BoundIntParam.between(5, 10, is_lower_inclusive=False, is_upper_inclusive=False)
    z = x - y
    assert_all_satisfied(z, [-5, -4, -3, -2])
    assert_none_satisfied(z, [-6, -1, 0])


def test_bound_int_param_subtraction_accepts_int_param_on_right() -> None:
    """Test subtraction of `BoundIntParam` with an `IntParam` on the right."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    y = IntParam.between(5, 10, is_lower_inclusive=True, is_upper_inclusive=True)
    z = x - y
    assert_all_satisfied(z, [-7, -3, 0])
    assert_none_satisfied(z, [-8, 1])


def test_bound_int_param_rsub_accepts_int_param_on_left() -> None:
    """Test reflected subtraction with `IntParam` on the left."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    y = IntParam.between(5, 10, is_lower_inclusive=True, is_upper_inclusive=True)
    z = y - x
    assert_all_satisfied(z, [0, 7])
    assert_none_satisfied(z, [-1, 8])


# =============================================================================
# Arithmetic — negation
# =============================================================================


def test_bound_int_param_negation_of_singleton_is_negated_singleton() -> None:
    """Test negation of a singleton `BoundIntParam` is a negated singleton."""
    x = BoundIntParam.exactly(4)
    z = -x
    assert_all_satisfied(z, [-4])
    assert_none_satisfied(z, [-3, -5])


def test_bound_int_param_negation_of_inclusive_interval_reflects_endpoints() -> None:
    """Test negation of an inclusive `BoundIntParam` interval reflects its endpoints."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=True)
    z = -x
    assert_all_satisfied(z, [-5, -4, -3])
    assert_none_satisfied(z, [-6, -2])


def test_bound_int_param_negation_of_strict_interval_uses_integer_semantics() -> None:
    """Test negation of a strict-interval `BoundIntParam` uses integer semantics."""
    x = BoundIntParam.between(3, 5, is_lower_inclusive=False, is_upper_inclusive=False)
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
    x = BoundIntParam.between(
        lower,
        upper,
        is_lower_inclusive=is_lower_inclusive,
        is_upper_inclusive=is_upper_inclusive,
    )
    y = BoundIntParam.between(
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
    x = BoundIntParam.between(
        lower,
        upper,
        is_lower_inclusive=is_lower_inclusive,
        is_upper_inclusive=is_upper_inclusive,
    )
    y = BoundIntParam.between(
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
    """Test `BoundIntParam` round-trips through dict serialization with constraints."""
    p = BoundIntParam.between(3, 5, is_lower_inclusive=True, is_upper_inclusive=False)
    dictionary = p.serialize_to_dict()
    restored = BoundIntParam.deserialize_from_dict(dictionary)
    assert_all_satisfied(restored, [3, 4])
    assert_none_satisfied(restored, [5])


# =============================================================================
# Keyword-only signatures
# =============================================================================


def test_bound_int_param_init_accepts_post_marker_args_as_keywords() -> None:
    """Test `BoundIntParam.__init__` accepts post-``*`` args as keywords."""
    BoundIntParam(name=Identifier("x"), prefer_inclusive=False)


def test_bound_int_param_init_rejects_name_passed_positionally() -> None:
    """Test `BoundIntParam.__init__` rejects ``name`` passed positionally."""
    with pytest.raises(TypeError):
        BoundIntParam(Identifier("x"))  # type: ignore[misc]  # test: keyword-only


@pytest.mark.parametrize(
    "callable_, positional_args",
    [
        pytest.param(BoundIntParam.between, (1, 2), id="between"),
        pytest.param(BoundIntParam.with_lower_bound, (1,), id="with-lower-bound"),
        pytest.param(BoundIntParam.with_upper_bound, (2,), id="with-upper-bound"),
        pytest.param(BoundIntParam.exactly, (1,), id="exactly"),
    ],
)
def test_bound_int_param_classmethod_rejects_name_passed_positionally(
    callable_: object, positional_args: tuple[int, ...]
) -> None:
    """Test each `BoundIntParam` classmethod rejects ``name`` passed positionally."""
    with pytest.raises(TypeError):
        callable_(*positional_args, Identifier("x"))  # type: ignore[operator,misc]  # test: keyword-only


# =============================================================================
# Default-flag invariants
# =============================================================================


@pytest.mark.parametrize(
    "factory, expected_substring",
    [
        # Bare ``BoundIntParam()`` exercises the `__init__` default directly;
        # classmethods carry their own default that masks the `__init__` one.
        pytest.param(
            lambda: BoundIntParam().add_lower_bound_constraint(3) + 1,
            ">=",
            id="init",
        ),
        pytest.param(lambda: BoundIntParam.between(3, 5) + 1, ">=", id="between"),
        pytest.param(
            lambda: BoundIntParam.with_lower_bound(3) + 1, ">=", id="with-lower-bound"
        ),
        pytest.param(
            lambda: BoundIntParam.with_upper_bound(5) + 1, "<=", id="with-upper-bound"
        ),
        pytest.param(lambda: BoundIntParam.exactly(3) + 1, ">=", id="exactly"),
    ],
)
def test_bound_int_param_default_prefer_inclusive_emits_inclusive_form(
    factory: Any, expected_substring: str
) -> None:
    """Test each public constructor defaults to ``prefer_inclusive=True``.

    Verified through the constraint form produced by an arithmetic operation:
    with the inclusive-form preference, the resulting param's repr embeds
    ``>=`` / ``<=`` rather than ``>`` / ``<``.
    """
    assert expected_substring in str(factory())


@pytest.mark.parametrize(
    "factory, boundary_value",
    [
        pytest.param(partial(BoundIntParam.between, 3, 5), 3, id="between-lower"),
        pytest.param(partial(BoundIntParam.between, 3, 5), 5, id="between-upper"),
        pytest.param(
            partial(BoundIntParam.with_lower_bound, 3), 3, id="with-lower-bound"
        ),
        pytest.param(
            partial(BoundIntParam.with_upper_bound, 5), 5, id="with-upper-bound"
        ),
    ],
)
def test_bound_int_param_default_inclusivity_admits_endpoint(
    factory: Any, boundary_value: int
) -> None:
    """Test each public constructor defaults to inclusive bounds (admits endpoint)."""
    assert factory().is_value_valid(boundary_value)


# =============================================================================
# Literal-on-left bound expressions and `_invert_binary_comparison_operation`
# =============================================================================


def _build_literal_left_constraint(
    variable: Identifier, op: BinaryOperation, literal_value: int
) -> EquationConstraint:
    """Build a ``literal op variable`` `EquationConstraint`."""
    return EquationConstraint(
        variable,
        BinaryExpression(
            op, LiteralExpression(literal_value), IdentifierExpression(variable)
        ),
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
    """Test `BoundIntParam` handles ``literal op variable`` constraints correctly.

    The four cases together drive `_invert_binary_comparison_operation`'s four
    branches via the ``literal op var`` arm of `_iter_bounds`. Asserting the
    resulting effective interval pins down each branch's return value: a
    mutated branch would either return the wrong inverted operation (flipping
    a lower bound to an upper bound or vice versa) or raise ``TypeError``
    (because pure `Enum` members do not support ordering).
    """
    var = Identifier("x")
    p = BoundIntParam(name=var).add_constraint(
        _build_literal_left_constraint(var, op, literal_value)
    )
    # Force `_iter_bounds` -> `_invert_binary_comparison_operation` via arithmetic.
    shifted = p + 0
    if expected_min is not None:
        assert shifted.is_value_valid(expected_min)
        assert not shifted.is_value_valid(expected_min - 1)
    if expected_max is not None:
        assert shifted.is_value_valid(expected_max)
        assert not shifted.is_value_valid(expected_max + 1)


def test_bound_int_param_iter_bounds_accepts_literal_on_left_constraint() -> None:
    """Test `_iter_bounds` accepts a well-formed ``literal op var`` constraint.

    Pins down the ``not isinstance(expression.left, LiteralExpression)`` guard
    in `_iter_bounds`: removing or inverting the ``not`` would raise
    ``RuntimeError`` for the (valid) literal-on-left case. Triggered through
    arithmetic, which is the public entry point that calls `_iter_bounds`.
    """
    var = Identifier("x")
    p = BoundIntParam(name=var).add_constraint(
        _build_literal_left_constraint(var, BinaryOperation.LESS_EQUAL, 1)
    )
    p + 0  # noqa: B015  # test: must not raise


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

    Pins down the two branches in `_create_param_from_min_max` against a
    branch flip on either side: with ``prefer_inclusive=True`` the result
    embeds ``x >= min_int`` / ``x <= max_int``; with ``False`` it embeds
    ``x > min_int - 1`` / ``x < max_int + 1``.
    """
    p = BoundIntParam.between(3, 5, prefer_inclusive=prefer_inclusive) + 1
    text = str(p)
    assert must_contain in text
    if must_not_contain is not None:
        assert must_not_contain not in text


# =============================================================================
# Structural equivalence — `_prefer_inclusive` flag
# =============================================================================


def test_bound_int_param_is_structurally_equivalent_to_self() -> None:
    """Test `BoundIntParam.is_structurally_equivalent` is reflexive.

    Reflexive equivalence pins down ``==`` against ``!=``, ``is not``, ``<``,
    and ``>`` on `_prefer_inclusive`: each of those mutations evaluates to
    ``False`` for ``True == True``, breaking reflexivity.
    """
    p = BoundIntParam.between(3, 5)
    assert p.is_structurally_equivalent(p)


def test_bound_int_param_is_not_equivalent_when_prefer_inclusive_differs() -> None:
    """Test `BoundIntParam`s with mismatched ``_prefer_inclusive`` are not equivalent.

    Constructs both params via ``between(..., is_lower_inclusive=True,
    is_upper_inclusive=True, prefer_inclusive=...)`` so the underlying
    constraint sets are identical and the only discriminator is the
    representation flag. Asserts non-equivalence in *both* directions to
    cover ``<=`` and ``>=`` mutations on ``bool`` operands.
    """
    shared_name = Identifier("x")
    shared_name_copy = Identifier.deserialize_from_dict(shared_name.serialize_to_dict())
    inclusive = BoundIntParam.between(
        3,
        5,
        name=shared_name,
        is_lower_inclusive=True,
        is_upper_inclusive=True,
        prefer_inclusive=True,
    )
    exclusive = BoundIntParam.between(
        3,
        5,
        name=shared_name_copy,
        is_lower_inclusive=True,
        is_upper_inclusive=True,
        prefer_inclusive=False,
    )
    assert not inclusive.is_structurally_equivalent(exclusive)
    assert not exclusive.is_structurally_equivalent(inclusive)


def test_bound_int_param_is_not_equivalent_when_super_constraints_differ() -> None:
    """Test two same-flag `BoundIntParam`s with different bounds are not equivalent.

    Pins down the ``and`` between ``isinstance(...)`` and
    ``super().is_structurally_equivalent(...)`` against an ``or`` weakening:
    with the mutation, the isinstance match alone would short-circuit to
    ``True`` regardless of constraint differences.
    """
    shared_name = Identifier("x")
    shared_name_copy = Identifier.deserialize_from_dict(shared_name.serialize_to_dict())
    smaller = BoundIntParam.between(3, 5, name=shared_name)
    larger = BoundIntParam.between(3, 10, name=shared_name_copy)
    assert not smaller.is_structurally_equivalent(larger)


def test_bound_int_param_is_not_structurally_equivalent_to_int_param() -> None:
    """Test a `BoundIntParam` is not structurally equivalent to a plain `IntParam`.

    Pins down the ``isinstance(other, BoundIntParam)`` short-circuit against
    an ``or`` weakening that would short-circuit on the
    ``_prefer_inclusive`` comparison and raise ``AttributeError`` when
    ``other`` does not carry that attribute.
    """
    shared_name = Identifier("x")
    shared_name_copy = Identifier.deserialize_from_dict(shared_name.serialize_to_dict())
    bound = BoundIntParam(name=shared_name)
    integer = IntParam(name=shared_name_copy)
    assert not bound.is_structurally_equivalent(integer)


# =============================================================================
# `validate_constraint` / `_is_valid_bound_expression`
# =============================================================================


def test_bound_int_param_add_constraint_rejects_non_equation_constraint() -> None:
    """Test `BoundIntParam.add_constraint` rejects non-`EquationConstraint`."""
    p = BoundIntParam()
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
        # Non-`BinaryExpression` falls through ``_is_valid_bound_expression`` early.
        pytest.param(lambda var: LiteralExpression(0), id="non-binary"),
        # Non-comparison binary operation (e.g. ``ADD``).
        pytest.param(
            lambda var: BinaryExpression(
                BinaryOperation.ADD, IdentifierExpression(var), LiteralExpression(0)
            ),
            id="non-comparison-op",
        ),
        # Identifier on both sides (no literal operand).
        pytest.param(
            lambda var: BinaryExpression(
                BinaryOperation.GREATER_EQUAL,
                IdentifierExpression(var),
                IdentifierExpression(Identifier("y")),
            ),
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
            lambda var: BinaryExpression(
                BinaryOperation.GREATER_EQUAL,
                IdentifierExpression(var),
                LiteralExpression(1.5),
            ),
            id="non-int-literal",
        ),
    ],
)
def test_bound_int_param_add_constraint_rejects_each_invalid_bound_expression(
    build_expression: Any,
) -> None:
    """Test `BoundIntParam.add_constraint` rejects each invalid bound-expression shape.

    Each parametrized case breaks one structural conjunct of
    ``_is_valid_bound_expression``: non-`BinaryExpression`, non-comparison
    operation, missing literal operand, missing identifier operand, or a
    non-``int`` literal value.
    """
    p = BoundIntParam()
    with pytest.raises(ParamError):
        p.add_constraint(
            _build_bound_constraint_with_expression(
                p.variable, build_expression(p.variable)
            )
        )


# =============================================================================
# Deserialization — structural rejection
# =============================================================================


def _wrap_bound_int_data(inner: dict[str, object]) -> dict[str, object]:
    return {"__type__": "bound_int_param", "__data__": inner}


@pytest.mark.parametrize(
    "build_inner_data",
    [
        # Missing ``prefer_inclusive``.
        pytest.param(
            lambda: {
                "variable": Identifier("x").serialize_to_dict(),
                "constraints": [],
            },
            id="missing-prefer-inclusive",
        ),
        # ``prefer_inclusive`` of the wrong type.
        pytest.param(
            lambda: {
                "variable": Identifier("x").serialize_to_dict(),
                "constraints": [],
                "prefer_inclusive": "not-a-bool",
            },
            id="prefer-inclusive-not-bool",
        ),
        # Inner ``is_valid_param_data`` fails (missing ``constraints``).
        pytest.param(
            lambda: {
                "variable": Identifier("x").serialize_to_dict(),
                "prefer_inclusive": True,
            },
            id="failing-inner-check",
        ),
    ],
)
def test_bound_int_param_deserialize_rejects_each_malformed_payload(
    build_inner_data: Any,
) -> None:
    """Test ``BoundIntParam.deserialize_from_dict`` rejects each malformed payload.

    Each parametrized case breaks exactly one conjunct of the validator's
    ``and``-chain: the missing ``prefer_inclusive`` field, a wrong-type
    ``prefer_inclusive``, or a failure in the inner ``is_valid_param_data``
    check.
    """
    payload = _wrap_bound_int_data(build_inner_data())
    with pytest.raises(DeserializationDictStructureError):
        BoundIntParam.deserialize_from_dict(payload)  # type: ignore[arg-type]  # test: dict shape


# =============================================================================
# Arithmetic with unbounded operands
# =============================================================================


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(
            lambda: BoundIntParam.with_upper_bound(5) + BoundIntParam(),
            id="addition-half-bounded-upper",
        ),
        pytest.param(
            lambda: BoundIntParam.with_lower_bound(3) - BoundIntParam(),
            id="subtraction-half-bounded-lower",
        ),
        pytest.param(
            lambda: BoundIntParam.with_upper_bound(5) - BoundIntParam(),
            id="subtraction-half-bounded-upper",
        ),
    ],
)
def test_bound_int_param_arithmetic_with_unbounded_operand_does_not_raise(
    operation: Any,
) -> None:
    """Test each arithmetic op with a half-bounded operand does not raise.

    Pins down the ``or`` in each ``None if (a is None or b is None)`` short-
    circuit against an ``and`` weakening: with ``and``, the half-bounded
    case would fall through and attempt ``int + None`` / ``int - None``,
    raising ``TypeError``.
    """
    operation()


# =============================================================================
# `_coerce_other` validation of `IntParam` operands
# =============================================================================


def test_bound_int_param_addition_rejects_int_param_with_non_bound_constraint() -> None:
    """Test addition rejects an `IntParam` operand carrying a non-bound constraint.

    Pins down the per-constraint validation loop in `_coerce_other` against a
    zero-iteration mutation: a skipped loop would let the non-bound
    constraint propagate to `_iter_bounds`, where it surfaces as a
    `RuntimeError` rather than the public-API `TypeError`.
    """
    integer = IntParam()
    integer = integer.add_constraint(
        EquationConstraint(
            integer.variable, (integer.variable_expression % 5).equals(0)
        )
    )
    bound = BoundIntParam.exactly(1)
    with pytest.raises(TypeError):
        bound + integer
