"""Tests for interval-integer parameter multiplication (``__mul__``/``__rmul__``).

Multiplication mirrors ``__add__``/``__sub__``: it is defined only for
interval-integer parameters, reuses the same coercion machinery, and
preserves ``non_negative`` only when both operands are non-negative. The
tricky part is the sign matrix and unbounded-end propagation, which these
tests pin down against hand-computed extended-integer products.
"""

import inspect

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import EquationConstraint
from fhy_core.symbolic.param import (
    Param,
    ParamError,
    create_categorical_param,
    create_integer_param,
    create_integer_param_between,
    create_interval_integer_param,
    create_interval_integer_param_between,
    create_interval_integer_param_exactly,
    create_interval_integer_param_with_lower_bound,
    create_interval_integer_param_with_upper_bound,
    create_interval_natural_param,
    create_real_param,
)

from .conftest import (
    assert_all_satisfied,
    assert_none_satisfied,
    assert_param_round_trips_in_all_formats,
    build_interval_integer_param,
)

# =============================================================================
# Sign matrix
# =============================================================================


def test_multiplication_of_positive_intervals_is_positive_product() -> None:
    """Test ``[2,3] * [4,5] -> [8,15]``."""
    x = create_interval_integer_param_between(2, 3)
    y = create_interval_integer_param_between(4, 5)

    z = x * y

    assert_all_satisfied(z, [8, 15])
    assert_none_satisfied(z, [7, 16])


def test_multiplication_of_mixed_sign_interval_uses_min_from_extreme_product() -> None:
    """Test ``[-2,3] * [4,5] -> [-10,15]``.

    The minimum comes from ``-2 * 5``, not from multiplying the two smallest
    magnitudes.
    """
    x = create_interval_integer_param_between(-2, 3)
    y = create_interval_integer_param_between(4, 5)

    z = x * y

    assert_all_satisfied(z, [-10, 15])
    assert_none_satisfied(z, [-11, 16])


def test_multiplication_of_two_negative_intervals_flips_to_positive() -> None:
    """Test ``[-2,-1] * [-3,-1] -> [1,6]``: negative times negative flips sign."""
    x = create_interval_integer_param_between(-2, -1)
    y = create_interval_integer_param_between(-3, -1)

    z = x * y

    assert_all_satisfied(z, [1, 6])
    assert_none_satisfied(z, [0, 7])


def test_multiplication_of_wholly_positive_and_wholly_negative_interval() -> None:
    """Test ``[2,5] * [-3,-1] -> [-15,-2]``.

    Distinct corner-selection path from the mixed-sign and negative-negative
    cases above: neither operand straddles zero, so the minimum comes from
    ``hi * lo`` (``5 * -3``) and the maximum from ``lo * hi`` (``2 * -1``).
    """
    x = create_interval_integer_param_between(2, 5)
    y = create_interval_integer_param_between(-3, -1)

    z = x * y

    assert_all_satisfied(z, [-15, -2])
    assert_none_satisfied(z, [-16, -1])


def test_multiplication_of_wholly_negative_and_wholly_positive_interval() -> None:
    """Test ``[-3,-1] * [2,5] -> [-15,-2]``: operand order mirrored from above."""
    x = create_interval_integer_param_between(-3, -1)
    y = create_interval_integer_param_between(2, 5)

    z = x * y

    assert_all_satisfied(z, [-15, -2])
    assert_none_satisfied(z, [-16, -1])


def test_multiplication_by_one_is_identity() -> None:
    """Test ``[2,5] * 1 -> [2,5]``: multiplying by the scalar ``1`` is an identity."""
    x = create_interval_integer_param_between(2, 5)

    z = x * 1

    assert_all_satisfied(z, [2, 5])
    assert_none_satisfied(z, [1, 6])


# =============================================================================
# Unbounded-end propagation
# =============================================================================


def test_multiplication_with_half_bounded_positive_operand_propagates_lower_bound() -> (
    None
):
    """Test ``[1,+inf) * [2,3] -> [2,+inf)``."""
    x = create_interval_integer_param_with_lower_bound(1)
    y = create_interval_integer_param_between(2, 3)

    z = x * y

    assert_all_satisfied(z, [2, 3, 10**6])
    assert_none_satisfied(z, [1])


def test_multiplication_with_fully_unbounded_operand_stays_fully_unbounded() -> None:
    """Test ``(-inf,+inf) * [2,3] -> (-inf,+inf)``."""
    x = create_interval_integer_param()
    y = create_interval_integer_param_between(2, 3)

    z = x * y

    assert_all_satisfied(z, [-(10**6), 0, 10**6])


def test_multiplication_of_zero_singleton_with_unbounded_operand_is_zero() -> None:
    """Test ``[0,0] * (-inf,+inf) -> [0,0]``: zero absorbs an unbounded operand."""
    x = create_interval_integer_param_exactly(0)
    y = create_interval_integer_param()

    z = x * y

    assert_all_satisfied(z, [0])
    assert_none_satisfied(z, [-1, 1])


def test_multiplication_of_non_negative_interval_with_unbounded_below_operand() -> None:
    """Test ``[0,5] * (-inf,0] -> (-inf,0]``."""
    x = create_interval_integer_param_between(0, 5)
    y = create_interval_integer_param_with_upper_bound(0)

    z = x * y

    assert_all_satisfied(z, [0, -(10**6)])
    assert_none_satisfied(z, [1])


@pytest.mark.parametrize(
    "left_bounds, right_bounds, admitted, rejected",
    [
        pytest.param((1, None), (-3, -1), [-(10**6), -1], [0], id="[1,+inf)*[-3,-1]"),
        pytest.param((-3, -1), (1, None), [-(10**6), -1], [0], id="[-3,-1]*[1,+inf)"),
        pytest.param((None, -1), (-3, -1), [1, 10**6], [0], id="(-inf,-1]*[-3,-1]"),
        pytest.param((-3, -1), (None, -1), [1, 10**6], [0], id="[-3,-1]*(-inf,-1]"),
        pytest.param((None, -1), (None, -1), [1, 10**6], [0], id="(-inf,-1]*(-inf,-1]"),
        pytest.param(
            (None, -1), (1, None), [-(10**6), -1], [0], id="(-inf,-1]*[1,+inf)"
        ),
        pytest.param(
            (1, None), (None, -1), [-(10**6), -1], [0], id="[1,+inf)*(-inf,-1]"
        ),
        pytest.param(
            (-2, None), (3, None), [-(10**6), -6, 10**6], [], id="[-2,+inf)*[3,+inf)"
        ),
        pytest.param(
            (3, None), (-2, None), [-(10**6), -6, 10**6], [], id="[3,+inf)*[-2,+inf)"
        ),
    ],
)
def test_multiplication_negative_finite_endpoint_signs_unbounded_corner_product(
    left_bounds: tuple[int | None, int | None],
    right_bounds: tuple[int | None, int | None],
    admitted: list[int],
    rejected: list[int],
) -> None:
    """Test a negative finite endpoint sets the sign of its unbounded corner product.

    Each case pairs a negative finite endpoint with an unbounded end of the
    other operand, in both operand orders. The unbounded side's extreme
    and the finite endpoint itself are admitted, and the first integer
    past the finite end is rejected, so a product that read the finite
    factor as positive would misplace both ends of the result.
    """
    x = build_interval_integer_param(*left_bounds)
    y = build_interval_integer_param(*right_bounds)

    z = x * y

    assert_all_satisfied(z, admitted)
    assert_none_satisfied(z, rejected)


# =============================================================================
# Scalar operands
# =============================================================================


def test_multiplication_with_int_scalar_on_right_scales_interval() -> None:
    """Test multiplication with a plain ``int`` on the right scales the interval."""
    x = create_interval_integer_param_between(3, 5)

    z = x * 2

    assert_all_satisfied(z, [6, 10])
    assert_none_satisfied(z, [5, 11])


def test_multiplication_with_int_scalar_on_left_scales_interval() -> None:
    """Test multiplication with a plain ``int`` on the left scales the interval."""
    x = create_interval_integer_param_between(3, 5)

    z = 2 * x

    assert_all_satisfied(z, [6, 10])
    assert_none_satisfied(z, [5, 11])


def test_multiplication_with_bool_operand_raises_type_error() -> None:
    """Test multiplication rejects a ``bool`` operand (``bool`` is not ``int`` here).

    The coercion step rejects the operand with its own typed ``TypeError``
    naming the offending type, rather than letting Python's operator
    protocol report a generic failure.
    """
    x = create_interval_integer_param_between(0, 1)

    with pytest.raises(TypeError, match=r"Unsupported operand type.*bool"):
        _ = x * True


def test_multiplication_with_bool_on_left_raises_type_error() -> None:
    """Test reflected multiplication rejects a ``bool`` operand on the left."""
    x = create_interval_integer_param_between(0, 1)

    with pytest.raises(TypeError, match=r"Unsupported operand type.*bool"):
        _ = True * x


# =============================================================================
# Unsupported operand kinds
# =============================================================================


def test_multiplication_with_unsupported_type_raises_type_error() -> None:
    """Test multiplication raises ``TypeError`` for an unsupported operand type.

    ``str`` defines its own ``__rmul__`` (sequence repetition), but the
    coercion step raises before Python can consult it, so the message names
    the rejected ``str`` rather than describing sequence repetition.
    """
    x = create_interval_integer_param_between(0, 1)

    with pytest.raises(TypeError, match=r"Unsupported operand type.*str"):
        _ = x * "nope"


def test_multiplication_with_real_param_raises_type_error() -> None:
    """Test multiplication raises ``TypeError`` against a real-valued parameter."""
    x = create_interval_integer_param_between(0, 1)
    y = create_real_param()

    with pytest.raises(TypeError, match=r"Unsupported operand type.*Param"):
        _ = x * y


def test_multiplication_with_categorical_param_raises_type_error() -> None:
    """Test multiplication raises ``TypeError`` against a categorical parameter."""
    x = create_interval_integer_param_between(0, 1)
    y = create_categorical_param({1, 2})

    with pytest.raises(TypeError, match=r"Unsupported operand type.*Param"):
        _ = x * y


# =============================================================================
# Reflected operator: a coercion failure pre-empts the other operand
# =============================================================================


class _ReflectedMultiplier:
    """Third-party type that would compose with a `Param` via `__rmul__`."""

    def __rmul__(self, other: object) -> str:
        del other
        return "reflected-mul"


def test_multiplication_pre_empts_a_third_party_operands_rmul() -> None:
    """Test `*` raises for an uncoercible operand instead of deferring to it.

    The coercion step raises its own typed ``TypeError`` as soon as the
    right operand has no interval form, so Python's operator protocol never
    reaches the operand's ``__rmul__``. Multiplication matches ``+`` and
    ``-`` here; all three share the coercion step.
    """
    x = create_interval_integer_param_between(0, 1)

    with pytest.raises(
        TypeError, match=r"Unsupported operand type.*_ReflectedMultiplier"
    ):
        _ = x * _ReflectedMultiplier()


# =============================================================================
# Coercion of a plain integer parameter (non-interval-integer)
# =============================================================================


def test_multiplication_accepts_plain_integer_param_on_right() -> None:
    """Test multiplication of interval-integer param with a plain integer param."""
    x = create_interval_integer_param_between(3, 5)
    y = create_integer_param_between(2, 3)

    z = x * y

    assert_all_satisfied(z, [6, 15])
    assert_none_satisfied(z, [5, 16])


def test_multiplication_accepts_plain_integer_param_on_left() -> None:
    """Test multiplication with a plain integer param on the left (via reflection)."""
    x = create_interval_integer_param_between(3, 5)
    y = create_integer_param_between(2, 3)

    z = y * x

    assert_all_satisfied(z, [6, 15])
    assert_none_satisfied(z, [5, 16])


def test_multiplication_of_two_non_interval_params_defers_to_python() -> None:
    """Test a non-interval left operand with no interval right operand defers.

    Neither side can coerce the other, so ``__mul__`` signals failure with
    ``NotImplemented`` and Python reports its own generic operator error
    rather than a diagnostic from the coercion step.
    """
    x = create_integer_param_between(2, 3)
    y = create_real_param()

    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = x * y


def test_multiplication_rejects_integer_param_with_non_bound_constraint() -> None:
    """Test multiplication rejects an integer operand with a non-bound constraint."""
    integer = create_integer_param()
    integer = integer.add_constraint(
        EquationConstraint((integer.variable_expression % 5).equals(0))
    )
    bound = create_interval_integer_param_exactly(2)

    with pytest.raises(
        TypeError, match="Cannot coerce an integer parameter with non-bound constraints"
    ):
        _ = bound * integer


# =============================================================================
# Class preservation: `non_negative` and `zero_included`
# =============================================================================


def test_multiplication_keeps_non_negative_when_both_operands_are() -> None:
    """Test ``non_negative`` is preserved when both operands are non-negative."""
    x = create_interval_natural_param(zero_included=False)
    x = x.add_lower_bound_constraint(1).add_upper_bound_constraint(3)
    y = create_interval_natural_param(zero_included=False)
    y = y.add_lower_bound_constraint(1).add_upper_bound_constraint(3)

    z = x * y

    assert isinstance(z.domain, type(x.domain))
    assert z.domain.non_negative  # type: ignore[attr-defined]


def test_multiplication_drops_non_negative_when_only_one_operand_is() -> None:
    """Test ``non_negative`` is dropped when only one operand is non-negative."""
    x = create_interval_natural_param()
    x = x.add_upper_bound_constraint(3)
    y = create_interval_integer_param_between(-2, 2)

    z = x * y

    assert not z.domain.non_negative  # type: ignore[attr-defined]


def test_multiplication_zero_included_is_or_of_operands() -> None:
    """Test the multiplication-specific ``zero_included = left or right`` rule.

    ``x`` is non-negative and excludes zero (``[1,3]``); ``y`` is non-negative
    and includes zero (``[0,3]``). Their product can be zero (``1 * 0``), so
    the result must include zero even though ``x`` alone would not.
    """
    x = create_interval_natural_param(zero_included=False)
    x = x.add_lower_bound_constraint(1).add_upper_bound_constraint(3)
    y = create_interval_natural_param(zero_included=True)
    y = y.add_upper_bound_constraint(3)

    z = x * y

    assert z.domain.non_negative  # type: ignore[attr-defined]
    assert z.domain.zero_included  # type: ignore[attr-defined]
    assert_all_satisfied(z, [0])


def test_multiplication_zero_included_is_or_of_operands_in_reversed_order() -> None:
    """Test ``[0,3] * [1,3]`` includes zero when only the LEFT operand admits it.

    The operand order of ``test_multiplication_zero_included_is_or_of_operands``
    swapped, so a result that read ``zero_included`` from the right
    operand alone would wrongly exclude zero here.
    """
    x = create_interval_natural_param(zero_included=True)
    x = x.add_upper_bound_constraint(3)
    y = create_interval_natural_param(zero_included=False)
    y = y.add_lower_bound_constraint(1).add_upper_bound_constraint(3)

    z = x * y

    assert z.domain.non_negative  # type: ignore[attr-defined]
    assert z.domain.zero_included  # type: ignore[attr-defined]
    assert_all_satisfied(z, [0])


def test_multiplication_zero_included_false_when_both_operands_exclude_zero() -> None:
    """Test ``zero_included`` stays ``False`` when neither operand admits zero."""
    x = create_interval_natural_param(zero_included=False)
    x = x.add_lower_bound_constraint(1).add_upper_bound_constraint(3)
    y = create_interval_natural_param(zero_included=False)
    y = y.add_lower_bound_constraint(1).add_upper_bound_constraint(3)

    z = x * y

    assert z.domain.non_negative  # type: ignore[attr-defined]
    assert not z.domain.zero_included  # type: ignore[attr-defined]


# =============================================================================
# Zero lower bound with ``prefer_inclusive=False`` (bound-rendering soundness)
# =============================================================================


def test_multiplication_squares_zero_included_exclusive_preference_natural() -> None:
    """Test squaring a zero-included, exclusive-preference natural param does not raise.

    A computed lower bound of ``0`` renders as the inclusive literal ``>= 0``
    rather than the exclusive ``> -1``, which the natural-number gate would
    otherwise reject even though it is an equivalent bound.
    """
    x = create_interval_natural_param(zero_included=True, prefer_inclusive=False)

    z = x * x

    assert isinstance(z.domain, type(x.domain))
    assert z.domain.non_negative  # type: ignore[attr-defined]
    assert_all_satisfied(z, [0])
    assert_none_satisfied(z, [-1])


@pytest.mark.parametrize(
    "x_zero_included, x_lower, y_zero_included, y_lower, expected_min",
    [
        pytest.param(
            True, None, False, 1, 0, id="left-zero-included-right-excludes-zero"
        ),
        pytest.param(
            False, 1, True, None, 0, id="left-excludes-zero-right-zero-included"
        ),
        pytest.param(
            True,
            1,
            True,
            1,
            1,
            id="both-zero-included-domain-but-bounded-away-from-zero",
        ),
    ],
)
def test_multiplication_zero_lower_bound_exclusive_preference_matrix(
    x_zero_included: bool,
    x_lower: int | None,
    y_zero_included: bool,
    y_lower: int | None,
    expected_min: int,
) -> None:
    """Test every zero/excludes-zero operand pairing with `prefer_inclusive=False`.

    Each combination drives the product's computed lower bound down to a
    value (``0`` or ``1``) whose raw exclusive rendering the natural-number
    gate would reject; ``_apply_interval_bounds`` must fall back to an
    inclusive rendering instead of crashing, in every case, and the result
    must still admit exactly ``[expected_min, 9]``.
    """
    x = create_interval_natural_param(
        zero_included=x_zero_included, prefer_inclusive=False
    )
    if x_lower is not None:
        x = x.add_lower_bound_constraint(x_lower)
    x = x.add_upper_bound_constraint(3)
    y = create_interval_natural_param(
        zero_included=y_zero_included, prefer_inclusive=False
    )
    if y_lower is not None:
        y = y.add_lower_bound_constraint(y_lower)
    y = y.add_upper_bound_constraint(3)

    z = x * y

    assert_all_satisfied(z, [expected_min, 9])
    assert_none_satisfied(z, [expected_min - 1, 10])


# =============================================================================
# `prefer_inclusive` rendering
# =============================================================================


def test_multiplication_rendering_follows_left_operand_prefer_inclusive() -> None:
    """Test the result's rendering follows the LEFT operand's `prefer_inclusive`.

    Each product mixes operands with DIFFERING ``prefer_inclusive`` flags
    (rather than both operands sharing the same flag), so a right-operand,
    OR, or AND mutant of the class-preservation logic is distinguishable:
    only the left operand's flag may determine the rendering.
    """
    x_incl = create_interval_integer_param_between(2, 3, prefer_inclusive=True)
    y_excl = create_interval_integer_param_between(4, 5, prefer_inclusive=False)
    x_excl = create_interval_integer_param_between(2, 3, prefer_inclusive=False)
    y_incl = create_interval_integer_param_between(4, 5, prefer_inclusive=True)

    z_left_incl = x_incl * y_excl
    z_left_excl = x_excl * y_incl

    for v in range(0, 20):
        assert z_left_incl.is_constraints_satisfied(
            v
        ) == z_left_excl.is_constraints_satisfied(v)
    assert str(z_left_incl) != str(z_left_excl)
    assert z_left_incl.domain.prefer_inclusive  # type: ignore[attr-defined]
    assert not z_left_excl.domain.prefer_inclusive  # type: ignore[attr-defined]


# =============================================================================
# Empty operand
# =============================================================================


def test_multiplication_with_empty_operand_raises_param_error() -> None:
    """Test multiplication raises ``ParamError`` when an operand is empty."""
    x = create_interval_integer_param()
    x = x.add_lower_bound_constraint(10).add_upper_bound_constraint(5)
    y = create_interval_integer_param_between(1, 2)

    with pytest.raises(ParamError):
        _ = x * y


# =============================================================================
# Result variable
# =============================================================================


def test_multiplication_result_uses_a_fresh_variable() -> None:
    """Test the multiplication result takes neither operand's variable.

    A product denotes its own quantity, so it is scoped to a freshly minted
    identifier rather than to either factor's. Every parameter here takes a
    freshly minted identifier, so the three are distinct by construction.
    """
    x = create_interval_integer_param_between(2, 3)
    y = create_interval_integer_param_between(4, 5)

    z = x * y

    assert z.variable != x.variable
    assert z.variable != y.variable
    assert isinstance(z.variable, Identifier)


def test_rmul_result_uses_a_fresh_variable() -> None:
    """Test ``int * Param`` takes neither the scalar's nor the interval's variable."""
    x = create_interval_integer_param_between(2, 3)

    z = 2 * x

    assert z.variable != x.variable
    assert isinstance(z.variable, Identifier)


# =============================================================================
# Brute-force property check
# =============================================================================


@pytest.mark.parametrize(
    "left_lower, left_upper, right_lower, right_upper",
    [
        pytest.param(0, 0, 0, 0, id="0-0-x-0-0"),
        pytest.param(0, 3, 0, 3, id="0-3-x-0-3"),
        pytest.param(-3, 3, -3, 3, id="neg3-3-x-neg3-3"),
        pytest.param(-3, -1, -3, -1, id="neg3-neg1-x-neg3-neg1"),
        pytest.param(1, 3, 1, 3, id="1-3-x-1-3"),
        pytest.param(-3, -1, 1, 3, id="neg3-neg1-x-1-3"),
        pytest.param(1, 3, -3, -1, id="1-3-x-neg3-neg1"),
        pytest.param(-2, 3, -3, 1, id="neg2-3-x-neg3-1"),
        pytest.param(0, 3, -3, -1, id="0-3-x-neg3-neg1"),
    ],
)
def test_multiplication_matches_brute_force_corner_products_over_bounded_intervals(
    left_lower: int, left_upper: int, right_lower: int, right_upper: int
) -> None:
    """Test multiplication's satisfied range matches the four-corner hull.

    Interval multiplication is an interval-hull operation, not exact set
    multiplication: for ``[a,b] * [c,d]`` the result is
    ``[min(ac,ad,bc,bd), max(ac,ad,bc,bd)]``. That hull can (and, self-
    multiplied over more than two values, generally does) admit integers
    that are not the product of any actual pair -- e.g. ``[1,3] * [1,3]``
    admits ``5``, which is not ``a * b`` for any ``a, b`` in ``{1,2,3}``.
    So this test checks the satisfied range against the corner-product hull
    directly, not against the brute-force set of actual products. The
    operands are given independently, so the asymmetric pairs exercise
    corners a self-product never reaches.
    """
    corners = [
        left_lower * right_lower,
        left_lower * right_upper,
        left_upper * right_lower,
        left_upper * right_upper,
    ]
    expected_min, expected_max = min(corners), max(corners)
    x = create_interval_integer_param_between(left_lower, left_upper)
    y = create_interval_integer_param_between(right_lower, right_upper)

    z = x * y

    for v in range(expected_min - 2, expected_max + 3):
        assert z.is_constraints_satisfied(v) == (expected_min <= v <= expected_max)


def test_signature_accepts_no_keyword_only_params_for_mul() -> None:
    """Test ``__mul__``/``__rmul__`` declare no keyword-only parameters.

    Python's operator protocol always calls a dunder with its operand
    positionally, so a keyword-only ``other`` parameter would silently
    break ``*``/reflected ``*`` dispatch; this inspects the actual
    signature rather than merely calling the methods, which would pass
    even if ``other`` were mistakenly marked keyword-only.
    """
    mul_parameters = inspect.signature(Param.__mul__).parameters.values()
    rmul_parameters = inspect.signature(Param.__rmul__).parameters.values()

    assert all(
        parameter.kind is not inspect.Parameter.KEYWORD_ONLY
        for parameter in mul_parameters
    )
    assert all(
        parameter.kind is not inspect.Parameter.KEYWORD_ONLY
        for parameter in rmul_parameters
    )


# =============================================================================
# Integration: serialization, subset/feasibility/assign interop, chaining
# =============================================================================


def test_multiplication_result_round_trips_through_serialization() -> None:
    """Test a multiplication result round-trips through DICT, JSON, and BINARY."""
    x = create_interval_integer_param_between(2, 3)
    y = create_interval_integer_param_between(4, 5)

    z = x * y

    assert_param_round_trips_in_all_formats(z)


@pytest.mark.z3
def test_multiplication_result_interoperates_with_is_subset() -> None:
    """Test ``[2,3] * [4,5]``'s result is a subset of a wider hand-built interval."""
    x = create_interval_integer_param_between(2, 3)
    y = create_interval_integer_param_between(4, 5)
    wider = create_interval_integer_param_between(0, 20)

    z = x * y

    assert z.is_subset(wider)
    assert not wider.is_subset(z)


@pytest.mark.z3
def test_multiplication_result_interoperates_with_is_feasible() -> None:
    """Test a non-empty multiplication result reports feasible."""
    x = create_interval_integer_param_between(2, 3)
    y = create_interval_integer_param_between(4, 5)

    z = x * y

    assert z.is_feasible()
    assert not z.is_empty()


def test_multiplication_result_interoperates_with_assign() -> None:
    """Test a value in the product interval can be assigned to the result."""
    x = create_interval_integer_param_between(2, 3)
    y = create_interval_integer_param_between(4, 5)

    z = x * y
    assignment = z.assign(10)

    assert assignment.value == 10
    with pytest.raises(ParamError):
        z.assign(7)


def test_chained_addition_then_multiplication() -> None:
    """Test ``(a + b) * c`` chains interval addition into interval multiplication."""
    a = create_interval_integer_param_between(1, 2)
    b = create_interval_integer_param_between(3, 4)
    c = create_interval_integer_param_between(2, 2)

    result = (a + b) * c

    # a + b -> [4, 6]; (a + b) * c -> [8, 12].
    assert_all_satisfied(result, [8, 12])
    assert_none_satisfied(result, [7, 13])
