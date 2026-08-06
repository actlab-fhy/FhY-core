"""Tests for natural-bound gating across the accepted bound literal types.

``add_lower_bound_constraint`` and ``add_upper_bound_constraint`` accept
``int``, ``float``, and string literals. On a non-negative integer domain all
three must pass through the same natural-number gates, so that no single bound
can build a parameter that admits no value at all.
"""

from decimal import Decimal
from typing import Any

import pytest

from fhy_core.symbolic.param import (
    Param,
    ParamError,
    create_integer_param,
    create_natural_param,
    create_real_param,
)

from .conftest import mock_identifier


def _create_natural_param(*, zero_included: bool = True) -> Param[int]:
    """Create a named natural param."""
    return create_natural_param(
        name=mock_identifier("x", 1), zero_included=zero_included
    )


def _add_bound(
    param: Param[int], bound: int | float | str, *, is_lower: bool, is_inclusive: bool
) -> Param[int]:
    """Add ``bound`` to ``param`` as a lower or upper bound."""
    if is_lower:
        return param.add_lower_bound_constraint(bound, is_inclusive=is_inclusive)
    return param.add_upper_bound_constraint(bound, is_inclusive=is_inclusive)


# =============================================================================
# Integer-valued floats gate exactly as the equivalent `int`
#
# A whole-valued float denotes the same bound as its integer counterpart, so it
# must reach the same verdict. These rows mirror the `int` boundary matrix in
# `test_nat_param.py`.
# =============================================================================


_INTEGRAL_FLOAT_CASES = [
    pytest.param(True, True, True, -1.0, "raises", id="lower-zero-incl-incl-neg1"),
    pytest.param(True, True, True, 0.0, "succeeds", id="lower-zero-incl-incl-0"),
    pytest.param(True, True, False, 0.0, "raises", id="lower-zero-incl-excl-0"),
    pytest.param(True, True, False, 1.0, "succeeds", id="lower-zero-incl-excl-1"),
    pytest.param(True, False, True, 0.0, "raises", id="lower-zero-excl-incl-0"),
    pytest.param(True, False, True, 1.0, "succeeds", id="lower-zero-excl-incl-1"),
    pytest.param(False, True, True, -1.0, "raises", id="upper-zero-incl-incl-neg1"),
    pytest.param(False, True, True, -5.0, "raises", id="upper-zero-incl-incl-neg5"),
    pytest.param(False, True, True, 0.0, "succeeds", id="upper-zero-incl-incl-0"),
    pytest.param(False, True, False, 0.0, "raises", id="upper-zero-incl-excl-0"),
    pytest.param(False, True, False, 1.0, "succeeds", id="upper-zero-incl-excl-1"),
    pytest.param(False, False, True, 0.0, "raises", id="upper-zero-excl-incl-0"),
    pytest.param(False, False, True, 1.0, "succeeds", id="upper-zero-excl-incl-1"),
    pytest.param(False, False, False, 1.0, "raises", id="upper-zero-excl-excl-1"),
    pytest.param(False, False, False, 2.0, "succeeds", id="upper-zero-excl-excl-2"),
]


@pytest.mark.parametrize(
    "is_lower, zero_included, is_inclusive, bound, kind", _INTEGRAL_FLOAT_CASES
)
def test_nat_param_gates_integral_float_bound_at_the_integer_threshold(
    is_lower: bool, zero_included: bool, is_inclusive: bool, bound: float, kind: str
) -> None:
    """Test a whole-valued float bound hits the same threshold as its ``int``."""
    param = _create_natural_param(zero_included=zero_included)

    if kind == "raises":
        with pytest.raises(ParamError):
            _add_bound(param, bound, is_lower=is_lower, is_inclusive=is_inclusive)
    else:
        _add_bound(param, bound, is_lower=is_lower, is_inclusive=is_inclusive)


@pytest.mark.parametrize(
    "is_lower, zero_included, is_inclusive, bound, kind", _INTEGRAL_FLOAT_CASES
)
def test_nat_param_gates_integral_float_bound_identically_to_int_bound(
    is_lower: bool, zero_included: bool, is_inclusive: bool, bound: float, kind: str
) -> None:
    """Test the float and ``int`` spellings of one bound agree on the verdict."""
    param = _create_natural_param(zero_included=zero_included)

    def determine_outcome(literal: int | float) -> str:
        try:
            _add_bound(param, literal, is_lower=is_lower, is_inclusive=is_inclusive)
        except ParamError:
            return "raises"
        return "succeeds"

    assert determine_outcome(bound) == determine_outcome(int(bound)) == kind


# =============================================================================
# Fractional bounds gate at the floor (upper) or ceiling (lower)
#
# Over the integers a fractional bound admits the same values as an inclusive
# bound at its floor or ceiling, so it is rejected only when that integer bound
# would be. `x <= 2.5` stays legal; `x <= -0.5` does not.
# =============================================================================


_FRACTIONAL_CASES = [
    pytest.param(False, True, True, 2.5, "succeeds", id="upper-2.5-admits-0-1-2"),
    pytest.param(False, True, False, 0.5, "succeeds", id="upper-excl-0.5-admits-0"),
    pytest.param(False, True, True, -0.5, "raises", id="upper-neg0.5-admits-nothing"),
    pytest.param(False, True, True, -5.5, "raises", id="upper-neg5.5-admits-nothing"),
    pytest.param(False, False, True, 0.5, "raises", id="upper-0.5-excludes-zero-only"),
    pytest.param(False, False, True, 1.5, "succeeds", id="upper-1.5-admits-1"),
    pytest.param(True, True, True, -0.5, "succeeds", id="lower-neg0.5-is-redundant"),
    pytest.param(True, True, True, 0.5, "succeeds", id="lower-0.5-admits-1-up"),
    pytest.param(True, True, False, 0.5, "succeeds", id="lower-excl-0.5-admits-1-up"),
]


@pytest.mark.parametrize(
    "is_lower, zero_included, is_inclusive, bound, kind", _FRACTIONAL_CASES
)
def test_nat_param_gates_fractional_float_bound_at_its_integer_equivalent(
    is_lower: bool, zero_included: bool, is_inclusive: bool, bound: float, kind: str
) -> None:
    """Test a fractional float bound is judged by the integers it actually admits."""
    param = _create_natural_param(zero_included=zero_included)

    if kind == "raises":
        with pytest.raises(ParamError):
            _add_bound(param, bound, is_lower=is_lower, is_inclusive=is_inclusive)
    else:
        _add_bound(param, bound, is_lower=is_lower, is_inclusive=is_inclusive)


# =============================================================================
# String literals
#
# The literal grammar is unsigned, so a string bound is never negative; the
# reachable failures are the ones that undercut an excluded zero.
# =============================================================================


_STRING_CASES = [
    pytest.param(True, "5", "succeeds", id="upper-5-admits-0-through-5"),
    pytest.param(True, "0", "succeeds", id="upper-0-admits-zero"),
    pytest.param(True, "1.5", "succeeds", id="upper-1.5-admits-0-and-1"),
    pytest.param(False, "0", "raises", id="upper-0-with-zero-excluded"),
    pytest.param(False, "0.5", "raises", id="upper-0.5-with-zero-excluded"),
    pytest.param(False, "1", "succeeds", id="upper-1-with-zero-excluded"),
    pytest.param(False, "1.5", "succeeds", id="upper-1.5-with-zero-excluded"),
]


@pytest.mark.parametrize("zero_included, bound, kind", _STRING_CASES)
def test_nat_param_gates_string_upper_bound_by_the_value_it_denotes(
    zero_included: bool, bound: str, kind: str
) -> None:
    """Test a string upper bound passes through the natural-number gates."""
    param = _create_natural_param(zero_included=zero_included)

    if kind == "raises":
        with pytest.raises(ParamError):
            param.add_upper_bound_constraint(bound)
    else:
        param.add_upper_bound_constraint(bound)


def test_nat_param_gates_zero_padded_string_bound_by_its_numeric_value() -> None:
    """Test a zero-padded string bound is read numerically, not lexically."""
    param = _create_natural_param(zero_included=False)

    with pytest.raises(ParamError):
        param.add_upper_bound_constraint("00")


# "nan", "Infinity", and "-Infinity" parse as `Decimal` but are non-finite;
# "not-a-number" fails to parse at all. Both kinds of failure are rejected
# directly by the natural-bound gate, naming the offending bound, rather than
# falling through to the literal grammar's error.
@pytest.mark.parametrize(
    "bound", ["not-a-number", "nan", "Infinity", "-Infinity"], ids=str
)
def test_nat_param_rejects_unparsable_or_non_finite_upper_bound_string(
    bound: str,
) -> None:
    """Test an unparsable or non-finite string upper bound raises ``ParamError``."""
    param = _create_natural_param()

    with pytest.raises(ParamError, match="Upper bound"):
        param.add_upper_bound_constraint(bound)


@pytest.mark.parametrize(
    "bound", ["not-a-number", "nan", "Infinity", "-Infinity"], ids=str
)
def test_nat_param_rejects_unparsable_or_non_finite_lower_bound_string(
    bound: str,
) -> None:
    """Test an unparsable or non-finite string lower bound raises ``ParamError``."""
    param = _create_natural_param()

    with pytest.raises(ParamError, match="Lower bound"):
        param.add_lower_bound_constraint(bound)


def test_nat_param_rejects_negative_numeric_string_lower_bound() -> None:
    """Test a negative numeric-string lower bound raises ``ParamError``."""
    param = _create_natural_param()

    with pytest.raises(ParamError, match="non-negative"):
        param.add_lower_bound_constraint("-5")


@pytest.mark.parametrize("bound", [Decimal("5"), None], ids=["decimal", "none"])
def test_nat_param_leaves_bound_of_an_unsupported_type_to_the_expression_layer(
    bound: Any,
) -> None:
    """Test a bound outside ``int | float | str`` is not silently gated.

    The gates recognize the literal types the bound can be stored as; anything
    else falls through to the expression layer, which rejects it by type.
    """
    param = _create_natural_param()

    with pytest.raises(ValueError, match="to an expression"):
        param.add_upper_bound_constraint(bound)


# =============================================================================
# Non-finite float bounds
# =============================================================================


@pytest.mark.parametrize(
    "bound", [float("inf"), float("-inf"), float("nan")], ids=["inf", "-inf", "nan"]
)
@pytest.mark.parametrize("is_lower", [True, False], ids=["lower", "upper"])
def test_nat_param_rejects_non_finite_bound(bound: float, is_lower: bool) -> None:
    """Test a non-finite float bound is rejected rather than stored."""
    param = _create_natural_param()

    with pytest.raises(ParamError, match="must be finite"):
        _add_bound(param, bound, is_lower=is_lower, is_inclusive=True)


# =============================================================================
# The invariant the gates exist to protect
# =============================================================================


@pytest.mark.parametrize(
    "bound",
    [-5, -5.0, -5.5, -0.5, 0, 0.0, 0.5, 1, 2.5, "0", "0.5", "5"],
    ids=str,
)
@pytest.mark.parametrize("zero_included", [True, False], ids=["zero-incl", "zero-excl"])
@pytest.mark.parametrize("is_inclusive", [True, False], ids=["incl", "excl"])
@pytest.mark.parametrize("is_lower", [True, False], ids=["lower", "upper"])
def test_nat_param_single_bound_either_raises_or_stays_feasible(
    bound: int | float | str, zero_included: bool, is_inclusive: bool, is_lower: bool
) -> None:
    """Test one bound on a fresh natural param never yields a silent dead param.

    A bound that admits no natural number must raise at construction. The
    gates are deliberately per-bound: crossing a lower and an upper bound is
    still allowed to build an infeasible param.
    """
    param = _create_natural_param(zero_included=zero_included)

    try:
        bounded = _add_bound(param, bound, is_lower=is_lower, is_inclusive=is_inclusive)
    except ParamError:
        return

    assert bounded.is_feasible(), (
        f"{'Lower' if is_lower else 'Upper'} bound {bound!r} "
        f"(is_inclusive={is_inclusive}, zero_included={zero_included}) was accepted "
        f"but admits no value"
    )


# =============================================================================
# Domains outside the natural gates are unaffected
# =============================================================================


def test_real_param_still_accepts_a_negative_float_bound() -> None:
    """Test the natural gates do not reach real-valued parameters."""
    param = create_real_param(name=mock_identifier("x", 1))

    assert param.add_upper_bound_constraint(-5.5).constraints


def test_signed_integer_param_still_accepts_a_negative_float_bound() -> None:
    """Test the natural gates do not reach integer params that allow negatives."""
    param = create_integer_param(name=mock_identifier("x", 1))

    assert param.add_upper_bound_constraint(-5.5).constraints


def test_real_param_still_accepts_a_non_finite_float_bound() -> None:
    """Test the finiteness check is part of the natural gates, not a global rule."""
    param = create_real_param(name=mock_identifier("x", 1))

    assert param.add_upper_bound_constraint(float("inf")).constraints
