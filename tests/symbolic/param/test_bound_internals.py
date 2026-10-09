"""Tests for the bound decoding interval-integer arithmetic reads.

The Rust core decodes each bound constraint of an interval param into its
side, integer and inclusivity. The rules are pinned here through the public
arithmetic, which reads the decoded interval, and the interval domain's
constraint check, which keeps every constraint the decoding reads a bound.
"""

from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import EquationConstraint, InSetConstraint
from fhy_core.symbolic.expression import (
    Expression,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.symbolic.param import (
    Param,
    ParamError,
    create_interval_integer_param,
)

from .conftest import mock_identifier

_Build = Callable[[Expression], Expression]


def _bounded(build: _Build) -> tuple[Param[int], Identifier]:
    """Return an interval param with the one bound `build` makes of its variable."""
    x = mock_identifier("x", 1)
    param = create_interval_integer_param(name=x)
    return param.add_constraint(EquationConstraint(build(IdentifierExpression(x)))), x


def _admitted(param: Any, values: range) -> list[int]:
    """Return the values of `values` the param admits."""
    return [value for value in values if param.is_value_valid(value)]


# =============================================================================
# A bound written with the literal on the left reads inverted
# =============================================================================


@pytest.mark.sympy
@pytest.mark.parametrize(
    ("build", "expected"),
    [
        pytest.param(lambda x: LiteralExpression(7) > x, list(range(3, 7)), id="gt-lt"),
        pytest.param(
            lambda x: LiteralExpression(7) >= x, list(range(3, 8)), id="ge-le"
        ),
        pytest.param(
            lambda x: LiteralExpression(4) < x, list(range(5, 10)), id="lt-gt"
        ),
        pytest.param(
            lambda x: LiteralExpression(4) <= x, list(range(4, 10)), id="le-ge"
        ),
    ],
)
def test_arithmetic_reads_a_literal_left_bound_inverted(
    build: _Build, expected: list[int]
) -> None:
    """Test a `k <cmp> x` bound reads as `x <inverse> k` in arithmetic."""
    param, _ = _bounded(build)

    result = param + 0

    assert _admitted(result, range(3, 10)) == expected


# =============================================================================
# Each comparison operator decodes into its side and inclusivity
# =============================================================================


@pytest.mark.sympy
@pytest.mark.parametrize(
    ("build", "expected"),
    [
        pytest.param(lambda x: x > 7, list(range(8, 12)), id="gt"),
        pytest.param(lambda x: x >= 7, list(range(7, 12)), id="ge"),
        pytest.param(lambda x: x < 7, list(range(3, 7)), id="lt"),
        pytest.param(lambda x: x <= 7, list(range(3, 8)), id="le"),
    ],
)
def test_arithmetic_decodes_each_comparison_operator(
    build: _Build, expected: list[int]
) -> None:
    """Test each comparison operator decodes into the interval it bounds."""
    param, _ = _bounded(build)

    result = param + 0

    assert _admitted(result, range(3, 12)) == expected


# =============================================================================
# The interval domain keeps every constraint the decoding reads a bound
# =============================================================================


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda x: x > LiteralExpression(1.5), id="non-int-literal"),
        pytest.param(lambda x: (x + 1).equals(3), id="non-comparison"),
    ],
)
def test_interval_domain_refuses_a_constraint_that_is_no_bound(build: _Build) -> None:
    """Test an interval param refuses an equation the decoding could not read."""
    x = mock_identifier("x", 1)
    param = create_interval_integer_param(name=x)
    constraint = EquationConstraint(build(IdentifierExpression(x)))

    with pytest.raises(ParamError, match="bound expressions"):
        param.add_constraint(constraint)


def test_interval_domain_refuses_a_set_constraint() -> None:
    """Test an interval param refuses a set constraint with `TypeError`."""
    x = mock_identifier("x", 1)

    with pytest.raises(TypeError, match="interval integer parameters"):
        create_interval_integer_param(name=x).add_constraint(InSetConstraint(x, {1}))


@pytest.mark.sympy
def test_interval_arithmetic_reads_the_core_constraints_not_a_slot() -> None:
    """Test a slot overwritten in place cannot reach the arithmetic.

    The arithmetic reads the constraints the Rust core validated, so the
    malformed state the Python guards once defended against cannot arise.
    """
    x = mock_identifier("x", 1)
    param = create_interval_integer_param(name=x).add_lower_bound_constraint(2)
    type(param).constraints.__set__(param, (InSetConstraint(x, {1}),))  # type: ignore[attr-defined]

    result = param + 0

    assert _admitted(result, range(0, 4)) == [2, 3]
