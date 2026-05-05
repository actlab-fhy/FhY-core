"""Tests for `Param.is_subset` across parameter types."""

from fhy_core.constraint import EquationConstraint
from fhy_core.param import IntParam, RealParam

# =============================================================================
# Same-class subset relations
# =============================================================================


def test_unconstrained_real_param_is_subset_of_unconstrained_real_param() -> None:
    """Test two unconstrained `RealParam`s are mutual subsets of each other."""
    left = RealParam()
    right = RealParam()
    assert left.is_subset(right)
    assert right.is_subset(left)


def test_constrained_real_param_is_subset_of_unconstrained_real_param() -> None:
    """Test a constrained `RealParam` is a subset of an unconstrained one only."""
    constrained = RealParam()
    constrained = constrained.add_constraint(
        EquationConstraint(constrained.variable, constrained.variable_expression > 0)
    )
    unconstrained = RealParam()
    assert constrained.is_subset(unconstrained)
    assert not unconstrained.is_subset(constrained)


def test_narrower_interval_real_param_is_subset_of_wider_interval_real_param() -> None:
    """Test a narrower-interval `RealParam` is a subset of a wider-interval one."""
    wider = RealParam()
    wider = wider.add_constraint(
        EquationConstraint(wider.variable, wider.variable_expression >= 0)
    )
    wider = wider.add_constraint(
        EquationConstraint(wider.variable, wider.variable_expression <= 3)
    )
    narrower = RealParam()
    narrower = narrower.add_constraint(
        EquationConstraint(narrower.variable, narrower.variable_expression >= 0)
    )
    narrower = narrower.add_constraint(
        EquationConstraint(narrower.variable, narrower.variable_expression <= 2)
    )
    assert not wider.is_subset(narrower)
    assert narrower.is_subset(wider)


# =============================================================================
# Cross-class subset relations
# =============================================================================


def test_is_subset_returns_false_for_different_param_classes() -> None:
    """Test `is_subset` returns ``False`` between different `Param` subclasses.

    Pins down the early-return guard against cross-class subset queries: an
    `IntParam` is not a subset of a `RealParam` even when both are unconstrained.
    """
    assert not IntParam().is_subset(RealParam())  # type: ignore[arg-type]  # test: cross class
    assert not RealParam().is_subset(IntParam())  # type: ignore[arg-type]  # test: cross class
