"""Tests for `Param.is_feasible` and `Param.is_empty` across param kinds.

A parameter is feasible when at least one value satisfies its domain and all of
its constraints; it is empty otherwise. Construction never rejects a jointly
empty parameter, so these tests confirm the empty constructions build before
querying feasibility.
"""

from fhy_core.symbolic.constraint import InSetConstraint, NotInSetConstraint
from fhy_core.symbolic.param import (
    create_categorical_param,
    create_integer_param,
    create_integer_param_between,
    create_ordinal_param,
    create_permutation_param,
)

from .conftest import mock_identifier

# =============================================================================
# Feasible cases
# =============================================================================


def test_unconstrained_integer_param_is_feasible() -> None:
    """Test an unconstrained integer parameter is feasible."""
    param = create_integer_param(name=mock_identifier("x", 1))

    assert param.is_feasible()
    assert not param.is_empty()


def test_bounded_integer_param_is_feasible() -> None:
    """Test a bounded integer parameter with a non-empty interval is feasible."""
    param = create_integer_param_between(0, 10, name=mock_identifier("x", 2))

    assert param.is_feasible()
    assert not param.is_empty()


def test_unnarrowed_ordinal_param_is_feasible() -> None:
    """Test an ordinal parameter with no narrowing constraints is feasible."""
    param = create_ordinal_param([1, 2, 3], name=mock_identifier("x", 3))

    assert param.is_feasible()
    assert not param.is_empty()


def test_unnarrowed_categorical_param_is_feasible() -> None:
    """Test a categorical parameter with no narrowing constraints is feasible."""
    param = create_categorical_param(["a", "b", "c"], name=mock_identifier("x", 4))

    assert param.is_feasible()
    assert not param.is_empty()


def test_unnarrowed_permutation_param_is_feasible() -> None:
    """Test a permutation parameter with no narrowing constraints is feasible."""
    param = create_permutation_param(["a", "b", "c"], name=mock_identifier("x", 5))

    assert param.is_feasible()
    assert not param.is_empty()


def test_categorical_param_with_admitting_constraint_is_feasible() -> None:
    """Test a categorical param stays feasible when a constraint admits a value."""
    param = create_categorical_param(["a", "b", "c"], name=mock_identifier("x", 6))
    narrowed = param.add_constraint(InSetConstraint(param.variable, {"a", "b"}))

    assert narrowed.is_feasible()
    assert not narrowed.is_empty()


def test_permutation_param_with_admitting_constraint_is_feasible() -> None:
    """Test a permutation param stays feasible when a constraint admits a value."""
    param = create_permutation_param(["a", "b", "c"], name=mock_identifier("x", 7))
    narrowed = param.add_constraint(InSetConstraint(param.variable, {("a", "b", "c")}))

    assert narrowed.is_feasible()
    assert not narrowed.is_empty()


# =============================================================================
# Empty / infeasible cases
# =============================================================================


def test_categorical_param_with_out_of_domain_in_set_is_empty() -> None:
    """Test a categorical param whose in-set allows only outside values is empty."""
    param = create_categorical_param(["a", "b"], name=mock_identifier("x", 8))
    # An in-set constraint that permits only a value the domain does not admit.
    # This constructs without error; the emptiness is only discoverable by query.
    narrowed = param.add_constraint(InSetConstraint(param.variable, {"z"}))

    assert not narrowed.is_feasible()
    assert narrowed.is_empty()


def test_integer_param_with_contradictory_bounds_is_empty() -> None:
    """Test a plain integer param with contradictory bounds is empty."""
    param = create_integer_param(name=mock_identifier("x", 9))
    # A plain integer domain permits arbitrary bound constraints, so these
    # jointly-empty bounds construct without error.
    narrowed = param.add_lower_bound_constraint(10).add_upper_bound_constraint(5)

    assert not narrowed.is_feasible()
    assert narrowed.is_empty()


def test_ordinal_param_excluding_all_members_is_empty() -> None:
    """Test an ordinal param whose not-in-set excludes every member is empty."""
    param = create_ordinal_param([1, 2, 3], name=mock_identifier("x", 10))
    narrowed = param.add_constraint(NotInSetConstraint(param.variable, {1, 2, 3}))

    assert not narrowed.is_feasible()
    assert narrowed.is_empty()


def test_permutation_param_excluding_all_permutations_is_empty() -> None:
    """Test a permutation param excluding every permutation is empty."""
    param = create_permutation_param(["a", "b"], name=mock_identifier("x", 11))
    narrowed = param.add_constraint(
        NotInSetConstraint(param.variable, {("a", "b"), ("b", "a")})
    )

    assert not narrowed.is_feasible()
    assert narrowed.is_empty()
