"""Alpha-equivalence tests for `Param` and `ParamAssignment`.

`Param.variable` is a binder that scopes over the ``constraints`` field, so two
params compare alpha-equivalent up to renaming of their bound variable while
remaining structurally distinct (structural equivalence compares the variable
nominally). These tests pin down that split across the numeric and non-numeric
kinds, verify the canonical constraint sort preserves position-wise
correspondence under renaming, cover the negatives, and confirm the recursion
into `ParamAssignment`.
"""

from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import EquationConstraint
from fhy_core.symbolic.param import (
    Param,
    create_categorical_param,
    create_integer_param,
    create_natural_param,
    create_ordinal_param,
    create_permutation_param,
    create_real_param,
)

from .conftest import mock_identifier


def _build_integer_param_with_bounds(
    name: Identifier,
    bound_builders: list[Callable[[Param[int]], EquationConstraint]],
) -> Param[int]:
    """Build an integer param whose constraints are added in list order.

    Each builder receives the base param so its constraint references the
    param's own variable.
    """
    base = create_integer_param(name=name)
    return base.add_constraints([build(base) for build in bound_builders])


# =============================================================================
# Structural equivalence is unchanged by the binder role
# =============================================================================


def test_structural_equivalence_still_false_when_variables_differ() -> None:
    """Test the binder still compares variables nominally in structural mode.

    Marking ``variable`` as a binder must not relax structural equivalence:
    two constraint-free params with different variables stay structurally
    distinct exactly as they did under the reference role.
    """
    left = create_integer_param(name=mock_identifier("x", 1))
    right = create_integer_param(name=mock_identifier("y", 2))

    assert not left.is_structurally_equivalent(right)


def test_structural_equivalence_still_true_for_equal_variable_copies() -> None:
    """Test params sharing a variable id remain structurally equivalent.

    A separately-constructed identifier with the same id is equal but
    distinct, standing in for a serialize/deserialize round-tripped copy.
    """
    left = create_integer_param(name=mock_identifier("x", 1))
    right = create_integer_param(name=mock_identifier("x", 1))

    assert left.is_structurally_equivalent(right)


# =============================================================================
# Alpha-equivalence holds across variable renaming
# =============================================================================


def test_alpha_equivalent_across_variable_rename_with_one_constraint() -> None:
    """Test ``x in Z | x > 0`` is alpha-equivalent to ``y in Z | y > 0``.

    Same domain and same constraint structure, different variables: the two
    variables bind to each other and the ``> 0`` constraint compares under
    the renaming. Alpha-equivalent, but structurally distinct.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    left = _build_integer_param_with_bounds(
        x, [lambda p: EquationConstraint(p.variable, p.variable_expression > 0)]
    )
    right = _build_integer_param_with_bounds(
        y, [lambda p: EquationConstraint(p.variable, p.variable_expression > 0)]
    )

    assert left.is_alpha_equivalent(right)
    assert not left.is_structurally_equivalent(right)


def test_alpha_equivalent_for_constraint_free_params_with_different_variables() -> None:
    """Test two constraint-free params with different variables are alpha-equivalent.

    With no constraints, only the domain (compared structurally regardless)
    and the bound variable remain; the variables rename onto each other.
    """
    left = create_integer_param(name=mock_identifier("x", 1))
    right = create_integer_param(name=mock_identifier("y", 2))

    assert left.is_alpha_equivalent(right)
    assert not left.is_structurally_equivalent(right)


def test_alpha_equivalent_renames_implied_natural_constraint() -> None:
    """Test the implied ``variable >= 0`` bound renames under alpha comparison.

    A natural param carries an implied non-negativity constraint that
    references the variable. Two natural params with different variables are
    alpha-equivalent only if that implied constraint is renamed too.
    """
    left = create_natural_param(name=mock_identifier("x", 1))
    right = create_natural_param(name=mock_identifier("y", 2))

    assert left.is_alpha_equivalent(right)
    assert not left.is_structurally_equivalent(right)


# =============================================================================
# Canonical-sort safety under renaming
# =============================================================================


@pytest.mark.parametrize(
    "left_order, right_order",
    [
        pytest.param((0, 1, 2), (0, 1, 2), id="same-order"),
        pytest.param((0, 1, 2), (2, 1, 0), id="reversed"),
        pytest.param((0, 1, 2), (1, 2, 0), id="rotated"),
        pytest.param((0, 1, 2), (2, 0, 1), id="permuted"),
    ],
)
def test_alpha_equivalent_with_multiple_constraints_in_different_orders(
    left_order: tuple[int, int, int], right_order: tuple[int, int, int]
) -> None:
    """Test multi-constraint params stay alpha-equivalent under reordering.

    Within a param all constraints share the variable, so the variable is a
    constant component of the canonical sort key and the ``__post_init__``
    sort yields the same position-wise ordering on both sides regardless of
    insertion order. Renaming then aligns the constraints element-wise.
    """
    builders: list[Callable[[Param[int]], EquationConstraint]] = [
        lambda p: EquationConstraint(p.variable, p.variable_expression > 0),
        lambda p: EquationConstraint(p.variable, p.variable_expression < 100),
        lambda p: EquationConstraint(p.variable, p.variable_expression >= 3),
    ]
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    left = _build_integer_param_with_bounds(x, [builders[i] for i in left_order])
    right = _build_integer_param_with_bounds(y, [builders[i] for i in right_order])

    assert left.is_alpha_equivalent(right)
    assert not left.is_structurally_equivalent(right)


# =============================================================================
# Negatives
# =============================================================================


def test_not_alpha_equivalent_for_different_domains() -> None:
    """Test an integer param and a real param are not alpha-equivalent.

    The domain has no identifiers and is compared structurally in both modes,
    so distinct value spaces are never reconciled by renaming.
    """
    integer = create_integer_param(name=mock_identifier("x", 1))
    real = create_real_param(name=mock_identifier("y", 2))

    assert not integer.is_alpha_equivalent(real)


def test_not_alpha_equivalent_for_differing_domain_flags() -> None:
    """Test a natural param is not alpha-equivalent to a plain integer param.

    Both are integer-valued, but the non-negativity flag differs (the natural
    param carries an implied ``>= 0`` the plain integer param lacks), so
    renaming cannot make them equal.
    """
    natural = create_natural_param(name=mock_identifier("x", 1))
    plain = create_integer_param(name=mock_identifier("y", 2))

    assert not natural.is_alpha_equivalent(plain)


def test_not_alpha_equivalent_for_differing_zero_inclusion_flags() -> None:
    """Test natural params with different zero-inclusion are not alpha-equivalent.

    ``zero_included`` changes the implied bound (``>= 0`` versus ``> 0``), so
    the constraint sets differ under any renaming.
    """
    zero_included = create_natural_param(
        name=mock_identifier("x", 1), zero_included=True
    )
    zero_excluded = create_natural_param(
        name=mock_identifier("y", 2), zero_included=False
    )

    assert not zero_included.is_alpha_equivalent(zero_excluded)


def test_not_alpha_equivalent_for_different_constraint_sets() -> None:
    """Test params with structurally different constraints are not alpha-equivalent.

    Same domain, renamable variable, but the bound literals differ (``> 0``
    versus ``> 5``); no renaming of the variable reconciles the literals.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    left = _build_integer_param_with_bounds(
        x, [lambda p: EquationConstraint(p.variable, p.variable_expression > 0)]
    )
    right = _build_integer_param_with_bounds(
        y, [lambda p: EquationConstraint(p.variable, p.variable_expression > 5)]
    )

    assert not left.is_alpha_equivalent(right)


def test_not_alpha_equivalent_for_different_constraint_counts() -> None:
    """Test params with different constraint counts are not alpha-equivalent."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    left = _build_integer_param_with_bounds(
        x,
        [
            lambda p: EquationConstraint(p.variable, p.variable_expression > 0),
            lambda p: EquationConstraint(p.variable, p.variable_expression < 100),
        ],
    )
    right = _build_integer_param_with_bounds(
        y, [lambda p: EquationConstraint(p.variable, p.variable_expression > 0)]
    )

    assert not left.is_alpha_equivalent(right)


def test_not_alpha_equivalent_for_non_param_other() -> None:
    """Test alpha-equivalence returns ``False`` (not raises) for a non-`Param`."""
    param = create_integer_param(name=mock_identifier("x", 1))

    assert not param.is_alpha_equivalent(object())
    assert not param.is_alpha_equivalent("not a param")


# =============================================================================
# Non-numeric kinds
# =============================================================================


def test_ordinal_params_alpha_equivalent_across_variable_rename() -> None:
    """Test ordinal params with equal value sets but different variables match.

    The ordered value set lives in the domain (no identifiers), so different
    variables rename onto each other and the params are alpha-equivalent yet
    structurally distinct.
    """
    left = create_ordinal_param([1, 2, 3], name=mock_identifier("x", 1))
    right = create_ordinal_param([1, 2, 3], name=mock_identifier("y", 2))

    assert left.is_alpha_equivalent(right)
    assert not left.is_structurally_equivalent(right)


def test_ordinal_params_not_alpha_equivalent_for_different_value_sets() -> None:
    """Test ordinal params with different value sets are not alpha-equivalent."""
    left = create_ordinal_param([1, 2, 3], name=mock_identifier("x", 1))
    right = create_ordinal_param([1, 2, 4], name=mock_identifier("y", 2))

    assert not left.is_alpha_equivalent(right)


def test_categorical_params_alpha_equivalent_across_variable_rename() -> None:
    """Test categorical params with equal categories and different variables match."""
    left = create_categorical_param({"a", "b", "c"}, name=mock_identifier("x", 1))
    right = create_categorical_param({"a", "b", "c"}, name=mock_identifier("y", 2))

    assert left.is_alpha_equivalent(right)
    assert not left.is_structurally_equivalent(right)


def test_categorical_params_not_alpha_equivalent_for_different_categories() -> None:
    """Test categorical params with different category sets are not alpha-equivalent."""
    left = create_categorical_param({"a", "b"}, name=mock_identifier("x", 1))
    right = create_categorical_param({"a", "c"}, name=mock_identifier("y", 2))

    assert not left.is_alpha_equivalent(right)


def test_permutation_params_alpha_equivalent_across_variable_rename() -> None:
    """Test permutation params with equal members and different variables match."""
    left = create_permutation_param(["n", "c", "h", "w"], name=mock_identifier("x", 1))
    right = create_permutation_param(["n", "c", "h", "w"], name=mock_identifier("y", 2))

    assert left.is_alpha_equivalent(right)
    assert not left.is_structurally_equivalent(right)


def test_permutation_params_not_alpha_equivalent_for_different_members() -> None:
    """Test permutation params with different members are not alpha-equivalent."""
    left = create_permutation_param(["n", "c"], name=mock_identifier("x", 1))
    right = create_permutation_param(["h", "w"], name=mock_identifier("y", 2))

    assert not left.is_alpha_equivalent(right)


# =============================================================================
# Consistency: reflexivity and structural-implies-alpha
# =============================================================================


def test_alpha_equivalence_is_reflexive() -> None:
    """Test every param kind is alpha-equivalent to itself."""
    x = mock_identifier("x", 1)
    params: list[Param[Any]] = [
        create_integer_param(name=x),
        create_natural_param(name=x),
        create_real_param(name=x),
        create_ordinal_param([1, 2, 3], name=x),
        create_categorical_param({"a", "b"}, name=x),
        create_permutation_param(["n", "c"], name=x),
    ]

    for param in params:
        assert param.is_alpha_equivalent(param)


def test_structural_equivalence_implies_alpha_equivalence() -> None:
    """Test structurally equivalent params are also alpha-equivalent.

    Two params sharing a variable id and constraint structure are
    structurally equivalent; alpha comparison with an empty renaming and the
    reflexive binding must agree.
    """
    left = _build_integer_param_with_bounds(
        mock_identifier("x", 1),
        [lambda p: EquationConstraint(p.variable, p.variable_expression > 0)],
    )
    right = _build_integer_param_with_bounds(
        mock_identifier("x", 1),
        [lambda p: EquationConstraint(p.variable, p.variable_expression > 0)],
    )

    assert left.is_structurally_equivalent(right)
    assert left.is_alpha_equivalent(right)


# =============================================================================
# ParamAssignment recursion
# =============================================================================


def test_param_assignments_alpha_equivalent_for_alpha_equivalent_params() -> None:
    """Test assignments recurse into the param and inherit alpha-equivalence.

    Two assignments over alpha-equivalent params bound to the same value are
    alpha-equivalent (the param field recurses) but structurally distinct
    (the underlying params carry different variables).
    """
    left_param = _build_integer_param_with_bounds(
        mock_identifier("x", 1),
        [lambda p: EquationConstraint(p.variable, p.variable_expression > 0)],
    )
    right_param = _build_integer_param_with_bounds(
        mock_identifier("y", 2),
        [lambda p: EquationConstraint(p.variable, p.variable_expression > 0)],
    )
    left = left_param.assign(5)
    right = right_param.assign(5)

    assert left.is_alpha_equivalent(right)
    assert not left.is_structurally_equivalent(right)


def test_param_assignments_not_alpha_equivalent_for_different_values() -> None:
    """Test assignments with alpha-equivalent params but different values differ.

    The bound value is compared by value regardless of the renaming, so a
    different value breaks alpha-equivalence even when the params match.
    """
    left_param = _build_integer_param_with_bounds(
        mock_identifier("x", 1),
        [lambda p: EquationConstraint(p.variable, p.variable_expression > 0)],
    )
    right_param = _build_integer_param_with_bounds(
        mock_identifier("y", 2),
        [lambda p: EquationConstraint(p.variable, p.variable_expression > 0)],
    )
    left = left_param.assign(5)
    right = right_param.assign(7)

    assert not left.is_alpha_equivalent(right)
