"""Hypothesis property tests for `ConstraintSystem`.

Kept in a dedicated module so a test environment without hypothesis
installed (the CI `tests` lane syncs only the `test` dependency group)
can still collect this package cleanly: the `pytest.importorskip` below
skips the whole module before the `hypothesis` import is attempted.
"""

from typing import Any

import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st

from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintMember,
    ConstraintOutcome,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
    build_constraint_ordering_key,
    create_constraint_system,
)
from fhy_core.symbolic.expression import (
    BinaryOperation,
    Expression,
    LiteralExpression,
    LiteralType,
    make_binary_expression,
)
from fhy_core.symbolic.symbol_type import SymbolType

from .conftest import mock_identifier

pytestmark = pytest.mark.property


@settings(max_examples=50, deadline=None)
@given(
    x_value=st.integers(min_value=-5, max_value=10),
    y_value=st.integers(min_value=-5, max_value=10),
)
def test_evaluate_with_bindings_matches_fold_of_member_outcomes(
    x_value: int, y_value: int
) -> None:
    """Test the conjunction outcome matches folding each member's own outcome."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    members: tuple[Constraint, ...] = (
        InSetConstraint(x, {1, 2, 3, 4}),
        InSetConstraint(y, {0, 1, 2}),
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, y)),
    )
    system = create_constraint_system(*members)
    bindings = {x: x_value, y: y_value}

    outcome = system.evaluate_with_bindings(bindings)

    member_outcomes = [member.evaluate_with_bindings(bindings) for member in members]
    if any(o is ConstraintOutcome.VIOLATED for o in member_outcomes):
        expected = ConstraintOutcome.VIOLATED
    elif all(o is ConstraintOutcome.SATISFIED for o in member_outcomes):
        expected = ConstraintOutcome.SATISFIED
    else:
        expected = ConstraintOutcome.UNDECIDED
    assert outcome is expected


@pytest.mark.z3
@settings(max_examples=50, deadline=None)
@given(threshold=st.integers(min_value=0, max_value=10))
def test_check_satisfiability_matches_brute_force_enumeration(threshold: int) -> None:
    """Test z3-backed satisfiability agrees with brute-force enumeration."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    domain = tuple(range(6))
    link_expression = make_binary_expression(
        BinaryOperation.EQUAL,
        make_binary_expression(BinaryOperation.ADD, x, threshold),
        y,
    )
    system = create_constraint_system(
        InSetConstraint(x, set(domain)),
        InSetConstraint(y, set(domain)),
        EquationConstraint(link_expression),
    )

    brute_force_satisfiable = any(a + threshold == b for a in domain for b in domain)

    outcome = system.check_satisfiability({x: SymbolType.INT, y: SymbolType.INT})

    expected = (
        ConstraintOutcome.SATISFIED
        if brute_force_satisfiable
        else ConstraintOutcome.VIOLATED
    )
    assert outcome is expected


# =============================================================================
# Canonical ordering key is constant on structural-equivalence classes
# =============================================================================

_VARIABLE_IDS = (0, 1)

# Weak literal-equivalence classes: every form within a tuple builds a
# `LiteralExpression` structurally equivalent to every other form in the same
# tuple. ``5``/``"5"``/``"05"`` share the integer bucket, ``"1.5"``/``"1.50"``
# the exact-decimal bucket, and ``0.0``/``-0.0`` the binary-float bucket.
_LITERAL_EQUIVALENCE_CLASSES: tuple[tuple[LiteralType, ...], ...] = (
    (5, "5", "05"),
    (4, "4", "0004"),
    ("1.5", "1.50", "1.500"),
    (1.5,),
    (0.0, -0.0),
    (True,),
    (False,),
)

# Members chosen so the type-strict classes are adjacent: ``1``, ``"1"``,
# ``1.0``, and ``True`` are four distinct members, not one.
_MEMBER_FORMS: tuple[ConstraintMember, ...] = (
    1,
    "1",
    1.0,
    True,
    2,
    (1, 2),
    frozenset({1}),
)

_SET_KINDS = (InSetConstraint, NotInSetConstraint)


def _build_literal_equation(
    variable: Any, form: LiteralType, wrap_in_comparison: bool
) -> Constraint:
    expression: Expression = LiteralExpression(form)
    if wrap_in_comparison:
        expression = make_binary_expression(BinaryOperation.EQUAL, variable, expression)
    return EquationConstraint(expression)


@st.composite
def _draw_constraint_pair(draw: Any) -> tuple[Constraint, Constraint]:
    """Draw two constraints of one shape, built from independently chosen forms.

    Both sides get their own ``mock_identifier`` for the drawn id, so the
    pair also exercises identifier keying by ``id`` rather than by object
    identity. The equation branch varies the literal form within one weak
    equivalence class; the set branch varies member order.
    """
    variable_id = draw(st.sampled_from(_VARIABLE_IDS))
    left_variable = mock_identifier("v", variable_id)
    right_variable = mock_identifier("v", variable_id)
    kind_index = draw(st.integers(min_value=0, max_value=2))
    if kind_index == 0:
        forms = draw(st.sampled_from(_LITERAL_EQUIVALENCE_CLASSES))
        wrap_in_comparison = draw(st.booleans())
        return (
            _build_literal_equation(
                left_variable, draw(st.sampled_from(forms)), wrap_in_comparison
            ),
            _build_literal_equation(
                right_variable, draw(st.sampled_from(forms)), wrap_in_comparison
            ),
        )
    kind = _SET_KINDS[kind_index - 1]
    members = draw(st.lists(st.sampled_from(_MEMBER_FORMS), min_size=0, max_size=3))
    shuffled = draw(st.permutations(members))
    return kind(left_variable, members), kind(right_variable, shuffled)


@settings(max_examples=50, deadline=None)
@given(pair=_draw_constraint_pair())
def test_ordering_key_is_constant_on_structural_equivalence_classes(
    pair: tuple[Constraint, Constraint],
) -> None:
    """Test the canonical ordering key agrees on structurally equivalent members.

    This is the invariant that makes ``ConstraintSystem``'s canonical
    member order well defined: sorting by a key that separates two
    equivalent members would leave two equivalent systems in different
    member orders, and hence not equivalent themselves.
    """
    left, right = pair
    if not left.is_structurally_equivalent(right):
        return
    assert build_constraint_ordering_key(left) == build_constraint_ordering_key(right)
