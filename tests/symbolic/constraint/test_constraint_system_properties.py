"""Hypothesis property tests for `ConstraintSystem`.

Kept in a dedicated module so a test environment without hypothesis
installed (the CI `tests` lane syncs only the `test` dependency group)
can still collect this package cleanly: the `pytest.importorskip` below
skips the whole module before the `hypothesis` import is attempted.
"""

import pytest

pytest.importorskip("hypothesis")
from hypothesis import example, given
from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintMember,
    ConstraintOutcome,
    ConstraintSystem,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
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

from ...strategies.constraints import (
    build_integer_bindings_strategy,
    draw_bound_equation_constraint,
    draw_constraint_system,
    draw_integer_set_constraint,
)
from ...strategies.identifiers import build_identifier_pool
from ...strategies.settings import cap_max_examples
from .conftest import mock_identifier

pytestmark = pytest.mark.property


# =============================================================================
# Conjunction outcome folds each member's own outcome
# =============================================================================

_PINNED_CONJUNCTION_X = mock_identifier("x", 0)
_PINNED_CONJUNCTION_Y = mock_identifier("y", 1)
_PINNED_CONJUNCTION_MEMBERS: tuple[Constraint, ...] = (
    InSetConstraint(_PINNED_CONJUNCTION_X, {1, 2, 3, 4}),
    InSetConstraint(_PINNED_CONJUNCTION_Y, {0, 1, 2}),
    EquationConstraint(
        make_binary_expression(
            BinaryOperation.LESS, _PINNED_CONJUNCTION_X, _PINNED_CONJUNCTION_Y
        )
    ),
)
_PINNED_CONJUNCTION_SYSTEM = create_constraint_system(*_PINNED_CONJUNCTION_MEMBERS)
"""A hand-picked system pinned as an example for the fold law below."""


@st.composite
def _draw_system_and_bindings(
    draw: st.DrawFn,
) -> tuple[ConstraintSystem, tuple[Constraint, ...], dict[Identifier, int]]:
    """Draw a constraint system, its members, and bindings over the same pool.

    The pool has 2 or 3 identifiers. Bindings cover every pool identifier
    with an integer, matching the full-assignment shape the fold law is
    checked against.
    """
    pool = build_identifier_pool(draw(st.sampled_from((2, 3))))
    system, members = draw(draw_constraint_system(pool))
    bindings = draw(build_integer_bindings_strategy(pool))
    return system, members, bindings


@example(
    drawn=(
        _PINNED_CONJUNCTION_SYSTEM,
        _PINNED_CONJUNCTION_MEMBERS,
        {_PINNED_CONJUNCTION_X: 1, _PINNED_CONJUNCTION_Y: 2},
    )
)
@example(
    drawn=(
        _PINNED_CONJUNCTION_SYSTEM,
        _PINNED_CONJUNCTION_MEMBERS,
        {_PINNED_CONJUNCTION_X: 5, _PINNED_CONJUNCTION_Y: 2},
    )
)
@given(drawn=_draw_system_and_bindings())
def test_evaluate_with_bindings_matches_fold_of_member_outcomes(
    drawn: tuple[ConstraintSystem, tuple[Constraint, ...], dict[Identifier, int]],
) -> None:
    """Test the conjunction outcome matches folding each member's own outcome."""
    system, members, bindings = drawn

    outcome = system.evaluate_with_bindings(bindings)

    member_outcomes = [member.evaluate_with_bindings(bindings) for member in members]
    if any(o is ConstraintOutcome.VIOLATED for o in member_outcomes):
        expected = ConstraintOutcome.VIOLATED
    elif all(o is ConstraintOutcome.SATISFIED for o in member_outcomes):
        expected = ConstraintOutcome.SATISFIED
    else:
        expected = ConstraintOutcome.UNDECIDED
    assert outcome is expected
    assert system.is_satisfied_with_bindings(bindings) == (
        outcome is ConstraintOutcome.SATISFIED
    )


# =============================================================================
# Z3-backed satisfiability matches brute-force enumeration
# =============================================================================

_SAT_DOMAIN_LIMIT = 6
_SAT_V0, _SAT_V1 = build_identifier_pool(2, name_prefix="sat_v")

_PINNED_THRESHOLD_LINKED_DOMAIN = tuple(range(_SAT_DOMAIN_LIMIT))
_PINNED_THRESHOLD = 3
_PINNED_THRESHOLD_LINKED_SYSTEM = create_constraint_system(
    InSetConstraint(_SAT_V0, set(_PINNED_THRESHOLD_LINKED_DOMAIN)),
    InSetConstraint(_SAT_V1, set(_PINNED_THRESHOLD_LINKED_DOMAIN)),
    EquationConstraint(
        make_binary_expression(
            BinaryOperation.EQUAL,
            make_binary_expression(BinaryOperation.ADD, _SAT_V0, _PINNED_THRESHOLD),
            _SAT_V1,
        )
    ),
)
"""A hand-picked threshold-linked system pinned as an example below."""


def _build_domain_strategy() -> st.SearchStrategy[tuple[int, ...]]:
    """Return a strategy for a non-empty subset of range(_SAT_DOMAIN_LIMIT)."""
    return st.lists(
        st.integers(min_value=0, max_value=_SAT_DOMAIN_LIMIT - 1),
        min_size=1,
        max_size=_SAT_DOMAIN_LIMIT,
        unique=True,
    ).map(tuple)


@st.composite
def _draw_further_member(draw: st.DrawFn) -> Constraint:
    """Draw one extra member: a bound equation, an integer set constraint, or a link.

    The link is an equation of the form ``_SAT_V0 + c == _SAT_V1``, the
    same shape as the pinned threshold-linked example below, with ``c``
    drawn here instead of fixed.
    """
    kind = draw(st.integers(min_value=0, max_value=2))
    if kind == 0:
        variable = draw(st.sampled_from((_SAT_V0, _SAT_V1)))
        return draw(draw_bound_equation_constraint(variable, limit=_SAT_DOMAIN_LIMIT))
    if kind == 1:
        variable = draw(st.sampled_from((_SAT_V0, _SAT_V1)))
        return draw(draw_integer_set_constraint(variable, limit=_SAT_DOMAIN_LIMIT))
    offset = draw(
        st.integers(min_value=-_SAT_DOMAIN_LIMIT, max_value=_SAT_DOMAIN_LIMIT)
    )
    return EquationConstraint(
        make_binary_expression(
            BinaryOperation.EQUAL,
            make_binary_expression(BinaryOperation.ADD, _SAT_V0, offset),
            _SAT_V1,
        )
    )


@st.composite
def _draw_domain_bound_system(
    draw: st.DrawFn,
) -> tuple[ConstraintSystem, tuple[int, ...], tuple[int, ...]]:
    """Draw a system with every pool variable in-set-constrained plus 1 to 3 more.

    ``_SAT_V0`` and ``_SAT_V1`` each carry an ``InSetConstraint`` over a
    drawn non-empty subset of ``range(_SAT_DOMAIN_LIMIT)``, built in
    directly rather than left to chance so the brute-force enumeration
    below always has a finite, non-empty domain to search.
    """
    domain0 = draw(_build_domain_strategy())
    domain1 = draw(_build_domain_strategy())
    further_count = draw(st.integers(min_value=1, max_value=3))
    further = [draw(_draw_further_member()) for _ in range(further_count)]
    system = create_constraint_system(
        InSetConstraint(_SAT_V0, set(domain0)),
        InSetConstraint(_SAT_V1, set(domain1)),
        *further,
    )
    return system, domain0, domain1


@pytest.mark.z3
# Z3-backed: check_satisfiability routes through the solver.
@cap_max_examples(50)
@example(
    drawn=(
        _PINNED_THRESHOLD_LINKED_SYSTEM,
        _PINNED_THRESHOLD_LINKED_DOMAIN,
        _PINNED_THRESHOLD_LINKED_DOMAIN,
    )
)
@given(drawn=_draw_domain_bound_system())
def test_check_satisfiability_matches_brute_force_enumeration(
    drawn: tuple[ConstraintSystem, tuple[int, ...], tuple[int, ...]],
) -> None:
    """Test z3-backed satisfiability agrees with brute-force enumeration."""
    system, domain0, domain1 = drawn

    brute_force_satisfiable = any(
        system.evaluate_with_bindings({_SAT_V0: a, _SAT_V1: b})
        is ConstraintOutcome.SATISFIED
        for a in domain0
        for b in domain1
    )
    outcome = system.check_satisfiability(
        {_SAT_V0: SymbolType.INT, _SAT_V1: SymbolType.INT}
    )

    assert outcome is not ConstraintOutcome.UNDECIDED, (
        f"check_satisfiability was UNDECIDED for system {system!r} over domains "
        f"{domain0!r} x {domain1!r}, but every member is fully bound over a "
        "finite integer domain, so brute force always decides it"
    )
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
    variable: Identifier, form: LiteralType, wrap_in_comparison: bool
) -> Constraint:
    expression: Expression = LiteralExpression(form)
    if wrap_in_comparison:
        expression = make_binary_expression(BinaryOperation.EQUAL, variable, expression)
    return EquationConstraint(expression)


@st.composite
def _draw_constraint_pair(draw: st.DrawFn) -> tuple[Constraint, Constraint]:
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
    assert left.build_ordering_key() == right.build_ordering_key()
