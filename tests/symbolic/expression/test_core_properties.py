"""Hypothesis property tests for expression-tree core laws.

Covers ``Expression.substitute``'s free-identifier specification (P6);
that ``is_structurally_equivalent`` and ``is_alpha_equivalent`` are
equivalence-relation-shaped (reflexive, symmetric), that structural
equivalence (a DICT round trip) implies alpha equivalence, that
renaming every free identifier breaks both unless declared through an
explicit free-renaming bijection; and that ``build_literal_equivalence_key``
is constant on the weak literal classes its docstring documents (P11).
"""

import pytest

pytest.importorskip("hypothesis")

from collections.abc import Sequence

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    Expression,
    IdentifierExpression,
    build_literal_equivalence_key,
)
from fhy_core.term import AlphaRenaming

from ...strategies.expressions import (
    build_numeric_expression_strategy,
    build_structural_expression_strategy,
)
from ...strategies.identifiers import build_identifier_pool
from ...strategies.literals import build_decimal_string_value_strategy
from .conftest import mock_identifier

pytestmark = pytest.mark.property

_POOL = build_identifier_pool(3)
_STRUCTURAL_MAX_LEAVES = 6

# A second pool with ids well above `_POOL`'s (10000-10002): calling
# `build_identifier_pool(3, name_prefix="w")` again would reuse ids
# 10000-10002 and alias `_POOL` (mock identifiers compare by id alone), so
# the renaming law below needs a pool built directly with an offset base.
_FRESH_POOL = tuple(mock_identifier(f"w{index}", 20_000 + index) for index in range(3))


# =============================================================================
# P6: substitute's free-identifier specification
# =============================================================================


@st.composite
def draw_expression_and_substitution(
    draw: st.DrawFn, identifiers: Sequence[Identifier], max_leaves: int = 6
) -> tuple[Expression, dict[Identifier, Expression]]:
    """Draw a numeric gate tree and a substitution over one or two pool identifiers.

    Every substituted value is itself a numeric gate tree over the same
    pool, so the substitution can both remove and reintroduce free
    identifiers, exercising both sides of the free-identifier law below.
    """
    expression = draw(build_numeric_expression_strategy(identifiers, max_leaves))
    domain_size = draw(st.integers(min_value=1, max_value=2))
    domain = draw(
        st.lists(
            st.sampled_from(identifiers),
            unique=True,
            min_size=domain_size,
            max_size=domain_size,
        )
    )
    substitution: dict[Identifier, Expression] = {
        variable: draw(build_numeric_expression_strategy(identifiers, max_leaves))
        for variable in domain
    }
    return expression, substitution


@given(draw_expression_and_substitution(_POOL))
def test_substitute_updates_free_identifiers_per_specification(
    pair: tuple[Expression, dict[Identifier, Expression]],
) -> None:
    """Test substitute's free-identifier law against set algebra on the test side.

    ``e.substitute(s).get_free_identifiers()`` must equal
    ``(free(e) - dom(s)) | union(free(s[v]) for v in free(e) & dom(s))``.
    Both sides are frozensets of mock identifiers, which hash and compare
    by id.
    """
    expression, substitution = pair
    free_before = expression.get_free_identifiers()
    domain = frozenset(substitution.keys())
    expected = free_before - domain
    for variable in free_before & domain:
        expected |= substitution[variable].get_free_identifiers()

    result = expression.substitute(substitution).get_free_identifiers()

    assert result == expected


# =============================================================================
# P11a/b: is_structurally_equivalent and is_alpha_equivalent are reflexive
# and symmetric
# =============================================================================


@given(build_structural_expression_strategy(_POOL, _STRUCTURAL_MAX_LEAVES))
def test_structural_and_alpha_equivalence_are_reflexive(expression: Expression) -> None:
    """Test is_structurally_equivalent and is_alpha_equivalent are reflexive."""
    assert expression.is_structurally_equivalent(expression)
    assert expression.is_alpha_equivalent(expression)


@given(
    build_structural_expression_strategy(_POOL, _STRUCTURAL_MAX_LEAVES),
    build_structural_expression_strategy(_POOL, _STRUCTURAL_MAX_LEAVES),
)
def test_structural_and_alpha_equivalence_are_symmetric(
    left: Expression, right: Expression
) -> None:
    """Test is_structurally_equivalent and is_alpha_equivalent are symmetric.

    Holds regardless of whether the two independently-drawn trees turn
    out equivalent or not: the boolean each relation reports must agree
    in both directions.
    """
    assert left.is_structurally_equivalent(right) == right.is_structurally_equivalent(
        left
    )
    assert left.is_alpha_equivalent(right) == right.is_alpha_equivalent(left)


# =============================================================================
# P11c: structural equivalence (a DICT round trip) implies alpha equivalence
# =============================================================================


@given(build_structural_expression_strategy(_POOL, _STRUCTURAL_MAX_LEAVES))
def test_dict_round_trip_is_structurally_and_therefore_alpha_equivalent(
    expression: Expression,
) -> None:
    """Test a DICT round trip is structurally equivalent to the original.

    Oracle: the specification that structural equivalence implies alpha
    equivalence, so the restored tree must also be alpha-equivalent.
    """
    restored = Expression.deserialize_from_dict(expression.serialize_to_dict())

    assert restored.is_structurally_equivalent(expression)
    assert restored.is_alpha_equivalent(expression)


# =============================================================================
# P11d: renaming every free identifier
# =============================================================================


@given(build_structural_expression_strategy(_POOL, _STRUCTURAL_MAX_LEAVES))
def test_renaming_free_identifiers_holds_only_under_a_declared_free_renaming(
    expression: Expression,
) -> None:
    """Test substituting every pool identifier with a fresh one.

    Oracle: the specification in
    ``fhy_core.term.alpha_equivalence.AlphaRenaming``. The renamed tree
    is always alpha-equivalent under the declared free-renaming
    bijection. When ``expression`` has at least one free identifier, the
    renaming also changes at least one leaf, so the renamed tree is
    neither structurally equivalent nor plainly alpha-equivalent (which
    compares free identifiers by identity); with no free identifiers,
    substitution changes nothing and all three hold.
    """
    rename_map = dict(zip(_POOL, _FRESH_POOL, strict=True))
    substitution: dict[Identifier, Expression] = {
        pool_identifier: IdentifierExpression(fresh_identifier)
        for pool_identifier, fresh_identifier in rename_map.items()
    }
    renamed = expression.substitute(substitution)

    is_structurally_equal = expression.is_structurally_equivalent(renamed)
    is_plainly_alpha_equal = expression.is_alpha_equivalent(renamed)
    is_alpha_equal_under_free_renaming = expression.is_alpha_equivalent_under(
        renamed, AlphaRenaming.with_free_renaming(rename_map)
    )

    assert is_alpha_equal_under_free_renaming
    if expression.get_free_identifiers():
        assert not is_structurally_equal
        assert not is_plainly_alpha_equal
    else:
        assert is_structurally_equal
        assert is_plainly_alpha_equal


# =============================================================================
# P11e: build_literal_equivalence_key is constant on weak literal classes
# =============================================================================


@given(
    value=st.integers(min_value=0, max_value=1000),
    padding=st.integers(min_value=0, max_value=5),
)
def test_literal_equivalence_key_agrees_for_int_and_digit_string_forms(
    value: int, padding: int
) -> None:
    """Test the key is constant across an int and its (zero-padded) digit string.

    Restricted to non-negative ``value``: the integer grammar
    ``build_literal_equivalence_key`` recognizes for a string is
    unsigned digits only (``LiteralExpression``'s own grammar,
    ``core.py`` around line 914), so ``str(value)`` for a negative
    ``value`` would not land in the integer bucket at all.
    """
    zero_padded = ("0" * padding) + str(value)

    integer_key = build_literal_equivalence_key(value)

    assert integer_key == build_literal_equivalence_key(str(value))
    assert integer_key == build_literal_equivalence_key(zero_padded)


@given(
    base=build_decimal_string_value_strategy(),
    extra_zeros=st.integers(min_value=0, max_value=5),
)
def test_literal_equivalence_key_is_unchanged_by_trailing_decimal_zeros(
    base: str, extra_zeros: int
) -> None:
    """Test appending zeros after the decimal point does not change the key.

    Every string ``build_decimal_string_value_strategy`` draws already
    ends after its decimal point (with a trailing digit or the point
    itself), so appending zeros at the end is exactly "appending zeros
    after the decimal point".
    """
    padded = base + ("0" * extra_zeros)

    assert build_literal_equivalence_key(base) == build_literal_equivalence_key(padded)


def test_literal_equivalence_key_treats_signed_and_unsigned_zero_as_equal() -> None:
    """Test 0.0 and -0.0 share an equivalence key.

    Documented explicitly in ``build_literal_equivalence_key``'s
    docstring: a binary float gains zero before classification, folding
    ``-0.0`` into the ``0.0`` it already equals.
    """
    assert build_literal_equivalence_key(0.0) == build_literal_equivalence_key(-0.0)
